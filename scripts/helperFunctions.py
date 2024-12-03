import re
#import rosbag 
#import sensor_msgs.point_cloud2 as pc2
import itertools
from operator import itemgetter
#from tf.transformations import euler_from_quaternion
import os
# import cupyx.scipy.signal
import numpy as np
import struct
from matplotlib.cm import get_cmap
from scipy import interpolate
# from transform import build_se3_transform
import matplotlib.pyplot as plt 
from scipy import signal, stats, spatial, ndimage 
import matplotlib.animation as animation
import open3d as o3d
import time
import pandas as pd
import matplotlib.patches as patches
from scipy.spatial.distance import cdist
from matplotlib.gridspec import GridSpec
import math
from concurrent.futures import ThreadPoolExecutor
# from velodyne import load_velodyne_raw, load_velodyne_binary, velodyne_raw_to_pointcloud
# from interpolate_poses import interpolate_vo_poses, interpolate_ins_poses
import ast 
import pykitti
import imutils
import random 
import pickle 
import json 
import cupy as cp
from cusignal import fftconvolve
# from cupyx.scipy.signal import oaconvolve
# import cupyx
import cv2
import skimage 
from sklearn.neighbors import KDTree
# import torch
# import torch.nn.functional as F

np.random.seed(1)

class attractorNetwork2D:
    '''defines 1D attractor network with N neurons, angles associated with each neurons 
    along with inhitory and excitatory connections to update the weights'''
    def __init__(self, N1, N2, num_links, excite_radius, activity_mag,inhibit_scale):
        self.excite_radius=int(excite_radius)
        self.num_links=int(num_links)
        self.N1=N1
        self.N2=N2
        self.N=(N1,N2)  
        self.activity_mag=activity_mag
        self.inhibit_scale=inhibit_scale

    def full_weights(self,radius, mx=0, my=0):
        len=(radius*2)+1
        x, y = np.meshgrid(np.linspace(-1,1,len), np.linspace(-1,1,len))
        sigma = 1.0
        return np.exp(-( ((x-mx)**2 + (y-my)**2) / ( 2.0 * sigma**2 ) ) )
    
    def inhibitions(self,weights):
        ''' constant inhibition scaled by amount of active neurons'''
        return np.sum(weights[weights>0]*self.inhibit_scale)

    def excitations(self, idx, idy, scale=1):
        '''A scaled 2D gaussian with excite radius is created at given neuron position with wraparound '''

        excite_rowvals = np.arange(-self.excite_radius, self.excite_radius+1)
        excite_colvals = np.arange(-self.excite_radius, self.excite_radius+1)
        excite_rowvals = (idx + excite_rowvals) % self.N[0]
        excite_colvals = (idy + excite_colvals) % self.N[1]

        gauss = self.full_weights(self.excite_radius)  # 2D gaussian scaled
        excite = np.zeros((self.N[0], self.N[1]))  # empty excite array
        excite[excite_rowvals[:, None], excite_colvals] = gauss
        return excite * scale

    def fractional_shift(self, M, delta_row, delta_col):
        M_new=np.zeros((self.N[0], self.N[1]))
        axiss=[1,0]
        for idx, delta in enumerate([delta_col,delta_row]):
            
            frac = delta % 1
            if frac == 0.0:
                M_new=M
                continue
            
            shift= 1 if delta > 0 else -1
            frac = frac if delta > 0 else (1-frac)
            M_new += (1 - frac) * M + frac * np.roll(M, shift, axis=axiss[idx])

        return M_new / (np.linalg.norm(M_new))
 
    def update_weights_dynamics_row_col(self,prev_weights, delta_row, delta_col):
        non_zero_rows, non_zero_cols=np.nonzero(prev_weights) # indexes of non zero prev_weights
        prev_max_col,prev_max_row=np.argmax(np.max(prev_weights, axis=0)),np.argmax(np.max(prev_weights, axis=1))

        func = lambda x: int(math.ceil(x)) if x < 0 else int(math.floor(x))

        '''copied and shifted activity'''
        full_shift=np.zeros((self.N1,self.N2))
        shifted_row_ids, shifted_col_ids=(non_zero_rows +func(delta_row))%self.N1, (non_zero_cols+ func(delta_col))%self.N2
        full_shift[shifted_row_ids, shifted_col_ids]=prev_weights[non_zero_rows, non_zero_cols]
        copy_shift=self.fractional_shift(full_shift,delta_row,delta_col)*self.activity_mag


        '''excitation'''
        copyPaste=copy_shift
        non_zero_copyPaste=np.nonzero(copyPaste)  
        # print(len(non_zero_copyPaste[0]))
        excited=np.zeros((self.N1,self.N2))
        # t=time.time()
        for row, col in zip(non_zero_copyPaste[0], non_zero_copyPaste[1]):
            excited+=self.excitations(row,col,copyPaste[row,col])
        # print(time.time()-t)
        
        # excited=np.sum(excited_array, axis=0)
        # print(np.shape(excited_array), np.shape(excited))
        '''inhibitions'''
        inhibit_val=0
        shift_excite=copy_shift+prev_weights+excited
        non_zero_inhibit=np.nonzero(shift_excite) 
        for row, col in zip(non_zero_inhibit[0], non_zero_inhibit[1]):
            inhibit_val+=shift_excite[row,col]*self.inhibit_scale
        inhibit_array=np.tile(inhibit_val,(self.N1,self.N2))

        '''update activity'''
        prev_weights+=copy_shift+excited-inhibit_val
        prev_weights[prev_weights<0]=0


        
        return prev_weights/np.linalg.norm(prev_weights) if np.sum(prev_weights) > 0 else [np.nan]
    
    def update_weights_dynamics(self,prev_weights, direction, speed, moreResults=None):
        non_zero_rows, non_zero_cols=np.nonzero(prev_weights) # indexes of non zero prev_weights
        # maxXPerScale, maxYPerScale=np.argmax(np.max(prev_weights, axis=0)),np.argmax(np.max(prev_weights, axis=1))
        prev_maxXPerScale, prev_maxYPerScale = np.argmax(np.max(prev_weights, axis=1)) , np.argmax(np.max(prev_weights, axis=0))
        prev_max_col=round(activityDecoding(prev_weights[prev_maxXPerScale, :],5,self.N2),0)
        prev_max_row=round(activityDecoding(prev_weights[:,prev_maxYPerScale],5,self.N1),0)

        delta_row=np.round(speed*np.sin(np.deg2rad(direction)),6)
        delta_col=np.round(speed*np.cos(np.deg2rad(direction)),6)
        
        func = lambda x: int(math.ceil(x)) if x < 0 else int(math.floor(x))

        '''copied and shifted activity'''
        full_shift=np.zeros((self.N1,self.N2))
        shifted_row_ids, shifted_col_ids=(non_zero_rows +func(delta_row))%self.N1, (non_zero_cols+ func(delta_col))%self.N2
        full_shift[shifted_row_ids, shifted_col_ids]=prev_weights[non_zero_rows, non_zero_cols]
        copy_shift=self.fractional_shift(full_shift,delta_row,delta_col)*self.activity_mag


        '''excitation'''
        copyPaste=copy_shift
        non_zero_copyPaste=np.nonzero(copyPaste)  
        # print(len(non_zero_copyPaste[0]))
        excited=np.zeros((self.N1,self.N2))
        # t=time.time()
        for row, col in zip(non_zero_copyPaste[0], non_zero_copyPaste[1]):
            excited+=self.excitations(row,col,copyPaste[row,col])
        # print(time.time()-t)
        
        # excited=np.sum(excited_array, axis=0)
        # print(np.shape(excited_array), np.shape(excited))
        '''inhibitions'''
        inhibit_val=0
        shift_excite=copy_shift+prev_weights+excited
        non_zero_inhibit=np.nonzero(shift_excite) 
        for row, col in zip(non_zero_inhibit[0], non_zero_inhibit[1]):
            inhibit_val+=shift_excite[row,col]*self.inhibit_scale
        inhibit_array=np.tile(inhibit_val,(self.N1,self.N2))

        '''update activity'''
        prev_weights+=copy_shift+excited-inhibit_val
        prev_weights[prev_weights<0]=0

 
        '''wrap around'''
        # maxXPerScale, maxYPerScale=np.argmax(np.max(prev_weights, axis=0)),np.argmax(np.max(prev_weights, axis=1))
        maxXPerScale, maxYPerScale = np.argmax(np.max(prev_weights, axis=1)) , np.argmax(np.max(prev_weights, axis=0))
        max_col=round(activityDecoding(prev_weights[maxXPerScale, :],5,self.N2),0)
        max_row=round(activityDecoding(prev_weights[:,maxYPerScale],5,self.N1),0)
        
        # print(f"col_prev_current {prev_max_col, max_col} row_prev_current {prev_max_row, max_row}")
        wrap_cols=0 
        if prev_max_col>max_col and (direction<=90 or direction>=270): #right 
            wrap_cols=1
        elif prev_max_col<max_col and (direction>=90 and direction<=270): #left
            wrap_cols=-1
            
        wrap_rows=0 
        if prev_max_row>max_row and (direction>=0 and direction<=180) : #up 
            wrap_rows=1
        elif prev_max_row<max_row and (direction>=180 and direction<=360) : #down 
            wrap_rows=-1

        if moreResults==True:
            return prev_weights/np.linalg.norm(prev_weights),copy_shift,excited,inhibit_array
        else:
            return prev_weights/np.linalg.norm(prev_weights) if np.sum(prev_weights) > 0 else [np.nan], wrap_rows, wrap_cols
        

def activityDecoding(prev_weights,radius,N):
    '''Isolating activity at a radius around the peak to decode position'''
    # if np.argmax(prev_weights)==0:
    #     return 0
    # else:
    neurons=np.arange(N)
    peak=np.argmax(prev_weights) 
    local_activity=np.zeros(N)
    local_activity_idx=[]
    for i in range(-radius,radius+1):
        local_activity_idx.append((peak + i) % N)
    local_activity[local_activity_idx]=prev_weights[local_activity_idx]

    x,y=local_activity*np.cos(np.deg2rad(neurons*360/N)), local_activity*np.sin(np.deg2rad(neurons*360/N))
    vect_sum=np.rad2deg(math.atan2(sum(y),sum(x))) % 360
    weighted_sum = N*(vect_sum/360)

    if weighted_sum>(N-1):
        weighted_sum=0

    return weighted_sum



class LiDAR_PlaceRecOLD():
    def __init__(self, datasetName, refDataset, queryDataset, refINS, queryINS, scanDimX, scanDimY, mapRes, horizNumScans, intensityFilter, pointInterval,scanScale, framesRefAll, framesRef, framesQuery): 
        self.mapRes=mapRes
        self.scanDimX=scanDimX
        self.scanDimY= scanDimY
        self.horizNumScans=horizNumScans
        self.intensityFilter=intensityFilter
        self.datasetName=datasetName
        self.pointInterval=pointInterval
        self.scanScale=scanScale
        self.framesRef=framesRef
        self.framesQuery=framesQuery

        # for kitti use pykitti to load the dataset and save to ref and query variable 
        # for OxfordRadar store the velodyne directory into ref and query variable
        self.framesRefAll= framesRefAll
        if datasetName == 'KittiOdom':
            self.refDataset= pykitti.odometry('./data/KittiOdometryDataset', refDataset, frames=framesRef)
            self.queryDataset= pykitti.odometry('./data/KittiOdometryDataset', queryDataset, frames=framesQuery)
        else: 
            self.refDataset= refDataset 
            self.refINS = refINS
            
            self.queryDataset= queryDataset 
            self.queryINS=queryINS

    def oxfordFileNames(self, velodyne_dir, frames):
        if velodyne_dir[-4:]=='left':
            suffix='.png'
        elif velodyne_dir[-4:]=='dmrs' or velodyne_dir[-4:]=='ront':
            suffix='.bin'

        timestamps_path = velodyne_dir + '.timestamps'
        if not os.path.isfile(timestamps_path):
            raise IOError("Could not find timestamps file: {}".format(timestamps_path))
        velodyne_timestamps = np.loadtxt(timestamps_path, delimiter=' ', usecols=[0], dtype=np.int64)

        filenames,timestamps=[], []
        for velodyne_timestamp in velodyne_timestamps:

            filenames.append(os.path.join(velodyne_dir, str(velodyne_timestamp) + suffix))
            timestamps.append(velodyne_timestamp)
        
        discreteFilenames= [filenames[i] for i in frames]
        discreteTimestamps= [timestamps[i] for i in frames]
        
        return discreteFilenames, discreteTimestamps

    def oxford_Velo_lms_ldmrs(self, velodyne_dir, pose_dir, frames, idx):
        filenames, DiscreteTimestamps=self.oxfordFileNames(velodyne_dir, frames)
        filename=filenames[idx]
        
        if filename[-3:]== 'png':
            ranges, intensities, angles, approximate_timestamps = load_velodyne_raw(filename)
            ptcld=velodyne_raw_to_pointcloud(ranges, intensities, angles)
            normReflectance=(ptcld[3, :] - np.min(ptcld[3, :])) / (np.max(ptcld[3, :]) - np.min(ptcld[3, :]))
            lidar_range = np.where(normReflectance[::self.pointInterval]> self.intensityFilter)[0]


            ptcld=ptcld[:3,lidar_range].T
            z_range=(np.min(ptcld[:,2]), np.max(ptcld[:,2])) 
            half_x = self.scanDimX * self.mapRes // 2
            half_y = self.scanDimY * self.mapRes // 2
            grid,_=np.histogramdd(ptcld, bins=(np.linspace(-half_x, half_x,  self.scanDimX+1),
                                                np.linspace(-half_y, half_y, self.scanDimY+1),
                                                np.arange(z_range[0], z_range[1],self.mapRes)))
            grid2d=np.sum(grid,axis=2).T
            grid2d[grid2d<=1]=0
            grid2d[grid2d>1]=1


            # xCoords, yCoords= np.array(ptcld[:,0]), np.array(ptcld[:,1])
            # mask_x = (xCoords >= -(self.scanDimX//2)) & (xCoords <= (self.scanDimX//2))
            # mask_y = (yCoords >= -(self.scanDimY//2)) & (yCoords <= (self.scanDimY//2))
            # cropped_x = xCoords[mask_x & mask_y]
            # cropped_y = yCoords[mask_x & mask_y] 
            # grid2d=np.zeros((self.scanDimY,self.scanDimX))
            # shiftedCol,shiftedRow=[int(x+(self.scanDimX//2)) for x in cropped_x], [int(y+(self.scanDimY//2)) for y in cropped_y]
            # grid2d[shiftedRow, shiftedCol]=1

            return grid2d

        elif velodyne_dir[-5:]=='ldmrs':
            scan_file = open(filename)
            scan = np.fromfile(scan_file, np.double)
            scan_file.close()
            scan = scan.reshape((len(scan) // 3, 3)).transpose()
            xCoords, yCoords=scan[0,::self.pointInterval], scan[1,::self.pointInterval]
        
        elif velodyne_dir[-5:]=='front':
            timestamps_path = velodyne_dir + '.timestamps'
            extrinsics_dir= './extrinsics'
            poses_file=pose_dir
            poses_type = re.search('(vo|ins|rtk)\.csv', poses_file).group(1)
            lidar = re.search('(lms_front|lms_rear|ldmrs|velodyne_left|velodyne_right)', velodyne_dir).group(0)
            timestamps = []
            
            print(DiscreteTimestamps[idx])

            with open(timestamps_path) as timestamps_file:
                for line in timestamps_file:
                    timestamp = int(line.split(' ')[0])
                    if DiscreteTimestamps[idx] <= timestamp <= DiscreteTimestamps[idx+1]:
                        timestamps.append(timestamp)

            print(idx, len(timestamps))

            with open(os.path.join(extrinsics_dir, lidar + '.txt')) as extrinsics_file:
                extrinsics = next(extrinsics_file)
            G_posesource_laser = build_se3_transform([float(x) for x in extrinsics.split(' ')])

            with open(os.path.join(extrinsics_dir, 'ins.txt')) as extrinsics_file:
                extrinsics = next(extrinsics_file)
                G_posesource_laser = np.linalg.solve(build_se3_transform([float(x) for x in extrinsics.split(' ')]),
                                                    G_posesource_laser)

            poses = interpolate_ins_poses(poses_file, timestamps, DiscreteTimestamps[idx], use_rtk=(poses_type == 'rtk'))
            ptcld=[]
            pointcloud = np.array([[0], [0], [0], [0]])
            for i in range(20):
                scan_file = open(filename)
                scan = np.fromfile(scan_file, np.double)
                scan_file.close()
                scan = scan.reshape((len(scan) // 3, 3)).transpose()
                reflectance = np.empty((0))
                reflectance = np.concatenate((reflectance, np.ravel(scan[2, :])))
                
            
                scan[2, :] = np.zeros((1, scan.shape[1]))
                scan = np.dot(np.dot(poses[i], G_posesource_laser), np.vstack([scan, np.ones((1, scan.shape[1]))]))
                pointcloud = np.hstack([pointcloud, scan])
                # ptcld.append(scan)
                # ptcld = [ptcld, scan[:2,:]]
                # lidar_range = np.where(scan[2, :][::self.pointInterval] > self.intensityFilter)
            pointcloud = pointcloud[:, 1:]
            print(pointcloud.shape)
            xCoords, yCoords=np.array(pointcloud[0,:]), np.array(pointcloud[1,:])
        return xCoords, yCoords

    def makingImageFromLidarCoords(self, xCoords, yCoords):
        '''Converting a single LiDAR scan into an image'''

        imgArray=np.zeros((self.scanDimY,self.scanDimX))
    
        shiftedCol,shiftedRow=[int(x+(self.scanDimX//2)) for x in xCoords], [int(y+(self.scanDimY//2)) for y in yCoords]
        # xidxs, yidsxs= (np.array(yCoords2)-min(yCoords2)).astype(int), (np.array(xCoords2)-min(xCoords2)).astype(int)
        imgArray[shiftedRow, shiftedCol]=1

        # resized_img = np.resize(imgArray, (int(self.scanDimY *self.scanScale), int(self.scanDimX *self.scanScale))  )
        resized_img =cv2.resize(imgArray, dsize=(self.scanDimY, self.scanDimX), interpolation=cv2.INTER_CUBIC)

        return resized_img

    def wildplacesFileNames(self, velodyne_dir, frames):
        filenames = os.listdir(velodyne_dir)
        timestamps= [file[:-4] for file in filenames] 
        # print(len(filenames), len(timestamps), len([i for i in frames]))

        discreteFilenames= [filenames[i] for i in frames]
        discreteTimestamps= [timestamps[i] for i in frames]
        
        return discreteFilenames, discreteTimestamps

    def jackal_LoadRosbag(self, bag_dir, frames, idx, topic='/velodyne_points'):
        timestamps=[]
        with rosbag.Bag(bag_dir, 'r') as bag:
            # Get messages from the specified topic
            messages = bag.read_messages(topics=[topic])
            ith_message = next(itertools.islice(messages, idx, None), None)

            if ith_message[1]._type == 'sensor_msgs/PointCloud2':
                timestamps.append(ith_message[1].header.stamp.to_sec())
                pc_data = pc2.read_points(ith_message[1], field_names=("x", "y", "z", "intensity"), skip_nans=True)
                ptcld = np.array(list(pc_data))
                intensity=ptcld[:, 3]
                normINTENSITY=(intensity-np.min(intensity))/(np.max(intensity)-np.min(intensity))
                _range = np.sqrt(np.square(ptcld[:, 0]) + np.square(ptcld[:, 1]))
                _good =  (_range<15)  & (ptcld[:, 2] < 1.8) & (ptcld[:, 2] > 0.1) #(normINTENSITY > self.intensityFilter) & 
                # lidar_range = np.where((ptcld[:, 2] > 0.1) &  (ptcld[:, 2] < 1.0) & (ptcld[:, 3] > self.intensityFilter))

                xCoords, yCoords=ptcld[:, 0][_good], ptcld[:, 1][_good]

                return xCoords, yCoords
    
    def jackal_LoadRosbagPoses(self, bag_dir, topic='/odom/true'):
        x,y, yaws= [],[], []
        odomTimestamps=[]
        with rosbag.Bag(bag_dir, 'r') as bag:
            # Get messages from the specified odometry topic
            messages = bag.read_messages(topics=[topic])
            # Iterate through odometry messages
            for _, msg, _ in messages:
                # Check if the message type is Odometry
                if msg._type == 'nav_msgs/Odometry':
                    orientation_quaternion = (
                        msg.pose.pose.orientation.x,
                        msg.pose.pose.orientation.y,
                        msg.pose.pose.orientation.z,
                        msg.pose.pose.orientation.w )
                    roll, pitch, yaw = euler_from_quaternion(orientation_quaternion)
                    odomTimestamps.append(msg.header.stamp.to_sec())
                    # Extract relevant information (x, y position)
                    x.append(msg.pose.pose.position.x)              
                    y.append(msg.pose.pose.position.y) 
                    yaws.append(yaw)
                    # print(f'{msg.pose.pose.position.x}, {msg.pose.pose.position.y}, {yaw}')

        return x, y, yaws, odomTimestamps
    
    def convert_pc_to_grid(self, xy_arr):
        map_res=self.mapRes
        scanDimY=int(self.scanDimY*self.scanScale)
        scanDimX=int(self.scanDimX*self.scanScale)
        # self.map_height     = int(np.ceil(self.map_height_m / self.map_res)) + 2
        # self.map_width      = int(np.ceil(self.map_width_m / self.map_res)) + 2

        assert xy_arr.shape[1] == 2

        # round xy pairs to nearest multiple of MAP_RESOLUTION:
        xy_rounded = np.array(map_res * np.around(xy_arr/map_res, 0))
        
        # remove duplicate point entries (only unique rows; parse through integer to avoid floating point equalities):
        xy_clean = np.array(np.unique(np.around(xy_rounded*100,0), axis=0)/100.0,dtype=float)

        grid = np.zeros((self.scanDimY, self.scanDimX), dtype=int)

        x_offset = int(self.scanDimX / 2.0)
        y_offset = int(self.scanDimY / 2.0)

        # populate map with pointcloud at correct cartesian coordinates
        for xy_pair in xy_clean:

            ix = int( np.round(xy_pair[0] / map_res )) + x_offset
            iy = int( np.round(xy_pair[1] / map_res )) + y_offset

            grid[iy-1:iy+1,ix-1:ix+1] = 1
        
        # resized_img =cv2.resize(grid, dsize=(scanDimY, scanDimX), interpolation=cv2.INTER_CUBIC)
        
        return grid #np.flipud(grid) #grid


    ''' Processing LiDAR'''
    def numRefScans(self):
        if self.datasetName=='Kitti':
           numScans=len(self.refDataset)

        elif self.datasetName=='OxfordRadar':
            filenames,timestamps=self.oxfordFileNames(self.refDataset,self.framesRef)
            numScans=len(filenames)
        
        elif self.datasetName=='WildPlaces':
            filenames,timestamps=self.wildplacesFileNames(self.refDataset,self.framesRef)
            numScans=len(filenames)
        
        elif self.datasetName=='Jackal':
            with rosbag.Bag(self.refDataset, 'r') as bag:
                numScans= bag.get_message_count('/velodyne_points')
        
        return numScans
    
    def loadingCroppingFiltering_2DScan(self, REForQUERY,idx):
        '''Loading + Filtering'''
        if self.datasetName=='Kitti':
            if REForQUERY=='REF':
                dataset=self.refDataset
            elif REForQUERY=='QUERY':
                dataset=self.queryDataset
            ptcld=dataset.get_velo(idx)
            lidar_range = np.where(ptcld[:, 3][::self.pointInterval] > self.intensityFilter)
            xCoords, yCoords=ptcld[lidar_range, 0], ptcld[lidar_range, 1]

        elif self.datasetName=='OxfordRadar':
            if REForQUERY=='REF':
                pose_dir=self.refINS
                velodyne_dir=self.refDataset
                frames=self.framesRef
            elif REForQUERY=='QUERY':
                pose_dir=self.queryINS
                velodyne_dir=self.queryDataset
                frames=self.framesQuery

            filenames, DiscreteTimestamps=self.oxfordFileNames(velodyne_dir, frames)
            filename=filenames[idx]

            ranges, intensities, angles, approximate_timestamps = load_velodyne_raw(filename)
            ptcld=velodyne_raw_to_pointcloud(ranges, intensities, angles)
            normReflectance=(ptcld[3, :] - np.min(ptcld[3, :])) / (np.max(ptcld[3, :]) - np.min(ptcld[3, :]))
            lidar_range = np.where(normReflectance[::self.pointInterval]> self.intensityFilter)[0]


            ptcld=ptcld[:3,lidar_range].T
            z_range=(np.min(ptcld[:,2]), np.max(ptcld[:,2])) 
            half_x = self.scanDimX * self.mapRes // 2
            half_y = self.scanDimY * self.mapRes // 2
            grid,_=np.histogramdd(ptcld, bins=(np.linspace(-half_x, half_x,  self.scanDimX+1),
                                                np.linspace(-half_y, half_y, self.scanDimY+1),
                                                np.arange(z_range[0], z_range[1],self.mapRes)))
            grid2d=np.sum(grid,axis=2).T
            grid2d[grid2d<=1]=0
            grid2d[grid2d>1]=1

            return grid2d
            
        elif self.datasetName=='WildPlaces':
            if REForQUERY=='REF':
                pcd_dir=self.refDataset
                frames=self.framesRef
            elif REForQUERY=='QUERY':
                pcd_dir=self.queryDataset
                frames=self.framesQuery

            filenames, discreteTimestamps=self.wildplacesFileNames(pcd_dir,frames)
            pcd = o3d.io.read_point_cloud(pcd_dir+'/'+filenames[idx])
            ptcld=np.asarray(pcd.points)
            # intensity_values = np.asarray(pcd.colors)[:, 0] 
            # normReflectance=(intensity_values - np.min(intensity_values)) / (np.max(intensity_values) - np.min(intensity_values))
            # lidar_range = np.where(normReflectance[::self.pointInterval]> self.intensityFilter)[0]
            # print(len(intensity_values), len(lidar_range), ptcld.shape())
            
            ptcld=ptcld[::self.pointInterval,:]

            z_range=(np.min(ptcld[:,2]), np.max(ptcld[:,2])) 
            half_x = self.scanDimX * self.mapRes // 2
            half_y = self.scanDimY * self.mapRes // 2
            grid,_=np.histogramdd(ptcld, bins=(np.linspace(-half_x, half_x, self.scanDimX+1),
                                              np.linspace(-half_y, half_y, self.scanDimY+1),
                                              np.arange(z_range[0], z_range[1], self.mapRes)))
            
            grid=np.sum(grid[:,:,:], axis=2)
            thresh=np.max(grid)//10
            grid[grid<=thresh]=0
            grid[grid>thresh]=1


            
            contours, _ = cv2.findContours(np.uint8(grid), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Create an empty image to draw contours
            grid = np.zeros_like(grid)

            # Draw contours on the empty image
            cv2.drawContours(grid, contours, -1, (1, 1, 1), 1)

            return grid 
        
        elif self.datasetName=='Jackal':
            if REForQUERY=='REF':
                bag_dir=self.refDataset
                frames=self.framesRef
            elif REForQUERY=='QUERY':
                bag_dir=self.queryDataset
                frames=self.framesQuery
            xCoords, yCoords=self.jackal_LoadRosbag(bag_dir, frames, idx)


        '''Cropping'''
        scanDimY=int(self.scanDimY*self.scanScale)
        scanDimX=int(self.scanDimX*self.scanScale)

        mask_x = (xCoords >= -(scanDimX//2)) & (xCoords <= (scanDimX//2))
        mask_y = (yCoords >= -(scanDimY//2)) & (yCoords <= (scanDimY//2))
        cropped_x = xCoords[mask_x & mask_y]
        cropped_y = yCoords[mask_x & mask_y] 

        return cropped_x, cropped_y 

    def scanPoses(self, REForQUERY):
        if REForQUERY == 'REF':
            poses_file=self.refINS
            velodyne_dir=self.refDataset
            frames=self.framesRef
        elif REForQUERY== 'QUERY':
            poses_file=self.queryINS
            velodyne_dir=self.queryDataset
            frames=self.framesQuery
        elif REForQUERY == 'REF_ALL':
            poses_file=self.refINS
            velodyne_dir=self.refDataset
            frames=self.framesRefAll

        if self.datasetName == 'OxfordRadar':
            timestamps_path=velodyne_dir + '.timestamps'
            timestampsAll=list(np.loadtxt(timestamps_path, delimiter=' ', usecols=[0], dtype=np.int64))
            timestamps=[timestampsAll[i] for i in frames]


            ins_data = pd.read_csv(poses_file)
            timestamps_np = ins_data['timestamp'].to_numpy()
            # Find the index of the closest timestamp for each target timestamp
            closest_indices = [np.argmin(np.abs(timestamps_np - time)) for time in timestamps]
            # Extract the corresponding data using NumPy indexing
            filtered_data_np = ins_data.iloc[closest_indices]
            # Extract desired columns
            x, y, yaw = filtered_data_np['easting'].to_numpy(), filtered_data_np['northing'].to_numpy(), filtered_data_np['yaw'].to_numpy()

            # gt_file=velodyne_dir.rsplit('/', 1)[0] + '/gt/radar_odometry.csv'
            # gt_data = pd.read_csv(gt_file)
            # odomTimestamps=gt_data['source_timestamp']
            # destTimestamps=gt_data['destination_timestamp']
            
            # closestTimestampsIds = [odomTimestamps.sub(time).abs().idxmin() for time in timestamps]
            # print(closestTimestampsIds)
            # # filtered_data = gt_data[gt_data['source_timestamp'].isin(closestTimestamps)]
            # # xAll,yAll, yawAll=np.cumsum(gt_data['x']),np.cumsum(gt_data['y']),np.cumsum(gt_data['yaw'])
            # x,y,theta=0,0,0
            # xAll, yAll, yawAll = [], [], []
            # for i in range(len(odomTimestamps)):
            #     dt=(odomTimestamps[i]-destTimestamps[i])/ 1_000_000_000
            #     transVel=np.sqrt((gt_data['x'][i])**2+ (gt_data['y'][i])**2)

            #     print(dt, transVel)

            #     x += gt_data['x'][i] * dt * math.cos(theta)
            #     y += gt_data['y'][i] * dt * math.sin(theta)
            #     theta += gt_data['yaw'][i] * dt
            #     xAll.append(x)
            #     yAll.append(y)
            #     yawAll.append(theta)

            # x=[xAll[time] for time in closestTimestampsIds]
            # y=[yAll[time] for time in closestTimestampsIds]
            # yaw= [yawAll[time] for time in closestTimestampsIds]
                
        if self.datasetName == 'WildPlaces':
            filenames,timestamps=self.wildplacesFileNames(velodyne_dir,frames)
            ins_data = pd.read_csv(poses_file)
            closestTimestamps= [min(ins_data['timestamp'], key=lambda x: abs(x - float(time))) for time in timestamps]
            filtered_data = ins_data[ins_data['timestamp'].isin(closestTimestamps)]
            x,y, yaw=filtered_data['x'].values.tolist(), filtered_data['y'].values.tolist(), filtered_data['qz'].values.tolist()
        
        if self.datasetName=='Jackal':
            x,y,yaws,odomTimestamps=self.jackal_LoadRosbagPoses(velodyne_dir)
            yaw=None
            veloTimestamps=[]
            with rosbag.Bag(velodyne_dir, 'r') as bag:
                # Get messages from the specified topic
                for _, msg, _ in bag.read_messages(topics=['/velodyne_points']):
                    if msg._type == 'sensor_msgs/PointCloud2':
                        veloTimestamps.append(msg.header.stamp.to_sec())

            # Adjust starting time stamps 
            diffStartTime=odomTimestamps[0] - veloTimestamps[0]
            odomTimestamps=[time-diffStartTime for time in odomTimestamps]  

            # Extract closest odom index for each scan 
            discreteVeloTimestamps= [veloTimestamps[i] for i in frames]
            closestTimestampsIdxs=[odomTimestamps.index(min(odomTimestamps, key=lambda x: abs(x - float(time)))) for time in discreteVeloTimestamps]
            get_indices = itemgetter(*closestTimestampsIdxs)
            print(f'odo start {odomTimestamps[0]}, velo start {veloTimestamps[0]}, diff: {odomTimestamps[0] - veloTimestamps[0]} ')
            x,y,yaw=get_indices(x),get_indices(y), get_indices(yaws)
            print()
        # print(f'closes time {closestTimestampsIdxs}')
        # plt.plot(odomTimestamps, [1]*len(odomTimestamps),'r.')
        # plt.plot( veloTimestamps, [0]*len(veloTimestamps),'b.')
        # plt.show()    

        return x,y,yaw
    
    def pathIntegration(self, startPose, speed, angVel):
        q=startPose#[0,0,2.645333]
        x_integ,y_integ, theta=[],[], []
        for i in range(len(speed)):
            q[0],q[1]=q[0]+speed[i]*np.cos(q[2]), q[1]+speed[i]*np.sin(q[2])
            q[2]+=angVel[i]
            x_integ.append(q[0])
            y_integ.append(q[1])
            theta.append(q[2])
        print(f'len(speed) {len(speed)}, lenOutput: {len(x_integ)}')
        return np.array(y_integ), np.array(x_integ), np.array(theta)
                                                              
    def scanPosesRadarOdom(self, REForQUERY):
        if REForQUERY == 'REF':
            insFile=self.refINS
            radarFile=os.path.dirname(self.refDataset)+'/gt/radar_odometry.csv'
            velodyne_dir=self.refDataset
            frames=self.framesRef
        elif REForQUERY== 'QUERY':
            insFile=self.queryINS
            radarFile=os.path.dirname(self.queryDataset)+'/gt/radar_odometry.csv'
            velodyne_dir=self.queryDataset
            frames=self.framesQuery
        elif REForQUERY == 'REF_ALL':
            insFile=self.refINS
            radarFile=os.path.dirname(self.refDataset)+'/gt/radar_odometry.csv'
            velodyne_dir=self.refDataset
            frames=self.framesRefAll

        if self.datasetName == 'OxfordRadar':
            timestamps_path=velodyne_dir + '.timestamps'
            timestampsAll=list(np.loadtxt(timestamps_path, delimiter=' ', usecols=[0], dtype=np.int64))
            timestamps=[timestampsAll[i] for i in frames]


            startYaw = pd.read_csv(insFile)['yaw'][0]
            startPose=[0,0, startYaw]
            radarData=pd.read_csv(radarFile)
            x,y,theta=self.pathIntegration(startPose, np.array(radarData['x']), np.array(radarData['yaw']))

            # closestTimestamps = [min(enumerate(radarData['destination_timestamp']), key=lambda x: abs(x[1] - time))[0] for time in timestamps]
            radar_timestamps_np = radarData['destination_timestamp'].to_numpy()
            closest_indices = [np.argmin(np.abs(radar_timestamps_np - time)) for time in timestamps]
     
            return np.array(x[closest_indices]), np.array(y[closest_indices]), np.array(theta[closest_indices])
            
    def translateScan(self, scan, deltaX, deltaY):
        height, width= scan.shape
        shifted_image = np.zeros_like(scan, dtype=int)
        x_start, x_end = max(0, deltaX), min(width, width + deltaX)
        y_start, y_end = max(0, deltaY), min(height, height + deltaY)
        shifted_image[y_start - deltaY:y_end - deltaY, x_start - deltaX:x_end - deltaX] = scan[y_start:y_end, x_start:x_end]

        return shifted_image
    
    def applyingRotation2D(self, angDEG,  xCoords, yCoords):
        '''Rotating a single LiDAR scan'''
        theta=np.deg2rad(angDEG)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Define the rotation matrix
        rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        xy=np.stack([np.array(xCoords).flatten(),np.array(yCoords).flatten()])

        XY = np.matmul(rotation_matrix, xy)

        rotX,rotY=XY[0,:], XY[1,:]

        scanDimY=int(self.scanDimY*self.scanScale)//2
        scanDimX=int(self.scanDimX*self.scanScale)//2
        mask_xy = (rotX >= -scanDimX) & (rotX <= scanDimX) & (rotY >= -scanDimY) & (rotY <= scanDimY)
 
        return np.array(rotX[mask_xy]), np.array(rotY[mask_xy])


    '''Scan Matching'''
    def makingReferenceGrid(self):
        '''Converting a dataset of LiDAR scans into a refernce grid image'''
        scanDimY=int(self.scanDimY*self.scanScale)
        scanDimX=int(self.scanDimX*self.scanScale)

        if self.datasetName=='Jackal':
            refRange=[i for i in self.framesRef]
            numScans=len(refRange)

            vertNumScans=np.ceil(numScans/self.horizNumScans)
            height,width=int(vertNumScans*self.scanDimY),int(self.horizNumScans*self.scanDimX)
            refernceGrid=np.zeros((height,width))
            for i,idx in enumerate(refRange):
                cropped_x, cropped_y= self.loadingCroppingFiltering_2DScan('REF', idx)
                xy_arr=np.transpose(np.stack([cropped_x, cropped_y],axis=0))
                refScan=self.convert_pc_to_grid(xy_arr)
                rowIdx=i//self.horizNumScans
                colIdx=i%self.horizNumScans
                xIdx,yIdx=colIdx*self.scanDimX, rowIdx*self.scanDimY
                refernceGrid[yIdx:yIdx+(self.scanDimY), xIdx:xIdx+(self.scanDimX)]=refScan

        elif self.datasetName=='WildPlaces' or self.datasetName=='OxfordRadar' :
            numScans=self.numRefScans()

            vertNumScans=np.ceil(numScans/self.horizNumScans)
            height,width=int(vertNumScans*self.scanDimY),int(self.horizNumScans*self.scanDimX)
            refernceGrid=np.zeros((height,width))
            for i in range(numScans):
                refScan= self.loadingCroppingFiltering_2DScan('REF', i)
                rowIdx=i//self.horizNumScans
                colIdx=i%self.horizNumScans
                xIdx,yIdx=colIdx*self.scanDimX, rowIdx*self.scanDimY
                refernceGrid[yIdx:yIdx+(self.scanDimY), xIdx:xIdx+(self.scanDimX)]=refScan

        else:
            numScans=self.numRefScans()
            vertNumScans=np.ceil(numScans/self.horizNumScans)
            height,width=int(vertNumScans*scanDimY),int(self.horizNumScans*scanDimX)
            refernceGrid=np.zeros((height,width))
            for i in range(numScans):
                cropped_x, cropped_y= self.loadingCroppingFiltering_2DScan('REF', i)
                refScan= self.makingImageFromLidarCoords(cropped_x,cropped_y)
                rowIdx=i//self.horizNumScans
                colIdx=i%self.horizNumScans
                xIdx,yIdx=colIdx*scanDimX, rowIdx*scanDimY
                refernceGrid[yIdx:yIdx+(scanDimY), xIdx:xIdx+(scanDimX)]=refScan
        
        return refernceGrid
     
    def singleQuery(self, idx):
        '''Making a query lidar scan kernel for a given index'''
        
        if self.datasetName=='Jackal':
            cropped_x, cropped_y= self.loadingCroppingFiltering_2DScan('QUERY', idx)
            xy_arr=np.transpose(np.stack([cropped_x, cropped_y],axis=0))
            localScan=self.convert_pc_to_grid(xy_arr)
        elif self.datasetName =='WildPlaces' or self.datasetName=='OxfordRadar':
            localScan= self.loadingCroppingFiltering_2DScan('QUERY', idx)
        else:
            cropped_x, cropped_y= self.loadingCroppingFiltering_2DScan('QUERY', idx)
            localScan= self.makingImageFromLidarCoords(cropped_x,cropped_y)
            
        return np.fliplr(np.flipud(localScan))
     
    def scanMatchWithConvolution(self, refernceGRid, idx, SEQorSINGLE, ONEorN='ONE'):
        if SEQorSINGLE == 'SINGLE':  
            kernel=self.singleQuery(idx) 
            maxXDiff = 0

        elif SEQorSINGLE == 'SEQ': 
            kernel=self.seqQuery(idx) 
            maxXDiff = 0#(0.5*(self.seqLength - 1))*self.scanDimX 
        
        convolved= signal.fftconvolve(refernceGRid, kernel, mode='same')
        if ONEorN == 'ONE':
            maxY, maxX = np.unravel_index(np.argmax(convolved), convolved.shape)
        else: 
            maxY, maxX = np.unravel_index(np.argsort(convolved, axis=None)[-ONEorN:], convolved.shape)


        return maxY, maxX, convolved
 
    def extractIDfromConv(self, maxX, maxY):
        centerXYs=self.scanCenter()
        dists=[]
        for centerX, centerY in centerXYs:
            dists.append(np.sqrt((centerY-maxY)**2+(centerX-maxX)**2))

        return np.argmin(dists)
   
    def scanCenter(self):
        numScans=self.numRefScans()
        
        scanDimY=int(self.scanDimY*self.scanScale)
        scanDimX=int(self.scanDimX*self.scanScale)

        centerXYs=[]
        for i in range(numScans):
            rowIdx=i//self.horizNumScans
            colIdx=i%self.horizNumScans
            xIdx,yIdx=colIdx*scanDimX, rowIdx*scanDimY
            centerXYs.append((xIdx+(scanDimX//2), yIdx+(scanDimX//2)))
        
        return centerXYs 

    def topMatchNeighbourReference(self, topNIds, searchRad):
        scanDimY=int(self.scanDimY*self.scanScale)
        scanDimX=int(self.scanDimX*self.scanScale)
        numScans=self.numRefScans()
        vertNumScans=len(topNIds)
        horizNumScans=(searchRad*2)+1

        height,width=int(vertNumScans*self.scanDimY),int(horizNumScans*self.scanDimX)
        refernceGrid=np.zeros((height,width))
        for rowIdx,topId in enumerate(topNIds):
            refIds=[i for i in range(topId-searchRad,topId+searchRad+1) if (i < numScans) and (i > 0)]
            # gridPos=[i for i in range(-searchRad,searchRad+1) if (topId+i < numScans) and (topId+i > 0)]
            for colIdx, id in enumerate(refIds):
                # print(f"gridrow:{rowIdx}, gridCol:{colIdx}, MatchId: {topId}, neighbour: {id}")
                cropped_x, cropped_y= self.loadingCroppingFiltering_2DScan('REF', id)
                if self.datasetName=='Jackal':
                    xy_arr=np.transpose(np.stack([cropped_x, cropped_y],axis=0))
                    refScan=self.convert_pc_to_grid(xy_arr)
                else:
                    refScan= self.makingImageFromLidarCoords(cropped_x,cropped_y)
                
            
                xIdx,yIdx=colIdx*self.scanDimX, rowIdx*self.scanDimY
                refernceGrid[yIdx:yIdx+(self.scanDimY), xIdx:xIdx+(self.scanDimX)]=refScan


        return refernceGrid

    def extractIDfromLocalGrid(self, maxX, maxY, topNIds, searchRad):
        # rowId=maxY/self.scanDimY
        # colId=maxX/self.scanDimX
        # idx=int(round((rowId * self.horizNumScans) + colId))
        scanDimY=int(self.scanDimY*self.scanScale)
        scanDimX=int(self.scanDimX*self.scanScale)

        vertNumScans=len(topNIds)
        horizNumScans=(searchRad*2)+1
        numScans=vertNumScans*horizNumScans

        centerXYs=[]
        dists=np.zeros((vertNumScans,horizNumScans))
        for rowIdx in range(vertNumScans):
            for colIdx in range(horizNumScans):
                xIdx,yIdx=colIdx*scanDimX, rowIdx*scanDimY
                centerX, centerY=xIdx+(scanDimX//2), yIdx+(scanDimY//2)
                centerXYs.append((centerX,centerY))
                dists[rowIdx,colIdx]=(np.sqrt((centerY-maxY)**2+(centerX-maxX)**2))
        min_row, min_col = np.unravel_index(np.argmin(dists), dists.shape)
        return topNIds[min_row], min_col-searchRad, centerXYs[np.argmin(dists)]


    '''Image Matching'''
    def patchNormalise(self, img1, patchLength):
        '''patchNormalising a single LiDAR scan image'''
        
        numZeroStd = []
        img1 = img1.astype(float)
        img2 = img1.copy()
        imgMask = np.ones(img1.shape,dtype=bool)
        
        if patchLength == 1:
            return img2

        for i in range(img1.shape[0]//patchLength):
            iStart = i*patchLength
            iEnd = (i+1)*patchLength
            for j in range(img1.shape[1]//patchLength):
                jStart = j*patchLength
                jEnd = (j+1)*patchLength
                tempData = img1[iStart:iEnd, jStart:jEnd].copy()
                mean1 = np.mean(tempData)
                std1 = np.std(tempData)
                tempData = (tempData - mean1)
                if std1 == 0:
                    std1 = 0.1 
                    numZeroStd.append(1)
                    imgMask[iStart:iEnd,jStart:jEnd] = np.zeros([patchLength,patchLength],dtype=bool)
                tempData /= std1
                img2[iStart:iEnd, jStart:jEnd] = tempData.copy()

        return img2
        
    def processImgs(self, REForQUERY):
        if REForQUERY == 'REF':
            poses_file=self.refINS
            velodyne_dir=self.refDataset
            frames=self.framesRef
        elif REForQUERY== 'QUERY':
            poses_file=self.queryINS
            velodyne_dir=self.queryDataset
            frames=self.framesQuery

        if self.datasetName == 'OxfordRadar':
            timestamps_path=velodyne_dir + '.timestamps'
            timestampsAll=list(np.loadtxt(timestamps_path, delimiter=' ', usecols=[0], dtype=np.int64))
            timestamps=[timestampsAll[i] for i in frames]


            images_dir=velodyne_dir.rsplit('velodyne_left', 1)[0] + 'mono_left'
            imgTimestamps=list(np.loadtxt(images_dir+'.timestamps', delimiter=' ', usecols=[0], dtype=np.int64))

            closestTimestamps= [min(imgTimestamps, key=lambda x: abs(x - time)) for time in timestamps]
            # filteredImgTime = imgTimestamps[imgTimestamps.isin(closestTimestamps)]
        
        Features=[]
        for time in closestTimestamps:
            im=cv2.imread(images_dir+'/'+str(time)+'.png')[:,:,::-1]
            im=cv2.resize(im, (64,64))
            im=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
            ft = self.patchNormalise(im,8)
            Features.append(ft.flatten())

        return np.array(Features)

    def matchQuerySAD(self, metric='euclidean'):

        # cropped_x, cropped_y= self.loadingCroppingFiltering_2DScan('QUERY', idx)
        # localScan= self.makingImageFromLidarCoords(cropped_x,cropped_y)
        # queryFeature=self.patchNormalise(localScan, 8).flatten()

        refFeatures= self.processImgs( 'REF')
        queryFeatures= self.processImgs('QUERY')
        print(np.shape(refFeatures))
        print(np.shape(queryFeatures))

        dMat=cdist(refFeatures,queryFeatures,metric)
        mIds=np.argsort(dMat,axis=0)[:1][0]

        return dMat, mIds



class LiDAR_PlaceRec():
    def __init__(self, datasetName, config, refNum, queryNum, framesQuery,  framesRef=[], refGridFilePath=None, n=None, background=None, blockSize=None, numPtsMultiplier=None, z_max=None, rotIncr=None, mapRes=None, HDMThresh=None): 
        '''Loading variables'''
        refIncr = config.get('refIncr')
        refFilenames = config.get('details', {}).get(str(refNum), [])
        queryFilenames = config.get('details', {}).get(str(queryNum), [])
        param = config.get('parameters', {})
        self.configParams=param
        refRad=param.get('refRadius', 0)
        # self.blockSize=blockSize#param.get('blkSize', 0)
        self.blockSize= param.get('blkSize', 0.0) if blockSize == None else blockSize
        self.rotIncr=param.get('rotIncr', 0) if rotIncr == None else rotIncr
        self.datasetName=datasetName
        self.scanDimX = int(param.get('scanDimX', 0)/self.blockSize)
        self.scanDimY = int(param.get('scanDimY', 0)/self.blockSize)
        self.mapRes = param.get('mapRes', 0.0) if mapRes == None else mapRes
        self.horizNumScans= param.get('horizNumScans', 0)
        self.intensityFilter= param.get('intensityFilter', 0)
        self.pointInterval= param.get('pointInterval', 0)
        self.HDMThresh = param.get('HDMThresh') if HDMThresh == None else HDMThresh
        self.randDwnsmple_dim=param.get('dim_randDwnsmple', 0)
        self.background= param.get('unoccWeight', 0.0) if background == None else background
        self.randDwnsmple_nMult= param.get('nMultiplier_randDwnsmpl', 0.0) if numPtsMultiplier == None else numPtsMultiplier
        self.z_max=param.get('zmax', 0.0) if z_max == None else z_max
        self.n=param.get('topN', 0.0) if n == None else n
        
        
        '''Loading files and indexes'''
        self.kitti_basdir='./data/KittiOdometryDataset'
        self.refDataset= refFilenames[0] 
        self.refINS = refFilenames[1]
        self.queryDataset= queryFilenames[0] 
        self.queryINS=queryFilenames[1]
        # self.xrAll,self.yrAll, self.rollR, self.pitchR, self.yawRAll= self.scanPoses('All')
        
        if datasetName=='Jackal':
            self.framesRef= range(0, refFilenames[1], refIncr)  
        elif datasetName[:10]=='WildPlaces': #or datasetName=='NCLT'
            self.xrAll,self.yrAll, self.rollR, self.pitchR, self.yawRAll= self.scanPoses('All')
            self.framesRef = self.subsample_everyN(self.xrAll, self.yrAll, refRad)

        elif datasetName == 'NCLT' or 'OxfordRadar':
            self.framesRef=framesRef
        else:
            filenames = os.listdir(refFilenames[0])
            filenames = [f for f in filenames if f[:-4].replace('.','',1).isdigit()]
            self.framesRef=range(0, len(filenames), refIncr)
        
        self.framesQuery=framesQuery
        self.discreteRefFilenames, self.discreteRefTimes=self.discretizeData(self.refDataset,self.framesRef)  
        self.discreteQueryFilenames, self.discreteQueryTimes=self.discretizeData(self.queryDataset,self.framesQuery)    

        self.pointDensity=0

        '''Making reference grid'''
        if refGridFilePath != None:
            if os.path.exists(refGridFilePath):
                refgrid=np.load(refGridFilePath)
                print('refGrid exsists')
            else:
                # curMinPoints=np.count_nonzero(self.loadingCroppingFiltering_2DScan('REF', 0))
                # for l in range(1,len(self.discreteRefFilenames)):
                #     refScan= self.loadingCroppingFiltering_2DScan('REF', l)
                #     numPoints=np.count_nonzero(refScan)
                #     if numPoints<curMinPoints:
                #         curMinPoints=numPoints
                #     print(numPoints,curMinPoints)
                refgrid=self.makingReferenceGrid()
                np.save(refGridFilePath, refgrid)

            
            refgrid[refgrid==0]=self.background
            self.poolFunc=np.mean
            refgridDownsample=skimage.measure.block_reduce(refgrid, self.blockSize, func=self.poolFunc)
            self.refgrid=cp.asarray(refgridDownsample)
            self.refgridFull=cp.asarray(refgrid)
        

    '''Loading data'''
    def numRefScans(self):
        if self.datasetName=='Kitti':
           numScans=len([i for i in self.framesRef])

        elif self.datasetName=='OxfordRadar' or self.datasetName[:10]=='WildPlaces' or self.datasetName=='Mulran' or self.datasetName=='BenchmarkOxford' or self.datasetName=='NCLT':
            filenames=self.discreteRefFilenames
            numScans=len(filenames)
        
        elif self.datasetName=='Jackal':
            with rosbag.Bag(self.refDataset, 'r') as bag:
                numScans= bag.get_message_count('/velodyne_points')
        
        return numScans


    def subsample_everyN(self, xr, yr, n):
        # Initialize the subsampled list with the first point
        points=list(zip(xr,yr))
        subsampled_points = [points[0]]
        
        # Iterate over the points and select the ones that are at least n meters away from the last selected point
        subsampleIds=[]
        for idx, point in enumerate(points[1:]):
            last_point = subsampled_points[-1]
            distance = np.sqrt((point[0] - last_point[0])**2 + (point[1] - last_point[1])**2)
            
            if distance >= n:
                subsampled_points.append(point)
                subsampleIds.append(idx)
        x,y = zip(*subsampled_points)
        
        return subsampleIds

        # # Build the KDTree
        # points=list(zip(xr,yr))
        # distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
        # # Cumulative sum of distances
        # cumulative_distance = np.insert(np.cumsum(distances), 0, 0)
        # # Create a KDTree for efficient querying
        # tree = KDTree(cumulative_distance[:, None])

        # # Find the indices of the points at every nth meter
        # target_distances = np.arange(0, cumulative_distance[-1], n)
        # indices = []
        # for distance in target_distances:
        #     idx = tree.query([[distance]], k=1)[1][0].tolist()[0]
        #     indices.append(idx)
        
        # return indices

        

    def subsample_Napart(self, xr, yr, n):
        # Initialize the list with the first position
        points=list(zip(xr,yr))
        filtered_positions = [points[0]]
        indices = []
        for idx, pos in enumerate(points[1:]):
            too_close = False
            for filtered_pos in filtered_positions:
                distance = np.linalg.norm(np.array(pos) - np.array(filtered_pos))
                if distance < n:
                    too_close = True
                    break
            if not too_close:
                filtered_positions.append(pos)
                indices.append(idx)

        return indices


    def discretizeData(self, velodyne_dir, frames, REForQUERY=None):
        
        if self.datasetName == 'OxfordRadar':
            timestamps_path = velodyne_dir + '.timestamps'
            velodyne_timestamps = np.loadtxt(timestamps_path, delimiter=' ', usecols=[0], dtype=np.int64)

            filenames,timestamps=[], []
            for velodyne_timestamp in velodyne_timestamps:

                filenames.append(os.path.join(velodyne_dir, str(velodyne_timestamp) + '.png'))
                timestamps.append(velodyne_timestamp)
            
            discreteFilenames= [filenames[i] for i in frames]
            discreteTimestamps= [timestamps[i] for i in frames]

        elif self.datasetName[:10]=='WildPlaces' or self.datasetName == 'Mulran' or self.datasetName == 'BenchmarkOxford' or self.datasetName == 'NCLT':
            start=time.time()
            filenames = os.listdir(velodyne_dir)
            filenames = [f for f in filenames if f[:-4].replace('.','',1).isdigit()]
            timestamps= [file[:-4] for file in filenames] 
            
            
            discreteFilenames= [filenames[i] for i in frames]
            discreteTimestamps= [timestamps[i] for i in frames]
        
        


        return discreteFilenames, discreteTimestamps


    def loadOxfordRadarPtcld(self,oxfordRadarFile):
        hdl32e_range_resolution = 0.002  # m / pixel
        hdl32e_minimum_range = 1.0
        hdl32e_elevations = np.array([-0.1862, -0.1628, -0.1396, -0.1164, -0.0930,
                                    -0.0698, -0.0466, -0.0232, 0., 0.0232, 0.0466, 0.0698,
                                    0.0930, 0.1164, 0.1396, 0.1628, 0.1862, 0.2094, 0.2327,
                                    0.2560, 0.2793, 0.3025, 0.3259, 0.3491, 0.3723, 0.3957,
                                    0.4189, 0.4421, 0.4655, 0.4887, 0.5119, 0.5353])[:, np.newaxis]
        hdl32e_base_to_fire_height = 0.090805
        hdl32e_cos_elevations = np.cos(hdl32e_elevations)
        hdl32e_sin_elevations = np.sin(hdl32e_elevations)

        example = cv2.imread(oxfordRadarFile, cv2.IMREAD_GRAYSCALE)
        intensities, ranges_raw, angles_raw, timestamps_raw = np.array_split(example, [32, 96, 98], 0)
        ranges = np.ascontiguousarray(ranges_raw.transpose()).view(np.uint16).transpose()
        ranges = ranges * hdl32e_range_resolution
        angles = np.ascontiguousarray(angles_raw.transpose()).view(np.uint16).transpose()
        angles = angles * (2. * np.pi) / 36000
        approximate_timestamps = np.ascontiguousarray(timestamps_raw.transpose()).view(np.int64).transpose()
        valid = ranges > hdl32e_minimum_range
        z = hdl32e_sin_elevations * ranges - hdl32e_base_to_fire_height
        xy = hdl32e_cos_elevations * ranges
        x = np.sin(angles) * xy
        y = -np.cos(angles) * xy

        xf = x[valid].reshape(-1)
        yf = y[valid].reshape(-1)
        zf = z[valid].reshape(-1)
        intensityf = intensities[valid].reshape(-1).astype(np.float32)
        ptcld = np.stack((xf, yf, zf, intensityf), 0)
        return ptcld


    def process_NCLT(self, file_path):
        hits = []
        with open(file_path, 'rb') as f_bin:
            while True:
                x_bytes = f_bin.read(2)
                if len(x_bytes) < 2:  # EOF or incomplete read
                    break

                y_bytes = f_bin.read(2)
                if len(y_bytes) < 2:  # EOF or incomplete read
                    break

                z_bytes = f_bin.read(2)
                if len(z_bytes) < 2:  # EOF or incomplete read
                    break

                i_byte = f_bin.read(1)
                if len(i_byte) < 1:  # EOF or incomplete read
                    break

                l_byte = f_bin.read(1)
                if len(l_byte) < 1:  # EOF or incomplete read
                    break

                x = struct.unpack('<H', x_bytes)[0]
                y = struct.unpack('<H', y_bytes)[0]
                z = struct.unpack('<H', z_bytes)[0]
                i = struct.unpack('B', i_byte)[0]
                l = struct.unpack('B', l_byte)[0]

                scaling = 0.005 # 5 mm
                offset = -100
                x = x * scaling + offset
                y = y * scaling + offset
                z = z * scaling + offset
                
                hits += [[x, y, z, l]]
                
        return np.asarray(hits)


    def jackal_LoadRosbag(self, bag_dir, frames, idx, topic='/velodyne_points'):
        timestamps=[]
        with rosbag.Bag(bag_dir, 'r') as bag:
            # Get messages from the specified topic
            messages = bag.read_messages(topics=[topic])
            ith_message = next(itertools.islice(messages, idx, None), None)

            if ith_message[1]._type == 'sensor_msgs/PointCloud2':
                timestamps.append(ith_message[1].header.stamp.to_sec())
                pc_data = pc2.read_points(ith_message[1], field_names=("x", "y", "z", "intensity"), skip_nans=True)
                ptcld = np.array(list(pc_data))
               
                return ptcld
    

    def jackal_LoadRosbagPoses(self, bag_dir, topic='/odom/true'):
        x,y, yaws= [],[], []
        odomTimestamps=[]
        with rosbag.Bag(bag_dir, 'r') as bag:
            # Get messages from the specified odometry topic
            messages = bag.read_messages(topics=[topic])
            # Iterate through odometry messages
            for _, msg, _ in messages:
                # Check if the message type is Odometry
                if msg._type == 'nav_msgs/Odometry':
                    orientation_quaternion = (
                        msg.pose.pose.orientation.x,
                        msg.pose.pose.orientation.y,
                        msg.pose.pose.orientation.z,
                        msg.pose.pose.orientation.w )
                    roll, pitch, yaw = euler_from_quaternion(orientation_quaternion)
                    odomTimestamps.append(msg.header.stamp.to_sec())
                    # Extract relevant information (x, y position)
                    x.append(msg.pose.pose.position.x)              
                    y.append(msg.pose.pose.position.y) 
                    yaws.append(yaw)
                    # print(f'{msg.pose.pose.position.x}, {msg.pose.pose.position.y}, {yaw}')
        return x, y, yaws, odomTimestamps


    ''' Processing LiDAR'''

    def loading2DScan(self, REForQUERY,idx, rotDeg=None, dim_3=False, raw=False):
        if REForQUERY=='REF':
            velodyne_dir=self.refDataset
            frames=self.framesRef
            filenames=self.discreteRefFilenames
        elif REForQUERY=='QUERY':
            velodyne_dir=self.queryDataset
            frames=self.framesQuery
            filenames=self.discreteQueryFilenames


        if self.datasetName=='OxfordRadar':
            filenames, discreteTimestamps=self.discretizeData(velodyne_dir, frames)
            filename=filenames[idx]
            ptcld= self.loadOxfordRadarPtcld(filename)
            # Filter based on conditions
            if raw==False: 
                normReflectance=(ptcld[3, :] - np.min(ptcld[3, :])) / (np.max(ptcld[3, :]) - np.min(ptcld[3, :]))
                _zfilter= (normReflectance > self.intensityFilter) & (ptcld[2, :] > 0.05) #& (ptcld[2, :] < 2) 
                _good=np.where(_zfilter[::self.pointInterval])[0]
                ptcld=ptcld[:3,_good].T
            else:
                ptcld=ptcld[:3,::2].T

            z_range=(np.min(ptcld[:,2]), np.max(ptcld[:,2])) 
            thresh=1
    
        elif self.datasetName[:10]=='WildPlaces':
            # filenames=self.discreteRefFilenames         
            pcd = o3d.io.read_point_cloud(velodyne_dir+'/'+filenames[idx])
            ptcld=np.asarray(pcd.points)
            _zfilter= (ptcld[:,2] > 0) & (ptcld[:,2] < self.z_max) 
            ptcld=ptcld[_zfilter,:]

                   
        elif self.datasetName=='BenchmarkOxford':
            filenames, discreteTimestamps=self.discretizeData(velodyne_dir,frames)
            pc = np.fromfile((velodyne_dir+'/'+filenames[idx]), dtype=np.float64)
            ptcld = np.reshape(pc,(pc.shape[0]//3,3))
            _zfilter= (ptcld[:, 2] < 0.3) & (ptcld[:, 2] > -0.3) 
            _good=np.where(_zfilter[::self.pointInterval])[0]
            ptcld=ptcld[:,:]
            z_range=(-0.3,0.3)
            thresh=0.5

        elif self.datasetName=='Mulran':
            filenames, discreteTimestamps=self.discretizeData(velodyne_dir,frames)
            ptcld = np.fromfile((velodyne_dir+'/'+filenames[idx]), dtype=np.float32).reshape(-1, 4)
            if raw == False: 
                normReflectance=(ptcld[:, 3] - np.min(ptcld[:, 3])) / (np.max(ptcld[:, 3]) - np.min(ptcld[:, 3]))
                _zfilter= (normReflectance > self.intensityFilter) & (ptcld[:, 2] > 0.1) #& (ptcld[:, 2] < 6)        
                _good=np.where(_zfilter[::self.pointInterval])[0]
                ptcld=ptcld[_good, :3]
            else: 
                ptcld=ptcld[:, :3]

            z_range=(np.min(ptcld[:,2]), np.max(ptcld[:,2])) 
            thresh=1
            
        elif self.datasetName=='Jackal':
            ptcld=self.jackal_LoadRosbag(velodyne_dir, frames, idx)
            intensity = ptcld[:, 3]
            norm_intensity = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))
            _range = np.sqrt(np.square(ptcld[:, 0]) + np.square(ptcld[:, 1]))
            _good = (_range < 15) & (ptcld[:, 2] < 1.8) & (ptcld[:, 2] > 0.1) & (norm_intensity > self.intensityFilter)
            ptcld = ptcld[(ptcld[:, 2] > 0.1) &(norm_intensity > self.intensityFilter), :3]

            z_range=(np.min(ptcld[:,2]), np.max(ptcld[:,2])) 
            thresh=1

        elif self.datasetName == 'Kitti':
            dataset = pykitti.odometry(self.kitti_basdir, velodyne_dir, frames=frames)
            pc = np.squeeze(dataset.get_velo(idx))
            normReflectance=(pc[:, 3] - np.min(pc[:, 3])) / (np.max(pc[:, 3]) - np.min(pc[:, 3]))
            _zfilter= (normReflectance > self.intensityFilter) & (pc[:, 2] > 0.1) #& (pc[2, :] < 2) 
            _good=np.where(_zfilter[::self.pointInterval])[0]
            ptcld=pc[_good,:3]

            z_range=(np.min(ptcld[:,2]), np.max(ptcld[:,2])) 
            thresh=1
        
        elif self.datasetName == 'NCLT':
            ptcld = self.process_NCLT(velodyne_dir+'/'+filenames[idx])
            _reflectFilter= (ptcld[:,3] < self.intensityFilter) #& (ptcld[:,2]>0)
            ptcld=ptcld[_reflectFilter,:3]
       
        return ptcld
    

    def processing2DScan(self, REForQUERY, inputPtcld, rotDeg=None, sameProcess=False):
        ptcld=inputPtcld.copy()
        if rotDeg!=None:
            x,y=self.applyingRotation2D(rotDeg, ptcld[:,1],ptcld[:,0])
            ptcld[:,1]=x
            ptcld[:,0]=y
        # if REForQUERY=='QU'
        newScanDimX, newScanDimY = int(self.scanDimX*self.blockSize), int(self.scanDimY*self.blockSize)
        z_range=(np.min(ptcld[:,2]), np.max(ptcld[:,2])) 
        half_x = newScanDimX * self.mapRes // 2
        half_y = newScanDimY * self.mapRes // 2
        grid3d,_=np.histogramdd(ptcld, bins=(np.linspace(-half_x, half_x, newScanDimX+1),
                                            np.linspace(-half_y, half_y, newScanDimY+1),
                                            np.arange(z_range[0], z_range[1],self.mapRes)))
        gridTopDown=np.sum(grid3d,axis=2) 
        gridTopDown[gridTopDown<=self.HDMThresh]=0
        gridTopDown[gridTopDown>self.HDMThresh]=1
        
        self.pointDensity=np.count_nonzero(gridTopDown)


        if REForQUERY=='REF' and self.randDwnsmple_nMult>0 and sameProcess==False:
            gridTopDown=random_downsample(gridTopDown, self.randDwnsmple_dim, numPtsMultiplier=self.randDwnsmple_nMult, downsampleType=self.configParams.get('downsampleType'))
            # pass
    

        if REForQUERY == 'QUERY' and sameProcess==False:
            
            # gridTopDown=random_downsample(gridTopDown, self.randDwnsmple_dim, numPtsMultiplier=self.randDwnsmple_nMult)
            gridTopDown[gridTopDown==0]=self.background
        


        return gridTopDown.T
    

    def loadingCroppingFiltering_2DScan(self, REForQUERY,idx, rotDeg=None, sameProcess=False):
        ptcld=self.loading2DScan(REForQUERY, idx)
        
        return self.processing2DScan(REForQUERY, ptcld, rotDeg=rotDeg, sameProcess=sameProcess)

        
    def scanPoses(self, REForQUERY):
        if REForQUERY == 'All':
            poses_file=self.refINS
            velodyne_dir=self.refDataset
            frames=range(0, len(os.listdir(self.refDataset)) )
            timestamps = [file[:-4] for file in os.listdir(velodyne_dir) if file[:-4].replace('.','',1).isdigit()]
        if REForQUERY == 'AllQuery':
            poses_file=self.queryINS
            velodyne_dir=self.queryDataset
            frames=range(0, len(os.listdir(self.queryDataset)) )
            timestamps = [file[:-4] for file in os.listdir(velodyne_dir) if file[:-4].replace('.','',1).isdigit()]

        elif REForQUERY == 'REF':
            poses_file=self.refINS
            velodyne_dir=self.refDataset
            frames=self.framesRef
            timestamps=self.discreteRefTimes
        elif REForQUERY== 'QUERY':
            poses_file=self.queryINS
            velodyne_dir=self.queryDataset
            frames=self.framesQuery
            timestamps=self.discreteQueryTimes

        if self.datasetName == 'OxfordRadar':
            timestamps_path=velodyne_dir + '.timestamps'
            timestampsAll=list(np.loadtxt(timestamps_path, delimiter=' ', usecols=[0], dtype=np.int64))
            timestamps=[timestampsAll[i] for i in frames]

            ins_data = pd.read_csv(poses_file)
            timestamps_np = ins_data['timestamp'].to_numpy()

            closest_indices = [np.argmin(np.abs(timestamps_np - time)) for time in timestamps]
            filtered_data_np = ins_data.iloc[closest_indices]
            x, y, yaw = filtered_data_np['easting'].to_numpy(), filtered_data_np['northing'].to_numpy(), filtered_data_np['yaw'].to_numpy()
            roll,pitch = filtered_data_np['roll'].to_numpy(), filtered_data_np['pitch'].to_numpy()
            return x,y,roll, pitch, yaw
                
        if self.datasetName[:10]=='WildPlaces':
            # filenames,timestamps=self.discretizeData(velodyne_dir,frames)
            ins_data = pd.read_csv(poses_file)
            timestamps_np = ins_data['timestamp'].to_numpy()

            closest_indices = [np.argmin(np.abs(timestamps_np - float(time))) for time in timestamps]
            filtered_data_np = ins_data.iloc[closest_indices]

            qx, qy, qz, qw = filtered_data_np['qx'].values.tolist(), filtered_data_np['qy'].values.tolist(), filtered_data_np['qz'].values.tolist(), filtered_data_np['qw'].values.tolist()
            yaw= [math.atan2(2 * (qw[i] * qz[i] + qx[i] * qy[i]), 1 - 2 * (qy[i]**2 + qz[i]**2)) for i in range(len(qx))]
            roll = [math.atan2(2 * (qw[i] * qx[i] + qy[i] * qz[i]), 1 - 2 * (qx[i]**2 + qy[i]**2)) for i in range(len(qx))]
            pitch = [math.asin(2 * (qw[i] * qy[i] - qz[i] * qx[i])) for i in range(len(qx))]
            x,y=filtered_data_np['x'].values.tolist(), filtered_data_np['y'].values.tolist()

            return x,y,roll, pitch, yaw
        
        if self.datasetName=='NCLT':
            ins_data=np.loadtxt(poses_file, delimiter=",")[2:,:]
            timestamps_np=ins_data[:,0]

            closest_indices = [np.argmin(np.abs(timestamps_np - float(time))) for time in timestamps]
            # filtered_data_np = ins_data.iloc[closest_indices]
            y = ins_data[closest_indices, 2]
            z = ins_data[closest_indices, 3]
            x = ins_data[closest_indices, 1]

            r = ins_data[closest_indices, 4]
            p = ins_data[closest_indices, 5]
            yaw = ins_data[closest_indices, 6]

            return x, y, r, p, yaw

        if self.datasetName == 'BenchmarkOxford':
            filenames,timestamps=self.discretizeData(velodyne_dir,frames)
            ins_data = pd.read_csv(poses_file)
            timestamps_np = ins_data['timestamp'].to_numpy()
            closest_indices = [np.argmin(np.abs(timestamps_np - float(time))) for time in timestamps]
            filtered_data_np = ins_data.iloc[closest_indices]
            x,y, yaw=filtered_data_np['easting'].values.tolist(), filtered_data_np['northing'].values.tolist(), None 
            return x,y,yaw

        if self.datasetName == 'Mulran':
            filenames,timestamps=self.discretizeData(velodyne_dir,frames)
            ins_data = pd.read_csv(poses_file)

            print(ins_data.shape)
            allX, allY=ins_data.iloc[:, 4].to_numpy(), ins_data.iloc[:, 8].to_numpy()
            timestamps_np=ins_data.iloc[:, 0].to_numpy()
            closest_indices = [np.argmin(np.abs(timestamps_np - float(time))) for time in timestamps]

            x,y=allX[closest_indices], allY[closest_indices]
            print(len(closest_indices), len(x))
            
            yaw=None
            return x,y,yaw

        if self.datasetName=='Jackal':
            x,y,yaws,odomTimestamps=self.jackal_LoadRosbagPoses(velodyne_dir)
            yaw=None
            veloTimestamps=[]
            with rosbag.Bag(velodyne_dir, 'r') as bag:
                # Get messages from the specified topic
                for _, msg, _ in bag.read_messages(topics=['/velodyne_points']):
                    if msg._type == 'sensor_msgs/PointCloud2':
                        veloTimestamps.append(msg.header.stamp.to_sec())

            # Adjust starting time stamps 
            diffStartTime=odomTimestamps[0] - veloTimestamps[0]
            odomTimestamps=[time-diffStartTime for time in odomTimestamps]  

            # Extract closest odom index for each scan 
            discreteVeloTimestamps= [veloTimestamps[i] for i in frames]
            closestTimestampsIdxs=[odomTimestamps.index(min(odomTimestamps, key=lambda x: abs(x - float(time)))) for time in discreteVeloTimestamps]
            get_indices = itemgetter(*closestTimestampsIdxs)
            print(f'odo start {odomTimestamps[0]}, velo start {veloTimestamps[0]}, diff: {odomTimestamps[0] - veloTimestamps[0]} ')
            x,y,yaw=get_indices(x),get_indices(y), get_indices(yaws)  
            return x,y,yaw

        if self.datasetName == 'Kitti':
            dataset = pykitti.odometry(self.kitti_basdir, velodyne_dir, frames=frames)
            x,y = [pose[0, 3] for pose in dataset.poses], [pose[2, 3] for pose in dataset.poses]
            yaw= [np.arctan2(pose[:3, :3][1, 0], pose[:3, :3][0, 0]) for pose in dataset.poses] 

            return x,y,yaw
    

    def translateScan(self, scan, deltaX, deltaY):
        height, width= scan.shape
        shifted_image = np.zeros_like(scan)
        x_start, x_end = max(0, deltaX), min(width, width + deltaX)
        y_start, y_end = max(0, deltaY), min(height, height + deltaY)
        shifted_image[y_start - deltaY:y_end - deltaY, x_start - deltaX:x_end - deltaX] = scan[y_start:y_end, x_start:x_end]

        return shifted_image
    

    def applyingRotation2D(self, angDEG,  xCoords, yCoords):
        '''Rotating a single LiDAR scan'''
        theta=np.deg2rad(angDEG)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Define the rotation matrix
        rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        xy=np.stack([np.array(xCoords).flatten(),np.array(yCoords).flatten()])

        XY = np.matmul(rotation_matrix, xy)

        return XY[0,:], XY[1,:]


    def extractIDfromConv(self, maxX, maxY):
        centerXYs=self.scanCenter()
        dists=[]
        for centerX, centerY in centerXYs:
            dists.append(np.sqrt((centerY-maxY)**2+(centerX-maxX)**2))

        return np.argmin(dists)
   

    def scanCenter(self, withPooling=True):
        if withPooling==True:
            scanDimX, scanDimY = self.scanDimX, self.scanDimY
        else: 
            scanDimX, scanDimY = int(self.scanDimX*self.blockSize),int( self.scanDimY*self.blockSize)
        numScans=self.numRefScans()
        centerXYs=[]
        for i in range(numScans):
            rowIdx=i//self.horizNumScans
            colIdx=i%self.horizNumScans
            xIdx,yIdx=colIdx*scanDimX, rowIdx*scanDimY
            centerXYs.append((xIdx+(scanDimX//2), yIdx+(scanDimX//2)))
        
        return centerXYs 


    def find_closest_point(self, points, x_target, y_target):
        min_distance = float('inf')  # Initialize with positive infinity
        closest_point = None

        for idx, (x, y) in enumerate(points):
            distance = math.sqrt((x - x_target)**2 + (y - y_target)**2)
            if distance < min_distance:
                min_distance = distance
                closeX, closeY = x, y
                closest_point_id = idx

        return closeX, closeY, closest_point_id, min_distance


    '''Scan Matching'''
    def makingReferenceGrid(self, rot=None, dim_3=False, minPoints=None):
        '''Converting a dataset of LiDAR scans into a refernce grid image'''
        if self.datasetName=='Jackal':
            refRange=[i for i in self.framesRef]
            numScans=len(refRange)

            vertNumScans=np.ceil(numScans/self.horizNumScans)
            height,width=int(vertNumScans*self.scanDimY),int(self.horizNumScans*self.scanDimX)
            refernceGrid=np.zeros((height,width))
            for i,idx in enumerate(refRange):
                refScan= self.loadingCroppingFiltering_2DScan('REF', idx)
                rowIdx=i//self.horizNumScans
                colIdx=i%self.horizNumScans
                xIdx,yIdx=colIdx*self.scanDimX, rowIdx*self.scanDimY
                refernceGrid[yIdx:yIdx+(self.scanDimY), xIdx:xIdx+(self.scanDimX)]=refScan

        elif dim_3 == True:
            numScans=self.numRefScans()
            vertNumScans=np.ceil(numScans/self.horizNumScans)
            depth,width=int(vertNumScans*self.scanDimY),int(self.horizNumScans*self.scanDimX)
            height= np.shape(self.loadingCroppingFiltering_2DScan('REF', 0, dim_3=True))[2]
            refernceGrid=np.zeros((width,depth,height))
            

            for i in range(numScans):
                refScan= self.loadingCroppingFiltering_2DScan('REF', i, dim_3=True)
                rowIdx=i//self.horizNumScans
                colIdx=i%self.horizNumScans
                xIdx,yIdx=colIdx*self.scanDimX, rowIdx*self.scanDimY
                refernceGrid[ xIdx:xIdx+(self.scanDimX), yIdx:yIdx+(self.scanDimY),:]=refScan
        else:
            numScans=self.numRefScans()
            vertNumScans=np.ceil(numScans/self.horizNumScans)
            height,width=int(vertNumScans*self.scanDimY*self.blockSize),int(self.horizNumScans*self.scanDimX*self.blockSize)
            refernceGrid=np.zeros((height,width))
            points=[]

            for i in range(numScans):
                t=time.time()
                if rot!=None:
                    refScan= self.loadingCroppingFiltering_2DScan('REF', i, rotDeg=-np.rad2deg(rot[i]))
                else:
                    refScan= self.loadingCroppingFiltering_2DScan('REF', i)
                # if minPoints is not None and np.count_nonzero(refScan == 1) > minPoints:
                #     downSampledRefScan = np.zeros_like(refScan)
                #     ones_indices = np.argwhere(refScan == 1)
                    
                #     if len(ones_indices) <= minPoints:
                #         downSampled_indices = ones_indices
                #     else:
                #         downSampled_indices = ones_indices[np.random.choice(len(ones_indices), minPoints, replace=False)]
                    
                #     for index in downSampled_indices:
                #         downSampledRefScan[tuple(index)] = 1
                        
                rowIdx=i//self.horizNumScans
                colIdx=i%self.horizNumScans
                xIdx,yIdx=colIdx*refScan.shape[0], rowIdx*refScan.shape[1]
                refernceGrid[yIdx:yIdx+(refScan.shape[1]), xIdx:xIdx+(refScan.shape[0])]=refScan
                points.append(self.pointDensity)
                print(f'ref: {i}, pointsCountAvg:{sum(points)/len(points)}, time: {time.time()-t}')   
        
        return refernceGrid
     

    # def scanMatchWithConvolution(self,refernceGRid, idx, rotQuery=None):
    #     # if rotQuery is not None and np.any(rotQuery):
    #     kernel=np.fliplr(np.flipud(rotQuery))
    #     # else:
    #     #     localScan= self.loadingCroppingFiltering_2DScan('QUERY', idx)
    #     #     kernel=np.fliplr(np.flipud(localScan))  
    #     convolved= signal.fftconvolve(refernceGRid, kernel, mode='same')


    #     maxY, maxX = np.unravel_index(np.argmax(convolved), convolved.shape)
    #     return maxY, maxX, convolved
    
     
    def scanMatchWithConvolution(self, referenceGrid, idx, rotQuery=None, globalSearch=True):
        # Move the reference grid and rotated query to the GPU
        # referenceGrid_gpu = cp.asarray(referenceGrid)
        
        if rotQuery is not None and np.any(rotQuery):
            kernel_gpu = cp.fliplr(cp.flipud(cp.asarray(rotQuery)))
        else:
            # Assuming self.loadingCroppingFiltering_2DScan is modified to return a GPU array
            localScan = cp.asarray(self.loadingCroppingFiltering_2DScan('QUERY', idx))
            kernel_gpu = cp.fliplr(cp.flipud(cp.asarray(localScan)))
        
        # Perform the convolution on the GPU
        convolved_gpu = fftconvolve(referenceGrid, kernel_gpu, mode='same')

        if self.n>1 and globalSearch==True:
            # Get the indices of the top n values in the flattened array
            top_n_indices = cp.argpartition(convolved_gpu.ravel(), -self.n)[-self.n:]
            # Sort these indices to get the actual top n values in order
            top_n_indices_sorted = top_n_indices[cp.argsort(convolved_gpu.ravel()[top_n_indices])[::-1]]
            # Convert flat indices back to 3D indices
            max_indices = cp.unravel_index(top_n_indices_sorted, convolved_gpu.shape)
            # Extract maxY and maxX coordinates
            
            maxY, maxX= list(cp.asnumpy(max_indices[0])), list(cp.asnumpy(max_indices[1]))
            align=[float(cp.asnumpy(convolved_gpu[maxY[m], maxX[m]])) for m in range(len(maxY))]

            maxY_cpu = maxY#cp.asnumpy(maxY)
            maxX_cpu = maxX#cp.asnumpy(maxX)
            align_cpu = align#cp.asnumpy(align)

        if self.n==1 or globalSearch==False:
            # Find the maximum location on the GPU
            maxY, maxX  = cp.unravel_index(cp.argpartition(convolved_gpu.ravel(), -1)[-1], convolved_gpu.shape)
            # align=convolved_gpu[maxY, maxX]
            
            # Move the result back to the CPU
            maxY_cpu = cp.asnumpy(maxY)
            maxX_cpu = cp.asnumpy(maxX)
            align_cpu = cp.asnumpy(convolved_gpu[maxY, maxX])
        # convolved_cpu = cp.asnumpy(convolved_gpu)

        return maxY_cpu, maxX_cpu, align_cpu, cp.asnumpy(convolved_gpu)
    
    
    
    def update(self, i, rotIncr, xr, yr, yawR, xq, yq, yawQ, centerXs, centerYs, closeIds, returnArray=False, returnConv=False ):
        global processTime, queryLoadTime, queryPtcld
        
        # t=time.time()
        # print('')
        '''Rotate Convolve'''
        processTime=0
        maxPos, alignments, rotations, meanAlignments=[], [], [], []
        queryPtcld = self.loading2DScan('QUERY', i)
        # print(np.shape(np.array(queryPtcld[:,:2])))
        # np.save('Venman3_Query_Ptcld.npy', np.array(queryPtcld))
        

        # '''To delete'''
        # numScans=1500
        # vertNumScans=np.ceil(numScans/self.horizNumScans)
        # height,width=int(vertNumScans*self.scanDimY*self.blockSize),int(self.horizNumScans*self.scanDimX*self.blockSize)
        # refgrid=skimage.measure.block_reduce(np.ones((height,width)), self.blockSize, func=self.poolFunc)
        # self.refgrid=cp.asarray(refgrid)
        # '''To delete'''

        queryLoadTime=0#time.time()-t
        secondLoadingTime=0
        t=time.time()
        # def process_scan(j, n=n):
        #     global processTime, queryLoadTime, queryPtcld
        for j in range(0, 360, rotIncr):
            tq=time.time()
            queryScan=self.processing2DScan('QUERY', queryPtcld, rotDeg=j)
            # queryScan=np.array(queryScan, dtype=np.uint8)
            queryScan=skimage.measure.block_reduce(queryScan, self.blockSize, func=self.poolFunc)
            queryLoadTime+=(time.time()-tq)

            ts=time.time()
            maxY, maxX, align = self.scanMatchWithConvolution(self.refgrid, i, rotQuery=queryScan)[:3]
            processTime+=(time.time()-ts)
        
            # print(f'singleConvTime: {round(singleConvTime,4)}, queryLoadTime:{queryLoadTime}')
            
            rotations.append(j)
            
            if self.n==1:
                maxPos.append((maxX, maxY))
                alignments.append(align)
            if self.n>1:
                maxPos.append([(x, y) for x, y in zip(maxX, maxY)])
                alignments.append(align)

            

        # with ThreadPoolExecutor(max_workers=2) as executor:
        #     executor.map(process_scan, range(0, 360, rotIncr))
        
        # processTime-=queryLoadTime

        t2=time.time()
        if self.n==1:
            headingIdx=np.argmax(alignments)
            heading=rotations[headingIdx]
            maxX,maxY=maxPos[headingIdx]
            print(headingIdx, maxX, maxY)
            idx=self.extractIDfromConv(maxX, maxY)

            tSeconfLoad=time.time()

            queryScan= self.loadingCroppingFiltering_2DScan('QUERY', i, rotDeg=heading)
            refScan= self.loadingCroppingFiltering_2DScan('REF', idx)

            deltaT=time.time()-tSeconfLoad
            secondLoadingTime+=deltaT


            refScanPlot=refScan
            refScan=cp.asarray(refScanPlot)

            maxY, maxX, alignRefmatch, refMatchCorr= self.scanMatchWithConvolution(refScan, i, rotQuery=queryScan, globalSearch=False)
            matchXDelta, matchYDelta = maxX-(refScan.shape[0]//2), maxY-(refScan.shape[1]//2)

            
            

        
        if self.n>1:
            flattened_alignments = cp.asnumpy(alignments).flatten()
            top_n_indices = np.argpartition(flattened_alignments, -self.n)[-self.n:]
            top_n_indices_sorted = top_n_indices[np.argsort(flattened_alignments[top_n_indices])[::-1]]
            

            flattened_maxPos = [pos for sublist in maxPos for pos in sublist]
            top_n_positions = [flattened_maxPos[idx] for idx in top_n_indices_sorted]

            flattened_rotations = np.repeat(rotations, self.n)
            top_n_rotations = [flattened_rotations[idx] for idx in top_n_indices_sorted]

            indices=[self.extractIDfromConv(maxX, maxY) for maxX,maxY in top_n_positions]

            height,width=int(self.scanDimX*self.blockSize),int(self.scanDimY*self.blockSize)
            alignments, newPos=[],[]
            for k in range(self.n):
                tSeconfLoad=time.time()
                queryScan= self.loadingCroppingFiltering_2DScan('QUERY', i, rotDeg=top_n_rotations[k])
                # queryScan=np.array(queryScan, dtype=np.uint8)
                refScan= self.loadingCroppingFiltering_2DScan('REF', indices[k])
                deltaT=time.time()-tSeconfLoad
                secondLoadingTime+=deltaT
                # refScan=np.array(refScan, dtype=np.uint8)
                refScan=cp.asarray(refScan)
                newMaxY, newMaxX, alignRefmatch, refMatchCorr = self.scanMatchWithConvolution(refScan, i, rotQuery=queryScan, globalSearch=False)
                alignments.append(alignRefmatch)
                newPos.append((newMaxX, newMaxY))
            maxX,maxY=newPos[np.argmax(alignments)]
            idx=indices[np.argmax(alignments)]
            heading=top_n_rotations[np.argmax(alignments)]
            matchXDelta, matchYDelta = maxX-(width//2), maxY-(height//2)
            rotations=top_n_rotations

            # queryScan= self.loadingCroppingFiltering_2DScan('QUERY', i)
            # height,width=int(self.scanDimX*self.blockSize),int(self.scanDimY*self.blockSize)
            # refMatchScan=np.zeros((height,n*width))
            # for k in range(n):
            #     refScan= self.loadingCroppingFiltering_2DScan('REF', indices[k], rotDeg=top_n_rotations[k]) #refMatchScan
            #     xIdx,yIdx=k*width, 0
            #     refMatchScan[yIdx:yIdx+height, xIdx:xIdx+width]=refScan

            # refMatchScanPlot=refMatchScan#np.array(refMatchScan, dtype=np.uint8)
            # refMatchScan=cp.asarray(refMatchScanPlot)

            # newheight,newwidth=int(self.scanDimX*self.blockSize),int(self.scanDimY*self.blockSize)
            # newMaxY, newMaxX, alignRefmatch = self.scanMatchWithConvolution(refMatchScan, i, rotQuery=queryScan)[:3]
            # orderInTopN=newMaxX//newwidth
            # idx=indices[orderInTopN]
            # matchXDelta, matchYDelta = (newMaxX%newwidth)-(newwidth//2), newMaxY-(newheight//2)


            # print('')
            # print(f'idx from top n {newMaxX//width}, top n ids {indices}, x center: {(newMaxX%newwidth)}, maxX:{newMaxX}, half: {(newwidth//2)}')
            # print(f'matchXYdelta: ({matchXDelta}, {matchYDelta}), hightwidth( {newheight},{newwidth})  maxXY ({newMaxX},{newMaxY}) ')
            # refMatchScanPlot=refMatchScanPlot[0:width, orderInTopN*height:(orderInTopN*height)+height]
            

        # fig, ax1= plt.subplots(1,1)

        # ax1.imshow(queryScan, cmap='Blues')
        # ax1.imshow(self.translateScan(refScanPlot, matchXDelta, matchYDelta), cmap='Reds', alpha=0.5)

        # # refgridNumpy=cp.asnumpy(self.refgrid)
        # # ax1.imshow(refgridNumpy[:60, :60], cmap='Greys')

        # # ax1.plot(xr,yr, '-', color='tab:blue')
        # # ax1.plot(xq,yq, '.', color='tab:green')
        
 
        # # Remove axis ticks and numbers
        # ax1.set_xticks([])
        # ax1.set_yticks([])
        # # Remove axis spines
        # ax1.spines['top'].set_visible(False)
        # ax1.spines['right'].set_visible(False)
        # ax1.spines['bottom'].set_visible(False)
        # ax1.spines['left'].set_visible(False)
        
        # # plt.savefig('highRes_refQueryPositions.png', dpi=300, bbox_inches='tight')
        # # plt.savefig('highRes_singleLowResRefGrid.png', dpi=300, bbox_inches='tight')
        # plt.show()
        # assert False

        
        
        '''Match and shift'''
        theta=yawR[idx]
        shiftedX = xr[idx] + (math.cos(theta)*(matchXDelta*self.mapRes)) - (math.sin(theta)*(matchYDelta*self.mapRes))
        shiftedY = yr[idx] + (math.sin(theta)*(matchXDelta*self.mapRes)) + (math.cos(theta)*(matchYDelta*self.mapRes))

        '''Distance'''
        dist=np.sqrt((yq[i]-shiftedY)**2+(xq[i]-shiftedX)**2)

        shiftTime=time.time()-t2 - secondLoadingTime
        duration=time.time()-t - secondLoadingTime
        # print(f'time for relativeShift: {shiftTime}')
        if returnArray == True: 
            '''Alignment'''
            queryScan= self.loadingCroppingFiltering_2DScan('QUERY', i, rotDeg=heading)
            maxY, maxX, alignMatch, allCor = self.scanMatchWithConvolution(self.refgridFull, i, rotQuery=queryScan, globalSearch=False)
            maxYq, maxXq, alignSame= self.scanMatchWithConvolution(cp.asarray(queryScan), i, rotQuery=queryScan, globalSearch=False)[:3]
            alignVals= (alignSame, None, alignMatch, allCor)

            '''Closest'''
            # closestScan= self.loadingCroppingFiltering_2DScan('REF', closeIds[i])
            # maxYclose, maxXsclose, aligncloseMatch = self.scanMatchWithConvolution(cp.asarray(closestScan), i, rotQuery=queryScan)[:3]
            # matchXDeltaClose, matchYDeltaClose = maxXsclose-(closestScan.shape[0]//2), maxYclose-(closestScan.shape[1]//2)
            refClosest=self.loadingCroppingFiltering_2DScan('REF', closeIds[i], sameProcess=True)
            # refClosest= self.translateScan(refClosest, matchXDeltaClose, matchYDeltaClose)

    


            '''RefMatch and query'''
            queryScan= self.loadingCroppingFiltering_2DScan('QUERY', i, rotDeg=heading, sameProcess=True)
            refScan= self.loadingCroppingFiltering_2DScan('REF', idx, sameProcess=True)
            refShiftMatchScan= self.translateScan(refScan, matchXDelta, matchYDelta)

            return queryScan, refScan, refShiftMatchScan, refClosest, idx, maxX, maxY, matchXDelta, matchYDelta, shiftedX, shiftedY, rotations, alignments, dist, alignVals
        
        elif returnConv == True: 
            # centerXY=self.scanCenter(withPooling=False)
            # centerXs, centerYs = zip(*centerXY)
            queryScan= self.loadingCroppingFiltering_2DScan('QUERY', i, rotDeg=heading)
            # queryScan=np.array(queryScan, dtype=np.uint8)
            maxY, maxX, alignMatch, allCor = self.scanMatchWithConvolution(self.refgridFull, i, rotQuery=queryScan, n=n)
            # matchConv=allCor[centerYs[idx]-(self.scanDimY):centerYs[idx]+(self.scanDimY), centerXs[idx]-(self.scanDimX):centerXs[idx]+(self.scanDimX)]
            return refMatchCorr, allCor, dist
        
        else:
            return idx, maxX, maxY, matchXDelta, matchYDelta, shiftedX, shiftedY, rotations, alignments, dist, queryLoadTime, processTime, shiftTime, duration



class LoopClose():
    def __init__(self, datasetName, refDataset, refINS, scanDimX, scanDimY, mapRes, horizNumScans, intensityFilter, pointInterval, framesRef): 
        self.mapRes=mapRes
        self.scanDimX=scanDimX
        self.scanDimY= scanDimY
        self.horizNumScans=horizNumScans
        self.intensityFilter=intensityFilter
        self.datasetName=datasetName
        self.pointInterval=pointInterval

        self.framesRef=framesRef
     
        self.refDataset= refDataset 
        self.refINS = refINS

        self.pointDistributionRef=self.distributionOfPoints('REF')


    '''Loading data'''
    def discretizeData(self, velodyne_dir, frames):
        
        if self.datasetName == 'OxfordRadar':
            timestamps_path = velodyne_dir + '.timestamps'
            velodyne_timestamps = np.loadtxt(timestamps_path, delimiter=' ', usecols=[0], dtype=np.int64)

            filenames,timestamps=[], []
            for velodyne_timestamp in velodyne_timestamps:

                filenames.append(os.path.join(velodyne_dir, str(velodyne_timestamp) + '.png'))
                timestamps.append(velodyne_timestamp)
            
            discreteFilenames= [filenames[i] for i in frames]
            discreteTimestamps= [timestamps[i] for i in frames]
        
        elif self.datasetName == 'WildPlaces' or self.datasetName == 'Mulran' or self.datasetName == 'BenchmarkOxford':
            filenames = os.listdir(velodyne_dir)
            # print(f'{velodyne_dir}, {len(filenames)}')
            timestamps= [file[:-4] for file in filenames] 

            discreteFilenames= [filenames[i] for i in frames]
            discreteTimestamps= [timestamps[i] for i in frames]
        
        
        return discreteFilenames, discreteTimestamps


    def loadOxfordRadarPtcld(self,oxfordRadarFile):
        hdl32e_range_resolution = 0.002  # m / pixel
        hdl32e_minimum_range = 1.0
        hdl32e_elevations = np.array([-0.1862, -0.1628, -0.1396, -0.1164, -0.0930,
                                    -0.0698, -0.0466, -0.0232, 0., 0.0232, 0.0466, 0.0698,
                                    0.0930, 0.1164, 0.1396, 0.1628, 0.1862, 0.2094, 0.2327,
                                    0.2560, 0.2793, 0.3025, 0.3259, 0.3491, 0.3723, 0.3957,
                                    0.4189, 0.4421, 0.4655, 0.4887, 0.5119, 0.5353])[:, np.newaxis]
        hdl32e_base_to_fire_height = 0.090805
        hdl32e_cos_elevations = np.cos(hdl32e_elevations)
        hdl32e_sin_elevations = np.sin(hdl32e_elevations)

        example = cv2.imread(oxfordRadarFile, cv2.IMREAD_GRAYSCALE)
        intensities, ranges_raw, angles_raw, timestamps_raw = np.array_split(example, [32, 96, 98], 0)
        ranges = np.ascontiguousarray(ranges_raw.transpose()).view(np.uint16).transpose()
        ranges = ranges * hdl32e_range_resolution
        angles = np.ascontiguousarray(angles_raw.transpose()).view(np.uint16).transpose()
        angles = angles * (2. * np.pi) / 36000
        approximate_timestamps = np.ascontiguousarray(timestamps_raw.transpose()).view(np.int64).transpose()
        valid = ranges > hdl32e_minimum_range
        z = hdl32e_sin_elevations * ranges - hdl32e_base_to_fire_height
        xy = hdl32e_cos_elevations * ranges
        x = np.sin(angles) * xy
        y = -np.cos(angles) * xy

        xf = x[valid].reshape(-1)
        yf = y[valid].reshape(-1)
        zf = z[valid].reshape(-1)
        intensityf = intensities[valid].reshape(-1).astype(np.float32)
        ptcld = np.stack((xf, yf, zf, intensityf), 0)
        return ptcld


    def jackal_LoadRosbag(self, bag_dir, frames, idx, topic='/velodyne_points'):
        timestamps=[]
        with rosbag.Bag(bag_dir, 'r') as bag:
            # Get messages from the specified topic
            messages = bag.read_messages(topics=[topic])
            ith_message = next(itertools.islice(messages, idx, None), None)

            if ith_message[1]._type == 'sensor_msgs/PointCloud2':
                timestamps.append(ith_message[1].header.stamp.to_sec())
                pc_data = pc2.read_points(ith_message[1], field_names=("x", "y", "z", "intensity"), skip_nans=True)
                ptcld = np.array(list(pc_data))
               
                return ptcld
    

    def jackal_LoadRosbagPoses(self, bag_dir, topic='/odom/true'):
        x,y, yaws= [],[], []
        odomTimestamps=[]
        with rosbag.Bag(bag_dir, 'r') as bag:
            # Get messages from the specified odometry topic
            messages = bag.read_messages(topics=[topic])
            # Iterate through odometry messages
            for _, msg, _ in messages:
                # Check if the message type is Odometry
                if msg._type == 'nav_msgs/Odometry':
                    orientation_quaternion = (
                        msg.pose.pose.orientation.x,
                        msg.pose.pose.orientation.y,
                        msg.pose.pose.orientation.z,
                        msg.pose.pose.orientation.w )
                    roll, pitch, yaw = euler_from_quaternion(orientation_quaternion)
                    odomTimestamps.append(msg.header.stamp.to_sec())
                    # Extract relevant information (x, y position)
                    x.append(msg.pose.pose.position.x)              
                    y.append(msg.pose.pose.position.y) 
                    yaws.append(yaw)
                    # print(f'{msg.pose.pose.position.x}, {msg.pose.pose.position.y}, {yaw}')
        return x, y, yaws, odomTimestamps


    def distributionOfPoints(self, REForQUERY):
        if REForQUERY=='REF':
            velodyne_dir=self.refDataset
            frames=self.framesRef
        elif REForQUERY=='QUERY':
            velodyne_dir=self.queryDataset
            frames=self.framesQuery

        if self.datasetName=='WildPlaces':
            filenames, discreteTimestamps=self.discretizeData(velodyne_dir,frames)
            numPoints=[]
            for i in range(len(filenames)):
                pcd = o3d.io.read_point_cloud(velodyne_dir+'/'+filenames[i])
                ptcld=np.asarray(pcd.points)
                _zfilter= (ptcld[:,2] > 1) & (ptcld[:,2] < 3) 
                ptcld=ptcld[_zfilter,:]
                numPoints.append(len(ptcld))

            return numPoints
        elif self.datasetName=='BenchmarkOxford':
            filenames, discreteTimestamps=self.discretizeData(velodyne_dir,frames)
            numPoints=[]
            for i in range(len(filenames)):
                pc = np.fromfile((velodyne_dir+'/'+filenames[i]), dtype=np.float64)
                ptcld = np.reshape(pc,(pc.shape[0]//3,3))
                z_range=(-0.3,0.3)
                half_x = self.scanDimX * self.mapRes // 2
                half_y = self.scanDimY * self.mapRes // 2
                grid3d,_=np.histogramdd(ptcld, bins=(np.linspace(-half_x, half_x, self.scanDimX+1),
                                                    np.linspace(-half_y, half_y, self.scanDimY+1),
                                                    np.arange(z_range[0], z_range[1], self.mapRes)))
                
                grid=np.sum(grid3d[:,:,:], axis=2)
                thresh=0.5
                grid[grid<=thresh]=0
                grid[grid>thresh]=1

                numPoints.append(np.count_nonzero(grid))

            return numPoints
        else: 
            print('Invalid dataset for point distribution')
                

    ''' Processing LiDAR'''
    def numRefScans(self):
        if self.datasetName=='Kitti':
           numScans=len(self.refDataset)

        elif self.datasetName=='OxfordRadar' or self.datasetName=='Mulran' or self.datasetName=='BenchmarkOxford':
            filenames,timestamps=self.discretizeData(self.refDataset,self.framesRef)
            numScans=len(filenames)
        
        elif self.datasetName=='Jackal':
            with rosbag.Bag(self.refDataset, 'r') as bag:
                numScans= bag.get_message_count('/velodyne_points')
        
        return numScans
    

    def loadingCroppingFiltering_2DScan(self, REForQUERY,idx, rotDeg=None, dim_3=False, raw=False):
        velodyne_dir=self.refDataset
        frames=self.framesRef



        if self.datasetName=='OxfordRadar':
            filenames, discreteTimestamps=self.discretizeData(velodyne_dir, frames)
            filename=filenames[idx]
            ptcld= self.loadOxfordRadarPtcld(filename)
            # Filter based on conditions
            if raw==False: 
                normReflectance=(ptcld[3, :] - np.min(ptcld[3, :])) / (np.max(ptcld[3, :]) - np.min(ptcld[3, :]))
                _zfilter= (normReflectance > self.intensityFilter) & (ptcld[2, :] > 0.05) #& (ptcld[2, :] < 2) 
                _good=np.where(_zfilter[::self.pointInterval])[0]
                ptcld=ptcld[:3,_good].T
            else:
                ptcld=ptcld[:3,::2].T

            z_range=(np.min(ptcld[:,2]), np.max(ptcld[:,2])) 
            thresh=1
    
        elif self.datasetName=='WildPlaces':
            filenames, discreteTimestamps=self.discretizeData(velodyne_dir,frames)
            pcd = o3d.io.read_point_cloud(velodyne_dir+'/'+filenames[idx])
            ptcld=np.asarray(pcd.points)
            # Filter based on conditions
            _zfilter= (ptcld[:,2] > 1) & (ptcld[:,2] < 3) 
            ptcld=ptcld[_zfilter,:]

            z_range=(np.min(ptcld[:,2]), np.max(ptcld[:,2])) 
            thresh=1

            # if REForQUERY=='REF':
            #     pointDistribution=self.pointDistributionRef
            #     indices = np.random.choice(pointDistribution[idx], int(np.min(pointDistribution)), replace=False)
            #     ptcld=ptcld[indices,:]
        
        elif self.datasetName=='BenchmarkOxford':
            filenames, discreteTimestamps=self.discretizeData(velodyne_dir,frames)
            pc = np.fromfile((velodyne_dir+'/'+filenames[idx]), dtype=np.float64)
            ptcld = np.reshape(pc,(pc.shape[0]//3,3))
            _zfilter= (ptcld[:, 2] < 0.3) & (ptcld[:, 2] > -0.3) 
            _good=np.where(_zfilter[::self.pointInterval])[0]
            ptcld=ptcld[:,:]
            z_range=(-0.3,0.3)
            thresh=0.5

        
        elif self.datasetName=='Mulran':
            filenames, discreteTimestamps=self.discretizeData(velodyne_dir,frames)
            ptcld = np.fromfile((velodyne_dir+'/'+filenames[idx]), dtype=np.float32).reshape(-1, 4)
            if raw == False: 
                normReflectance=(ptcld[:, 3] - np.min(ptcld[:, 3])) / (np.max(ptcld[:, 3]) - np.min(ptcld[:, 3]))
                _zfilter= (normReflectance > self.intensityFilter) & (ptcld[:, 2] > 0.1) #& (ptcld[:, 2] < 6)        
                _good=np.where(_zfilter[::self.pointInterval])[0]
                ptcld=ptcld[_good, :3]
            else: 
                ptcld=ptcld[:, :3]

            z_range=(np.min(ptcld[:,2]), np.max(ptcld[:,2])) 
            thresh=1
            
        elif self.datasetName=='Jackal':
            ptcld=self.jackal_LoadRosbag(velodyne_dir, frames, idx)
            intensity = ptcld[:, 3]
            norm_intensity = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))
            _range = np.sqrt(np.square(ptcld[:, 0]) + np.square(ptcld[:, 1]))
            _good = (_range < 15) & (ptcld[:, 2] < 1.8) & (ptcld[:, 2] > 0.1) & (norm_intensity > self.intensityFilter)
            ptcld = ptcld[(ptcld[:, 2] > 0.1) &(norm_intensity > self.intensityFilter), :3]

            z_range=(np.min(ptcld[:,2]), np.max(ptcld[:,2])) 
            thresh=1


        if rotDeg != None:
            x,y=self.applyingRotation2D(rotDeg, ptcld[:,1],ptcld[:,0])
            ptcld[:,1]=x
            ptcld[:,0]=y

        half_x = self.scanDimX * self.mapRes // 2
        half_y = self.scanDimY * self.mapRes // 2
        grid3d,_=np.histogramdd(ptcld, bins=(np.linspace(-half_x, half_x, self.scanDimX+1),
                                            np.linspace(-half_y, half_y, self.scanDimY+1),
                                            np.arange(z_range[0], z_range[1], self.mapRes)))
        
        grid=np.sum(grid3d[:,:,:], axis=2)
        grid[grid<=thresh]=0
        grid[grid>thresh]=1

        numPoints2d=1000
        if np.count_nonzero(grid)>numPoints2d:
            ones_indices = np.argwhere(grid == 1)
            selected_indices = np.random.choice(len(ones_indices), numPoints2d, replace=False)
            grid = np.zeros_like(grid)
            for index in selected_indices:
                grid[ones_indices[index][0], ones_indices[index][1]] = 1


        if dim_3==True:
            grid3d[grid3d<=thresh]=0
            grid3d[grid3d>thresh]=1
            return grid3d
        else: 
            return grid.T
        

    def scanPoses(self, REForQUERY):
        if REForQUERY == 'REF':
            poses_file=self.refINS
            velodyne_dir=self.refDataset
            frames=self.framesRef
        elif REForQUERY== 'QUERY':
            poses_file=self.queryINS
            velodyne_dir=self.queryDataset
            frames=self.framesQuery

        if self.datasetName == 'OxfordRadar':
            timestamps_path=velodyne_dir + '.timestamps'
            timestampsAll=list(np.loadtxt(timestamps_path, delimiter=' ', usecols=[0], dtype=np.int64))
            timestamps=[timestampsAll[i] for i in frames]

            ins_data = pd.read_csv(poses_file)
            timestamps_np = ins_data['timestamp'].to_numpy()

            closest_indices = [np.argmin(np.abs(timestamps_np - time)) for time in timestamps]
            filtered_data_np = ins_data.iloc[closest_indices]
            x, y, yaw = filtered_data_np['easting'].to_numpy(), filtered_data_np['northing'].to_numpy(), filtered_data_np['yaw'].to_numpy()
                
        if self.datasetName == 'WildPlaces':
            filenames,timestamps=self.discretizeData(velodyne_dir,frames)
            ins_data = pd.read_csv(poses_file)
            timestamps_np = ins_data['timestamp'].to_numpy()

            closest_indices = [np.argmin(np.abs(timestamps_np - float(time))) for time in timestamps]
            filtered_data_np = ins_data.iloc[closest_indices]

            qx, qy, qz, qw = filtered_data_np['qx'].values.tolist(), filtered_data_np['qy'].values.tolist(), filtered_data_np['qz'].values.tolist(), filtered_data_np['qw'].values.tolist()
            yaws= [math.atan2(2 * (qw[i] * qz[i] + qx[i] * qy[i]), 1 - 2 * (qy[i]**2 + qz[i]**2)) for i in range(len(qx))]
            x,y, yaw=filtered_data_np['x'].values.tolist(), filtered_data_np['y'].values.tolist(), yaws
        
        if self.datasetName == 'BenchmarkOxford':
            filenames,timestamps=self.discretizeData(velodyne_dir,frames)
            ins_data = pd.read_csv(poses_file)
            timestamps_np = ins_data['timestamp'].to_numpy()
            closest_indices = [np.argmin(np.abs(timestamps_np - float(time))) for time in timestamps]
            filtered_data_np = ins_data.iloc[closest_indices]
            x,y, yaw=filtered_data_np['easting'].values.tolist(), filtered_data_np['northing'].values.tolist(), None 


        if self.datasetName == 'Mulran':
            filenames,timestamps=self.discretizeData(velodyne_dir,frames)
            ins_data = pd.read_csv(poses_file)

            print(ins_data.shape)
            allX, allY=ins_data.iloc[:, 4].to_numpy(), ins_data.iloc[:, 8].to_numpy()
            timestamps_np=ins_data.iloc[:, 0].to_numpy()
            closest_indices = [np.argmin(np.abs(timestamps_np - float(time))) for time in timestamps]

            x,y=allX[closest_indices], allY[closest_indices]
            print(len(closest_indices), len(x))
            
            yaw=None

        

        if self.datasetName=='Jackal':
            x,y,yaws,odomTimestamps=self.jackal_LoadRosbagPoses(velodyne_dir)
            yaw=None
            veloTimestamps=[]
            with rosbag.Bag(velodyne_dir, 'r') as bag:
                # Get messages from the specified topic
                for _, msg, _ in bag.read_messages(topics=['/velodyne_points']):
                    if msg._type == 'sensor_msgs/PointCloud2':
                        veloTimestamps.append(msg.header.stamp.to_sec())

            # Adjust starting time stamps 
            diffStartTime=odomTimestamps[0] - veloTimestamps[0]
            odomTimestamps=[time-diffStartTime for time in odomTimestamps]  

            # Extract closest odom index for each scan 
            discreteVeloTimestamps= [veloTimestamps[i] for i in frames]
            closestTimestampsIdxs=[odomTimestamps.index(min(odomTimestamps, key=lambda x: abs(x - float(time)))) for time in discreteVeloTimestamps]
            get_indices = itemgetter(*closestTimestampsIdxs)
            print(f'odo start {odomTimestamps[0]}, velo start {veloTimestamps[0]}, diff: {odomTimestamps[0] - veloTimestamps[0]} ')
            x,y,yaw=get_indices(x),get_indices(y), get_indices(yaws)  

        return x,y,yaw


    def translateScan(self, scan, deltaX, deltaY):
        height, width= scan.shape
        shifted_image = np.zeros_like(scan, dtype=int)
        x_start, x_end = max(0, deltaX), min(width, width + deltaX)
        y_start, y_end = max(0, deltaY), min(height, height + deltaY)
        shifted_image[y_start - deltaY:y_end - deltaY, x_start - deltaX:x_end - deltaX] = scan[y_start:y_end, x_start:x_end]

        return shifted_image
    

    def applyingRotation2D(self, angDEG,  xCoords, yCoords):
        '''Rotating a single LiDAR scan'''
        theta=np.deg2rad(angDEG)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Define the rotation matrix
        rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        xy=np.stack([np.array(xCoords).flatten(),np.array(yCoords).flatten()])

        XY = np.matmul(rotation_matrix, xy)

        return XY[0,:], XY[1,:]




    '''Scan Matching'''
    def makingReferenceGrid(self, rot=None, dim_3=False):
        '''Converting a dataset of LiDAR scans into a refernce grid image'''
        if self.datasetName=='Jackal':
            refRange=[i for i in self.framesRef]
            numScans=len(refRange)

            vertNumScans=np.ceil(numScans/self.horizNumScans)
            height,width=int(vertNumScans*self.scanDimY),int(self.horizNumScans*self.scanDimX)
            refernceGrid=np.zeros((height,width))
            for i,idx in enumerate(refRange):
                refScan= self.loadingCroppingFiltering_2DScan('REF', idx)
                rowIdx=i//self.horizNumScans
                colIdx=i%self.horizNumScans
                xIdx,yIdx=colIdx*self.scanDimX, rowIdx*self.scanDimY
                refernceGrid[yIdx:yIdx+(self.scanDimY), xIdx:xIdx+(self.scanDimX)]=refScan

        elif dim_3 == True:
            numScans=self.numRefScans()
            vertNumScans=np.ceil(numScans/self.horizNumScans)
            depth,width=int(vertNumScans*self.scanDimY),int(self.horizNumScans*self.scanDimX)
            height= np.shape(self.loadingCroppingFiltering_2DScan('REF', 0, dim_3=True))[2]
            refernceGrid=np.zeros((width,depth,height))
            

            for i in range(numScans):
                refScan= self.loadingCroppingFiltering_2DScan('REF', i, dim_3=True)
                rowIdx=i//self.horizNumScans
                colIdx=i%self.horizNumScans
                xIdx,yIdx=colIdx*self.scanDimX, rowIdx*self.scanDimY
                refernceGrid[ xIdx:xIdx+(self.scanDimX), yIdx:yIdx+(self.scanDimY),:]=refScan
        else:
            numScans=self.numRefScans()
            vertNumScans=np.ceil(numScans/self.horizNumScans)
            height,width=int(vertNumScans*self.scanDimY),int(self.horizNumScans*self.scanDimX)
            refernceGrid=np.zeros((height,width))
            

            for i in range(numScans):
                if rot!=None:
                    refScan= self.loadingCroppingFiltering_2DScan('REF', i, rotDeg=-np.rad2deg(rot[i]))
                else:
                    refScan= self.loadingCroppingFiltering_2DScan('REF', i)
                    
                rowIdx=i//self.horizNumScans
                colIdx=i%self.horizNumScans
                xIdx,yIdx=colIdx*self.scanDimX, rowIdx*self.scanDimY
                refernceGrid[yIdx:yIdx+(self.scanDimY), xIdx:xIdx+(self.scanDimX)]=refScan
        
        return refernceGrid
     
    
    def scanMatchWithConvolution(self, refernceGRid, idx, rotQuery=None):
        if rotQuery is not None and np.any(rotQuery):
            kernel=np.fliplr(np.flipud(rotQuery))
        else:
            localScan= self.loadingCroppingFiltering_2DScan('QUERY', idx)
            kernel=np.fliplr(np.flipud(localScan))
            

        convolved= signal.fftconvolve(refernceGRid, kernel, mode='same')
        maxY, maxX = np.unravel_index(np.argmax(convolved), convolved.shape)

        return maxY, maxX, convolved
 

    def extractIDfromConv(self, maxX, maxY):
        centerXYs=self.scanCenter()
        dists=[]
        for centerX, centerY in centerXYs:
            dists.append(np.sqrt((centerY-maxY)**2+(centerX-maxX)**2))

        return np.argmin(dists)
   

    def scanCenter(self):
        numScans=self.numRefScans()
        centerXYs=[]
        for i in range(numScans):
            rowIdx=i//self.horizNumScans
            colIdx=i%self.horizNumScans
            xIdx,yIdx=colIdx*self.scanDimX, rowIdx*self.scanDimY
            centerXYs.append((xIdx+(self.scanDimX//2), yIdx+(self.scanDimX//2)))
        
        return centerXYs 



def Oxford_MatchShift(i, LPR, refgrid, centerXYs, xr, yr, yawR):
    '''Lidar Match'''
    maxY, maxX, convolved= LPR.scanMatchWithConvolution(refgrid, i, 'SINGLE')
    idx=LPR.extractIDfromConv(maxX, maxY)

    '''Finding realtive shift'''
    centerXs,centerYs=zip(*centerXYs)
    centerX, centerY = centerXs[idx], centerYs[idx]
    matchXDelta, matchYDelta = maxX-centerX, maxY-centerY

    '''Query Scan '''
    cropped_x, cropped_y= LPR.loadingCroppingFiltering_2DScan('QUERY', i)
    queryScan=LPR.makingImageFromLidarCoords(cropped_x, cropped_y) 
    
    '''Ref match scans'''
    cropped_x, cropped_y= LPR.loadingCroppingFiltering_2DScan('REF', idx)
    refConvMatchScan= LPR.makingImageFromLidarCoords(cropped_x, cropped_y)
    refShiftMatchScan= LPR.translateScan(refConvMatchScan, matchXDelta, matchYDelta)
    
    '''Scan align percentage '''
    maxAlignCount=np.min([np.count_nonzero(queryScan==1), np.count_nonzero(refShiftMatchScan==1)])
    alignFrac=(np.count_nonzero((queryScan+refShiftMatchScan)==2)/np.count_nonzero(queryScan==1))

    '''Shifted Pos'''
    theta=yawR[idx]-np.pi
    shiftedX = xr[idx] + (math.sin(theta)*(matchXDelta)) + (math.cos(theta)*(matchYDelta))
    shiftedY = yr[idx] + (math.cos(theta)*(matchXDelta)) - (math.sin(theta)*(matchYDelta)) 

    return idx, matchXDelta, matchYDelta, shiftedX, shiftedY, alignFrac


def Jackal_MatchShift(i, LPR, refgrid, centerXYs, xr, yr, yawR, framesQuery, framesRef, mapRes):
    queryIds=[j for j in framesQuery]
    refIDs=[j for j in framesRef]

    cropped_x, cropped_y= LPR.loadingCroppingFiltering_2DScan('QUERY', queryIds[i])
    xy_arr=np.transpose(np.stack([cropped_x, cropped_y],axis=0))
    queryScan=LPR.convert_pc_to_grid(xy_arr)
    
    maxY, maxX, convolved= LPR.scanMatchWithConvolution(refgrid, queryIds[i], 'SINGLE')
    idx=LPR.extractIDfromConv(maxX, maxY)

    '''Finding realtive shift'''
    centerXs,centerYs=zip(*centerXYs)
    centerX, centerY = centerXs[idx], centerYs[idx]
    matchXDelta, matchYDelta = maxX-centerX, maxY-centerY
    
    '''Ref match scans'''
    cropped_x, cropped_y= LPR.loadingCroppingFiltering_2DScan('REF', refIDs[idx])
    xy_arr=np.transpose(np.stack([cropped_x, cropped_y],axis=0))
    refConvMatchScan=LPR.convert_pc_to_grid(xy_arr)
    refShiftMatchScan= LPR.translateScan(refConvMatchScan, matchXDelta, matchYDelta)

    '''Shifted Pos'''
    shiftedX = xr[idx] + (math.cos(yawR[idx])*(matchXDelta*mapRes)) - (math.sin(yawR[idx])*(matchYDelta*mapRes))
    shiftedY = yr[idx] + (math.sin(yawR[idx])*(matchXDelta*mapRes)) + (math.cos(yawR[idx])*(matchYDelta*mapRes))
    
    '''Scan align percentage '''
    maxAlignCount=np.min([np.count_nonzero(queryScan==1), np.count_nonzero(refShiftMatchScan==1)])
    alignFrac=(np.count_nonzero((queryScan+refShiftMatchScan)==2))/np.count_nonzero(queryScan==1)

    return idx, matchXDelta, matchYDelta, shiftedX, shiftedY, alignFrac


def extractResultsParam(results_filename, errTolerance=20):
    df = pd.read_excel(results_filename)
    refPositions, queryPositions, matchIds, matchPositions = df['referencePositions'].dropna(), df['queryPose'], df['matchID'], df['matchPose'],
    matchShiftedPositions, scanAlignment, canPeaks, canVariences = df['matchShiftedPosition'], df['scanAlignment'], df['canPeak'],df['canVarience']
    
    
    refX, refY=zip(*refPositions.apply(lambda x: (ast.literal_eval(x)[0], ast.literal_eval(x)[1])))
    queryX, queryY=zip(*queryPositions.apply(lambda x: (ast.literal_eval(x)[0], ast.literal_eval(x)[1])))
    lprX, lprY=zip(*matchPositions.apply(lambda x: (ast.literal_eval(x)[0], ast.literal_eval(x)[1])))
    lprShiftX, lprShiftY=zip(*matchShiftedPositions.apply(lambda x: (ast.literal_eval(x)[0], ast.literal_eval(x)[1])))
    canX, canY=zip(*canPeaks.apply(lambda x: (ast.literal_eval(x)[0], ast.literal_eval(x)[1])))
  
    canError=np.sqrt(    (np.array(canX)-np.array(queryX))**2    +   (np.array(canY)-np.array(queryY))**2   )
    lprShiftError=np.sqrt(   (np.array(lprShiftX)-np.array(queryX))**2   +   (np.array(lprShiftY)-np.array(queryY))**2   )
    lprError=np.sqrt(   (np.array(lprX)-np.array(queryX))**2   +   (np.array(lprY)-np.array(queryY))**2   )


    lprMag=np.sqrt(np.array(np.gradient(lprX))**2+np.array(np.gradient(lprY))**2)


    lidarLowErrorIds=np.where(lprError<errTolerance)[0]
    lidarLowShiftErrorIds=np.where(lprShiftError<errTolerance)[0]
    lidarLowMagIds=np.where(lprMag<errTolerance)[0]
    canLowErrorIds=np.where(canError<errTolerance)[0]
    
    lidarHighErrorIds=np.where(lprError>errTolerance)[0]
    lidarHighShiftErrorIds=np.where(lprShiftError>errTolerance)[0]
    lidarHighMagIds=np.where(lprMag>errTolerance)[0]
    canHighErrorIds=np.where(canError>errTolerance)[0]

    result_dict = {
    'refX': refX,
    'refY': refY,
    'queryX': queryX,
    'queryY': queryY,
    'lprX': lprX,
    'lprY': lprY,
    'lprShiftX': lprShiftX,
    'lprShiftY': lprShiftY,
    'canX': canX,
    'canY': canY,
    'canError': canError,
    'lprShiftError': lprShiftError,
    'lprError': lprError,
    'lprMag': lprMag,
    'lidarLowErrorIds': lidarLowErrorIds,
    'lidarHighErrorIds': lidarHighErrorIds,
    'lidarLowShiftErrorIds': lidarLowShiftErrorIds,
    'lidarHighShiftErrorIds': lidarHighShiftErrorIds,
    'lidarLowMagIds': lidarLowMagIds,
    'lidarHighMagIds': lidarHighMagIds,
    'canLowErrorIds': canLowErrorIds,
    'canHighErrorIds': canHighErrorIds, 
    'scanAlignment': scanAlignment
    }

    return result_dict
    

def find_closest_point(points, x_target, y_target):
    min_distance = float('inf')  # Initialize with positive infinity
    closest_point = None

    for idx, (x, y) in enumerate(points):
        distance = math.sqrt((x - x_target)**2 + (y - y_target)**2)
        if distance < min_distance:
            min_distance = distance
            closeX, closeY = x, y
            closest_point_id = idx

    return closeX, closeY, closest_point_id, min_distance


def closestN_Ids(points, x_target, y_target, num_points=25):
    distances = []

    for idx, (x, y) in enumerate(points):
        distance = math.sqrt((x - x_target)**2 + (y - y_target)**2)
        distances.append((idx, distance))

    # Sort distances and get the top 25 closest indices
    sorted_distances = sorted(distances, key=lambda x: x[1])
    top_indices = [index for index, _ in sorted_distances[:num_points]]

    return top_indices


def random_downsample(array, n, numPtsMultiplier, downsampleType=None):
    m_orig, n_orig = array.shape
    # print(downsampleType)
    
    # Calculate the dimensions of the downsampled array
    m_downsampled = (m_orig + n - 1) // n
    n_downsampled = (n_orig + n - 1) // n
    
    downsampled_array = np.zeros((m_downsampled * n, n_downsampled * n), dtype=int)

    for i in range(m_downsampled):
        for j in range(n_downsampled):
            # Select the region in the original array
            region = array[i * n: min((i + 1) * n, m_orig), j * n: min((j + 1) * n, n_orig)]
            # Count the number of ones in the fig.
            num_ones = np.count_nonzero(region)
            # Randomly downsample the region if there are more than n ones
            if num_ones > n*numPtsMultiplier:
                # Get indices of ones in the region
                ones_indices = np.argwhere(region == 1)
                if downsampleType =='ordered':
                    # Determine the number of points to select
                    num_points_to_select = n * numPtsMultiplier
                    # Select evenly spaced points
                    step = max(1, len(ones_indices) // num_points_to_select)
                    sampled_indices = ones_indices[::step][:num_points_to_select]
                else:
                    # Randomly select n indices
                    sampled_indices = np.random.choice(len(ones_indices), int(n*numPtsMultiplier), replace=False)
                
                # Set only the selected ones, others remain zero
                for index in sampled_indices:
                    x, y = ones_indices[index]
                    downsampled_array[i * n + x, j * n + y] = 1
            else:
                # Copy the region as it is
                downsampled_array[i * n: min((i + 1) * n, m_orig), j * n: min((j + 1) * n, n_orig)] = region

    return downsampled_array


def count_excel_rows(file_path):
    try:
        df = pd.read_excel(file_path)
        num_rows = df.shape[0]  # Number of rows
        return num_rows
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return 0
    

def runWildPlaces(datasetName, refNum, queryNum, savePath, refNpyName, querySaveFilepath=None, refSaveFilepath=None, ablate=False, rotInc=None, n=None, background=None, blockSize=None, numPtsMultiplier=None, z_max=None, mapRes=None, HDMThresh=None, returnTimes=False):
    
    with open('./scripts/config.json') as f:
        config = json.load(f)
    config=config.get(datasetName, {})
    refIncr = config.get('refIncr')
    queryIncr = config.get('queryIncr')
    queryFilenames = config.get('details', {}).get(str(queryNum), [])
    param = config.get('parameters', {})
    queryRad=param.get('queryRadius', 0.0)
    refRad=param.get('refRadius', 0.0)
    refFilenames = config.get('details', {}).get(str(refNum), [])
    refGridFilePath=refFilenames[0].rsplit('/', 1)[:-1][0] + refNpyName 
    mapRes = param.get('mapRes', 0.0) if mapRes == None else mapRes
    HDMThresh=param.get('HDMThresh') if HDMThresh == None else HDMThresh
    background= param.get('unoccWeight', 0.0) if background == None else background
    blockSize= param.get('blkSize', 0.0) if blockSize == None else blockSize
    numPtsMultiplier= param.get('nMultiplier_randDwnsmpl', 0.0) if numPtsMultiplier == None else numPtsMultiplier
    z_max=param.get('zmax', 0.0) if z_max == None else z_max
    n=param.get('topN', 0.0) if n == None else n
    errTolerance = config.get('errTolerance')
    scanDimX= param.get('scanDimX')
    rotInc=param.get('rotIncr', 0) if rotInc == None else rotInc
       
    if datasetName[:10]=='WildPlaces' and ablate == False:
        with open(config.get('evalInfoFile'), 'rb') as f:
            evalInfo = pickle.load(f)[queryNum-1]
        
        all_files = os.listdir(queryFilenames[0])
        framesQuery = []
        for k in range(len(evalInfo)):
            try:
                if evalInfo[k][refNum-1] != []:
                    index = all_files.index(evalInfo[k]['query'].split('/')[-1])
                    framesQuery.append(index)
            except ValueError:
                pass  
        LPR=LiDAR_PlaceRec(datasetName, config, refNum, queryNum, framesQuery, refGridFilePath=refGridFilePath, background=background ) 
        centerXs,centerYs=zip(*LPR.scanCenter())
        xq,yq, rollQ, pitchQ, yawQ=LPR.scanPoses('QUERY')
        xr,yr,rollR, pitchR, yawR=LPR.scanPoses('REF')
        closeIds=[LPR.find_closest_point(zip(xr,yr), xq[i], yq[i])[2] for i in range(len(xq)) ]
        minDist=[LPR.find_closest_point(zip(xr,yr), xq[i], yq[i])[3] for i in range(len(xq))]
        minDistAll=[find_closest_point(zip(LPR.xrAll,LPR.yrAll), xq[i], yq[i])[3] for i in range(len(xq))]
        filteredFramesQuery=[framesQuery[i] for i in range(len(framesQuery)) if minDist[i] < errTolerance ]
        print(f'{len(LPR.framesQuery)-len(filteredFramesQuery)} queries removed from {len(LPR.framesQuery)} for being outside {errTolerance}m from ref')

    
    if ablate==True:
        if savePath !=None:
            savePath+=(f'/{datasetName}_R:{refNum}_Q:{queryNum}_RInc:{refIncr}_Rrad:{refRad}_QInc:{queryIncr}_Res:{mapRes}_Dim:{scanDimX}_blkAvg:{blockSize}_N2ndsearch:{n}_rotInc{rotInc}_UnoccWeight{background}_nptsMult{numPtsMultiplier}_zMax{z_max}.xlsx')
        framesQuery=list(np.load(querySaveFilepath))
        
        LPR=LiDAR_PlaceRec(datasetName, config, refNum, queryNum, framesQuery, refGridFilePath=refGridFilePath, mapRes=mapRes, HDMThresh=HDMThresh,
                           n=n, background=background, blockSize=blockSize, numPtsMultiplier=numPtsMultiplier, z_max=z_max, rotIncr=rotInc) 
        centerXs,centerYs=zip(*LPR.scanCenter())
        xq,yq, rollQ, pitchQ, yawQ=LPR.scanPoses('QUERY')
        xr,yr,rollR, pitchR, yawR=LPR.scanPoses('REF')
        closeIds=[LPR.find_closest_point(zip(xr,yr), xq[i], yq[i])[2] for i in range(len(xq)) ]
        minDist=[LPR.find_closest_point(zip(xr,yr), xq[i], yq[i])[3] for i in range(len(xq))]
        minDistAll=[find_closest_point(zip(LPR.xrAll,LPR.yrAll), xq[i], yq[i])[3] for i in range(len(xq))]
        filteredFramesQuery=[framesQuery[i] for i in range(len(framesQuery)) if minDist[i] < errTolerance ]
        print(f'{len(LPR.framesQuery)-len(filteredFramesQuery)} queries removed from {len(LPR.framesQuery)} for being outside {errTolerance}m from ref')
        print(f'topN:{LPR.n}, unoccWeight:{LPR.background}, poolSize:{LPR.blockSize}, rotIncr:{LPR.rotIncr}, numPtsMultiplier:{LPR.randDwnsmple_nMult*LPR.randDwnsmple_dim}, zmax={LPR.z_max}')
 
        
    elif datasetName == 'NCLT'or datasetName == 'OxfordRadar':
        framesQuery=list(np.load(querySaveFilepath))
        framesRef=list(np.load(refSaveFilepath))

        print(f'mapRes:{mapRes}, HDM_thresh:{HDMThresh}')
        LPR=LiDAR_PlaceRec(datasetName, config, refNum, queryNum, framesQuery=framesQuery, framesRef=framesRef, refGridFilePath=refGridFilePath, mapRes=mapRes, HDMThresh=HDMThresh) 
        centerXs,centerYs=zip(*LPR.scanCenter())
        xq,yq, rollQ, pitchQ, yawQ=LPR.scanPoses('QUERY')
        xr,yr,rollR, pitchR, yawR=LPR.scanPoses('REF')
        closeIds=[LPR.find_closest_point(zip(xr,yr), xq[i], yq[i])[2] for i in range(len(xq)) ]
        minDist=[LPR.find_closest_point(zip(xr,yr), xq[i], yq[i])[3] for i in range(len(xq))]  
        filteredFramesQuery=[framesQuery[i] for i in range(len(framesQuery)) if minDist[i] < errTolerance ]
        print(f'{len(LPR.framesQuery)-len(filteredFramesQuery)} queries removed from {len(LPR.framesQuery)} for being outside {errTolerance}m from ref')
        # np.save(f'./results/LPR_NCLT/FramesFiles/framesQuery_qInc:{queryIncr}_qRad:{queryRad}.npy',LPR.framesQuery)
        # np.save(f'./results/LPR_NCLT/FramesFiles/framesRef_rInc:{refIncr}_rRad:{refRad}.npy',LPR.framesRef)


    dataStorage=[]
    timelapse,err,corr,numEval, processTimelapse, shiftTimelapse, queryLoadTimelapse=0,0,0,0, 0, 0, 0
    
    if savePath is None:
        start=len(LPR.framesQuery)-5*queryIncr
    elif (savePath is not None) and (count_excel_rows(savePath)*queryIncr < len(filteredFramesQuery)):
        numRows=count_excel_rows(savePath)*queryIncr
        uptoSaved=minDist[:numRows]
        skippedUpToSaved=len([uptoSaved[i] for i in range(len(uptoSaved)) if uptoSaved[i]>errTolerance]) 
        print('skipped len up to saved',  len([uptoSaved[i] for i in range(len(uptoSaved)) if uptoSaved[i]>errTolerance])  )
        start=math.ceil((numRows+skippedUpToSaved) / queryIncr) * queryIncr 
    else: 
        start=len(LPR.framesQuery)
    
    
    print(f'starting {start} - refNum:{refNum}, queryNum:{queryNum}')
    for i in range(start,len(LPR.framesQuery), queryIncr):
        if minDist[i]<errTolerance:
            idx, maxX, maxY, matchXDelta, matchYDelta, shiftedX, shiftedY, rotations, alignment, dist, queryLoadTime, processTime, shiftTime, duration = LPR.update(
                    i, rotInc, xr, yr, yawR, xq, yq, yawQ, centerXs, centerYs, closeIds)
            

            numEval+=1
            if dist>errTolerance:
                err+=1
            else:
                corr+=1
            recall=round(corr/numEval, 2)
            
            '''Print info'''
            
            timelapse+=duration
            processTimelapse+=processTime
            shiftTimelapse+=shiftTime
            queryLoadTimelapse+=queryLoadTime
            print(f'{i}/{len(LPR.framesQuery)}, err:{err}, recall: {recall},  distErr:{round(dist,2)}, idx:{idx} closestId: {closeIds[i]}, minDist: {round(minDist[i],2)}, time {round(duration, 3)}, loadTime: {round(queryLoadTimelapse/numEval, 3)}, avgConv={round(processTimelapse/numEval, 3)},avgShift={round(shiftTimelapse/numEval, 3)}, avgTime={round(timelapse/numEval, 3)}')


            testResult= { 'queryPose': (xq[i], yq[i], yawQ[i]), 'matchID':idx, 'refgridPeak': (maxX, maxY), 'matchPose':(xr[idx], yr[idx], yawR[idx]), 'shiftAmount': (matchXDelta, matchYDelta),  
                        'matchShiftedPosition':(shiftedX, shiftedY), 'dist': dist, 'peakRotation': rotations[np.argmax(alignment)],  'peakAlignment': np.max(alignment), 'closestId': closeIds[i], 
                        'closePose': (xr[closeIds[i]], yr[closeIds[i]], yawR[closeIds[i]]), 'closeDist': minDist[i], 'timePerQuery': round(duration, 3), 'descGenTime': round(queryLoadTime, 3), 'searchTime': round(processTime, 3), 'poseCorrectTime': round(shiftTime,3) }
            dataStorage.append(testResult)

            if savePath != None:
                if os.path.exists(savePath):
                    df1 = pd.read_excel(savePath)
                    df1 = pd.concat([df1, pd.DataFrame([testResult])], ignore_index=True)#df1.append(testResult, ignore_index=True)
                else:
                    df1= pd.DataFrame(dataStorage)
                with pd.ExcelWriter(savePath) as writer:
                    df1.to_excel(writer, index=False)  # Adjust sheet_name as needed
            
    # print(f'done - refNum:{refNum}, queryNum:{queryNum}' )
    if returnTimes==True:
        if numEval==0:
            return 0, 0, 0, 0, 0, 0
        else:
            return queryLoadTimelapse/numEval, processTimelapse/numEval, shiftTimelapse/numEval, timelapse/numEval, corr, numEval
    


def runWildPlacesHPC(datasetName, refNum, queryNum, envName, savePath, configPath='./scripts/config.json'):
    import pickle5 as pickle 
    with open('./scripts/config.json') as f:
        config = json.load(f)
    config=config.get(datasetName, {})
    print('loaded config')

    with open(config.get('evalInfoFile'), 'rb') as f:
        evalInfo = pickle.load(f)[queryNum-1]
    queryIncr = config.get('queryIncr')
    print('got eval info')

    queryFilenames = config.get('details', {}).get(str(queryNum), [])
    print(queryFilenames)
    all_files = os.listdir(queryFilenames[0])
    framesQuery = []
    for k in range(len(evalInfo)):
        try:
            if evalInfo[k][refNum-1] != []:
                index = all_files.index(evalInfo[k]['query'].split('/')[-1])
                framesQuery.append(index)
        except ValueError:
            pass
    print(len(framesQuery))

    param = config.get('parameters', {})
    scanDimX = param.get('scanDimX', 0)
    scanDimY = param.get('scanDimY', 0)
    mapRes = param.get('mapRes', 0.0)
    

    LPR=LiDAR_PlaceRec(datasetName, config, refNum, queryNum, framesQuery ) 
    centerXs,centerYs=zip(*LPR.scanCenter())
    xq,yq, rollQ, pitchQ, yawQ=LPR.scanPoses('QUERY')
    xr,yr,rollR, pitchR, yawR=LPR.scanPoses('REF')
    closeIds=[LPR.find_closest_point(zip(xr,yr), xq[i], yq[i])[2] for i in range(len(xq)) ]
    minDist=[LPR.find_closest_point(zip(xr,yr), xq[i], yq[i])[3] for i in range(len(xq))]
    
    refIncr = config.get('refIncr')
    refFilenames = config.get('details', {}).get(str(refNum), [])
    print(refFilenames)
    refGridFilePath=refFilenames[0].rsplit('/', 1)[:-1][0] +f'/refGrid_incr{refIncr}_dim{scanDimX}_mapRes{mapRes}_downSampHalf.npy'
    print( refGridFilePath)
    
    if os.path.exists(refGridFilePath):
        refgrid=np.load(refGridFilePath)
        print('refGrid exsists')
    else:
        print('making grid')
        refgrid=LPR.makingReferenceGrid(rot=None, dim_3=False)
        np.save(refGridFilePath, refgrid)
    
    refgrid=np.array(refgrid, dtype=np.uint8)
    print(f'{refgrid.dtype}, {np.amax(refgrid)}, {np.amin(refgrid)}')
    dataStorage=[]
    timelapse,err,corr,numEval=0,0,0,0
    start=count_excel_rows(savePath)*queryIncr
    
    print(f'starting - refNum:{refNum}, queryNum:{queryNum}')
    for i in range(start, len(framesQuery), queryIncr):
        t=time.time()
        rotIncr=10
        idx, maxX, maxY, matchXDelta, matchYDelta, shiftedX, shiftedY, rotations, alignment, dist = LPR.update(refgrid, 
                i, rotIncr, mapRes, xr, yr, yawR, xq, yq, yawQ, centerXs, centerYs, closeIds )
        
        numEval+=1
        if dist>3:
            err+=1
        else:
            corr+=1
        recall=round(corr/numEval, 2)
        
        '''Print info'''
        duration=time.time()-t
        timelapse+=duration
        print(f'{i}, err:{err}, recall: {recall},  distErr:{round(dist,2)}, idx:{idx} closestId: {closeIds[i]}, minDist: {round(minDist[i],2)}, time {round(time.time()-t, 2)}, accumTime: {round(timelapse/60,2)}')


        testResult= { 'queryPose': (xq[i], yq[i], yawQ[i]), 'matchID':idx, 'refgridPeak': (maxX, maxY), 'matchPose':(xr[idx], yr[idx], yawR[idx]), 'shiftAmount': (matchXDelta, matchYDelta),  
                    'matchShiftedPosition':(shiftedX, shiftedY), 'dist': dist, 'peakRotation': rotations[np.argmax(alignment)],  'peakAlignment': np.max(alignment), 'closestId': closeIds[i], 
                    'closePose': (xr[closeIds[i]], yr[closeIds[i]], yawR[closeIds[i]]), 'closeDist': minDist[i]}
        dataStorage.append(testResult)

        if os.path.exists(savePath):
            df1 = pd.read_excel(savePath, engine='openpyxl')
            df1 = pd.concat([df1, pd.DataFrame([testResult])], ignore_index=True)#df1.append(testResult, ignore_index=True)
        else:
            df1= pd.DataFrame(dataStorage)
        with pd.ExcelWriter(savePath, engine='openpyxl') as writer:
                df1.to_excel(writer, index=False)  # Adjust sheet_name as needed
        
    print(f'done - refNum:{refNum}, queryNum:{queryNum}' )


def saveRefQuerySplit(datasetName, refNum, queryNum, querySaveFilepath, refSaveFilepath):
    

    with open('./scripts/config.json') as f:
        config = json.load(f)
    config=config.get(datasetName, {})
    # refIncr = config.get('refIncr')
    queryIncr = config.get('queryIncr')
    queryFilenames = config.get('details', {}).get(str(queryNum), [])
    param = config.get('parameters', {})
    queryRad=param.get('queryRadius', 0.0)
    refRad=param.get('refRadius', 0.0)
    # refFilenames = config.get('details', {}).get(str(refNum), [])
    # refNpyName=f"/refGrid_dim{scanDimX}_mapRes{mapRes}_refIncr{refIncr}_refRad{refrad}_rThresh{refThresh}_patchDim{dim_randDwnsmple}_nptsMult{nMultiplier_randDwnsmpl}.npy"
    # refGridFilePath=refFilenames[0].rsplit('/', 1)[:-1][0] + refNpyName 
    

    
    # framesQuery=list(np.load(querySaveFilepath))
    # framesRef=list(np.load(refSaveFilepath))
    # LPR=LiDAR_PlaceRec(datasetName, config, refNum, queryNum, framesQuery, refGridFilePath, framesRef=framesRef)
    
    if not os.path.exists(refSaveFilepath):
        print('Creating file to store reference frames', refSaveFilepath)
        filenames=os.listdir(queryFilenames[0])
        filenames = [f for f in filenames if f[:-4].replace('.','',1).isdigit()]
        framesQuery=range(0, len(filenames), queryIncr)
        LPR=LiDAR_PlaceRec(datasetName, config, refNum, queryNum, framesQuery=framesQuery, refGridFilePath=None)
        xrAll,yrAll, rollR, pitchR, yawRAll= LPR.scanPoses('All')
        framesRef = LPR.subsample_Napart(xrAll, yrAll, refRad)
        np.save(refSaveFilepath,np.array(framesRef))
        print(framesRef)
        print(f'ref len {len(framesRef)}')
    else:
        print(len(np.load(refSaveFilepath)))

    if not os.path.exists(querySaveFilepath):
        print('Creating file to store query frames', querySaveFilepath)
        filenames=os.listdir(queryFilenames[0])
        filenames = [f for f in filenames if f[:-4].replace('.','',1).isdigit()]
        framesQuery=range(0, len(filenames), queryIncr)
        framesRef=list(np.load(refSaveFilepath))
        LPR=LiDAR_PlaceRec(datasetName, config, refNum, queryNum, framesQuery, refGridFilePath=None, framesRef=framesRef)
        xrAll,yrAll, rollR, pitchR, yawRAll= LPR.scanPoses('All')

        xqAll,yqAll, rollQ, pitchQ, yawQAll= LPR.scanPoses('AllQuery')
        framesQuery = LPR.subsample_everyN(xqAll, yqAll, queryRad)
        print(f'query len {len(framesQuery)}, ref len: {len(framesRef)}')
        np.save(querySaveFilepath,np.array(framesQuery))
    else:
        print(len(np.load(querySaveFilepath)))
    
    

def ablateRefQuerySplit(datasetName, refNum, queryNum, querySaveFilepath):
    if not os.path.exists(querySaveFilepath):
        with open('./scripts/config.json') as f:
            config = json.load(f)
        config=config.get(datasetName, {})
        queryFilenames = config.get('details', {}).get(str(queryNum), [])


        with open(config.get('evalInfoFile'), 'rb') as f:
            evalInfo = pickle.load(f)[queryNum-1]
        
        all_files = os.listdir(queryFilenames[0])
        framesQuery = []
        for k in range(len(evalInfo)):
            try:
                if evalInfo[k][refNum-1] != []:
                    index = all_files.index(evalInfo[k]['query'].split('/')[-1])
                    framesQuery.append(index)
            except ValueError:
                pass  
        
        framesAblateQuery=[]
        for j in range(framesQuery[0], framesQuery[-1]):
            if j not in framesQuery:
                framesAblateQuery.append(j)

        np.save(querySaveFilepath,np.array(framesAblateQuery))


def ablateRefQuerySplitUrban(querySaveFilepath, queryAblateSaveFilepath):
    framesQuery=np.load(querySaveFilepath)

    framesAblateQuery=[]
    for j in range(framesQuery[0], framesQuery[-1]):
        if j not in framesQuery:
            framesAblateQuery.append(j)
    framesAblateQuery=framesAblateQuery[::100]
    print(len(framesAblateQuery))

    np.save(queryAblateSaveFilepath,np.array(framesAblateQuery))
    