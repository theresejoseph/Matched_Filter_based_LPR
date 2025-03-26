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
np.random.seed(1)


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
 

def ablateWithWildPlaces(refNum, queryNum):
    datasetName = 'WildPlaces_Venman'
    envName='Venman'
    querySaveFilepath=f'./results/LPR_Wildplaces/FramesFiles/R:{refNum}_Q:{queryNum}_framesQueryAblation.npy'
    ablateRefQuerySplit(datasetName, refNum, queryNum, querySaveFilepath)

    with open('./scripts/config.json') as f:
        config = json.load(f)
    datasetName=f'WildPlaces_{envName}'
    config=config.get(datasetName, {})
    param = config.get('parameters', {})
    refrad=param.get('refRadius', 0)
    blockSize=param.get('blkSize', 0)
    scanDimX = param.get('scanDimX', 0)
    scanDimY = param.get('scanDimY', 0)
    mapRes = param.get('mapRes', 0.0)
    queryIncr = config.get('queryIncr')
    refIncr = config.get('refIncr')
    queryThresh = param.get('queryThresh', 0)
    numPtsMultiplier = param.get('nMultiplier_randDwnsmpl')

    saveFolder=f'./results/LPR_Wildplaces/Ablate_TrainingSet_TimeIncl_1'
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)
        
    runWildPlaces(datasetName, refNum, queryNum, None, 
                  f"/refGrid_dim{scanDimX}_mapRes{mapRes}_refIncr{refIncr}_refRad{refrad}_rThresh2_nptsMult{numPtsMultiplier}_AvgDownsamp.npy", 
                  querySaveFilepath=querySaveFilepath, n=1, ablate=True)
    assert False
    
    '''Varying topN'''
    NumMatches=[10,8,5,2,1]
    for NumMatch in NumMatches:
        refNpyName=f"/refGrid_dim{scanDimX}_mapRes{mapRes}_refIncr{refIncr}_refRad{refrad}_rThresh2_nptsMult{numPtsMultiplier}_AvgDownsamp.npy"
        runWildPlaces(datasetName, refNum, queryNum, saveFolder, refNpyName, querySaveFilepath=querySaveFilepath, ablate=True, n=NumMatch)
    
    '''Varying pool size'''
    blockSizes=[5,4,3,2,1]
    for blksz in blockSizes:
        refNpyName=f"/refGrid_dim{scanDimX}_mapRes{mapRes}_refIncr{refIncr}_refRad{refrad}_rThresh2_nptsMult{numPtsMultiplier}_AvgDownsamp.npy"
        runWildPlaces(datasetName, refNum, queryNum, saveFolder, refNpyName, querySaveFilepath=querySaveFilepath, ablate=True, blockSize=blksz)
    

    '''Varying backgroung/ unoccupied cell weight'''
    unoccVals=[ -0.3, -0.25,-0.2,-0.15, -0.1, -0.05,0]
    for val in unoccVals:
        refNpyName=f"/refGrid_dim{scanDimX}_mapRes{mapRes}_refIncr{refIncr}_refRad{refrad}_rThresh2_nptsMult{numPtsMultiplier}_AvgDownsamp.npy"
        runWildPlaces(datasetName, refNum, queryNum, saveFolder, refNpyName, querySaveFilepath=querySaveFilepath, ablate=True,  background=val)

    '''Varying rotIncr'''
    rotIncrs=[1,5,10,20,30]
    for rot in rotIncrs:
        refNpyName=f"/refGrid_dim{scanDimX}_mapRes{mapRes}_refIncr{refIncr}_refRad{refrad}_rThresh2_nptsMult{numPtsMultiplier}_AvgDownsamp.npy"
        runWildPlaces(datasetName, refNum, queryNum, saveFolder, refNpyName, querySaveFilepath=querySaveFilepath, ablate=True, rotInc=rot)


    '''Varying max occupied cells per patch'''
    numPtsMultipliers=[0.1,0.5,0.75,1.5,2,4,6,8,10]
    for nMult in numPtsMultipliers:
        refNpyName=f"/refGrid_dim{scanDimX}_mapRes{mapRes}_refIncr{refIncr}_refRad{refrad}_rThresh2_nptsMult{nMult}_AvgDownsamp.npy"
        runWildPlaces(datasetName, refNum, queryNum, saveFolder, refNpyName, querySaveFilepath=querySaveFilepath, ablate=True, numPtsMultiplier=nMult)
    
    '''Varying z threshold'''
    z_maxs=[2,3,4,6,8,10,15,20,30]
    for z in z_maxs:
        refNpyName=f"/refGrid_dim{scanDimX}_mapRes{mapRes}_refIncr{refIncr}_refRad{refrad}_rThresh2_nptsMult{numPtsMultiplier}_zMax{z}_AvgDownsamp.npy"         
        runWildPlaces(datasetName, refNum, queryNum, saveFolder, refNpyName, querySaveFilepath=querySaveFilepath, ablate=True, z_max=z)
    
    