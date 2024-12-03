import os
import numpy as np
import cv2
from matplotlib.cm import get_cmap
from scipy import interpolate
import open3d
import matplotlib.pyplot as plt 
from scipy import signal, stats, spatial 
import matplotlib.animation as animation
import open3d as o3d
import time
import pandas as pd
import matplotlib.patches as patches
from scipy.spatial.distance import cdist
from matplotlib.gridspec import GridSpec
import math
from helperFunctions import activityDecoding, LiDAR_PlaceRec, find_closest_point,closestN_Ids, random_downsample, saveRefQuerySplit
from scipy.stats import entropy
from scipy.ndimage import rotate
import ast 
import imutils
import scipy
from concurrent.futures import ThreadPoolExecutor
import json 
import pickle 
import cProfile
import cupy as cp
import skimage 
from sklearn.preprocessing import StandardScaler
# import torch
# import torch.nn.functional as F

def load_from_pickle(pickle_path):
    with open(pickle_path, 'rb') as f:
        file = pickle.load(f)
    return file 

'''---------------------------------------------------------------------------------------------------------------------'''

# dataset_name='BenchmarkOxford'
# refNum=0
# queryNum=1
# increment=5

# BASE_DIR='./data/benchmark_datasets/oxford/'
# all_folders=sorted(os.listdir(BASE_DIR))
# index_list=[5,6,7,9,10,11,12,13,14,15,16,17,18,19,22,24,31,32,33,38,39,43,44]
# folders=[all_folders[index] for index in index_list]

# refDataset=BASE_DIR + folders[refNum] + "/pointcloud_20m/"
# queryDataset=BASE_DIR + folders[queryNum] + "/pointcloud_20m/"
# refGndtru=BASE_DIR + folders[refNum] + "/pointcloud_locations_20m.csv"
# queryGndtru=BASE_DIR + folders[queryNum] + "/pointcloud_locations_20m.csv"

# print(refDataset, queryDataset, refGndtru, queryGndtru)

# queryLen=len(os.listdir(queryDataset))
# queryIncr=5
# scanDimX, scanDimY, mapRes, horizNumScans, intensityFilter, pointInterval =50, 50, 0.1, 25, 0,1
# framesRef, framesQuery= range(0,len(os.listdir(refDataset)),increment), range(0,queryLen,queryIncr)



# LPR=LiDAR_PlaceRec(dataset_name, refDataset, queryDataset, refGndtru, queryGndtru, scanDimX, scanDimY, mapRes, 
#                    horizNumScans, intensityFilter, pointInterval, framesRef, framesQuery)

# refGridFilePath=BASE_DIR + folders[refNum] +'/refGrid.npy'
# if os.path.exists(refGridFilePath):
#     refgrid=np.load(refGridFilePath)
# else:
#     refgrid=LPR.makingReferenceGrid(rot=None, dim_3=False)
#     np.save(refGridFilePath, refgrid)

'''---------------------------------------------------------------------------------------------------------------------'''



# fig, ((ax1,ax2, ax3), (ax4,ax5,ax6))= plt.subplots(2,3, figsize=(10,8))
# plt.tight_layout(pad=2)
# ax1.scatter(xr, yr, color='c', s=3,alpha=0.5)
# ax1.scatter(xq, yq, color='g', s=3, alpha=0.5)
def update_GT_check(i):
    global err, correct, centerXYs
    if i > 0:
        t=time.time()
        # ,ax4.clear(),
        ax4.clear(), ax5.clear(), ax6.clear()
        # ax1.invert_yaxis()
        # ax1.plot(xr, yr, 'y--')

        queryScan= LPR.loadingCroppingFiltering_2DScan('QUERY', i)
        
        
        maxY, maxX, convolved= LPR.scanMatchWithConvolution(refgrid, i)
        idx=LPR.extractIDfromConv(maxX, maxY)
        matchIds.append(idx)
        

        '''Finding realtive shift'''
        centerXs,centerYs=zip(*centerXYs)
        centerX, centerY = centerXs[idx], centerYs[idx]
        matchXDelta, matchYDelta = maxX-centerX, maxY-centerY
        theta=yawR[idx] -np.pi#+(np.pi*(1/2))
        # shiftedX, shiftedY = zerodXR[idx]-matchXDelta, zerodYR[idx]-matchYDelta
        
        GTxdelta, GTydelta= round(xq[i]-xr[idx], 2), round(yq[i]-yr[idx], 2)
        xtrans=round((math.cos(theta)*(matchXDelta)) - (math.sin(theta)*(matchYDelta)), 2)
        ytrans= round((math.sin(theta)*(matchXDelta)) + (math.cos(theta)*(matchYDelta)), 2)

        '''Ref match scans'''
        refConvMatch= LPR.loadingCroppingFiltering_2DScan('REF', idx)
        refShiftMatchScan= LPR.translateScan(refConvMatch, matchXDelta, matchYDelta)
        refClosest= refgrid[centerYs[closeIds[i]]-(scanDimY//2):centerYs[closeIds[i]]+(scanDimY//2), centerXs[closeIds[i]]-(scanDimX//2):centerXs[closeIds[i]]+(scanDimX//2)]

        '''Shifted Pos'''
        shiftedX = xr[idx] + (math.sin(theta)*(matchXDelta*mapRes)) + (math.cos(theta)*(matchYDelta*mapRes))
        shiftedY = yr[idx] + (math.cos(theta)*(matchXDelta*mapRes)) - (math.sin(theta)*(matchYDelta*mapRes)) 
        
        '''Scan align percentage '''
        maxAlignCount=np.min([np.count_nonzero(queryScan==1), np.count_nonzero(refShiftMatchScan==1)])
        alignFrac=(np.count_nonzero((queryScan+refShiftMatchScan)==2))/np.count_nonzero(queryScan==1)
        alignSum=convolved[maxY, maxX]#(np.count_nonzero((queryScan+refShiftMatchScan)==2))
        currConv=convolved[centerY-(scanDimY//2):centerY+(scanDimY//2), centerX-(scanDimX//2):centerX+(scanDimX//2)]

        '''Metrics'''
        dist=np.sqrt((yq[i]-yr[idx])**2+(xq[i]-xr[idx])**2)
        distShift=np.sqrt((yq[i]-shiftedY)**2+(xq[i]-shiftedX)**2)
        distClosest=np.sqrt((yq[i]-yr[closeIds[i]])**2+(xq[i]-xr[closeIds[i]])**2)
        top25Ids=closestN_Ids(zip(xr,yr), xq[i], yq[i])
        if idx in top25Ids:
            correct+=1
        # if dist>10:
        #     err+=1


        '''Ploting'''
        ax1.plot(xq[i], yq[i], 'g.')
        ax1.plot(shiftedX, shiftedY, 'b.')
        ax1.plot([xq[i],shiftedX], [yq[i], shiftedY], color='tab:brown', linestyle='-', alpha=0.5)
        
        ax2.plot(i, minDist[i],'.',color='tab:purple')
        ax2.set_title(f'Distance of closest match: {round(minDist[i],2)}')
        # ax2.set_ylim([0,1.1])

        ax3.plot(i,distShift, '.')
        ax3.set_title(f'Distance of match: {round(distShift,2)}')

        ax4.imshow(queryScan, cmap='Blues', alpha=1)
        ax4.imshow(refShiftMatchScan, cmap='Reds',  alpha=0.5)
        ax4.set_title(f'Query and Match: {idx}')
        
        ax5.imshow(queryScan, cmap='Blues', alpha=1)
        ax5.imshow(refClosest, cmap='Greens',  alpha=0.5)
        ax5.set_title(f'Query and closest Reference: {closeIds[i]}')

        ax6.imshow(convolved)
        ax6.plot(maxX,maxY, 'r.')
        ax6.set_title(f'convMatch')

        

        print(f'{i}, idx:{idx}, err: {err}, distShift: {round(distShift,2)}, yaw: {round(np.rad2deg(theta-yawR[0]),2)%360}, matchXY: {xtrans, ytrans} GTxy: {GTxdelta, GTydelta}, time {time.time()-t}')
        # plt.pause(0.1)


def update_ScanMatch_wildplaces(i):
    global err, centerXYs
    t=time.time()
    ax6.clear(),ax3.clear(), ax5.clear(), ax4.clear()
    # ax1.invert_yaxis()
    # ax1.plot(xr, yr, 'y--')
    

    queryScan= LPR.loadingCroppingFiltering_2DScan('QUERY', i, rotDeg=-np.rad2deg(yawQ[i]))
    # queryScan=imutils.rotate(queryScan, angle=-np.rad2deg(yawQ[i])) 

    

    '''Match and shift'''
    maxY, maxX, convolved= LPR.scanMatchWithConvolution(refgrid, i, rotQuery=queryScan)
    idx=LPR.extractIDfromConv(maxX, maxY)
    matchIds.append(idx)
    centerXs,centerYs=zip(*centerXYs)
    centerX, centerY = centerXs[idx], centerYs[idx]
    matchXDelta, matchYDelta = maxX-centerX, maxY-centerY
    theta=yawR[idx]
    shiftedX = xr[idx] + (math.sin(theta)*(matchXDelta*mapRes)) + (math.cos(theta)*(matchYDelta*mapRes))
    shiftedY = yr[idx] + (math.cos(theta)*(matchXDelta*mapRes)) - (math.sin(theta)*(matchYDelta*mapRes)) 


    '''Ref match scans'''
    refCoords=zip(xr,yr)
    closeX, closeY, closeId, minDist=find_closest_point(refCoords, xq[i], yq[i])
    refConvMatchScan= LPR.loadingCroppingFiltering_2DScan('REF', closeId, rotDeg=-np.rad2deg(yawR[closeId]))
    # refConvMatchScan=imutils.rotate(refConvMatchScan, angle=-np.rad2deg(yawR[closeId])) 

    refConv= refgrid[centerY-(scanDimY//2):centerY+(scanDimY//2), centerX-(scanDimX//2):centerX+(scanDimX//2)]
    refShiftMatchScan= LPR.translateScan(refConv, matchXDelta, matchYDelta)
    
    '''Scan align percentage '''
    maxAlignCount=np.min([np.count_nonzero(queryScan==1), np.count_nonzero(refConvMatchScan==1)])
    alignFrac=(np.count_nonzero((queryScan+refConvMatchScan)==2))/np.count_nonzero(queryScan==1)
    # alignSum=convolved[maxY, maxX]/(scanDimX*scanDimY)#(np.count_nonzero((queryScan+refShiftMatchScan)==2))

    '''Match Entropy'''
    # currConv=convolved[centerY-(scanDimY//2):centerY+(scanDimY//2), centerX-(scanDimX//2):centerX+(scanDimX//2)]
    # epsilon=1e-10
    # normConv=(currConv/np.linalg.norm(currConv, 'fro'))+epsilon
    # average_entropy = np.mean(normConv)

    # dist=np.sqrt((yq[i]-yr[idx])**2+(xq[i]-xr[idx])**2)
    dist=np.sqrt((yq[i]-shiftedY)**2+(xq[i]-shiftedX)**2)
    alignThresh=0.035 #for 100 width #0.025 # for 150 width 0.015 for 200 width
    if dist>20:
        err+=1

    print(f'{i}, idx:{idx}, err:{err}, distErr:{dist}, time {time.time()-t}')

    '''Ploting'''
    ax1.plot(xq[i],yq[i], 'g.')
    # ax1.plot(closeX, closeY, 'y.')
    ax1.plot(shiftedX, shiftedY, 'b.')
    ax1.plot([xq[i],shiftedX], [yq[i], shiftedY], color='tab:brown', linestyle='-', alpha=0.5)
    ax1.set_title(f'Dist:{round(dist,2)}')

    ax3.imshow(queryScan, cmap='Blues', alpha=1)
    ax3.imshow(refShiftMatchScan, cmap='Reds', alpha=0.5)
    ax3.set_title(f'Query and Match')

    ax2.set_title('Distance error')
    ax2.plot(i,dist, '.', color='tab:orange')
    
    # ax2.imshow(weights)
    ax4.imshow(queryScan, cmap='Blues', alpha=0.5)
    ax4.set_title('Query')

    # ax5.imshow(queryScan, cmap='Blues', alpha=1)
    ax5.imshow(refConvMatchScan, cmap='Greens', alpha=0.5)
    ax5.set_title(f'Nearest Neighbour')

    ax6.imshow(queryScan, cmap='Blues', alpha=1)
    ax6.imshow(refConvMatchScan, cmap='Greens', alpha=0.5)
    ax6.set_title('Query and Nearest Neigbour')


def update_Rotate_wildplaces(i):
    if i>0:
        global err, centerXYs, correct 
        t=time.time()
        ax4.clear(), ax5.clear(), ax3.clear()
        i=i*queryIncr
        print(i)
        # ax5.clear(), ax6.clear(),
        '''Rotate Convolve'''
        rotIncr=10
        maxPos, alignment, rotations=[], [], []
        def process_scan(j):
            # print(i, j)
            
            queryScan = LPR.loadingCroppingFiltering_2DScan('QUERY', i, rotDeg=j)
            maxY, maxX, convolved = LPR.scanMatchWithConvolution(refgrid, i, rotQuery=queryScan)
            # kernel=np.fliplr(np.flipud(queryScan))
            # convolved= signal.fftconvolve(refgrid, kernel, mode='valid')[::scanDimY, ::scanDimX]
            # maxY, maxX = np.unravel_index(np.argmax(convolved), convolved.shape)
            rotations.append(j)
            maxPos.append((maxX, maxY))
            alignment.append(convolved[maxY, maxX])
        
        
        with ThreadPoolExecutor(max_workers=6) as executor:
            executor.map(process_scan, range(0, 360, rotIncr))


        '''Match and shift'''
        headingIdx=np.argmax(alignment)#np.argmin(SAD)
        heading=rotations[headingIdx]
        maxX,maxY=maxPos[headingIdx] 
   

        idx=LPR.extractIDfromConv(maxX, maxY)
        centerX, centerY = centerXs[idx], centerYs[idx]
        matchXDelta, matchYDelta = maxX-centerX, maxY-centerY
        theta=yawR[idx] #+np.deg2rad(heading)#- (np.pi/2)
        shiftedX = xr[idx] + (math.cos(theta)*(matchXDelta*mapRes)) - (math.sin(theta)*(matchYDelta*mapRes))
        shiftedY = yr[idx] + (math.sin(theta)*(matchXDelta*mapRes)) + (math.cos(theta)*(matchYDelta*mapRes))


        '''Ref match scans'''
        queryScan= LPR.loadingCroppingFiltering_2DScan('QUERY', i, rotDeg=heading)
        refConv= refgrid[centerY-(scanDimY//2):centerY+(scanDimY//2), centerX-(scanDimX//2):centerX+(scanDimX//2)]
        refClosest= refgrid[centerYs[closeIds[i]]-(scanDimY//2):centerYs[closeIds[i]]+(scanDimY//2), centerXs[closeIds[i]]-(scanDimX//2):centerXs[closeIds[i]]+(scanDimX//2)]
        refShiftMatchScan= LPR.translateScan(refConv, matchXDelta, matchYDelta)
        
        '''Error Metrics'''
        # dist=np.sqrt((yq[i]-yr[idx])**2+(xq[i]-xr[idx])**2)
        dist=np.sqrt((yq[i]-shiftedY)**2+(xq[i]-shiftedX)**2)
        if dist>5:
            err+=1
        top25Ids=closestN_Ids(zip(xr,yr), xq[i], yq[i])
        recall=round(((i//queryIncr)-err)/(i//queryIncr), 2)
        if idx in top25Ids:
            correct+=1
        print(f'{i}, err:{err}, correct: {correct}, recall: {recall},  distErr:{round(dist,2)}, idx:{idx} closestId: {closeIds[i]}, minDist: {minDist[i]}, time {round(time.time()-t, 2)}')

        '''Ploting'''
        if dist<5:
            # ax1.plot(shiftedX, shiftedY, 'b.')
            ax1.plot(xr[idx],yr[idx], 'b.')
        else: 
            # ax1.plot(shiftedX, shiftedY, 'r.')
            ax1.plot(xr[idx],yr[idx], 'r.')
        # ax1.plot([xq[i],shiftedX], [yq[i], shiftedY], color='tab:brown', linestyle='-', alpha=0.5)
        ax1.set_title(f'Dist:{round(dist,2)}')

        ax2.plot(i, recall,'.',color='tab:purple')
        ax2.set_title(f'Average recall over traverse: {recall}')
        ax2.set_ylim([0,1.1])

        ax3.plot(rotations,alignment, 'm.')
        ax3.plot(rotations[np.argmax(alignment)], np.max(alignment), 'g*')
        ax3.set_title(f'Est Rot: {rotations[np.argmax(alignment)]}')
        
        ax4.imshow(queryScan, cmap='Blues', alpha=1)
        ax4.imshow(refShiftMatchScan, cmap='Reds',  alpha=0.5)
        ax4.set_title(f'Query and Match: {idx}')
        
        ax5.imshow(queryScan, cmap='Blues', alpha=1)
        ax5.imshow(refClosest, cmap='Greens',  alpha=0.5)
        ax5.set_title(f'Query and closest Reference: {closeIds[i]}')

        ax6.plot(i,dist, '.',color='tab:brown')
        ax6.set_title(f'Distance')

   
def update_benchmarkOxford(i):
    if i>0:
        global err, centerXYs, correct 
        t=time.time()
        ax3.clear(), ax4.clear(), ax5.clear(), ax6.clear(), ax2.clear()
        # ax5.clear(), ax6.clear(),
        # if datasetName=='Jacakl': 
        #     queryIds=[j for j in framesQuery]
        #     i=queryIds[i]
  

        '''Match and shift'''
        queryScan = LPR.loadingCroppingFiltering_2DScan('QUERY', i, dim_3=False)
        # queryRaw=LPR.loadingCroppingFiltering_2DScan('QUERY', i, raw=True)
        maxY, maxX, convolved = LPR.scanMatchWithConvolution(refgrid, i)
        # kernel=np.flip(queryScan, axis=(0, 1, 2))
        # convolved= signal.fftconvolve(refgrid, kernel, mode='same')
        # convolved=np.sum(convolved[:,:,:], axis=2)
        # maxY, maxX = np.unravel_index(np.argmax(convolved), convolved.shape)

        # convolved= signal.fftconvolve(refgrid, np.fliplr(np.flipud(queryScan)), mode='valid')
        # convCenters=convolved[::scanDimY, ::scanDimX]
        # maxY, maxX = np.unravel_index(np.argmax(convCenters), convCenters.shape)
        # idx=np.argmax(convCenters)

        idx=LPR.extractIDfromConv(maxX, maxY)
        centerX, centerY = centerXs[idx], centerYs[idx]
        matchXDelta, matchYDelta = maxX-centerX, maxY-centerY
        # theta=yawR[idx] - (np.pi/2)
        # shiftedX = xr[idx] + (math.sin(theta)*(matchXDelta*mapRes)) - (math.cos(theta)*(matchYDelta*mapRes))
        # shiftedY = yr[idx] + (math.cos(theta)*(matchXDelta*mapRes)) + (math.sin(theta)*(matchYDelta*mapRes)) 
        # matchRaw=LPR.loadingCroppingFiltering_2DScan('REF', idx, raw=True)
        # matchRawShift=LPR.translateScan(matchRaw, matchXDelta, matchYDelta)

        '''Ref match scans'''
        refConv= refgrid[centerY-(scanDimY//2):centerY+(scanDimY//2), centerX-(scanDimX//2):centerX+(scanDimX//2)]
        refClosest= refgrid[centerYs[closeIds[i]]-(scanDimY//2):centerYs[closeIds[i]]+(scanDimY//2), centerXs[closeIds[i]]-(scanDimX//2):centerXs[closeIds[i]]+(scanDimX//2)]
        refShiftMatchScan= LPR.translateScan(refConv, matchXDelta, matchYDelta)
        
        '''Error Metrics'''
        dist=np.sqrt((yq[i]-yr[idx])**2+(xq[i]-xr[idx])**2)
        # dist=np.sqrt((yq[i]-shiftedY)**2+(xq[i]-shiftedX)**2)
        if dist>20:
            err+=1
        top25Ids=closestN_Ids(zip(xr,yr), xq[i], yq[i])
        print(top25Ids)
        if idx in top25Ids:
            correct+=1
        print(f'{idx}, err:{err}, correct: {correct}, recall: {round(correct/i, 2)},  distErr:{round(dist,2)}, refCount: {np.count_nonzero(refConv)}, queryCount:{np.count_nonzero(queryScan)} time {round(time.time()-t, 2)}')

        '''Ploting'''
        if dist<20:
            # ax1.plot(shiftedX, shiftedY, 'b.')
            ax1.plot(xr[idx],yr[idx], 'b.')
        else: 
            # ax1.plot(shiftedX, shiftedY, 'r.')
            ax1.plot(xr[idx],yr[idx], 'r.')
        ax1.set_title(f'Recall: {round(correct/i, 2)}, Dist:{round(dist,2)}')


        # ax2.imshow(queryScan, cmap='viridis')
        # ax3.imshow(queryRaw, cmap='Purples', alpha=0.75)
        # ax4.imshow(matchRaw, cmap='Greens', alpha=0.75)
        # ax5.imshow(matchRawShift, cmap='Greens', alpha=1)
        # ax5.imshow(queryRaw, cmap='Purples', alpha=0.5)

        ax4.imshow(queryScan, cmap='Blues', alpha=1)
        ax4.imshow(refShiftMatchScan, cmap='Reds',  alpha=0.5)
        ax4.set_title(f'Query and Match: {idx}')

        ax5.imshow(queryScan, cmap='Blues', alpha=1)
        ax5.imshow(refClosest, cmap='Greens',  alpha=0.5)
        ax5.set_title(f'Query and closest Reference: {closeIds[i]}')

        ax6.imshow(convolved)
        ax6.plot(maxX,maxY, 'r.')
        ax6.set_title(f'convMatch')

    
def update_Kitti(i, rotQuery=True):
    global err, correct, centerXYs, TP, TN, FP, FN
    if i > 0:
        t=time.time()
        # ,ax4.clear(),
        ax6.clear(), ax5.clear()
        # ax1.invert_yaxis()
        # ax1.plot(xr, yr, 'y--')

        queryScan= LPR.loadingCroppingFiltering_2DScan('QUERY', i)
        
        
        # maxY, maxX, convolved= LPR.scanMatchWithConvolution(refgrid, i)
        # idx=LPR.extractIDfromConv(maxX, maxY)
        if rotQuery==True:
            rotIncr=10
            maxPos, alignment, rotations=[], [], []
            def process_scan(j): 
                print(i, j)
                
                queryScan = LPR.loadingCroppingFiltering_2DScan('QUERY', i, rotDeg=j)
                kernel=np.fliplr(np.flipud(queryScan))
                convolved= signal.fftconvolve(refgrid, kernel, mode='valid')[::scanDimY, ::scanDimX]
                maxY, maxX = np.unravel_index(np.argmax(convolved), convolved.shape)
                
                rotations.append(j)
                maxPos.append((maxX, maxY))
                alignment.append(convolved[maxY, maxX])
            
            with ThreadPoolExecutor(max_workers=6) as executor:
                executor.map(process_scan, range(0, 360, rotIncr))
        

            headingIdx=np.argmax(alignment)#np.argmin(SAD)
            heading=rotations[headingIdx]
            maxX,maxY=maxPos[headingIdx] 
        else:
            queryScan = LPR.loadingCroppingFiltering_2DScan('QUERY', i)
            kernel=np.fliplr(np.flipud(queryScan))
            convolved= signal.fftconvolve(refgrid, kernel, mode='valid')[::scanDimY, ::scanDimX]
            maxY, maxX = np.unravel_index(np.argmax(convolved), convolved.shape)
            heading=0
        idx=(maxY * horizNumScans) + maxX
        # idx=LPR.extractIDfromConv(maxX,maxY)
        matchIds.append(idx)
        

        '''Finding realtive shift'''
        centerXs,centerYs=zip(*centerXYs)
        centerX, centerY = centerXs[idx], centerYs[idx]
        matchXDelta, matchYDelta = maxX-centerX, maxY-centerY
        matchDist=np.sqrt((matchXDelta*mapRes)**2+(matchYDelta*mapRes)**2)
        theta=yawR[idx] #-np.pi#+(np.pi*(1/2))
        # shiftedX, shiftedY = zerodXR[idx]-matchXDelta, zerodYR[idx]-matchYDelta
        
        '''Metrics'''
        dist=np.sqrt((yq[i]-yr[idx])**2+(xq[i]-xr[idx])**2)
        distClosest=np.sqrt((yq[i]-yr[closeIds[i]])**2+(xq[i]-xr[closeIds[i]])**2)
        # distShift=np.sqrt((yq[i]-shiftedY)**2+(xq[i]-shiftedX)**2)
        

        '''Ref match scans'''
        refConvMatch= LPR.loadingCroppingFiltering_2DScan('REF', idx, rotDeg=heading)
        # refShiftMatchScan= LPR.translateScan(refConvMatch, matchXDelta, matchYDelta)
        refClosest= refgrid[centerYs[closeIds[i]]-(scanDimY//2):centerYs[closeIds[i]]+(scanDimY//2), centerXs[closeIds[i]]-(scanDimX//2):centerXs[closeIds[i]]+(scanDimX//2)]

        '''Shifted Pos'''
        # shiftedX = xr[idx] + (math.sin(theta)*(matchXDelta*mapRes)) - (math.cos(theta)*(matchYDelta*mapRes))
        # shiftedY = yr[idx] + (math.cos(theta)*(mak0tchXDelta*mapRes)) + (math.sin(theta)*(matchYDelta*mapRes)) 
        
        '''Scan align percentage '''
        maxAlignCount=np.min([np.count_nonzero(queryScan==1), np.count_nonzero(refClosest==1)])
        alignFrac=(np.count_nonzero((queryScan+refClosest)==2))/maxAlignCount

        # alignSum=max(alignment)/ min(np.count_nonzero(queryScan), np.count_nonzero(refConvMatch)) 
        alignSum=(np.count_nonzero((queryScan+refConvMatch)==2))/np.min([np.count_nonzero(queryScan==1), np.count_nonzero(refConvMatch==1)])
        # currConv=convolved[centerY-(scanDimY//2):centerY+(scanDimY//2), centerX-(scanDimX//2):centerX+(scanDimX//2)]
        

        # if alignSum > thresh and dist<revisitThresh:
        #     TP+=1
        # if alignSum <thresh and dist<revisitThresh:
        #     FN+=1
        # if alignSum <thresh and dist>revisitThresh:
        #     TN+=1
        # if alignSum > thresh and dist>revisitThresh:
        #     FP+=1
        
        # precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        # recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1_score=0
        # f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0



        '''Ploting'''
        # ax1.plot(xq[i], yq[i], 'g.')
        # ax1.plot(shiftedX, shiftedY, 'b.')
        # ax1.plot([xq[i],shiftedX], [yq[i], shiftedY], color='tab:brown', linestyle='-', alpha=0.5)
        # if alignFrac<thresh:
        #     ax1.plot(xr[idx], yr[idx], 'k.', alpha=0.5)
            # ax1.plot(shiftedX, shiftedY, 'k.', alpha=0.5)
        # else: 
            # ax1.plot(xr[idx], yr[idx], 'b.')
            # ax1.plot(shiftedX, shiftedY, 'b.')
        # if alignSum>thresh:
        #     ax1.plot(xr[idx], yr[idx], 'b.')  
        # ax1.set_title(f'Dist: {round(dist,2)}')
        
        # # ax2.plot(i, alignFrac,'.',color='tab:purple')
        # if alignSum > thresh and dist<revisitThresh:
        #     ax2.plot(i, alignSum,'*',color='tab:green')
        # if alignSum < thresh and dist<revisitThresh:
        #     ax2.plot(i, alignSum,'.',color='tab:olive')
        # if alignSum < thresh and dist>revisitThresh:
        #     ax2.plot(i, alignSum,'*',color='tab:red')
        # if alignSum > thresh and dist>revisitThresh:
        #     ax2.plot(i, alignSum,'.',color='tab:orange')
        # ax2.set_title(f'AlignFrac: {round(alignFrac,2)}, AlignSum: {round(alignSum,2)}')
        # ax2.set_ylim([0,1.1])

        ax3.plot(i,dist, 'b.')
        ax3.plot(i,distClosest, 'g.')
        ax3.set_title(f'Dist: {round(dist,2)}, DistClose: {round(distClosest,2)}')

        

        # ax4.plot(i, f1_score, '.', color='tab:pink')
        # ax4.set_title(f'f1:{round(f1_score,2)}')

        ax5.imshow(queryScan, cmap='Blues', alpha=1)
        ax5.imshow(refClosest, cmap='Greens',  alpha=0.5)
        ax5.set_title(f'Query and Close: {closeIds[i]}')


        ax6.imshow(queryScan, cmap='Blues', alpha=1)
        ax6.imshow(refConvMatch, cmap='Reds',  alpha=0.5)
        ax6.set_title(f'Query and Match: idx={idx} ')

        

        print(f'{i}, idx:{idx}, F1={round(f1_score,2)} ,dist: {round(dist,2)}, distClose: {round(distClosest,2)}, alignSum:{round(alignSum,2)}, alignFrac:{round(alignFrac,2)},  time {round(time.time()-t,2)}')
        # plt.pause(0.1)


def update_helperFunc(i):
    i=i*queryIncr
    global err, centerXYs, correct, count
    t=time.time()
    
    rotIncr=10
    
    if minDist[i]<errTolerance:
        queryScan, refScan, refShiftMatchScan, refClosest, idx, maxX, maxY, matchXDelta, matchYDelta, shiftedX, shiftedY, rotations, alignments, dist, alignVals= LPR.update( 
            i, rotIncr, xr, yr, yawR, xq, yq, yawQ, centerXs, centerYs, closeIds, n=1, returnArray=True)
        heading=rotations[np.argmax(alignments)]
        query= LPR.loadingCroppingFiltering_2DScan('QUERY', i, rotDeg=heading, sameProcess=False)
        match= LPR.loadingCroppingFiltering_2DScan('REF', idx, sameProcess=False)
        shiftMatch= LPR.translateScan(match, matchXDelta, matchYDelta)

        '''Error Metrics'''
        dist=np.sqrt((yq[i]-shiftedY)**2+(xq[i]-shiftedX)**2)
        if dist>3:
            err+=1
        
        '''Ploting'''
        ax4.clear(), ax5.clear(), ax6.clear()
        if dist<errTolerance:
            ax1.plot(shiftedX,shiftedY, 'b.')
        else: 
            ax1.plot(shiftedX,shiftedY, 'r.')
        ax1.set_title(f'Dist:{round(dist,2)}')
        if i>0:
            recall=1-(err/(i/queryIncr))
            ax2.plot(i, recall,'.',color='tab:purple')
            ax2.set_title(f'Average recall over traverse: {recall}')
            ax2.set_ylim([0,1.1])
            print(f'{i}/{len(xq)}, err:{err},  recall:{round(recall,2)}, distErr:{round(dist,2)}, idx:{idx} closestId: {closeIds[i]}, minDist: {minDist[i]}, time {round(time.time()-t, 2)}')
        
        ax3.plot(i,dist, '.',color='tab:brown')
        ax3.set_title(f'Distance')

        
        
        ax4.imshow(queryScan, cmap='Blues', alpha=1)
        ax4.imshow(refShiftMatchScan, cmap='Reds',  alpha=0.5)
        ax4.set_title(f'Query and Match: {idx}')
        
        ax5.imshow(queryScan, cmap='Blues', alpha=1)
        ax5.imshow(refClosest, cmap='Greens',  alpha=0.5)
        ax5.set_title(f'Query and closest Reference: {closeIds[i]}')
        
        ax6.imshow(query, cmap='Blues', alpha=1)
        ax6.imshow(shiftMatch, cmap='Reds', alpha=0.5)
        ax6.set_title(f'Query and Match Downsamp: {idx}')
        # ax3.plot(rotations,alignment, 'm.')
        # ax3.plot(rotations[np.argmax(alignment)], np.max(alignment), 'g*')
        # ax3.set_title(f'Est Rot: {rotations[np.argmax(alignment)]}')



def update_saveGoodBad(i):
    i=i*queryIncr
    global err, centerXYs, correct, saveBadCount, saveGoodCount
    t=time.time()
    
    rotInc=10
    
    if minDist[i]<errTolerance:
        queryScan, refScan, refShiftMatchScan, refClosest, idx, maxX, maxY, matchXDelta, matchYDelta, shiftedX, shiftedY, rotations, alignment, dist, alignVals = LPR.update(
                i, rotInc, xr, yr, yawR, xq, yq, yawQ, centerXs, centerYs, closeIds, n=NumMatches,returnArray=True )
        

 
        '''Error Metrics'''
        dist=np.sqrt((yq[i]-shiftedY)**2+(xq[i]-shiftedX)**2)
        if dist>errTolerance:
            err+=1
        else:
            correct+=1
       
        '''Ploting'''
        ax1.clear(), ax2.clear()
        for ax in [ax1,ax2]:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])

        ax1.imshow(queryScan, cmap='Blues', alpha=1)
        ax1.imshow(refClosest, cmap='Greens',  alpha=0.5)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)

        ax2.imshow(queryScan, cmap='Blues', alpha=1)
        ax2.imshow(refShiftMatchScan, cmap='Reds',  alpha=0.5)
        # ax1.set_title(f'Query and Match: {idx}')
        if dist>errTolerance:
            ax2.spines['top'].set_color('red')
            ax2.spines['right'].set_color('red')
            ax2.spines['left'].set_color('red')
            ax2.spines['bottom'].set_color('red')
            saveBadCount+=1
            if saveBadCount<10:
                plt.savefig(f'./goodbadMatchExamples/Bad_{datasetName}_{refNum}_{queryNum}_Matched_{idx}whileClosest{closeIds[i]}.png', dpi=300)
        else:
            ax2.spines['top'].set_color('green')
            ax2.spines['right'].set_color('green')
            ax2.spines['left'].set_color('green')
            ax2.spines['bottom'].set_color('green')
            saveGoodCount+=1
            if saveGoodCount<10:
                plt.savefig(f'./goodbadMatchExamples/Good_{datasetName}_{refNum}_{queryNum}_Matched_{idx}whileClosest{closeIds[i]}.png', dpi=300)
       
        
        recall=round(correct/(saveBadCount+saveGoodCount), 2)
        print(f'{i}/{len(LPR.framesQuery)}, err:{err}, recall: {recall},  distErr:{round(dist,2)}, idx:{idx} closestId: {closeIds[i]}, minDist: {round(minDist[i],2)}')



def plotClosestandMatch(i):

    rotIncr=10
    # if minDist[i]<3:
   
    queryScan, refScan, refShiftMatchScan, refClosest, idx, maxX, maxY, matchXDelta, matchYDelta, shiftedX, shiftedY, rotations, alignment, dist, alignVals = LPR.update( 
        i, rotIncr, xr, yr, yawR, xq, yq, yawQ, centerXs, centerYs, closeIds, n=1, returnArray=True)
    
    print(f'{i}/{len(xq)}, err:{err}, distErr:{round(dist,2)}, idx:{idx} closestId: {closeIds[i]}, minDist: {minDist[i]}')
    headingIdx=np.argmax(alignment)
    heading=rotations[headingIdx]

    # closest= LPR.loadingCroppingFiltering_2DScan('REF', closeIds[i], sameProcess=True)
    matchedScan= LPR.loadingCroppingFiltering_2DScan('REF', idx, sameProcess=True)
    queryScanNoRot= LPR.loadingCroppingFiltering_2DScan('QUERY', i, sameProcess=True)

    fig, (ax1,ax2, ax3, ax4, ax5)= plt.subplots(1,5, figsize=(14,3))
    plt.tight_layout(pad=2)
    # ax3.plot(rotations,alignment, 'm.')
    # ax3.plot(rotations[np.argmax(alignment)], np.max(alignment), 'g*')
    # ax3.set_title(f'Est Rot: {rotations[np.argmax(alignment)]}')

    ax1.imshow(queryScan, cmap='Blues',  alpha=0.5)
    ax1.set_title(f'Query')

    ax2.imshow(refClosest, cmap='Greens',  alpha=0.5)
    ax2.set_title(f'Closest:{closeIds[i]}') #pos: {round(xr[closeIds[i]],1)}, {round(yr[closeIds[i]],1)}
    
    ax3.set_title(f'Match:{idx}') #pos: {round(xr[idx],1)}, {round(yr[idx],1)}
    ax3.imshow(refShiftMatchScan, cmap='Reds',  alpha=0.5)
    
    ax4.imshow(queryScanNoRot, cmap='Blues', alpha=1)
    ax4.imshow(refClosest, cmap='Greens',  alpha=0.5)
    ax4.set_title(f'Query and closest')

    ax5.imshow(queryScan, cmap='Blues', alpha=1)
    ax5.imshow(refShiftMatchScan, cmap='Reds',  alpha=0.5)
    ax5.set_title(f'Query and Match')

    # plt.savefig('result.png')
    plt.show()


def plotCorrelationScores(ids):
    # Create a figure and axes
    fig, ax = plt.subplots(2, 3, figsize=(10, 6))
    axes = ax.flatten()

    # Initialize a list to store handles and labels for the legend
    handles_list = []
    labels_list = []

    # Assuming `ids`, `LPR`, and other necessary variables are defined
    for m, ax in enumerate(axes):
        i = ids[m]
        queryScan, refScan, refShiftMatchScan, refClosest, idx, maxX, maxY, matchXDelta, matchYDelta, shiftedX, shiftedY, rotations, alignment, dist, alignVals  = LPR.update(
            i, rotIncr, xr, yr, yawR, xq, yq, yawQ, centerXs, centerYs, closeIds, n=1, returnArray=True)
        
        print(f'{i}/{len(xq)}, err:{err}, distErr:{round(dist,2)}, idx:{idx} closestId: {closeIds[i]}, minDist: {minDist[i]}')

        allCorrVals = alignVals[3].flatten()
        matchIDX = np.argmax(allCorrVals)
        closestIDX = np.where(allCorrVals == alignVals[1])[0]

        if isinstance(closestIDX, np.ndarray):
            closestIDX = closestIDX[np.argmin(np.abs(np.array(closestIDX) - matchIDX))]
        print(alignVals[1])

        ax.plot(allCorrVals, 'k-', label='All Correlation Scores')
        ax.plot(matchIDX, allCorrVals[matchIDX], 'r.', label='Match')
        ax.plot(closestIDX, allCorrVals[closestIDX], 'g.', label='Closest')
        ax.plot(0, alignVals[0], 'b.', label='Query')
        ax.plot(1000000, alignVals[4], '.', color='navy', alpha=0.5, label='Add Noise Query')
        ax.plot(5, alignVals[5], '.', color='lightblue', label='Remove Noise Query')

        if dist > 3:
            for spine in ax.spines.values():
                spine.set_color('red')

        # Add title and labels
        ax.set_title(f'qIDX{i}_mIDX{idx}_cIDX{closeIds[i]}')
        ax.set_ylabel('Correlation score')

        ax.set_ylim([0,1500])

        # Collect handles and labels for the legend
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label not in labels_list:
                handles_list.append(handle)
                labels_list.append(label)

    # Create a single legend outside the plot
    fig.legend(handles_list, labels_list, loc='lower center', ncols=6)

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()


def plotConvOutputForDifferentBackgrounds(i):
    if datasetName[:10]=='WildPlaces':
        framesRef=[]
    else:
        framesRef=list(np.load(refSaveFilepath))
    fig, ax = plt.subplots(2, 5, figsize=(18, 6), gridspec_kw={'width_ratios': [1, 1, 1, 1, 1], 'height_ratios': [1, 1]})
    backgroundVals = [0, -0.1, -0.25, -0.5, -1]
    vmin, vmax = 0, 1
    allcormin, allcorrmax= 0, 1

    for val in backgroundVals:
        LPR = LiDAR_PlaceRec(datasetName, config, refNum, queryNum, framesQuery=framesQuery, framesRef=framesRef, refGridFilePath=refGridFilePath, background=val)
        convOutput, allCorr, dist = LPR.update(i, rotIncr, xr, yr, yawR, xq, yq, yawQ, centerXs, centerYs, closeIds, n=1, returnConv=True)
        if convOutput.min() < vmin:
            vmin = convOutput.min()
        if convOutput.max() > vmax:
            vmax = convOutput.max()

        if allCorr.min() < allcormin:
            allcormin = allCorr.min()
        if allCorr.max() > allcorrmax:
            allcorrmax = allCorr.max()

    for m, val in enumerate(backgroundVals):
        # First row: imshow
        LPR = LiDAR_PlaceRec(datasetName, config, refNum, queryNum, framesQuery=framesQuery, framesRef=framesRef, refGridFilePath=refGridFilePath, background=val)
        convOutput, allCorr,dist = LPR.update(i, rotIncr, xr, yr, yawR, xq, yq, yawQ, centerXs, centerYs, closeIds, n=1, returnConv=True)
        im = ax[0, m].imshow(allCorr, vmin=allcormin, vmax=allcorrmax, cmap='plasma')
        ax[0, m].set_title(f'Unocc-Val: {val}')

        # Second row: histogram
        ax[1, m].plot(allCorr.ravel(),color='purple')
        ax[1, m].set_title(f'Distrib. for {val}')
        ax[1, m].set_ylabel('Amplitude')
        # ax[1, m].set_ylim([allcormin, allcorrmax+20])

    # fig.colorbar(im, ax=ax[0, :].ravel().tolist())
    plt.tight_layout(pad=1.0)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    
    # Adjust layout to make room for the colorbar
    plt.subplots_adjust(right=0.9)

    plt.show()
            

def plotReforQueryDownSampeld(ids):
    if datasetName[:10]=='WildPlaces':
        framesRef=[]
    else:
        framesRef=list(np.load(refSaveFilepath))
    fig, ax = plt.subplots(1, 5, figsize=(18, 4), gridspec_kw={'width_ratios': [1, 1, 1, 1, 1]})
    for m, i in enumerate(ids):
        # First row: imshow
        LPR = LiDAR_PlaceRec(datasetName, config, refNum, queryNum, framesQuery=framesQuery, framesRef=framesRef, refGridFilePath=refGridFilePath, background=0)
        convOutput, allCorr, dist = LPR.update(i, rotIncr, xr, yr, yawR, xq, yq, yawQ, centerXs, centerYs, closeIds, n=1, returnConv=True)
        

        # Second row: histogram
        ax[m].plot(allCorr.ravel(), color='black')
        ax[m].plot(np.argmax(allCorr.ravel()), np.max(allCorr.ravel()),'*', color='red')
        ax[m].set_title(f'Distrib. for idx:{i}')
        ax[m].set_ylabel('Amplitude')

        if dist > 3:
            for spine in ax[m].spines.values():
                spine.set_color('red')

    plt.tight_layout(pad=1.0)
    plt.show()


def plotProcessingScan(i):
    if datasetName[:10]=='WildPlaces':
        framesRef=[]
    else:
        framesRef=list(np.load(refSaveFilepath))
    LPR = LiDAR_PlaceRec(datasetName, config, refNum, queryNum, framesQuery=framesQuery, framesRef=framesRef, refGridFilePath=refGridFilePath, background=0)
    ptcld=LPR.loading2DScan('REF', i)
    newScanDimX, newScanDimY = int(LPR.scanDimX*LPR.blockSize), int(LPR.scanDimY*LPR.blockSize)
    z_range=(np.min(ptcld[:,2]), np.max(ptcld[:,2])) 
    half_x = newScanDimX * LPR.mapRes // 2
    half_y = newScanDimY * LPR.mapRes // 2
    grid3d,_=np.histogramdd(ptcld, bins=(np.linspace(-half_x, half_x, newScanDimX+1),
                                        np.linspace(-half_y, half_y, newScanDimY+1),
                                        np.arange(z_range[0], z_range[1],LPR.mapRes)))
    a=np.sum(grid3d,axis=2) 
    # row_sums = a.sum(axis=1)
    # gridTopDown = a / row_sums[:, np.newaxis]
    # print(np.unique(gridTopDown))
    # gridTopDown[(gridTopDown <= 0.01) & (gridTopDown != 0)] = 0.01

    scaler = StandardScaler()
    gridTopDown=scaler.fit_transform(a)
    gridTopDown[(gridTopDown <= 0.01)] = 0

    fig,ax  = plt.subplots(1,1)
    ax.imshow(gridTopDown, cmap='gist_heat_r') #, 
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.savefig('wildplaces_scan.png', dpi=300)

    thresh=2
    gridTopDown=np.sum(grid3d,axis=2) 
    gridTopDown[gridTopDown<=thresh]=0
    gridTopDown[gridTopDown>thresh]=1

    fig,ax  = plt.subplots(1,1)
    ax.imshow(gridTopDown, cmap='gist_heat_r') #, 
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.savefig('wildplaces_scan2.png', dpi=300)

    gridTopDown=random_downsample(gridTopDown,10,2)
    fig,ax  = plt.subplots(1,1)
    ax.imshow(gridTopDown, cmap='gist_heat_r') #, 
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.savefig('wildplaces_scan3.png', dpi=300)

    gridTopDown=skimage.measure.block_reduce(gridTopDown, LPR.blockSize, func=LPR.poolFunc)
    # gridTopDown[gridTopDown==0]=-0.1
    fig,ax  = plt.subplots(1,1)
    ax.imshow(gridTopDown, cmap='gist_heat_r') #, 
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.savefig('wildplaces_scan4.png', dpi=300)


def plotCorrOutput(i):
    LPR = LiDAR_PlaceRec(datasetName, config, refNum, queryNum, framesQuery=framesQuery, framesRef=framesRef, refGridFilePath=refGridFilePath, background=-0.1)
    convOutput, allCorr, dist = LPR.update(i, rotIncr, xr, yr, yawR, xq, yq, yawQ, centerXs, centerYs, closeIds, n=1, returnConv=True)
    # im = ax[0, m].imshow(allCorr, vmin=allcormin, vmax=allcorrmax, cmap='plasma')

    fig,ax  = plt.subplots(1,1)
    ax.imshow(convOutput, cmap='viridis') #, 
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.savefig('convOutput.png', dpi=300)

'''---------------------------------------------------------------------------------------------------------------------'''
# scanDimX, scanDimY, mapRes, horizNumScans, intensityFilter, pointInterval =150, 150,0.5, 25, 0,1
# datasetName='Kitti'
# datasetDetails = {
#     0: ['00', range(0,3000), range(3200, 4541)],
#     1: ['02', range(0,3400), range(3600+500, 4661)], 
#     2: ['05', range(0,1000), range(1200,2751)], 
#     3: ['06', range(0,600), range(800,1101)],
#   }
# thresh= 0.55 #0.25 for seq '05'
# revisitThresh=3


# queryLen=len([i for i in datasetDetails[datasetNum][2]])
# queryIncr=1

# LPR=LiDAR_PlaceRec(datasetName, datasetDetails[datasetNum][0], datasetDetails[datasetNum][0], None, None, scanDimX, scanDimY, mapRes, 
#                 horizNumScans, intensityFilter, pointInterval, datasetDetails[datasetNum][1], datasetDetails[datasetNum][2])

# refGridFilePath='./data/KittiOdometryDataset/sequences/'+ datasetDetails[datasetNum][0]+f'/refGrid_{mapRes}.npy'
# print( refGridFilePath)
# if os.path.exists(refGridFilePath):
#     refgrid=np.load(refGridFilePath)
# else:
#     refgrid=LPR.makingReferenceGrid(rot=None, dim_3=False)
#     np.save(refGridFilePath, refgrid)

'''---------------------------------------------------------------------------------------------------------------------'''
envName='Karawatha'
datasetName=f'WildPlaces_{envName}'
# datasetName='NCLT'
# datasetName='OxfordRadar'
refNum, queryNum=1,2
errTolerance=3
# refNum, queryNum=11,2
# saveRefQuerySplit(datasetName, refNum, queryNum, None)   


with open('./scripts/config.json') as f:
    config = json.load(f)
config=config.get(datasetName, {})
refIncr = config.get('refIncr')
queryIncr = config.get('queryIncr')
refFilenames = config.get('details', {}).get(str(refNum), [])
queryFilenames = config.get('details', {}).get(str(queryNum), [])
param = config.get('parameters', {})
queryRad=param.get('queryRadius', 0.0)
refRad=param.get('refRadius', 0.0)
scanDimX = param.get('scanDimX', 0)
scanDimY = param.get('scanDimY', 0)
mapRes = param.get('mapRes', 0.0)
horizNumScans= param.get('horizNumScans', 0) 
refThresh= param.get('refThresh', 0)
queryThresh = param.get('queryThresh', 0)
dim_randDwnsmple= param.get('dim_randDwnsmple', 0)
numPtsMultiplier = param.get('nMultiplier_randDwnsmpl', 0)
blockSize=param.get('blkSize', 0)
NumMatches=param.get('topN')
background=param.get('unoccWeight')


#Oxford      
# querySaveFilepath=f'./results/LPR_{datasetName}/FramesFiles/R:{refNum}_Q:{queryNum}_framesQuery_qInc:{queryIncr}_qRad:{queryRad}_everyN.npy'
# refSaveFilepath=f'./results/LPR_{datasetName}/FramesFiles/R:{refNum}_framesRef_rInc:{refIncr}_rRad:{refRad}_everyN.npy'
# saveRefQuerySplit(datasetName, refNum, queryNum,querySaveFilepath, refSaveFilepath,None)      
# refNpyName=f"/refGrid_dim{scanDimX}_mapRes{mapRes}_refIncr{refIncr}_refRad{refRad}_rThresh{refThresh}_patchDim{dim_randDwnsmple}_nptsMult{numPtsMultiplier}.npy"


if datasetName[:10]=='WildPlaces':
    refNpyName=f"/refGrid_dim{scanDimX}_mapRes{mapRes}_refIncr{refIncr}_refRad{refRad}_rThresh{refThresh}_nptsMult{numPtsMultiplier}_AvgDownsamp.npy"#_negativeBackground
    refGridFilePath=refFilenames[0].rsplit('/', 1)[:-1][0] +refNpyName

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

elif datasetName == 'NCLT' or datasetName=='OxfordRadar':
    
    # refNpyName=f"/refGrid_dim{scanDimX}_mapRes{mapRes}_refIncr{refIncr}_refRad{refRad}_rThresh{refThresh}_patchDim{dim_randDwnsmple}_nptsMult{numPtsMultiplier}.npy"
    refNpyName=f"/refGrid_dim{scanDimX}_mapRes{mapRes}_refIncr{refIncr}_refRad{refRad}_rThresh{refThresh}_patchDim{dim_randDwnsmple}_nptsMult{numPtsMultiplier}.npy"
    refGridFilePath=refFilenames[0].rsplit('/', 1)[:-1][0] + refNpyName 

    querySaveFilepath=f'./results/LPR_{datasetName}/FramesFiles/Q:{queryNum}_framesQuery_qInc:{queryIncr}_qRad:{queryRad}_everyN.npy'
    refSaveFilepath=f'./results/LPR_{datasetName}/FramesFiles/R:{refNum}_framesRef_rInc:{refIncr}_rRad:{refRad}_Napart.npy'

    framesQuery=list(np.load(querySaveFilepath))
    framesRef=list(np.load(refSaveFilepath))
    LPR=LiDAR_PlaceRec(datasetName, config, refNum, queryNum, framesQuery=framesQuery, framesRef=framesRef, refGridFilePath=refGridFilePath, background=-0.1 ) 
    
    centerXYs=LPR.scanCenter(withPooling=False)
    centerXs,centerYs=zip(*centerXYs)
    xq,yq, rollQ, pitchQ, yawQ=LPR.scanPoses('QUERY')
    xr,yr,rollR, pitchR, yawR=LPR.scanPoses('REF')
    print(f'query len no incr: {len(framesQuery)}, ref len: {len(xr)}')
    closeIds=[find_closest_point(zip(xr,yr), xq[i], yq[i])[2] for i in range(len(xq)) ]
    minDist=[find_closest_point(zip(xr,yr), xq[i], yq[i])[3] for i in range(len(xq))]
    filteredFramesQuery=[framesQuery[i] for i in range(len(framesQuery)) if minDist[i] < errTolerance ]
    print(f'{len(LPR.framesQuery)-len(filteredFramesQuery)} queries removed from {len(LPR.framesQuery)} for being outside {errTolerance}m from ref')

    # '''Filter futher away query''' 
    # xrAll,yrAll, rollR, pitchR, yawRAll= LPR.scanPoses('All')
    # minDistAll=[find_closest_point(zip(xrAll,yrAll), xq[i], yq[i])[3] for i in range(len(xq))]
    # newframesQuery=[]#range(0, len(os.listdir(queryFilenames[0])), queryIncr)
    # for k in range(len(framesQuery)):
    #     if minDistAll[k]<errTolerance:
    #         newframesQuery.append(framesQuery[k])
    # print('Old query len',len(framesQuery),'New query len',len(newframesQuery))

rotIncr=10
matchIds,err, errCAN, errShift, correct, count=[], 0, 0, 0, 0, 0
TP, TN, FP, FN = 0,0,0,0

# plotClosestandMatch(50)
# plotCorrelationScores([20,200,500,750,900,1200])
# plotConvOutputForDifferentBackgrounds(50)
# plotReforQueryDownSampeld([20,500,750,900,1200])
# plotProcessingScan(2168)
# plotCorrOutput(100)
# assert False 


# fig, ((ax1,ax2, ax3), (ax4,ax5,ax6))= plt.subplots(2,3, figsize=(10,8))
# plt.tight_layout(pad=2)
# # ax1.scatter(xr, yr, color='c', s=3,alpha=0.5)
# ax1.scatter(xq, yq, color='g', s=3, alpha=0.5) 

fig, (ax1,ax2)= plt.subplots(2,1, figsize=(4,7))
plt.tight_layout(pad=1)
saveGoodCount, saveBadCount =0,0
ani = animation.FuncAnimation(fig, update_saveGoodBad,  frames=range(0, len(framesQuery)//queryIncr), repeat=False) #len(framesQuery)//queryIncr
plt.show()
# writervideo = animation.FFMpegWriter(fps=4, bitrate=-1) 
# ani.save(f'./{datasetName}_PATCHREF_{refNum}_QUERY{queryNum}_RES{mapRes}_DIM{scanDimX}_qIncr{queryIncr}_refIncr{refIncr}.mp4', writer=writervideo)
# print(f"ref: {refNum, refIncr}, query: {queryNum, queryIncr}, recall: {1-(err/count)}") 


assert False 

for i in range(2,3):
    for j in range(2,5):
            if i !=j:      
                refNum, queryNum=i,j
                with open('./scripts/config.json') as f:
                    config = json.load(f)
                config=config.get(datasetName, {})

                refIncr = config.get('refIncr')
                queryIncr = config.get('queryIncr')
                refFilenames = config.get('details', {}).get(str(refNum), [])
                queryFilenames = config.get('details', {}).get(str(queryNum), [])
                param = config.get('parameters', {})
                scanDimX = param.get('scanDimX', 0)
                scanDimY = param.get('scanDimY', 0)
                mapRes = param.get('mapRes', 0.0)
                horizNumScans= param.get('horizNumScans', 0) 

                print(refFilenames)
                framesRef= range(0, refFilenames[1], refIncr) if datasetName=='Jackal' else range(0, len(os.listdir(refFilenames[0])), refIncr)
                # evalInfo=load_from_pickle('/media/therese/TOSHIBA EXT/data/WildPlaces/data/Venman/Venman_evaluation_query.pickle')[queryNum-1] 
                evalInfo=load_from_pickle('/media/therese/TOSHIBA EXT/data/WildPlaces/data/Karawatha/Karawatha_evaluation_query.pickle')[queryNum-1] 
                all_files = os.listdir(queryFilenames[0])
                framesQuery=[]
                for k in range(len(evalInfo)):
                        try:
                            index = all_files.index(evalInfo[k]['query'].split('/')[-1])
                            framesQuery.append(index)
                        except ValueError:
                            pass

                # framesQuery=[ all_files.index(evalInfo[i]['query'].split('/')[-1]) for i in range(len(evalInfo))]
                print(len(framesQuery))


                LPR=LiDAR_PlaceRec(datasetName, config, refNum, queryNum, framesQuery ) 

                refGridFilePath=refFilenames[0].rsplit('/', 1)[:-1][0] +f'/refGrid_incr{refIncr}_dim{scanDimX}_mapRes{mapRes}_patchDownsample42.npy'
                # os.remove(refGridFilePath)

                if os.path.exists(refGridFilePath):
                    refgrid=np.load(refGridFilePath)
                    print('refGrid exsists')
                else:
                    refgrid=LPR.makingReferenceGrid(rot=None, dim_3=False)
                    np.save(refGridFilePath, refgrid)

                matchIds,err, errCAN, errShift, correct, count=[], 0, 0, 0, 0, 0
                TP, TN, FP, FN = 0,0,0,0
                centerXYs=LPR.scanCenter()
                centerXs,centerYs=zip(*centerXYs)

                xq,yq, rollQ, pitchQ, yawQ=LPR.scanPoses('QUERY')
                xr,yr,rollR, pitchR, yawR=LPR.scanPoses('REF')

                closeIds=[find_closest_point(zip(xr,yr), xq[i], yq[i])[2] for i in range(len(xq)) ]
                minDist=[find_closest_point(zip(xr,yr), xq[i], yq[i])[3] for i in range(len(xq))]
                # orderedYawR= [yawR[i] for i in closeIds]
                # matchIds.append(closeIds[0])


                fig, ((ax1,ax2, ax3), (ax4,ax5,ax6))= plt.subplots(2,3, figsize=(10,8))
                plt.tight_layout(pad=2)
                ax1.scatter(xr, yr, color='c', s=3,alpha=0.5)
                ax1.scatter(xq, yq, color='g', s=3, alpha=0.5) 
                ani = animation.FuncAnimation(fig, update_helperFunc, frames=len(framesQuery)//queryIncr, repeat=False) #len(framesQuery)//queryIncr
                # plt.show()
                writervideo = animation.FFMpegWriter(fps=4, bitrate=-1) 
                ani.save(f'./wildplaces_REF{refNum}_QUERY{queryNum}_RES{mapRes}_qIncr{queryIncr}_dwnSampleBoth.mp4', writer=writervideo)
                print(f"ref: {refNum, refIncr}, query: {queryNum, queryIncr}, recall: {1-(err/count)}") 






'''---------------------------------------------------------------------------------------------------------------------'''



# '''Load LPR data'''
# startID=int((datasetDetails[queryNum][2]/increment)*0.88)
# duration=100
# filename=f'./results/MatchShiftResults/Ref:{refNum}_Query:{queryNum}_Incr:{increment}_LPRparam:1.xlsx'   
# df = pd.read_excel(filename)
# refPositions, queryPositions, matchIds, matchPositions = df['referencePositions'].dropna(), df['queryPose'], df['matchID'], df['matchPose'],
# matchShiftedPositions, scanAlignment, shiftAmounts = df['matchShiftedPosition'], df['scanAlignment'], df['shiftAmount']

# zerodXR, zerodYR=zip(*refPositions.apply(lambda x: (ast.literal_eval(x)[0], ast.literal_eval(x)[1])))
# zerodXQ, zerodYQ, yawQ=zip(*queryPositions.apply(lambda x: (ast.literal_eval(x)[0], ast.literal_eval(x)[1], ast.literal_eval(x)[2])))
# matchX, matchY, matchYaw=zip(*matchPositions.apply(lambda x: (ast.literal_eval(x)[0], ast.literal_eval(x)[1], ast.literal_eval(x)[1])))
# zerodShiftedX, zerodShiftedY=zip(*matchShiftedPositions.apply(lambda x: (ast.literal_eval(x)[0], ast.literal_eval(x)[1])))
# matchXDeltas, matchYDeltas=zip(*shiftAmounts.apply(lambda x: (ast.literal_eval(x)[0], ast.literal_eval(x)[1])))
# idx=matchIds[startID]
# convFilename=f'./results/MatchShiftResults/{datasetName}_Ref:{refNum}_Query:{queryNum}_Incr:{increment}_LPRparam:1.npz'
# convOuputsSaved=np.load(convFilename, allow_pickle=True)

# '''Initiliase CAN'''
# N1, N2 = int(math.ceil(max(zerodYR) / 100.0)) * 100, int(math.ceil(max(zerodXR) / 100.0)) * 100 
# colDiff, rowDiff = int(N1-max(zerodYR)), int(N2-max(zerodXR))
# num_links,excite,activity_mag,inhibit_scale=10,2,0.06,6.5e-05 #10,2,0.06510262708,6.51431074e-05,2 #with decimals 200 iters
# # num_links,excite,activity_mag,inhibit_scale=8,  2,  6.76660488e-02,  2.39649509e-05
# num_links,excite,activity_mag,inhibit_scale=4,  5,  8.76770871e-03,  6.18866427e-05
# net=attractorNetwork2D(N1, N2, num_links, excite, activity_mag,inhibit_scale)
# weights=net.excitations(int(zerodYR[idx]), int(zerodXR[idx]))*4
# x,y=zerodXR[idx], zerodYR[idx]
# combinedWeights=np.zeros((N1,N2)) 

# fig, (ax1,ax2, ax4)= plt.subplots(1,3, figsize=(12,7))
def update_CAN_fitler(i): 
    global startID, weights, zerodXR, zerodYR, zerodXQ, zerodYQ,x,y, combinedWeights
    t=time.time()
    ax4.clear()

    i=i+startID
    idx=matchIds[i]

    '''CAN inputs'''
    delY, delX = zerodShiftedY[i]-y, zerodShiftedX[i]-x
    mag=(np.sqrt(delX**2 + delY**2))*2
    ang=np.rad2deg(math.atan2(delY, delX))

    '''Cropped conv'''
    convDim=20
    currConv=convOuputsSaved[i][50-convDim:50+convDim, 50-convDim:50+convDim]
    currConv=rotate(currConv, -ang, reshape=False) 
    # currConv=convolved[maxY-convDim:maxY+convDim, maxX-convDim:maxX+convDim]
    currConv=currConv/np.linalg.norm(currConv)

    '''CAN weights'''
    weights,wrap_rows, wrap_cols=net.update_weights_dynamics(weights, ang, mag)
    weights= weights/np.linalg.norm(weights)
    # weights[int(zerodShiftedY[i])-convDim:int(zerodShiftedY[i])+convDim, int(zerodShiftedX[i])-convDim:int(zerodShiftedX[i])+convDim]=currConv*0.6
    weights+=net.excitations(int(zerodShiftedY[i]), int(zerodShiftedX[i]))*scanAlignment[i]*0.1
    # weights+=net.fractional_shift(net.excitations(int(shiftedY), int(shiftedX))*0.05, shiftedY,shiftedX)
    # weights= weights/np.linalg.norm(weights)
    weights[weights<np.max(weights)*0.1]=0
    print(f'weights range: max {np.max(weights)}, min{np.min(weights)}')
    x=activityDecoding(weights[np.argmax(np.max(weights, axis=1)), :],5,N2)
    y=activityDecoding(weights[:, np.argmax(np.max(weights, axis=0)) ],5,N1)

    dist=np.sqrt((zerodYQ[i]-zerodYR[idx])**2+(zerodXQ[i]-zerodXR[idx])**2)
    distCAN=np.sqrt((zerodYQ[i]-y)**2+(zerodXQ[i]-x)**2)

    # combinedWeights= combinedWeights/np.linalg.norm(combinedWeights)
    # distQuery=np.sqrt((zerodYQ[i]-zerodYR[i-1])**2+(zerodXQ[i]-zerodXR[i-1])**2)
    distQuery=np.sqrt((delY)**2+(delX)**2)
    if distQuery>2:
        combinedWeights+=(weights/np.linalg.norm(weights))

    '''Ploting'''
    ax1.plot(zerodXQ[i], zerodYQ[i], 'g.')
    ax1.plot(x, y, 'b.')
    ax1.plot([zerodXQ[i], x], [zerodYQ[i], y], color='cyan', linestyle='-', alpha=0.5)
    ax1.legend(['Ref GT','Conv Match Pos'])
    
    ax2.plot(i,abs(zerodYQ[i]-y), '.',color='tab:red')
    ax2.plot(i,abs(zerodXQ[i]-x), '.',color='tab:pink')
    ax2.set_ylim([0,5])
    ax2.legend(['yerror', 'xerror'])
    ax2.set_title('GT to shifted match distance')


    # ax3.imshow(weights)
    # ax3.invert_yaxis()
    # ax3.set_title('CAN Dynamics')

    ax4.imshow(combinedWeights)
    ax4.invert_yaxis()
    ax4.set_title('Combined CAN')

    
   
    print(f'{i}, maqtchID: {idx}, matchDist {int(dist)}, canDist {int(distCAN)}, yaw: {np.rad2deg(yawQ[i]-(np.pi))}, heading:{ang} queryDist: {distQuery},  time {time.time()-t}')
    # plt.pause(0.1)

# fig, ((ax1,ax2, ax3))= plt.subplots(1,3, figsize=(8,5))
def update_realtiveOffset(i):
    global err, errCAN, errShift, centerXYs, weights, zerodXR, zerodYR, zerodXQ, zerodYQ
    t=time.time()
    ax3.clear()
    # ax1.plot(zerodXR, zerodYR, 'k.', alpha=0.3)
    
    queryScan= LPR.loadingCroppingFiltering_2DScan('QUERY', i)
    
    maxY, maxX, convolved= LPR.scanMatchWithConvolution(refgrid, i)
    idx=LPR.extractIDfromConv(maxX, maxY)
    matchIds.append(idx)
    

    '''Finding realtive shift'''
    centerXs,centerYs=zip(*centerXYs)
    centerX, centerY = centerXs[idx], centerYs[idx]
    matchXDelta, matchYDelta = maxX-centerX, maxY-centerY
    # shiftedX, shiftedY = zerodXR[idx]-matchYDelta, zerodYR[idx]-matchXDelta
    angle=np.arctan2(zerodYR[idx-1]-zerodYR[idx], zerodXR[idx-1]-zerodXR[idx])
    shiftedX = zerodXR[idx] + (math.cos(angle)*(matchXDelta)) - (math.sin(angle)*(matchYDelta))
    shiftedY = zerodYR[idx] + (math.sin(angle)*(matchXDelta)) + (math.cos(angle)*(matchYDelta))

    
    # cos_theta = np.cos(-yawR[idx])
    # sin_theta = np.sin(-yawR[idx])
    # rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    # xy=np.stack([matchXDelta, matchYDelta])
    # XY = np.matmul(rotation_matrix, xy)
    # matchXDelta, matchYDelta= int(XY[0]), int(XY[1])
    # shiftedX, shiftedY = zerodXR[idx]+matchXDelta, zerodYR[idx]+matchYDelta0


    refMatchScan= LPR.loadingCroppingFiltering_2DScan('REF', idx)
    shiftedImage=LPR.translateScan(refMatchScan,matchXDelta, matchYDelta)


    dist=np.sqrt((zerodYQ[i]-zerodYR[idx])**2+(zerodXQ[i]-zerodXR[idx])**2)
    distShift=np.sqrt((zerodYQ[i]-shiftedY)**2+(zerodXQ[i]-shiftedX)**2)
    if dist>10:
        err+=1

    if distShift > 10:
        errShift+=1

    '''Ploting'''
    ax1.plot(zerodXQ[i], zerodYQ[i], 'g.')
    ax1.plot(zerodXR[idx], zerodYR[idx], 'b.')
    ax1.plot([zerodXQ[i], zerodXR[idx]], [zerodYQ[i], zerodYR[idx]], color='cyan', linestyle='-', alpha=0.5)
    ax1.legend(['Query GT', 'Conv Match Pos'])
    
    ax2.plot(zerodXQ[i], zerodYQ[i], 'g.')
    ax2.plot(shiftedX,shiftedY,marker='.', color='tab:purple')
    ax2.plot([zerodXQ[i],shiftedX], [zerodYQ[i], shiftedY], color='tab:brown', linestyle='-', alpha=0.5)
    ax2.legend(['Query GT', 'Conv Shift Match Pos'])


    ax3.imshow(queryScan, cmap='grey')
    ax3.imshow(shiftedImage, cmap='Purples_r', alpha=0.7)
    ax3.set_title(f'Query and shift ref {int(matchXDelta)}, {int(matchYDelta)}')

    

    print(f'{i}, errCount {err}, shiftErrCount {errShift}, matchDist {int(dist)}, shiftDist {int(distShift)}, matchID {idx}, time {time.time()-t}')
    # plt.pause(0.1)


def update_imageLidarCombine(i): 
    global err, centerXYs, distCANTot,weights,x,y
    t=time.time()
    ax3.clear(), ax5.clear()

    maxY, maxX, convolved= LPR.scanMatchWithConvolution(refgrid, i)
    idx=LPR.extractIDfromConv(maxX, maxY)
    matchIds.append(idx)

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
    alignedSum=(np.count_nonzero((queryScan+refShiftMatchScan)==2)/maxAlignCount)
    matchConfidence= 0.2 if alignedSum>0.5 else 0.05 

    '''Shifted Pos'''
    angle=np.arctan2(zerodYR[idx-1]-zerodYR[idx], zerodXR[idx-1]-zerodXR[idx])
    shiftedX = zerodXR[idx] + (math.cos(angle)*(matchXDelta)) - (math.sin(angle)*(matchYDelta))
    shiftedY = zerodYR[idx] + (math.sin(angle)*(matchXDelta)) + (math.cos(angle)*(matchYDelta))

    '''CAN'''
    # delY, delX = (zerodYR[idx]-zerodYR[matchIds[-2]]), (zerodXR[idx]-zerodXR[matchIds[-2]])
    delY, delX = shiftedY-y, shiftedX-x
    # delY, delX  = zerodYQ[i]-y, zerodXQ[i]-x
    mag=(np.sqrt(delX**2 + delY**2))*2
    ang=np.rad2deg(math.atan2(delY, delX))


    weights,wrap_rows, wrap_cols=net.update_weights_dynamics(weights, ang, mag)
    weights= weights/np.linalg.norm(weights)
    weights+=net.excitations(int(shiftedY), int(shiftedX))*alignedSum*0.15

    # weights+=net.excitations(int(zerodYR[mIds[i]]), int(zerodXR[mIds[i]]))*0.02
    # weights[weights<np.percentile(weights, 50)]=0
    # weights= (weights/np.linalg.norm(weights))
    oldX, oldY=x,y
    x=activityDecoding(weights[np.argmax(np.max(weights, axis=1)), :],5,N2)
    y=activityDecoding(weights[:, np.argmax(np.max(weights, axis=0)) ],5,N1)
    
    '''Spread'''
    non_zero_rows, non_zero_cols = np.nonzero(weights)
    # Calculate distances from the maximum row and column to all non-zero rows and columns
    row_distances = np.abs(non_zero_rows - y)
    col_distances = np.abs(non_zero_cols - x)
    averageSpread=np.mean(row_distances)+np.mean(col_distances)
    
    '''Error dists'''
    distImg=np.sqrt((yq[i]-yr[mIds[i]])**2+(xq[i]-xr[mIds[i]])**2)
    distMatch=np.sqrt((yq[i]-yr[idx])**2+(xq[i]-xr[idx])**2)

    distCAN=np.sqrt((zerodYQ[i]-y)**2+(zerodXQ[i]-x)**2)
    distShift=np.sqrt((zerodYQ[i]-shiftedY)**2+(zerodXQ[i]-shiftedX)**2)
    distShifts.append(distShift)
    
    distCANTot+=distCAN
    if distCAN>20:
        err+=1

    '''Ploting'''
    ax1.plot(zerodXQ[i], zerodYQ[i], 'g.')
    ax1.plot(shiftedX, shiftedY, 'c.')
    ax1.plot([zerodXQ[i],shiftedX], [zerodYQ[i], shiftedY], color='tab:brown', linestyle='-', alpha=0.5)

    ax2.plot(averageSpread, distCAN, '.',color='tab:purple')
    ax2.set_title('CAN spread vs error')
   
    ax3.imshow(queryScan, cmap='gray')
    ax3.imshow(refShiftMatchScan, cmap='Purples_r', alpha=0.7)
    ax3.set_title(f'Query and Shifted: {matchXDelta, matchYDelta}, align:{round(alignedSum,2)})')

    ax4.plot(zerodXQ[i], zerodYQ[i], 'g.')
    ax4.plot(x,y, 'b.')
    ax4.plot([zerodXQ[i],x], [zerodYQ[i], y], color='tab:brown', linestyle='-', alpha=0.5)
    # ax1.legend(['Query GT', 'Closest Ref to Query', 'Conv Match Pos'])   

    ax5.imshow(weights)
    ax5.invert_yaxis()
    ax5.set_title('CAN weights')
    

    # ax6.plot(i, abs(yq[i]-yr[mIds[i]]), '.',color='tab:purple')
    # ax6.plot(i, abs(xq[i]-xr[mIds[i]]), '.',color='tab:pink')
    ax6.plot(i,distShift, 'k.')
    ax6.plot(i,distCAN, 'y.')
    # ax6.set_ylim([0,20])
    ax6.set_title('error lidar(black) CAN(yellow) match')

    

    print(f'{i}, truePos: {round(zerodXQ[i],2), round(zerodYQ[i],2)}, PosWithDels: {round(oldX+delX,2), round(oldY+delY,2)}, usedDels: {round(delX,2), round(delY,2)},')# trueDels: {round(trueDelX,2), round(trueDelY,2)}')
    print(f' distMatch, distCAn: {round(distShift,2), round(distCAN,2)}, err_CAN, err_MatchShift: {round(distCANTot,2), round(np.sum(distShifts),2)}, align: {round(alignedSum,2)}, time {round(time.time()-t,2)}')
    print('')
    # plt.pause(0.1)

# fig, ((ax1,ax2, ax3),(ax4,ax5,ax6))= plt.subplots(2,3, figsize=(10,7))
# ax1.plot(zerodXR,zerodYR, 'k--', alpha=0.3)
# ax4.plot(zerodXQ, zerodYQ, 'k--', alpha=0.3)
# netLocalUpdate=attractorNetwork2D(scanDimY, scanDimX, num_links, excite, activity_mag,inhibit_scale)
# weightsLocalUpdate=netLocalUpdate.excitations(int(scanDimY//2), int(scanDimX//2))
def update_CAN_twoScale_relativeOffset(i):
    global err, errCAN, centerXYs, weights, weightsLocalUpdate, zerodXR, zerodYR, zerodXQ, zerodYQ
    t=time.time()
    ax2.clear(), ax3.clear(), ax5.clear(), ax6.clear()
    # ax1.plot(zerodXR, zerodYR, 'k.', alpha=0.3)

    '''Query scan'''
    cropped_x, cropped_y= LPR.loadingCroppingFiltering_2DScan('QUERY', i)
    queryScan=LPR.makingImageFromLidarCoords(cropped_x,cropped_y)
    
    
    '''Conv Matching'''
    maxY, maxX, convolved= LPR.scanMatchWithConvolution(refgrid, i)
    idx=LPR.extractIDfromConv(maxX, maxY)
    matchIds.append(idx)
    

    '''Finding realtive shift'''
    centerXs,centerYs=zip(*centerXYs)
    centerX, centerY = centerXs[idx], centerYs[idx]
    matchXDelta, matchYDelta = maxX-centerX, maxY-centerY
    shiftedX, shiftedY = zerodXR[idx]-matchXDelta, zerodYR[idx]-matchYDelta

    currConv=convolved[centerY-(scanDimY//2):centerY+(scanDimY//2), centerX-(scanDimX//2):centerX+(scanDimX//2)]


    '''Ref match scans'''
    cropped_x, cropped_y= LPR.loadingCroppingFiltering_2DScan('REF', idx)
    refConvMatchScan=LPR.makingImageFromLidarCoords(cropped_x,cropped_y)
    refShiftMatchScan= LPR.translateScan(refConvMatchScan, matchXDelta, matchYDelta)
    

    '''CAN'''
    delY, delX = (zerodYR[idx]-zerodYR[matchIds[-2]]), (zerodXR[idx]-zerodXR[matchIds[-2]])
    mag=(np.sqrt(delX**2 + delY**2))*2
    ang=np.rad2deg(math.atan2(delY, delX))
    weights,wrap_rows, wrap_cols=net.update_weights_dynamics(weights, ang, mag)
    weights= weights/np.linalg.norm(weights)
    weights+=net.excitations(int(zerodYR[idx]), int(zerodXR[idx]))*0.05
    # weights+=net.fractional_shift(net.excitations(int(shiftedY), int(shiftedX))*0.05, shiftedY,shiftedX)
    weights= (weights/np.linalg.norm(weights))

    print(currConv.shape, weightsLocalUpdate.shape)
    weightsLocalUpdate+=currConv
    weightsLocalUpdate= (weightsLocalUpdate/np.linalg.norm(weightsLocalUpdate))    


    xShift=activityDecoding(weightsLocalUpdate[np.argmax(np.max(weightsLocalUpdate, axis=1)), :],5,scanDimX)-(scanDimX//2)
    yShift=activityDecoding(weightsLocalUpdate[:, np.argmax(np.max(weightsLocalUpdate, axis=0)) ],5,scanDimY)-(scanDimY//2)

    x=activityDecoding(weights[np.argmax(np.max(weights, axis=1)), :],5,N2)-xShift
    y=activityDecoding(weights[:, np.argmax(np.max(weights, axis=0)) ],5,N1)-yShift


    dist=np.sqrt((zerodYQ[i]-zerodYR[idx])**2+(zerodXQ[i]-zerodXR[idx])**2) 
    distCAN=np.sqrt((zerodYQ[i]-y)**2+(zerodXQ[i]-x)**2)
    if dist>10:
        err+=1

    if distCAN>10:
        errCAN+=1

    '''Ploting'''
    ax1.plot(x,y,'b.')
    ax1.legend(['Ref Path','CAN Match Pos'])
    
    ax2.imshow(weights)
    ax2.set_title('Coarse CAN')

    ax3.imshow(weightsLocalUpdate)
    ax3.set_title(f'Local Update CAN: {int(xShift), int(yShift)}')

    
    ax4.scatter(zerodXQ[i], zerodYQ[i], c='g', s=4)
    ax4.scatter(x,y,c='b', s=4)
    ax4.plot([zerodXQ[i],x], [zerodYQ[i], y], color='cyan', linestyle='-', alpha=0.5)
    ax4.legend(['Query Path', 'Query Pos', 'CAN Match Pos'])


    ax5.imshow(queryScan, cmap='Blues', alpha=1)
    ax5.imshow(refConvMatchScan, cmap='Reds', alpha=0.5)
    ax5.set_title('No Alignment')

    ax6.imshow(queryScan, cmap='Blues', alpha=1)
    ax6.imshow(refShiftMatchScan, cmap='Reds', alpha=0.5)
    ax6.set_title('Aligned')

    

    print(f'{i}, matchErrCount {err},  matchDist {int(dist)}, canErrCount {errCAN}, canDist {int(distCAN)}, canXY {int(x), int(y)}, trueXy {int(zerodXQ[i]),int(zerodYQ[i])}, shift {int(xShift), int(yShift)}, time {time.time()-t}')
    # plt.pause(0.1)


# fig, ((ax1,ax2,ax3, ax4),(ax5,ax6,ax7,ax8))= plt.subplots(2,4, figsize=(10,7))
# ax1.plot(xr, yr, 'c--')
# ax1.plot(xq, yq, 'g--')
# # ax1.axis('equal')
# ax5.plot(xr, yr, 'c--')
# ax5.plot(xq, yq, 'g--')
# ax4.axis('equal')
def update_JackalLidarCheck(i):
    global err, centerXYs
    t=time.time()
    ax4.clear(), ax8.clear(),
    # ax1.invert_yaxis()
    # ax1.plot(xr, yr, 'y--')
    queryIds=[j for j in framesQuery]
    refIDs=[j for j in framesRef]

    queryScan= LPR.loadingCroppingFiltering_2DScan('QUERY', queryIds[i])
    # xy_arr=np.transpose(np.stack([cropped_x, cropped_y],axis=0))
    # queryScan=LPR.convert_pc_to_grid(xy_arr)
    
    maxY, maxX, convolved= LPR.scanMatchWithConvolution(refgrid, queryIds[i])
    idx=LPR.extractIDfromConv(maxX, maxY)
    matchIds.append(idx)
    print(idx, len(xr))

    '''Finding realtive shift'''
    centerXs,centerYs=zip(*centerXYs)
    centerX, centerY = centerXs[idx], centerYs[idx]
    matchXDelta, matchYDelta = maxX-centerX, maxY-centerY
    
    '''Ref match scans'''
    refConvMatchScan= LPR.loadingCroppingFiltering_2DScan('REF', refIDs[idx])
    # xy_arr=np.transpose(np.stack([cropped_x, cropped_y],axis=0))
    # refConvMatchScan=LPR.convert_pc_to_grid(xy_arr)
    refShiftMatchScan= LPR.translateScan(refConvMatchScan, matchXDelta, matchYDelta)

    '''Shifted Pos'''
    shiftedX = xr[idx] + (math.cos(yawR[idx])*(matchXDelta*mapRes)) - (math.sin(yawR[idx])*(matchYDelta*mapRes))
    shiftedY = yr[idx] + (math.sin(yawR[idx])*(matchXDelta*mapRes)) + (math.cos(yawR[idx])*(matchYDelta*mapRes))

    '''Match velocity'''
    delY, delX = (yr[idx]-yr[matchIds[-2]]), (xr[idx]-xr[matchIds[-2]])
    mag=(np.sqrt(delX**2 + delY**2))
    
    '''Scan align percentage '''
    maxAlignCount=np.min([np.count_nonzero(queryScan==1), np.count_nonzero(refShiftMatchScan==1)])
    alignedSum=(np.count_nonzero((queryScan+refShiftMatchScan)==2)/np.count_nonzero(queryScan==1))

    dist=np.sqrt((yq[i]-shiftedY)**2+(xq[i]-shiftedX)**2)
    distMatch=np.sqrt((yq[i]-yr[idx])**2+(xq[i]-xr[idx])**2)
    if dist>0.5:
        err+=1
    # xShiftErr=abs(xq[i]-shiftedX)
    # xErr=abs(xq[i]-xr[idx])
    # yShiftErr=abs(yq[i]-shiftedY)
    # yErr=abs(yq[i]-yr[idx])

    '''Ploting'''
    ax1.plot(xq[i], yq[i], 'g.')
    ax1.plot(shiftedX,shiftedY, 'b.')
    ax1.plot([xq[i],shiftedX], [yq[i], shiftedY], color='tab:brown', linestyle='-', alpha=0.5)
    ax1.legend(['Query Pos', 'Shifted Ref Pos'])
    ax1.set_title('With relative shift')

    
    ax2.plot(i,dist, 'm.' )
    ax2.plot(np.arange(i-1,i+1), [0.5]*2, 'k--')
    # ax2.legend(['xError [m]','yError [m]'])
    ax2.set_title(f'Error match+shif ') #- align% {round(alignedSum,3)}
    ax2.set_ylim([0,3.5])

    if dist>0.5: 
        ax3.plot(i, mag, '*',color='r', alpha=0.5)
    else: 
        ax3.plot(i, mag, '.',color='tab:purple')
    ax3.set_title(f'Velocity')
    ax3.set_xticks(np.arange(i), minor=True)
    ax3.grid(axis='x', linewidth=1)

    ax4.imshow(queryScan, cmap='Blues', alpha=1)
    ax4.imshow(refShiftMatchScan, cmap='Reds', alpha=0.5)
    ax4.set_title(f'Aligned: { matchXDelta, matchYDelta} - {round(alignedSum,3)}%')

    ax5.plot(xq[i], yq[i], 'g.')
    ax5.plot(xr[idx], yr[idx], 'b.')
    ax5.plot([xq[i],xr[idx]], [yq[i], yr[idx]], color='tab:brown', linestyle='-', alpha=0.5)
    ax5.legend(['Query Pos', 'Matched Ref Pos'])
    ax5.set_title('No relative shift')
 
    
    # ax5.plot(i,xErr, 'r.' )
    ax6.plot(i,distMatch, 'm.' )
    # ax5.legend(['xError [m]','yError [m]'])
    ax6.set_title('Error match (only)')
    ax6.set_ylim([0,3.5])
    ax6.plot(np.arange(i-1,i+1), [0.5]*2, 'k--')

    if dist>0.5: 
        ax7.plot(i, alignedSum, '*',color='r', alpha=0.5)
    else: 
        ax7.plot(i, alignedSum, '.',color='tab:olive')
    ax7.set_title('Alignability')
    ax7.grid(axis='x', linewidth=1)
    ax7.set_xticks(np.arange(i), minor=True)

    ax8.imshow(queryScan, cmap='Blues', alpha=1)
    ax8.imshow(refConvMatchScan, cmap='Reds', alpha=0.5)
    ax8.set_title('No Alignment')


    print(f'{i}, matchID {idx} time {time.time()-t}, yaw {np.rad2deg(yawR[idx])}, errCount {err}, matchDist {round(dist,3)} ')
    # plt.pause(0.1)


def update_JackalCAN(i):
    global err, centerXYs, weights 
    t=time.time()
    ax4.clear(), ax8.clear(), ax5.clear()
    # ax1.invert_yaxis()
    # ax1.plot(xr, yr, 'y--')

    cropped_x, cropped_y= LPR.loadingCroppingFiltering_2DScan('QUERY', i*increment)
    xy_arr=np.transpose(np.stack([cropped_x, cropped_y],axis=0))
    queryScan=LPR.convert_pc_to_grid(xy_arr)
    
    maxY, maxX, convolved= LPR.scanMatchWithConvolution(refgrid, i*increment)
    idx=LPR.extractIDfromConv(maxX, maxY)
    matchIds.append(idx)

    '''Finding realtive shift'''
    centerXs,centerYs=zip(*centerXYs)
    centerX, centerY = centerXs[idx], centerYs[idx]
    matchXDelta, matchYDelta = maxX-centerX, maxY-centerY
    

    '''Ref match scans'''
    cropped_x, cropped_y= LPR.loadingCroppingFiltering_2DScan('REF', idx*increment)
    xy_arr=np.transpose(np.stack([cropped_x, cropped_y],axis=0))
    refConvMatchScan=LPR.convert_pc_to_grid(xy_arr)
    refShiftMatchScan= LPR.translateScan(refConvMatchScan, matchXDelta, matchYDelta)

    '''Shifted Pos'''
    shiftedX = zerodXR[idx] + (math.cos(yawR[idx])*(matchXDelta*mapRes)) - (math.sin(yawR[idx])*(matchYDelta*mapRes))
    shiftedY = zerodYR[idx] + (math.sin(yawR[idx])*(matchXDelta*mapRes)) + (math.cos(yawR[idx])*(matchYDelta*mapRes))


    '''Scan align percentage '''
    maxAlignCount=np.min([np.count_nonzero(queryScan==1), np.count_nonzero(refShiftMatchScan==1)])
    alignedSum=(np.count_nonzero((queryScan+refShiftMatchScan)==2)/maxAlignCount)

    '''Match lin and ang velocity'''
    delY, delX = (zerodYR[idx]-zerodYR[matchIds[-2]]), (zerodXR[idx]-zerodXR[matchIds[-2]])
    mag=(np.sqrt(delX**2 + delY**2))*2
    ang=np.rad2deg(math.atan2(delY, delX))

    '''CAN'''
    weights,wrap_rows, wrap_cols=net.update_weights_dynamics(weights, ang, mag)
    weights= weights/np.linalg.norm(weights)
    # weights+=net.excitations(int(zerodYR[idx]), int(zerodXR[idx]))*0.05
    # weights+=net.fractional_shift(net.excitations(int(shiftedY), int(shiftedX))*0.05, shiftedY,shiftedX)
    weights= (weights/np.linalg.norm(weights))
    if alignedSum>0.4:
        weights+=net.excitations(int(shiftedY), int(shiftedX))
        weights= (weights/np.linalg.norm(weights))

    x=activityDecoding(weights[np.argmax(np.max(weights, axis=1)), :],5,N2)
    y=activityDecoding(weights[:, np.argmax(np.max(weights, axis=0)) ],5,N1)

 
    dist=np.sqrt((yq[i]-y)**2+(xq[i]-x)**2)
    distMatch=np.sqrt((yq[i]-yr[idx])**2+(xq[i]-xr[idx])**2)
    if dist>0.5:
        err+=1
    # xShiftErr=abs(xq[i]-shiftedX)
    # xErr=abs(xq[i]-xr[idx])
    # yShiftErr=abs(yq[i]-shiftedY)
    # yErr=abs(yq[i]-yr[idx])

    '''Ploting'''
    ax1.plot(xq[i], yq[i], 'g.')
    ax1.plot(x,y, 'b.')
    ax1.plot([xq[i],x], [yq[i], y], color='tab:brown', linestyle='-', alpha=0.5)
    ax1.legend(['Query Pos', 'CAN Ref Pos'])
    ax1.set_title('With relative shift')

    
    ax2.plot(i,dist, 'm.' )
    ax2.plot(np.arange(i-1,i+1), [0.5]*2, 'k--')
    # ax2.legend(['xError [m]','yError [m]'])
    ax2.set_title(f'Error match+shif ') #- align% {round(alignedSum,3)}
    # ax2.set_ylim([0,3.5])

    if dist>0.5: 
        ax3.plot(i, mag, '*',color='r', alpha=0.5)
    else: 
        ax3.plot(i, mag, '.',color='tab:purple')
    ax3.set_title(f'Velocity')
    ax3.set_xticks(np.arange(i), minor=True)
    ax3.grid(axis='x', linewidth=1)

    ax4.imshow(queryScan, cmap='Blues', alpha=1)
    ax4.imshow(refShiftMatchScan, cmap='Reds', alpha=0.5)
    ax4.set_title(f'Aligned: { matchXDelta, matchYDelta} - {round(alignedSum,3)}%')

    ax5.imshow(weights)
    ax5.set_title('CAN')
 
    
    # ax5.plot(i,xErr, 'r.' )
    ax6.plot(i,distMatch, 'm.' )
    # ax5.legend(['xError [m]','yError [m]'])
    ax6.set_title('Error match (only)')
    # ax6.set_ylim([0,3.5])
    ax6.plot(np.arange(i-1,i+1), [0.5]*2, 'k--')

    if dist>0.5: 
        ax7.plot(i, alignedSum, '*',color='r', alpha=0.5)
    else: 
        ax7.plot(i, alignedSum, '.',color='tab:olive')
    ax7.set_title('Alignability')
    ax7.grid(axis='x', linewidth=1)
    ax7.set_xticks(np.arange(i), minor=True)

    ax8.imshow(queryScan, cmap='Blues', alpha=1)
    ax8.imshow(refConvMatchScan, cmap='Reds', alpha=0.5)
    ax8.set_title('No Alignment')


    print(f'{i}, matchID {idx} time {time.time()-t}, yaw {np.rad2deg(yawR[idx])}, errCount {err}, matchDist {round(dist,3)} ')
    # plt.pause(0.1)


def update_Jackal_CoarseFineSearch(i):
    global err, centerXYs
    t=time.time()
    ax4.clear(), ax8.clear(),
    # ax1.invert_yaxis()
    # ax1.plot(xr, yr, 'y--')
    queryIds=[j for j in framesQuery]
    refIDs=[j for j in framesRef]

    '''Query scans'''
    cropped_x, cropped_y= LPR.loadingCroppingFiltering_2DScan('QUERY', queryIds[i])
    xy_arr=np.transpose(np.stack([cropped_x, cropped_y],axis=0))
    queryScan=LPR.convert_pc_to_grid(xy_arr)

    '''Coarse Search'''
    # '''Scan matching'''
    maxY, maxX, convolved= LPR.scanMatchWithConvolution(refgrid, queryIds[i])
    idx=LPR.extractIDfromConv(maxX, maxY)
    matchIds.append(idx)

    # '''Finding realtive shift'''
    centerXs,centerYs=zip(*centerXYs)
    centerX, centerY = centerXs[idx], centerYs[idx]
    matchXDelta, matchYDelta = maxX-centerX, maxY-centerY
    
    shiftedX = xr[idx] + (math.cos(yawR[idx])*(matchXDelta*mapRes)) - (math.sin(yawR[idx])*(matchYDelta*mapRes))
    shiftedY = yr[idx] + (math.sin(yawR[idx])*(matchXDelta*mapRes)) + (math.cos(yawR[idx])*(matchYDelta*mapRes))

    # '''Ref match scans'''
    cropped_x, cropped_y= LPR.loadingCroppingFiltering_2DScan('REF', refIDs[idx])
    xy_arr=np.transpose(np.stack([cropped_x, cropped_y],axis=0))
    refConvMatchScan=LPR.convert_pc_to_grid(xy_arr)
    refShiftMatchScan= LPR.translateScan(refConvMatchScan, matchXDelta, matchYDelta)

    # '''Match velocity'''
    delY, delX = (yr[idx]-yr[matchIds[-2]]), (xr[idx]-xr[matchIds[-2]])
    mag=(np.sqrt(delX**2 + delY**2))
    
    # '''Scan align percentage '''
    maxAlignCount=np.min([np.count_nonzero(queryScan==1), np.count_nonzero(refShiftMatchScan==1)])
    alignedSum=(np.count_nonzero((queryScan+refShiftMatchScan)==2)/maxAlignCount)

    '''Fine search'''
    topN=10
    searchradius=10
    maxYs, maxXs, convolved= LPR.scanMatchWithConvolution(refgrid, queryIds[i], 'SINGLE', ONEorN=topN)
    idxes=np.unique([LPR.extractIDfromConv(maxXs[i], maxYs[i]) for i in range(len(maxYs))])
    
    localRefGrid=LPR.topMatchNeighbourReference(idxes,searchradius)
    maxYLocal, maxXLocal, convolvedLocal= LPR.scanMatchWithConvolution(localRefGrid, queryIds[i], 'SINGLE')
    bestMatch, fineSearchShift, fineCenterXYs=LPR.extractIDfromLocalGrid(maxXLocal, maxYLocal,idxes, searchradius)
    xFine, yFine, yawFine = xrAll[(bestMatch*increment)+fineSearchShift], yrAll[(bestMatch*increment)+fineSearchShift], yawRAll[(bestMatch*increment)+fineSearchShift]
    print(f"idx: {idx}: idx other: {idxes}, bestmatch: {bestMatch} idxShift: {fineSearchShift}")

    # '''Finding realtive shift'''
    centerX,centerY=fineCenterXYs
    # centerX, centerY = centerXs[bestMatch+fineSearchShift], centerYs[bestMatch+fineSearchShift]
    matchXDelta, matchYDelta = maxXLocal-centerX, maxYLocal-centerY
    
    shiftedXFine = xFine #+ (math.cos(yawFine)*(matchXDelta*mapRes)) - (math.sin(yawFine)*(matchYDelta*mapRes))
    shiftedYFine = yFine #+ (math.sin(yawFine)*(matchXDelta*mapRes)) + (math.cos(yawFine)*(matchYDelta*mapRes))

    dist=np.sqrt((yq[i]-shiftedY)**2+(xq[i]-shiftedX)**2)
    distMatch=np.sqrt((yq[i]-shiftedYFine)**2+(xq[i]-shiftedXFine)**2)
    if dist>0.5:
        err+=1
    # xShiftErr=abs(xq[i]-shiftedX)
    # xErr=abs(xq[i]-xr[idx])
    # yShiftErr=abs(yq[i]-shiftedY)
    # yErr=abs(yq[i]-yr[idx])

    '''Ploting'''
    ax1.plot(xq[i], yq[i], 'g.')
    ax1.plot(shiftedX,shiftedY, 'b.')
    ax1.plot([xq[i],shiftedX], [yq[i], shiftedY], color='tab:brown', linestyle='-', alpha=0.5)
    ax1.legend(['Query Pos', 'Shifted Ref Pos'])
    ax1.set_title('With relative shift')

    
    ax2.plot(i,dist, 'm.' )
    ax2.plot(np.arange(i-1,i+1), [0.5]*2, 'k--')
    # ax2.legend(['xError [m]','yError [m]'])
    ax2.set_title(f'Error match+shif ') #- align% {round(alignedSum,3)}
    ax2.set_ylim([0,3.5])

    if dist>0.5: 
        ax3.plot(i, mag, '*',color='r', alpha=0.5)
    else: 
        ax3.plot(i, mag, '.',color='tab:purple')
    ax3.set_title(f'Velocity')
    ax3.set_xticks(np.arange(i), minor=True)
    ax3.grid(axis='x', linewidth=1)

    ax4.imshow(queryScan, cmap='Blues', alpha=1)
    ax4.imshow(refShiftMatchScan, cmap='Reds', alpha=0.5)
    ax4.set_title(f'Aligned: { matchXDelta, matchYDelta} - {round(alignedSum,3)}%')

    ax5.plot(xq[i], yq[i], 'g.')
    ax5.plot(shiftedXFine, shiftedYFine, 'b.')
    ax5.plot([xq[i],shiftedXFine], [yq[i], shiftedYFine], color='tab:brown', linestyle='-', alpha=0.5)
    ax5.legend(['Query Pos', 'Matched Ref Pos'])
    ax5.set_title('Fine search')
 
    
    # ax5.plot(i,xErr, 'r.' )
    ax6.plot(i,distMatch, 'm.' )
    # ax5.legend(['xError [m]','yError [m]'])
    ax6.set_title('Error fine match (only)')
    ax6.set_ylim([0,3.5])
    ax6.plot(np.arange(i-1,i+1), [0.5]*2, 'k--')

    if dist>0.5: 
        ax7.plot(i, alignedSum, '*',color='r', alpha=0.5)
    else: 
        ax7.plot(i, alignedSum, '.',color='tab:olive')
    ax7.set_title('Alignability')
    ax7.grid(axis='x', linewidth=1)
    ax7.set_xticks(np.arange(i), minor=True)

    ax8.imshow(queryScan, cmap='Blues', alpha=1)
    ax8.imshow(refConvMatchScan, cmap='Reds', alpha=0.5)
    ax8.set_title('No Alignment')


    print(f'{i}, matchID {idx} time {time.time()-t}, yaw {np.rad2deg(yawR[idx])}, errCount {err}, matchDist {round(dist,3)} ')
    # plt.pause(0.1)



# "details": {
#   "1": ["/media/therese/TOSHIBA EXT/data/NCLT/2012-01-08/velodyne_sync/", "/media/therese/TOSHIBA EXT/data/NCLT/groundtruth_2012-01-08.csv", 8226],
#   "2": ["/media/therese/TOSHIBA EXT/data/NCLT/2012-06-15/velodyne_sync/", "/media/therese/TOSHIBA EXT/data/NCLT/groundtruth_2012-06-15.csv", 8941],
#   "3": ["/media/therese/TOSHIBA EXT/data/NCLT/2012-02-05/velodyne_sync/", "/media/therese/TOSHIBA EXT/data/NCLT/groundtruth_2012-02-05.csv", 8941]
# },