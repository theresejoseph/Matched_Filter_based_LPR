import pandas as pd
import matplotlib.patches as patches
from scipy.spatial.distance import cdist
from matplotlib.gridspec import GridSpec
import math
from helperFunctions import activityDecoding, attractorNetwork2D, LiDAR_PlaceRec, Oxford_MatchShift, Jackal_MatchShift, runWildPlaces, saveRefQuerySplit, ablateRefQuerySplit, ablateRefQuerySplitUrban
import numpy as np
import time 
import matplotlib.pyplot as plt 
import ast
from scipy.signal import find_peaks
from matplotlib.ticker import PercentFormatter
import os
import pickle
import json 
from concurrent.futures import ThreadPoolExecutor


'''LPR Params'''
# datasetName='OxfordRadar'
# datasetDetails = {
#     1: ['./data/OxfordRadarDataset/2019-01-10-11-46-21-radar-oxford-10k/velodyne_left', './data/OxfordRadarDataset/2019-01-10-11-46-21-radar-oxford-10k/gps/ins.csv', 44414, 2220],
#     2: ['./data/OxfordRadarDataset/2019-01-10-12-32-52-radar-oxford-10k/velodyne_left', './data/OxfordRadarDataset/2019-01-10-12-32-52-radar-oxford-10k/gps/ins.csv', 43143, 2157], 
#     3: ['./data/OxfordRadarDataset/2019-01-10-14-02-34-radar-oxford-10k/velodyne_left', './data/OxfordRadarDataset/2019-01-10-14-02-34-radar-oxford-10k/gps/ins.csv'  , 39766, 1988], 
#     4: ['./data/OxfordRadarDataset/2019-01-16-13-42-28-radar-oxford-10k/velodyne_left', './data/OxfordRadarDataset/2019-01-16-13-42-28-radar-oxford-10k/gps/ins.csv', 38271, 1913],
#     5: ['./data/OxfordRadarDataset/2019-01-18-15-20-12-radar-oxford-10k/velodyne_left', './data/OxfordRadarDataset/2019-01-18-15-20-12-radar-oxford-10k/gps/ins.csv', 41510, 2075],
# }
# scanDimX, scanDimY, mapRes, horizNumScans, intensityFilter, pointInterval,scanScale,  =100, 100, 0.1, 25, 0.4, 20, 1, 

# datasetName='Jackal'
# rootLocaltion='./data/JackalDataset/'
# datasetDetails = {
#     1: [rootLocaltion+'run5_ccw.bag', 2628],
#     2: [rootLocaltion+'run10_ccw.bag', 2459], 
#     3: [rootLocaltion+'s5_ccw_1.bag', 1688,4538], 
#     4: [rootLocaltion+'s6_ccw_1.bag', 1640, 4408], 
#     5: [rootLocaltion+'sim_cw_3.bag', 1058],
#     6: [rootLocaltion+'sim_cw_5.bag', 1498],
#     7: [rootLocaltion+'run4_new.bag', 3690],
#     8: [rootLocaltion+'run3_new.bag', 4074],
# }
# scanDimX, scanDimY, mapRes, horizNumScans, intensityFilter, pointInterval,scanScale =302, 302, 0.05, 20, 0, 1, 1# jackal

def load_from_pickle(pickle_path):
    with open(pickle_path, 'rb') as f:
        file = pickle.load(f)
    return file 

'''CAN params'''
AttractorNetworkParams={
    1: [10, 2, 0.06, 6.5e-05 ],
    2: [8,  2,  6.76660488e-02,  2.39649509e-05],
    3: [1,  3,  2.25719914e-02,  8.26681953e-05],
    4: [1,  4,  2.25719914e-02,  8.26681953e-05],
    6: [4,  5,  8.76770871e-03,  6.18866427e-05],
}

def runOxford(filename, NetworkNum, increment, refNum, queryNum):
    if datasetName == 'OxfordRadar':
        framesRef, framesQuery, framesRefAll= range(0,datasetDetails[refNum][2],increment), range(0,datasetDetails[queryNum][2],increment), range(0,datasetDetails[refNum][2])
    elif datasetName == 'Jackal':
        framesRefAll, framesRef, framesQuery= range(0,datasetDetails[refNum][1]), range(0,datasetDetails[refNum][1],increment), range(0,datasetDetails[queryNum][1],increment)

    # framesRef, framesQuery= range(0,1000,increment), range(0,800,increment)
    '''Initialise LPR positions, Scans and Images'''
    LPR=LiDAR_PlaceRec(datasetName, datasetDetails[refNum][0], datasetDetails[queryNum][0], datasetDetails[refNum][1], datasetDetails[queryNum][1]
                    ,scanDimX, scanDimY, mapRes,horizNumScans, intensityFilter, pointInterval, scanScale, framesRefAll, framesRef, framesQuery)

    xq,yq, yawQ=LPR.scanPoses('QUERY')
    xr,yr,yawR=LPR.scanPoses('REF')
    zerodXR, zerodYR=np.array(xr)-np.min(xr)+20,np.array(yr)-np.min(yr)+20
    zerodXQ, zerodYQ=np.array(xq)-np.min(xr)+20,np.array(yq)-np.min(yr)+20
    centerXYs=LPR.scanCenter()
    refgrid=LPR.makingReferenceGrid()


    '''Initialise Attractor Network'''
    N1, N2 = int(math.ceil(max(zerodYR) / 100.0)) * 100, int(math.ceil(max(zerodXR) / 100.0)) * 100 
    num_links=AttractorNetworkParams[NetworkNum][0]
    excite=AttractorNetworkParams[NetworkNum][1]
    activity_mag=AttractorNetworkParams[NetworkNum][2]
    inhibit_scale=AttractorNetworkParams[NetworkNum][3]
    net=attractorNetwork2D(N1, N2, num_links, excite, activity_mag,inhibit_scale)

    maxY, maxX, convolved= LPR.scanMatchWithConvolution(refgrid, 0, 'SINGLE')
    idx=LPR.extractIDfromConv(maxX, maxY)
    weights=net.excitations(int(zerodYR[idx]), int(zerodXR[idx]))*4
    x,y=zerodXR[idx], zerodYR[idx]



    dataStorage=[]
    timelapse,err=0,0
    for i in range(len(xq)):
        t=time.time()
        if datasetName == 'OxfordRadar':
            idx, matchXDelta, matchYDelta, shiftedX, shiftedY, alignFrac=Oxford_MatchShift(i, LPR, refgrid, centerXYs, xr, yr, yawR)
        elif datasetName == 'Jackal':
            idx, matchXDelta, matchYDelta, shiftedX, shiftedY, alignFrac=Jackal_MatchShift(i, LPR, refgrid, centerXYs, xr, yr, yawR, framesQuery, framesRef, mapRes)

        zerodShiftedX, zerodShiftedY = shiftedX-np.min(xr), shiftedY-np.min(yr)
        
        '''CAN'''
        delY, delX = zerodShiftedY-y, zerodShiftedX-x
        mag=(np.sqrt(delX**2 + delY**2))*2
        ang=np.rad2deg(math.atan2(delY, delX))

        weights,wrap_rows, wrap_cols=net.update_weights_dynamics(weights, ang, mag)
        weights= weights/np.linalg.norm(weights)
        weights+=net.excitations(int(zerodShiftedY), int(zerodShiftedX))*alignFrac*0.1
        weights= (weights/np.linalg.norm(weights))
        x=activityDecoding(weights[np.argmax(np.max(weights, axis=1)), :],5,N2)
        y=activityDecoding(weights[:, np.argmax(np.max(weights, axis=0)) ],5,N1)

        '''Spread'''
        non_zero_rows, non_zero_cols = np.nonzero(weights)
        row_distances = np.abs(non_zero_rows - y)
        col_distances = np.abs(non_zero_cols - x)
        canVarience=np.mean(row_distances)+np.mean(col_distances)

        '''Save excel'''
        testResult= { 'queryPose': (zerodXQ[i], zerodYQ[i], yawQ[i]), 'matchID':idx, 'matchPose':(zerodXR[idx], zerodYR[idx], yawR[idx]), 'shiftAmount': (matchXDelta, matchYDelta),  
                        'matchShiftedPosition':(zerodShiftedX, zerodShiftedY),'scanAlignment': alignFrac, 'canMagnitude': mag, 'canPeak':(x,y), 'canVarience': canVarience }
    
        dataStorage.append(testResult)
        refPos=list(zip(zerodXR, zerodYR))
        pdColumn=pd.Series(refPos, name='referencePositions')
        df1 = pd.DataFrame(dataStorage)
        df1['referencePositions']=pdColumn
        with pd.ExcelWriter(filename) as writer:
            df1.to_excel(writer)

        '''Print info'''
        duration=time.time()-t
        timelapse+=duration
        distCAN=np.sqrt((zerodYQ[i]-y)**2+(zerodXQ[i]-x)**2)
        if datasetName == 'OxfordRadar' and distCAN>20:
            err+=1
        elif datasetName == 'Jackal' and distCAN>1:
            err+=1
        print(f'{i}, duration:{duration}, timelapse: {round(timelapse,2)}, NumErrors: {err}, canDist: {distCAN}')


'''RUN'''
def timeSingleConvolution():
    for i in range(1,5):
        envName='Venman'
        datasetName=f'WildPlaces_{envName}'
        refNum, queryNum=1,2
        blocksize=i
        refRad=2
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

        framesRef= range(0, refFilenames[1], refIncr) if datasetName=='Jackal' else range(0, len(os.listdir(refFilenames[0])), refIncr)
        # evalInfo=load_from_pickle('/media/therese/TOSHIBA EXT/data/WildPlaces/data/Venman/Venman_evaluation_query.pickle')[queryNum-1] 
        evalInfo=load_from_pickle(f'/media/therese/TOSHIBA EXT/data/WildPlaces/data/{envName}/{envName}_evaluation_query.pickle')[queryNum-1] 
        all_files = os.listdir(queryFilenames[0])
        framesQuery=[]
        for k in range(len(evalInfo)):
                try:
                    index = all_files.index(evalInfo[k]['query'].split('/')[-1])
                    framesQuery.append(index)
                except ValueError:
                    pass


        refGridFilePath=refFilenames[0].rsplit('/', 1)[:-1][0] +f'/refGrid_incr{refIncr}_dim{scanDimX}_mapRes{mapRes}_refIncr{refIncr}_refRad{refRad}_rThresh2_nptsMult2_AvgDownsamp.npy'
        LPR=LiDAR_PlaceRec(datasetName, config, refNum, queryNum, framesQuery, refGridFilePath=refGridFilePath,refRad=refRad, blockSize=blocksize ) 

        queryScan = LPR.loadingCroppingFiltering_2DScan('QUERY', 1, rotDeg=10)
        rotIncr=10
        i=1
        # cProfile.run('LPR.update( i, rotIncr, mapRes, xr, yr, yawR, xq, yq, yawQ, centerXs, centerYs, closeIds, returnArray=False )')
        # cProfile.run('LPR.scanMatchWithConvolution(LPR.refgrid, i, rotQuery=queryScan)')
        t= time.time()
        LPR.scanMatchWithConvolution(LPR.refgrid, i, rotQuery=queryScan)
        print(time.time()-t)


def testVariousTopN():
    topNList=[2,5,10,20]
    for NumMatches in topNList:
        count, avgTime, avgShift, avgConv, avgLoad=0,0,0,0,0
        for i in range(1,5):
            for j in range(1,5):
                if i!=j:
                    count+=1
                    envName='Venman'
                    refNum, queryNum = i,j
                    print(i,j)
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
                    # horizNumScans = param.get('horizNumScans', 0)

                    saveFolder=f'./results/LPR_Wildplaces/Ablate_Res:{mapRes}_Dim:{scanDimX}_RInc:{refIncr}_Rrad:{refrad}'
                    if not os.path.exists(saveFolder):
                        os.makedirs(saveFolder)
                    refNpyName=f"/refGrid_incr{refIncr}_dim{scanDimX}_mapRes{mapRes}_refIncr{refIncr}_refRad{refrad}_rThresh2_nptsMult2_AvgDownsamp.npy"
                    savePath = saveFolder+f'/{envName}_R:{refNum}_Q:{queryNum}_RInc:{refIncr}_Rrad:{refrad}_QInc:{queryIncr}_Res:{mapRes}_Dim:{scanDimX}_blkAvg:{blockSize}_N2ndsearch:{NumMatches}.xlsx'
                    # savePath=None 
                    # os.remove(config.get('details', {}).get(str(refNum), [])[0].rsplit('/', 1)[:-1][0] + refNpyName)
                    if os.path.exists(savePath):
                        os.remove(savePath)
                    loadTimeperExp, processTimeperExp, shiftTimeperExp, avgTimeperExp=runWildPlaces(datasetName, refNum, queryNum , savePath, refNpyName, n=NumMatches, returnTimes=True)
                    avgTime+=avgTimeperExp
                    avgShift+=shiftTimeperExp
                    avgConv+=processTimeperExp
                    avgLoad+=loadTimeperExp
                    # print(f'Average Time for {i},{j}: overall: {avgTime/count}, load: {avgLoad/count}, conv: {avgConv/count}, shift: {avgShift/count}')
        print(f'Average Time for venman exps with {NumMatches}- overall: {avgTime/count}, load: {avgLoad/count}, conv: {avgConv/count}, shift: {avgShift/count}')


def testWildplaces():
    count, avgTime, avgShift, avgConv, avgLoad=0,0,0,0,0
    for envName in ['Venman', 'Karawatha']:
        for i in range(1,5):
            for j in range(1,5):
                if i!=j:
                    count+=1
                    # envName='Venman'
                    refNum, queryNum = i,j
                    print(i,j)
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
                    refThresh = param.get('refThresh', 0)
                    numPtsMultiplier = param.get('nMultiplier_randDwnsmpl')
                    NumMatches=param.get('topN')
                    background=param.get('unoccWeight')
                    queryRad= param.get('queryRadius', 0)
                    # horizNumScans = param.get('horizNumScans', 0)

                    saveFolder=f'./results/LPR_Wildplaces/SameHyperParamNewZ'
                    if not os.path.exists(saveFolder):
                        os.makedirs(saveFolder)
                        
                    refNpyName=f"/refGrid_dim{scanDimX}_mapRes{mapRes}_refIncr{refIncr}_refRad{refrad}_rThresh{refThresh}_nptsMult{numPtsMultiplier}_AvgDownsamp.npy"#_negativeBackground
                    savePath = saveFolder+f'/{envName}_R:{refNum}_Q:{queryNum}_RInc:{refIncr}_Rrad:{refrad}_QInc:{queryIncr}_Res:{mapRes}_Dim:{scanDimX}_blkAvg:{blockSize}_N2ndsearch:{NumMatches}_nptsMult{numPtsMultiplier}_background{background}.xlsx'
                    # querySaveFilepath=f'./results/LPR_{datasetName}/FramesFiles/Q:{queryNum}_framesQuery_qInc:{queryIncr}_qRad:{queryRad}_everyN.npy'
                    # refSaveFilepath=f'./results/LPR_{datasetName}/FramesFiles/R:{refNum}_framesRef_rInc:{refIncr}_rRad:{refrad}_Napart.npy'

                    loadTimeperExp, processTimeperExp, shiftTimeperExp, avgTimeperExp, corr, numEval=runWildPlaces(datasetName, refNum, queryNum, savePath, refNpyName, n=NumMatches, errTolerance=3, background=background, returnTimes=True, z_max=6)
        
                    avgTime+=avgTimeperExp
                    avgShift+=shiftTimeperExp
                    avgConv+=processTimeperExp
                    avgLoad+=loadTimeperExp
                    # print(f'Average Time for {i},{j}: overall: {avgTime/count}, load: {avgLoad/count}, conv: {avgConv/count}, shift: {avgShift/count}')
        print(f'Average Time for venman exps- overall: {avgTime/count}, load: {avgLoad/count}, conv: {avgConv/count}, shift: {avgShift/count}')


def testNCLT(refNum=None, queryNum=None, errToler=25):
    datasetName='NCLT'
    with open('./scripts/config.json') as f:
        config = json.load(f)
    config=config.get(datasetName, {})
    refIncr = config.get('refIncr')
    queryIncr = config.get('queryIncr')
    param = config.get('parameters', {})
    scanDimX = param.get('scanDimX', 0)
    scanDimY = param.get('scanDimY', 0)
    mapRes = param.get('mapRes', 0.0)
    refrad= param.get('refRadius', 0)
    blockSize = param.get('blkSize', 0)
    refThresh= param.get('refThresh', 0)
    queryThresh = param.get('queryThresh', 0)
    dim_randDwnsmple= param.get('dim_randDwnsmple', 0)
    nMultiplier_randDwnsmpl = param.get('nMultiplier_randDwnsmpl', 0)
    NumMatches=param.get('topN', 0)
    queryRad= param.get('queryRadius', 0)
    background=-0.15
    saveFolder=f'./results/LPR_{datasetName}/SameHyperParam'
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)
    
    if (refNum!= None) and (queryNum != None):
        refNpyName=f"/refGrid_dim{scanDimX}_mapRes{mapRes}_refIncr{refIncr}_refRad{refrad}_rThresh{refThresh}_patchDim{dim_randDwnsmple}_nptsMult{nMultiplier_randDwnsmpl}.npy"
        savePath = saveFolder+f'/{datasetName}_R:{refNum}_Q:{queryNum}_RInc:{refIncr}_Rrad:{refrad}_QInc:{queryIncr}_Qrad:{queryRad}_Res:{mapRes}_Dim:{scanDimX}_blkAvg:{blockSize}_N2ndsearch:{NumMatches}_background:{background}_errToler:{errToler}.xlsx'  #__queryClose2Ref #_everyNref_queryClose2Ref
        querySaveFilepath=f'./results/LPR_{datasetName}/FramesFiles/Q:{queryNum}_framesQuery_qInc:{queryIncr}_qRad:{queryRad}_everyN.npy'
        refSaveFilepath=f'./results/LPR_{datasetName}/FramesFiles/R:{refNum}_framesRef_rInc:{refIncr}_rRad:{refrad}_Napart.npy'

        

        saveRefQuerySplit(datasetName, refNum, queryNum, querySaveFilepath, refSaveFilepath)           
        runWildPlaces(datasetName, refNum, queryNum, savePath, refNpyName, refSaveFilepath=refSaveFilepath, querySaveFilepath=querySaveFilepath, n=NumMatches, errTolerance=errToler, background=background)

    else:
        for i in range(1,10):
            for j in [10,11]:
                refNum, queryNum=j,i
                refNpyName=f"/refGrid_dim{scanDimX}_mapRes{mapRes}_refIncr{refIncr}_refRad{refrad}_rThresh{refThresh}_patchDim{dim_randDwnsmple}_nptsMult{nMultiplier_randDwnsmpl}.npy"
                savePath = saveFolder+f'/{datasetName}_R:{refNum}_Q:{queryNum}_RInc:{refIncr}_Rrad:{refrad}_QInc:{queryIncr}_Qrad:{queryRad}_Res:{mapRes}_Dim:{scanDimX}_blkAvg:{blockSize}_N2ndsearch:{NumMatches}_background:{background}.xlsx'
                querySaveFilepath=f'./results/LPR_{datasetName}/FramesFiles/Q:{queryNum}_framesQuery_qInc:{queryIncr}_qRad:{queryRad}_everyN.npy'
                refSaveFilepath=f'./results/LPR_{datasetName}/FramesFiles/R:{refNum}_framesRef_rInc:{refIncr}_rRad:{refrad}_Napart.npy'

                saveRefQuerySplit(datasetName, refNum, queryNum, querySaveFilepath, refSaveFilepath)
                runWildPlaces(datasetName, refNum, queryNum, savePath, refNpyName, refSaveFilepath=refSaveFilepath, querySaveFilepath=querySaveFilepath, n=NumMatches, errTolerance=errToler, background=background)
        
    
def testOxfordRadar(refNum=None, queryNum=None):
    datasetName='OxfordRadar'
    folderName='LPR_OxfordRadar'
     
    with open('./scripts/config.json') as f:
        config = json.load(f)
    config=config.get(datasetName, {})
    refIncr = config.get('refIncr')
    queryIncr = config.get('queryIncr')
    param = config.get('parameters', {})
    scanDimX = param.get('scanDimX', 0)
    scanDimY = param.get('scanDimY', 0)
    mapRes = param.get('mapRes', 0.0)
    refrad= param.get('refRadius', 0)
    blockSize = param.get('blkSize', 0)
    refThresh= param.get('refThresh', 0)
    queryThresh = param.get('queryThresh', 0)
    dim_randDwnsmple= param.get('dim_randDwnsmple', 0)
    nMultiplier_randDwnsmpl = param.get('nMultiplier_randDwnsmpl', 0)
    NumMatches=param.get('topN', 0)
    queryRadius= param.get('queryRadius', 0)
    background=-0.15

    if (refNum== None) and (queryNum== None):
        for i in range(1,6):
            for j in [6,7]:
                print(f'ref{j}, query{i}')
                refNum, queryNum=j,i

                saveFolder=f'./results/{folderName}/SameHyperParam'
                if not os.path.exists(saveFolder):
                    os.makedirs(saveFolder)
                refNpyName=f"/refGrid_dim{scanDimX}_mapRes{mapRes}_refIncr{refIncr}_refRad{refrad}_rThresh{refThresh}_patchDim{dim_randDwnsmple}_nptsMult{nMultiplier_randDwnsmpl}.npy"
                savePath = saveFolder+f'/{datasetName}_R:{refNum}_Q:{queryNum}_RInc:{refIncr}_Rrad:{refrad}_QInc:{queryIncr}_Qrad:{queryRadius}_Res:{mapRes}_Dim:{scanDimX}_blkAvg:{blockSize}_N2ndsearch:{NumMatches}_background:{background}.xlsx'

                querySaveFilepath=f'./results/LPR_{datasetName}/FramesFiles/Q:{queryNum}_framesQuery_qInc:{queryIncr}_qRad:{queryRadius}_everyN.npy'
                refSaveFilepath=f'./results/LPR_{datasetName}/FramesFiles/R:{refNum}_framesRef_rInc:{refIncr}_rRad:{refrad}_Napart.npy'


                saveRefQuerySplit(datasetName, refNum, queryNum, querySaveFilepath, refSaveFilepath)  
                runWildPlaces(datasetName, refNum, queryNum, savePath, refNpyName, querySaveFilepath, refSaveFilepath, n=NumMatches, errTolerance=25)


    else: 
        saveFolder=f'./results/{folderName}/Ablate_Res:{mapRes}_Dim:{scanDimX}_RInc:{refIncr}_Rrad:{refrad}'
        if not os.path.exists(saveFolder):
            os.makedirs(saveFolder)
        refNpyName=f"/refGrid_dim{scanDimX}_mapRes{mapRes}_refIncr{refIncr}_refRad{refrad}_rThresh{refThresh}_patchDim{dim_randDwnsmple}_nptsMult{nMultiplier_randDwnsmpl}.npy"
        savePath = saveFolder+f'/{datasetName}_R:{refNum}_Q:{queryNum}_RInc:{refIncr}_Rrad:{refrad}_QInc:{queryIncr}_Qrad:{queryRadius}_Res:{mapRes}_Dim:{scanDimX}_blkAvg:{blockSize}_N2ndsearch:{NumMatches}.xlsx'

        querySaveFilepath=f'./results/LPR_{datasetName}/FramesFiles/Q:{queryNum}_framesQuery_qInc:{queryIncr}_qRad:{queryRadius}_everyN.npy'
        refSaveFilepath=f'./results/LPR_{datasetName}/FramesFiles/R:{refNum}_framesRef_rInc:{refIncr}_rRad:{refrad}_Napart.npy'


        saveRefQuerySplit(datasetName, refNum, queryNum, querySaveFilepath, refSaveFilepath)  
        runWildPlaces(datasetName, refNum, queryNum, savePath, refNpyName, querySaveFilepath, refSaveFilepath, n=NumMatches, errTolerance=25)


def testVariousRotDeg():
    rotDegs=[1,5,10,20,30]
    for rot in rotDegs:
        count, avgTime, avgShift, avgConv, avgLoad=0,0,0,0,0
        for i in range(1,5):
            for j in range(1,5):
                if i!=j:
                    count+=1
                    envName='Venman'
                    refNum, queryNum = i,j
                    print(i,j)
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
                    NumMatches=1
                    # horizNumScans = param.get('horizNumScans', 0)

                    saveFolder=f'./results/LPR_Wildplaces/Ablate_VariousRot_Res:{mapRes}_Dim:{scanDimX}_RInc:{refIncr}_Rrad:{refrad}'
                    if not os.path.exists(saveFolder):
                        os.makedirs(saveFolder)
                    refNpyName=f"/refGrid_incr{refIncr}_dim{scanDimX}_mapRes{mapRes}_refIncr{refIncr}_refRad{refrad}_rThresh2_nptsMult2_AvgDownsamp.npy"
                    savePath = saveFolder+f'/{envName}_R:{refNum}_Q:{queryNum}_RInc:{refIncr}_Rrad:{refrad}_QInc:{queryIncr}_Res:{mapRes}_Dim:{scanDimX}_blkAvg:{blockSize}_N2ndsearch:{NumMatches}_rotDeg{rot}.xlsx'
                    # os.remove(config.get('details', {}).get(str(refNum), [])[0].rsplit('/', 1)[:-1][0] + refNpyName)
                    # if os.path.exists(savePath):
                    #     os.remove(savePath)
                    savePath=None 
                    loadTimeperExp, processTimeperExp, shiftTimeperExp, avgTimeperExp=runWildPlaces(datasetName, refNum, queryNum , savePath, refNpyName, n=NumMatches, rotInc=rot, returnTimes=True)
                    avgTime+=avgTimeperExp
                    avgShift+=shiftTimeperExp
                    avgConv+=processTimeperExp
                    avgLoad+=loadTimeperExp
                    # print(f'Average Time for {i},{j}: overall: {avgTime/count}, load: {avgLoad/count}, conv: {avgConv/count}, shift: {avgShift/count}')
        print(f'Average Time for venman exps with {rot}- overall: {avgTime/count}, load: {avgLoad/count}, conv: {avgConv/count}, shift: {avgShift/count}')


def testNegativeBackground():

    backgroundVals=[-0.25]
    for val in backgroundVals:
        count, avgTime, avgShift, avgConv, avgLoad, corrs, numEvals=0,0,0,0,0,0,0
        for i in range(1,5):
            for j in range(1,5):
                if i!=j:
                    count+=1
                    envName='Venman'
                    refNum, queryNum = i,j
                    print(i,j)
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
                    numPtsMultiplier = param.get('nMultiplier_randDwnsmpl', 0)
                    NumMatches=1
                    # horizNumScans = param.get('horizNumScans', 0)

                    saveFolder=f'./results/LPR_Wildplaces/Ablate_NegativeBackground_Res:{mapRes}_Dim:{scanDimX}_RInc:{refIncr}_Rrad:{refrad}'
                    if not os.path.exists(saveFolder):
                        os.makedirs(saveFolder)
                        
                    refNpyName=f"/refGrid_dim{scanDimX}_mapRes{mapRes}_refIncr{refIncr}_refRad{refrad}_rThresh2_nptsMult{numPtsMultiplier}_AvgDownsamp.npy"
                    savePath = saveFolder+f'/{envName}_R:{refNum}_Q:{queryNum}_RInc:{refIncr}_Rrad:{refrad}_QInc:{queryIncr}_Res:{mapRes}_Dim:{scanDimX}_blkAvg:{blockSize}_N2ndsearch:{NumMatches}_background{val}_nptsMult{numPtsMultiplier}.xlsx'
                    # os.remove(config.get('details', {}).get(str(refNum), [])[0].rsplit('/', 1)[:-1][0] + refNpyName)
                    # if os.path.exists(savePath):
                    #     os.remove(savePath)
                    # savePath=None 
                    loadTimeperExp, processTimeperExp, shiftTimeperExp, avgTimeperExp, corrOut, numEvalOut=runWildPlaces(datasetName, refNum, queryNum , savePath, refNpyName, n=NumMatches, background=val, returnTimes=True)
                    avgTime+=avgTimeperExp
                    avgShift+=shiftTimeperExp
                    avgConv+=processTimeperExp
                    avgLoad+=loadTimeperExp
                    corrs+=corrOut
                    numEvals+=numEvalOut
                    print(corrs, numEvals)
                    # print(f'Average Time for {i},{j}: overall: {avgTime/count}, load: {avgLoad/count}, conv: {avgConv/count}, shift: {avgShift/count}')
        print(f'Average Time for venman exps with {val}- recall: {corrs/numEvals}, overall: {avgTime/count}, load: {avgLoad/count}, conv: {avgConv/count}, shift: {avgShift/count}')


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
    
    
    
def ablatePatchDownSample(refNum, queryNum):
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
    # NumMatches=param.get('topN')
    saveFolder=f'./results/LPR_Wildplaces/Ablate_TrainingSet'
    '''Varying max occupied cells per patch'''
    numPtsMultipliers=[0.1,0.5,0.75,1.5,2,4,6,8,10]
    NumMatch, rotInc, background=2, 10, -0.15
    for numPtsMultiplier in numPtsMultipliers:
        if not os.path.exists(saveFolder):
            os.makedirs(saveFolder)
        refNpyName=f"/refGrid_dim{scanDimX}_mapRes{mapRes}_refIncr{refIncr}_refRad{refrad}_rThresh2_nptsMult{numPtsMultiplier}_AvgDownsamp.npy"
        savePath = saveFolder+(f'/{envName}_R:{refNum}_Q:{queryNum}_RInc:{refIncr}_Rrad:{refrad}_QInc:'
        f'{queryIncr}_Res:{mapRes}_Dim:{scanDimX}_blkAvg:{blockSize}_N2ndsearch:{NumMatch}_rotInc{rotInc}_UnoccWeight{background}_nptsMult{numPtsMultiplier}.xlsx')           

        runWildPlaces(datasetName, refNum, queryNum, savePath, refNpyName, querySaveFilepath=querySaveFilepath, 
                      ablate=True, errTolerance=3, n=NumMatch, blockSize=blockSize, numPtsMultiplier=numPtsMultiplier)
    

    # '''ordered versus random downsampling'''
    # refNpyName=f"/refGrid_dim{scanDimX}_mapRes{mapRes}_refIncr{refIncr}_refRad{refrad}_rThresh2_nptsMult{numPtsMultiplier}_orderedPatch.npy"
    # savePath = saveFolder+(f'/{envName}_R:{refNum}_Q:{queryNum}_RInc:{refIncr}_Rrad:{refrad}_QInc:'
    # f'{queryIncr}_Res:{mapRes}_Dim:{scanDimX}_orderedPatchDwnSample1.xlsx')           

    # runWildPlaces(datasetName, refNum, queryNum, savePath, refNpyName, querySaveFilepath=querySaveFilepath, 
    #                 ablate=True, errTolerance=3, numPtsMultiplier=2)

        
def analysingPtcldDensity(datasetName, refNum, queryNum, ablate=True, errTolerance=3, fileNotSaved=False, Incr=None, fixedVoxelSzie=True, ablateHDMThresh=None, QUERYorREF='QUERY'):
    if fileNotSaved==True:
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
        blockSize=param.get('blkSize', 0)
        scanDimX = param.get('scanDimX', 0)
        scanDimY = param.get('scanDimY', 0)
        mapRes = param.get('mapRes', 0.0)
        refrad=param.get('refRadius', 0)
        numPtsMultiplier = param.get('nMultiplier_randDwnsmpl')
        refNpyName=f"/refGrid_dim{scanDimX}_mapRes{mapRes}_refIncr{refIncr}_refRad{refrad}_rThresh2_nptsMult{numPtsMultiplier}_AvgDownsamp.npy"
        print(refFilenames)
        refGridFilePath=refFilenames[0].rsplit('/', 1)[:-1][0] + refNpyName 
        
        

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
            framesRef=None
        
        elif datasetName[:10]=='WildPlaces' and ablate==True:
            querySaveFilepath=f'./results/LPR_Wildplaces/FramesFiles/R:{refNum}_Q:{queryNum}_framesQueryAblation.npy'
            framesQuery=list(np.load(querySaveFilepath))
            framesRef=None 

        elif datasetName == 'NCLT'or datasetName == 'OxfordRadar':
            querySaveFilepath=f'./results/LPR_{datasetName}/FramesFiles/Q:{queryNum}_framesQuery_qInc:{queryIncr}_qRad:{queryRad}_everyN.npy'
            framesQuery=list(np.load(querySaveFilepath))
            refSaveFilepath=f'./results/LPR_{datasetName}/FramesFiles/R:{refNum}_framesRef_rInc:{refIncr}_rRad:{refrad}_Napart.npy'
            framesRef=list(np.load(refSaveFilepath))
        
        
            

        LPR=LiDAR_PlaceRec(datasetName, config, refNum, queryNum, framesQuery, framesRef=framesRef, refGridFilePath=None ) 
        # xq,yq, rollQ, pitchQ, yawQ=LPR.scanPoses('QUERY')
        # xr,yr,rollR, pitchR, yawR=LPR.scanPoses('REF')
        # minDist=[LPR.find_closest_point(zip(xr,yr), xq[i], yq[i])[3] for i in range(len(xq))]  

        # rawPtcldDensityQuery,descriptorDensityQuery =[],[]
        # for i in range(0,len(LPR.framesQuery), queryIncr):
        #     if minDist[i]<errTolerance:
        #         ptcld=LPR.loading2DScan( 'QUERY',i)
        #         rawPtcldDensityQuery.append(len(ptcld))
        #         topDown=LPR.processing2DScan('QUERY',ptcld)
        #         descriptorDensityQuery.append(np.count_nonzero(topDown > 0))
        # np.save('./results/LPR_Wildplaces/Ablate_TrainingSet/ptcldDenisty/descriptorDensityQuery.npy',descriptorDensityQuery )   
        # np.save('./results/LPR_Wildplaces/Ablate_TrainingSet/ptcldDenisty/rawPtcldDensityQuery.npy', rawPtcldDensityQuery)   

        
        rawPtcldDensityRef,descriptorDensityRef=[],[]
        print(LPR.numRefScans())
        print(len(framesQuery))
        max_z=0
        Incr= 10 if Incr == None else Incr
        if QUERYorREF == 'QUERY':
            endRange= len(framesQuery)
        elif QUERYorREF == 'REF':
            endRange= LPR.numRefScans()
        for j in range(0,endRange,Incr):
            ptcld=LPR.loading2DScan( QUERYorREF,j)
            # print(ptcld[:, 2])
            max_z= np.max(abs(ptcld[:, 2])) if np.max(abs(ptcld[:, 2]))>max_z else max_z
            # print(j, '/', LPR.numRefScans(), len(ptcld), 'max_z:', max_z)
            rawPtcldDensityRef.append(len(np.unique(ptcld[:, :2], axis=0)))
            if fixedVoxelSzie == True: 
                LPR.mapRes=1
            if ablateHDMThresh!= None:
                LPR.HDMThresh = ablateHDMThresh
            topDown=LPR.processing2DScan(QUERYorREF,ptcld)
            descriptorDensityRef.append(np.count_nonzero(topDown > 0))

            # print(j,'/',LPR.numRefScans(), "    mapres", LPR.mapRes, 'raw:', len(ptcld), 'descriptor:', np.count_nonzero(topDown > 0))

        np.save(f'./results/ptcldDensity/{datasetName}_{queryNum}_descriptorDensity_{QUERYorREF}_Incr{Incr}_HDMThresh_{ablateHDMThresh}.npy', descriptorDensityRef)   
        # np.save(f'./results/ptcldDensity/{datasetName}_{queryNum}_uniqueXY.npy', rawPtcldDensityRef)   
        print( 'dataset', datasetName,'thresh', ablateHDMThresh, 'avg descriptior', np.mean(descriptorDensityRef), LPR.mapRes, LPR.HDMThresh)

    

# testVariousRotDeg()
# testOxfordRadar(refNum=6, queryNum=1)
# testOxfordRadar()

# testNCLT(refNum=2,queryNum=3, errToler=10)
# testNCLT(refNum=2,queryNum=6, errToler=10)
# testNCLT(refNum=10, queryNum=1)
# testNCLT()

# testNegativeBackground()
# testWildplaces()
ablateWithWildPlaces(2, 3)
# ablatePatchDownSample(2, 3)

# for thresh in [1,2,3]:
#     analysingPtcldDensity("OxfordRadar", 6,3,ablate=True, fileNotSaved=True, fixedVoxelSzie=False, ablateHDMThresh=thresh, QUERYorREF='REF')
#     analysingPtcldDensity("NCLT", 10,3,ablate=True, fileNotSaved=True, fixedVoxelSzie=False, ablateHDMThresh=thresh, QUERYorREF='REF')
#     analysingPtcldDensity("WildPlaces_Venman", 2,3,ablate=False, fixedVoxelSzie=False, fileNotSaved=True, Incr=20, ablateHDMThresh=thresh, QUERYorREF='REF')

def averageDesitiesAllDataset():
    # for q in range(1,5): 
    #     r=1 if q ==2 else 2   
    #     analysingPtcldDensity("WildPlaces_Karawatha", r, q,ablate=False, fileNotSaved=True,Incr=50)
    #     analysingPtcldDensity("WildPlaces_Venman", r, q,ablate=False, fileNotSaved=True, Incr=50)
    # for r in range(1,10):
    #     q=11 #if r ==2 else 2       
    #     analysingPtcldDensity("NCLT", q, r,ablate=True, fileNotSaved=True)
    # for r in range(1,6): 
    #     q=7 #if r ==2 else 2  
    #     analysingPtcldDensity("OxfordRadar", q, r,ablate=True, fileNotSaved=True)

    avgDensities=[]
    for q in range(1, 5):
        densities_karawatha = np.load(f'./results/ptcldDensity/WildPlaces_Karawatha_{q}_descriptorDensityRef.npy')
        densities_venman = np.load(f'./results/ptcldDensity/WildPlaces_Venman_{q}_descriptorDensityRef.npy')
        avgDensities.extend(densities_karawatha)
        avgDensities.extend(densities_venman)

    # Calculate and print average density
    average_density = np.mean(avgDensities)
    print("Average Density WildPlaces:", average_density)

    avgDensities = []
    # Loop through files in the NCLT dataset
    for r in range(1, 10):
        density_values = np.load(f'./results/ptcldDensity/NCLT_{r}_descriptorDensityRef.npy')
        avgDensities.extend(density_values)  # Extend list with loaded values

    # Calculate and print the average for the NCLT dataset
    nclt_avg_density = np.mean(avgDensities)
    print("Average Density for NCLT dataset:", nclt_avg_density)

    # Repeat the same process for the OxfordRadar dataset
    avgDensities = []
    for r in range(1, 6):
        density_values = np.load(f'./results/ptcldDensity/OxfordRadar_{r}_descriptorDensityRef.npy')
        avgDensities.extend(density_values)

    # Calculate and print the average for the OxfordRadar dataset
    oxford_avg_density = np.mean(avgDensities)
    print("Average Density for OxfordRadar dataset:", oxford_avg_density)


def ablateWithUrban(datasetName, refNum, queryNum):
    with open('./scripts/config.json') as f:
        config = json.load(f)
    config=config.get(datasetName, {})
    param = config.get('parameters', {})
    refIncr = config.get('refIncr')
    queryIncr = config.get('queryIncr')
    refrad= param.get('refRadius', 0)
    queryRad= param.get('queryRadius', 0)
    scanDimX = param.get('scanDimX', 0)
    

    saveFolder=f'./results/LPR_{datasetName}/Ablate_TrainingSet_TimeIncl'
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)

    querySaveFilepath=f'./results/LPR_{datasetName}/FramesFiles/Q:{queryNum}_framesQuery_qInc:{queryIncr}_qRad:{queryRad}_everyN.npy'
    queryAblateSaveFilepath=f'./results/LPR_{datasetName}/FramesFiles/Q:{queryNum}_framesQuery_qInc:{queryIncr}_qRad:{queryRad}_everyN.npy'
    refSaveFilepath=f'./results/LPR_{datasetName}/FramesFiles/R:{refNum}_framesRef_rInc:{refIncr}_rRad:{refrad}_Napart.npy'
    framesQuery=ablateRefQuerySplitUrban(querySaveFilepath, queryAblateSaveFilepath)

    mapReses=[0.3,0.5,0.75,1,1.3,1.5]
    HDM_thresh=param.get('HDMThresh')
    for mapRes in mapReses:
        refNpyName=f"/refGrid_dim{scanDimX}_mapRes{mapRes}_refIncr{refIncr}_refRad{refrad}_rThresh2.npy"
        savePath = saveFolder+(f'/{datasetName}_R:{refNum}_Q:{queryNum}_RInc:{refIncr}_Rrad:{refrad}_QInc:'
        f'{queryIncr}_Res:{mapRes}_Dim:{scanDimX}_map_res{mapRes}_HDMthresh{HDM_thresh}.xlsx')  
        runWildPlaces(datasetName, refNum, queryNum, savePath, refNpyName, refSaveFilepath=refSaveFilepath, querySaveFilepath=queryAblateSaveFilepath, errTolerance=25, mapRes=mapRes)

    HDM_threshes=[2,3,4]
    mapRes=param.get('mapRes')
    for HDMthresh in HDM_threshes:
        refNpyName=f"/refGrid_dim{scanDimX}_mapRes{mapRes}_refIncr{refIncr}_refRad{refrad}_HDMthresh{HDMthresh}.npy"
        savePath = saveFolder+(f'/{datasetName}_R:{refNum}_Q:{queryNum}_RInc:{refIncr}_Rrad:{refrad}_QInc:'
        f'{queryIncr}_Res:{mapRes}_Dim:{scanDimX}_map_res{mapRes}_HDMthresh{HDMthresh}.xlsx')  
        runWildPlaces(datasetName, refNum, queryNum, savePath, refNpyName, refSaveFilepath=refSaveFilepath,querySaveFilepath=queryAblateSaveFilepath, errTolerance=25, HDMThresh=HDMthresh)

def averageDesitiesAllDataset():
    # for q in range(1,5): 
    #     r=1 if q ==2 else 2   
    #     analysingPtcldDensity("WildPlaces_Karawatha", r, q,ablate=False, fileNotSaved=True,Incr=50)
    #     analysingPtcldDensity("WildPlaces_Venman", r, q,ablate=False, fileNotSaved=True, Incr=50)
    # for r in range(1,10):
    #     q=11 #if r ==2 else 2       
    #     analysingPtcldDensity("NCLT", q, r,ablate=True, fileNotSaved=True)
    # for r in range(1,6): 
    #     q=7 #if r ==2 else 2  
    #     analysingPtcldDensity("OxfordRadar", q, r,ablate=True, fileNotSaved=True)

    avgDensities=[]
    for q in range(1, 5):
        densities_karawatha = np.load(f'./results/ptcldDensity/WildPlaces_Karawatha_{q}_descriptorDensityRef.npy')
        densities_venman = np.load(f'./results/ptcldDensity/WildPlaces_Venman_{q}_descriptorDensityRef.npy')
        avgDensities.extend(densities_karawatha)
        avgDensities.extend(densities_venman)

    # Calculate and print average density
    average_density = np.mean(avgDensities)
    print("Average Density WildPlaces:", average_density)

    avgDensities = []
    # Loop through files in the NCLT dataset
    for r in range(1, 10):
        density_values = np.load(f'./results/ptcldDensity/NCLT_{r}_descriptorDensityRef.npy')
        avgDensities.extend(density_values)  # Extend list with loaded values

    # Calculate and print the average for the NCLT dataset
    nclt_avg_density = np.mean(avgDensities)
    print("Average Density for NCLT dataset:", nclt_avg_density)

    # Repeat the same process for the OxfordRadar dataset
    avgDensities = []
    for r in range(1, 6):
        density_values = np.load(f'./results/ptcldDensity/OxfordRadar_{r}_descriptorDensityRef.npy')
        avgDensities.extend(density_values)

    # Calculate and print the average for the OxfordRadar dataset
    oxford_avg_density = np.mean(avgDensities)
    print("Average Density for OxfordRadar dataset:", oxford_avg_density)



# refNum, queryNum = 2,3
# envName='Karawatha'
# avgTimeperExp=runWildPlaces(datasetName, refNum, queryNum , savePath)

# ablateWithUrban('NCLT', 11, 1)
# ablateWithUrban('OxfordRadar', 7, 3)
# averageDesitiesAllDataset()
        
assert False

for i in range(1,5):
    if i!=1:
        refNum, queryNum = 1,i
        runWildPlaces("WildPlaces_Venman", refNum, queryNum ,  f'./results/LPR_PosEstimation/Venman_Ref:{refNum}_Query:{queryNum}_Inc:1_Res:{mapRes}_downnsample.xlsx')

for i in range(1,5):
    if i!=1:
        refNum, queryNum = 1,i
        runWildPlaces("WildPlaces_Karawatha", refNum, queryNum ,  f'./results/LPR_PosEstimation/Karawatha_Ref:{refNum}_Query:{queryNum}_Inc:1_downnsample.xlsx')

# refNum, queryNum = 2,3
# runWildPlaces("WildPlaces_Venman", refNum, queryNum ,  f'./results/LPR_PosEstimation/Ref:{refNum}_Query:{queryNum}.xlsx')

# refNum, queryNum = 2,3
# runWildPlaces("WildPlaces_Karawatha", refNum, queryNum ,  f'./results/LPR_PosEstimation/Karawatha_Ref:{refNum}_Query:{queryNum}_noNorm.xlsx')

# refNum, queryNum = 2,3
# runWildPlaces("WildPlaces_Karawatha", refNum, queryNum )

# envName='Karawatha'
# refNum, queryNum = 3,1
# runWildPlaces(f"WildPlaces_{envName}", refNum, queryNum, envName )

# refNum, queryNum = 4,1
# runWildPlaces(f"WildPlaces_{envName}", refNum, queryNum, envName )
