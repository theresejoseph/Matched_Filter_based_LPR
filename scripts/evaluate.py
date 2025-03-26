import pandas as pd
import matplotlib.patches as patches
from scipy.spatial.distance import cdist
from matplotlib.gridspec import GridSpec
import math
from helper import LiDAR_PlaceRec
import numpy as np
import time 
import matplotlib.pyplot as plt 
import ast
from scipy.signal import find_peaks
from matplotlib.ticker import PercentFormatter
import os 
import json 
import pickle
import matplotlib.animation as animation
from matplotlib.lines import Line2D

def extractSingleReuslt(results_path, errTolerance):
    df = pd.read_excel(results_path)
    queryPositions, matchIds, matchPositions = df['queryPose'], df['matchID'], df['matchPose'],
    matchShiftedPositions= df['matchShiftedPosition']
    closePositions=df['closePose']
    dists=np.array(df['dist'])
    matchIds=np.array(df['matchID'])
    closestIds=np.array(df['closestId'])
    yawShifts=np.array(df['peakRotation'])
    closeDists=np.array(df['closeDist'])

    matchX, matchY, matchYaw=zip(*matchPositions.apply(lambda x: (ast.literal_eval(x)[0], ast.literal_eval(x)[1],  ast.literal_eval(x)[2])))
    queryX, queryY, queryYaw= zip(*queryPositions.apply(lambda x: (ast.literal_eval(x)[0], ast.literal_eval(x)[1], ast.literal_eval(x)[2])))
    lprShiftX, lprShiftY=zip(*matchShiftedPositions.apply(lambda x: (ast.literal_eval(x)[0], ast.literal_eval(x)[1])))
    closeX, closeY=zip(*closePositions.apply(lambda x: (ast.literal_eval(x)[0], ast.literal_eval(x)[1])))
    
    lprShiftError=np.sqrt(   (np.array(lprShiftX)-np.array(queryX))**2   +   (np.array(lprShiftY)-np.array(queryY))**2   )
    lprMatchError=np.sqrt(   (np.array(matchX)-np.array(queryX))**2   +   (np.array(matchY)-np.array(queryY))**2   )
    refHasCloseMatch=np.where(closeDists<errTolerance)[0]

    # lprShiftError=lprShiftError[refHasCloseMatch]
    lidarLowShiftErrorIds=np.where(lprShiftError<errTolerance)[0]
    lidarLowMatchErrorIds=np.where(lprMatchError<errTolerance)[0]
    lidarHighShiftErrorIds=np.where(lprShiftError>errTolerance)[0]
    correctMathces=np.where(closeDists<errTolerance)[0]
    correctShiftMatches=np.where(dists<errTolerance)[0]

    recall=len(correctShiftMatches)/len(dists)
    # print(f'recall: {recall}')
    # print(lidarHighShiftErrorIds)
    output={
        'matchX':np.array(matchX),
        'matchY':np.array(matchY),
        'matchYaw':np.rad2deg(np.array(matchYaw)),
        'queryX':np.array(queryX), 
        'queryY':np.array(queryY), 
        'queryYaw':np.rad2deg(np.array(queryYaw)), 
        'shiftedX': np.array(lprShiftX),
        'shiftedY': np.array(lprShiftY),
        'yawShift':yawShifts,
        'closeX':np.array(closeX), 
        'closeY':np.array(closeY), 
        'recall': recall,
        'dists': dists,
        'correctShiftedIds':lidarLowShiftErrorIds,
        'correctMatchIds':lidarLowMatchErrorIds,
        'matchIds': matchIds,
        'closestIds': closestIds,

    }
    return output


def plotRealtiveShift(numRefTraverse=4, numqueryTraverse=4, datasetName='Wildplaces',mapRes=0.3, scanDimX=120, refIncr=1, queryIncr=1, refRad=2, blockSize=2, NumMatches=2, numPtsMultiplier=2, background=-0.15):
    shiftErrorsAll, matchErrorsAll = [], []
    for envName in ['Venman','Karawatha']:
        for i in range(1,numRefTraverse+1):
            for j in range(1,numqueryTraverse+1):
                if i!=j:
                    # try: 
                    results_dir=f'./results/LPR_Wildplaces/EvaluateTestSet'
                    results_filename=f'/{envName}_R:{i}_Q:{j}_RInc:{refIncr}_Rrad:{refRad}_QInc:{queryIncr}_Res:{mapRes}_Dim:{scanDimX}_blkAvg:{blockSize}_N2ndsearch:{NumMatches}_nptsMult{numPtsMultiplier}_background{background}.xlsx'
                    output= extractSingleReuslt(results_dir+results_filename, errTolerance=3)
                    recall=output['recall']
                    shiftErrorsAll+=list(np.sqrt((output['shiftedX']-output['queryX'])**2   +   (output['shiftedY']-output['queryY'])**2)[output['correctShiftedIds']])
                    matchErrorsAll+=list(np.sqrt((output['matchX']-output['queryX'])**2   +   (output['matchY']-output['queryY'])**2)[output['correctShiftedIds']])
                
    print(len(matchErrorsAll), len(shiftErrorsAll))
    kwargs = dict(histtype='step', alpha=1, bins=60, density=True, linewidth=1.5)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(matchErrorsAll, **kwargs, ec='blue', label='Matches before relative shift')
    ax.hist(shiftErrorsAll, **kwargs, ec='magenta', label='Matches after relative shift')
    ax.set_xlabel('Position Error [m]', fontsize=16)
    ax.set_ylabel('Density of Distribution', fontsize=16)
    handles, labels = ax.get_legend_handles_labels()
    new_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles]
    ax.legend(new_handles, labels, fontsize=14)
    # ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    plt.savefig('./paperFigures/DistributionOfErrorsRelativeShift.png')
    plt.savefig('./paperFigures/DistributionOfErrorsRelativeShift.pdf')


def recallOverDistances(numRefTraverse=4, numqueryTraverse=4, datasetName='Wildplaces', mapRes=0.3, scanDimX=120, refIncr=1, queryIncr=1, refRad=2, blockSize=2,  NumMatches=1):
    recallListV, recallListK = [], []
    distances=[0.1 ,0.5, 0.75, 1, 2, 3, 5,10]

    envName='Venman' 
    for d in distances:
        count, currRecall=0, 0
        for i in range(1,numRefTraverse+1):
            for j in range(1,numqueryTraverse+1):
                if i!=j:
                    results_dir=f'./results/LPR_{datasetName}/Ablate_Res:{mapRes}_Dim:{scanDimX}_RInc:{refIncr}_Rrad:{refRad}'
                    results_filename=f'/{envName}_R:{i}_Q:{j}_RInc:{refIncr}_Rrad:{refRad}_QInc:{queryIncr}_Res:{mapRes}_Dim:{scanDimX}_blkAvg:{blockSize}.xlsx'
                    refGridFilename=f"/refGrid_incr{refIncr}_dim{scanDimX}_mapRes{mapRes}_refIncr{refIncr}_refRad2_rThresh2_nptsMult2_AvgDownsamp.npy"
                    output= extractSingleReuslt(f'{envName}', i,j,results_dir+results_filename, refGridFilename, errTolerance=d)
                    currRecall+=output['recall']
                    count+=1
        recallListV.append((currRecall/count)*100)
    
    envName='Karawatha'                
    for d in distances:
        count, currRecall=0, 0
        for i in range(1,numRefTraverse+1):
            for j in range(1,numqueryTraverse+1):
                if i!=j:
                    results_dir=f'./results/LPR_{datasetName}/Ablate_Res:{mapRes}_Dim:{scanDimX}_RInc:{refIncr}_Rrad:{refRad}'
                    results_filename=f'/{envName}_R:{i}_Q:{j}_RInc:{refIncr}_Rrad:{refRad}_QInc:{queryIncr}_Res:{mapRes}_Dim:{scanDimX}_blkAvg:{blockSize}.xlsx'
                    refGridFilename=f"/refGrid_incr{refIncr}_dim{scanDimX}_mapRes{mapRes}_refIncr{refIncr}_refRad2_rThresh2_nptsMult2_AvgDownsamp.npy"
                    output= extractSingleReuslt(f'{envName}', i,j,results_dir+results_filename, refGridFilename, errTolerance=d)
                    currRecall+=output['recall']
                    count+=1
        recallListK.append((currRecall/count)*100)
                    
    kwargs = dict(histtype='step', alpha=1, bins=60, density=True, linewidth=1.5)
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10, 6), dpi=300)
    ax1.plot(distances, recallListV, '.-', color='blue')
    ax1.set_xlabel('Distance Threshold [meters]', fontsize=14)
    ax1.set_ylabel('Average Recall', fontsize=14)
    ax1.set_ylim([0,100])
    # ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.set_title('Venman', fontweight='bold')
    ax1.set_xticks(np.arange(0,11,1))
    ax1.set_yticks(np.arange(0,101,5))
    ax1.grid(True, which='both',linestyle='--', alpha=0.3)

    ax2.set_ylabel('Recall', fontsize=14)
    ax2.set_xlabel('Distance Threshold [meters]', fontsize=14)
    ax2.set_ylim([0,100])
    ax2.plot(distances, recallListK, '.-', color='darkviolet')
    ax2.set_title('Karawatha', fontweight='bold')
    ax2.set_xticks(np.arange(0,11,1))
    ax2.set_yticks(np.arange(0,101,5))
    ax2.grid(True, which='both',linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig('./paperFigures/recallOverDistances.png')
    plt.savefig('./paperFigures/recallOverDistances.pdf')


def wildplacesOverTime(mapRes=0.3, scanDimX=120, refIncr=1, queryIncr=1, refRad=2, blockSize=2, NumMatches=2, numPtsMultiplier=2, background=-0.15):
    ourVenman, ourKarawatha = [], []
    percentVal = 100
    j = 2
    for i in [1, 3, 4]:
        envName = 'Venman'
        results_dir = './results/LPR_Wildplaces/EvaluateTestSet'
        results_filename = f'/{envName}_R:{i}_Q:{j}_RInc:{refIncr}_Rrad:{refRad}_QInc:{queryIncr}_Res:{mapRes}_Dim:{scanDimX}_blkAvg:{blockSize}_N2ndsearch:{NumMatches}_nptsMult{numPtsMultiplier}_background{background}.xlsx'
        outputNamesVenman = extractSingleReuslt(results_dir + results_filename, errTolerance=3)
        ourVenman.append(outputNamesVenman['recall'] * percentVal)
        print(envName,i,j,outputNamesVenman['recall'] )
    for i in [1, 3, 4]:
        envName = 'Karawatha'
        results_dir = './results/LPR_Wildplaces/EvaluateTestSet'
        results_filename = f'/{envName}_R:{i}_Q:{j}_RInc:{refIncr}_Rrad:{refRad}_QInc:{queryIncr}_Res:{mapRes}_Dim:{scanDimX}_blkAvg:{blockSize}_N2ndsearch:{NumMatches}_nptsMult{numPtsMultiplier}_background{background}.xlsx'
        outputNamesKarawatha = extractSingleReuslt(results_dir + results_filename, errTolerance=3)
        ourKarawatha.append(outputNamesKarawatha['recall'] * percentVal)
        print(envName,i,j,outputNamesKarawatha['recall'] )

    scancontextVenman, scancontextKarawatha = [63.66, 27.95, 18.41], [49.77, 25.72, 35.58]
    transloc3dVenman, transloc3dKarawatha = [59.42, 37.69, 40.34], [45.43, 36.32]
    minkloc3dv2Venman, minkloc3dv2Karawatha = [90.46, 67.36, 69.17], [73.62, 68.21, 60.58]
    logg3dVenman, logg3dKarawatha = [90.36, 70.75, 73.83], [77.54, 73.62, 65.14]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=300, sharey=True)

    # Custom labels and markers
    markers = ['o', 's', '^']
    colors = ["tab:red", "tab:blue", "tab:green"]
    y_labels = ['ScanContext', 'TransLoc3D', 'MinkLoc3Dv2', 'LoGG3D-Net', 'Ours']

    # Plot for ax1
    for n, results in enumerate([scancontextVenman, transloc3dVenman, minkloc3dv2Venman, logg3dVenman, ourVenman]):
        ax1.hlines(n, min(results), max(results), linestyle='--', color='gray', alpha=0.6)
        for x, marker, col in zip(results, markers, colors):
            ax1.plot(x, n, marker=marker, markersize=14, markeredgecolor='k', alpha=1, color=col)

    # Set custom y-axis labels for ax1
    ax1.set_yticks(range(len(y_labels)))
    ax1.set_yticklabels(y_labels, rotation=0)
    ax1.set_xlabel('Recall@1', fontsize=14)
    ax1.set_xticks(np.arange(10, 110, 10))
    ax1.set_xlim([15, 105])
    ax1.grid(True, which='both', linestyle='--', alpha=0.4)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.set_title('Venman', fontsize=16, fontweight='bold')

    # Plot for ax2
    for n, results in enumerate([scancontextKarawatha, transloc3dKarawatha, minkloc3dv2Karawatha, logg3dKarawatha, ourKarawatha]):
        if results == transloc3dKarawatha:
            markers = ['s', '^']
            colors = ["tab:blue", "tab:green"]
        else:
            markers = ['o', 's', '^']
            colors = ["tab:red", "tab:blue", "tab:green"]
        ax2.hlines(n, min(results), max(results), linestyle='--', color='gray', alpha=0.6)
        for x, marker, col in zip(results, markers, colors):
            ax2.plot(x, n, marker=marker, markersize=14, markeredgecolor='k', alpha=1, color=col)

    # Set custom y-axis labels for ax2
    ax2.set_yticks(range(len(y_labels)))
    ax2.set_xlabel('Recall@1', fontsize=14)
    ax2.set_xticks(np.arange(10, 110, 10))
    ax2.set_xlim([15, 105])
    ax2.grid(True, which='both', linestyle='--', alpha=0.4)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.set_title('Karawatha', fontsize=16, fontweight='bold')

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[0], markeredgecolor='k', markersize=14, label='Same Day'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=colors[1], markeredgecolor='k', markersize=14, label='6 Months'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=colors[2], markeredgecolor='k', markersize=14, label='14 Months')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=14, bbox_to_anchor=(0.5, 0))
    plt.tight_layout(rect=[0, 0.07, 1, 1])  # Adjust the rect parameter to make space at the bottom
    plt.savefig('./paperFigures/WildplacesRecallOverTime.png')
    plt.savefig('./paperFigures/WildplacesRecallOverTime.pdf')


def extractResultNCLT(refNum,queryNum,mapRes=1, scanDimX=120, refIncr=1, queryIncr=1, refRad=2, queryRad=5, blockSize=2, NumMatches=2, background=-0.15, refThresh=1, dim_randDwnsmple=10, numPtsMultiplier=2, errToler=25,variationToName=None):
    datasetName='NCLT'
    # results_dir=f'./results/LPR_{datasetName}/Res:{mapRes}_Dim:{scanDimX}_RInc:{refIncr}_Rrad:{refRad}'
    # results_filename=f'/{datasetName}_R:{refNum}_Q:{queryNum}_RInc:{refIncr}_Rrad:{refRad}_QInc:{queryIncr}_Qrad:{queryRad}_Res:{mapRes}_Dim:{scanDimX}_blkAvg:{blockSize}_N2ndsearch:{NumMatches}_background:{background}.xlsx' #
    # refGridFilename=f"/refGrid_incr{refIncr}_dim{scanDimX}_mapRes{mapRes}_refIncr{refIncr}_refRad2_rThresh2_nptsMult2_AvgDownsamp_noReverse.npy"#
    # refGridFilename=f"/refGrid_dim{scanDimX}_mapRes{mapRes}_refIncr{refIncr}_refRad{refRad}_rThresh{refThresh}_patchDim{dim_randDwnsmple}_nptsMult{numPtsMultiplier}_noReverse.npy"
    
    refGridFilename=f"/refGrid_dim{scanDimX}_mapRes{mapRes}_refIncr{refIncr}_refRad{refRad}_rThresh{refThresh}_patchDim{dim_randDwnsmple}_nptsMult{numPtsMultiplier}.npy"
    # saveFolder=f'./results/LPR_{datasetName}/Res:{mapRes}_Dim:{scanDimX}_RInc:{refIncr}_Rrad:{refRad}'
    saveFolder= f'./results/LPR_{datasetName}/EvaluateTestSet'
    # savePath = saveFolder+f'/{datasetName}_R:{refNum}_Q:{queryNum}_RInc:{refIncr}_Rrad:{refRad}_QInc:{queryIncr}_Qrad:{queryRad}_Res:{mapRes}_Dim:{scanDimX}_blkAvg:{blockSize}_N2ndsearch:{NumMatches}_background:{background}.xlsx'
    if variationToName!=None:
        savePath = saveFolder+f'/{datasetName}_R:{refNum}_Q:{queryNum}_RInc:{refIncr}_Rrad:{refRad}_QInc:{queryIncr}_Qrad:{queryRad}_Res:{mapRes}_Dim:{scanDimX}_blkAvg:{blockSize}_N2ndsearch:{NumMatches}_background:{background}{variationToName}.xlsx'  #__queryClose2Ref #_everyNref_queryClose2Ref
    
    else:
        savePath = saveFolder+f'/{datasetName}_R:{refNum}_Q:{queryNum}_RInc:{refIncr}_Rrad:{refRad}_QInc:{queryIncr}_Qrad:{queryRad}_Res:{mapRes}_Dim:{scanDimX}_blkAvg:{blockSize}_N2ndsearch:{NumMatches}_background:{background}.xlsx'  #__queryClose2Ref #_everyNref_queryClose2Ref
    
    return extractSingleReuslt(savePath, errTolerance=errToler)

def recallNCLT():
    totRecall,count=0,0
    for i in [10,11]:
        for j in range(1,10):
            
            count+=1  
            output=extractResultNCLT(i,j, refRad=4, queryRad=10, mapRes=0.75)
            recall=output['recall']
            print(f'   ref:{i}, query:{j}, recall:{recall}   '  )
            totRecall+=output['recall']
    print(f'average recall: {totRecall/count}')  

def recallOxfordRadar():
    totRecall,count=0,0
    for i in [6,7]:
        for j in range(1,6):
            
            count+=1  
            datasetName='OxfordRadar'
            mapRes=1
            scanDimX=120
            refIncr=1
            queryIncr=1
            refRad=4
            queryRad=10
            blockSize=2
            NumMatches=2
            background=-0.15
            refThresh=1
            dim_randDwnsmple=10
            nMultiplier_randDwnsmpl=2
            results_dir=f'./results/LPR_{datasetName}/EvaluateTestSet'
            results_filename=f'/{datasetName}_R:{i}_Q:{j}_RInc:{refIncr}_Rrad:{refRad}_QInc:{queryIncr}_Qrad:{queryRad}_Res:{mapRes}_Dim:{scanDimX}_blkAvg:{blockSize}_N2ndsearch:{NumMatches}_background:{background}.xlsx'
            refGridFilename=refNpyName=f"/refGrid_dim{scanDimX}_mapRes{mapRes}_refIncr{refIncr}_refRad{refRad}_rThresh{refThresh}_patchDim{dim_randDwnsmple}_nptsMult{nMultiplier_randDwnsmpl}.npy"
            output= extractSingleReuslt(results_dir+results_filename, errTolerance=25)
            print(results_filename, output['recall'])
            totRecall+=output['recall']
    print(f'average recall: {totRecall/count}')  

def recallWildplaces():
    for envName in ['Karawatha', 'Venman']:
        totRecall,count=0,0
        for i in range(1,5):
            for j in range(1,5):
                if i!=j: 
                    count+=1  
                    datasetName=f'WildPlaces_{envName}'
                    mapRes=0.3
                    scanDimX=120
                    refIncr=1
                    queryIncr=1
                    refRad=2
                    blockSize=2
                    NumMatches=2
                    numPtsMultiplier=2
                    background=-0.15
                    results_dir=f'./results/LPR_Wildplaces/Trial'
                    results_filename=f'/{envName}_R:{i}_Q:{j}_RInc:{refIncr}_Rrad:{refRad}_QInc:{queryIncr}_Res:{mapRes}_Dim:{scanDimX}_blkAvg:{blockSize}_N2ndsearch:{NumMatches}_nptsMult{numPtsMultiplier}_background{background}.xlsx'
                    
                    output= extractSingleReuslt(results_dir+results_filename, errTolerance=3)
                    print(output['recall'])
                    totRecall+=output['recall']
        print(f'average recall: {totRecall/count}')  

def ablationResultsPlot():
    top_n_recall = []
    top_n_values = [1,2,5,10]
    
    for n in top_n_values:
        savePath = f'./results/LPR_Wildplaces/Ablate_TrainingSet/Venman_R:2_Q:3_RInc:1_Rrad:2_QInc:10_Res:0.3_Dim:120_blkAvg:2_N2ndsearch:{n}_rotInc10_UnoccWeight-0.15.xlsx'
        output = extractSingleReuslt(savePath, errTolerance=3)
        recall = output['recall']
        top_n_recall.append(recall)
    
    unocc_weight_recall = []
    unocc_weight_values = [0, -0.05, -0.1, -0.15, -0.2, -0.25, -0.3]
    
    for val in unocc_weight_values:
        savePath = f'./results/LPR_Wildplaces/Ablate_TrainingSet/Venman_R:2_Q:3_RInc:1_Rrad:2_QInc:10_Res:0.3_Dim:120_blkAvg:2_N2ndsearch:2_rotInc10_UnoccWeight{val}.xlsx'
        output = extractSingleReuslt(savePath, errTolerance=3)
        recall = output['recall']
        unocc_weight_recall.append(recall)
    
    pool_size_recall = []
    pool_size_values = [1, 2, 3, 4, 5]
    
    for blockSize in pool_size_values:
        savePath = f'./results/LPR_Wildplaces/Ablate_TrainingSet/Venman_R:2_Q:3_RInc:1_Rrad:2_QInc:10_Res:0.3_Dim:120_blkAvg:{blockSize}_N2ndsearch:2_rotInc10_UnoccWeight-0.15.xlsx'
        output = extractSingleReuslt(savePath, errTolerance=3)
        recall = output['recall']
        pool_size_recall.append(recall)
    
    rot_incr_recall = []
    rot_incr_values = [1, 5, 10, 20, 30]
    
    for rot in rot_incr_values:
        savePath = f'./results/LPR_Wildplaces/Ablate_TrainingSet/Venman_R:2_Q:3_RInc:1_Rrad:2_QInc:10_Res:0.3_Dim:120_blkAvg:2_N2ndsearch:2_rotInc{rot}_UnoccWeight-0.15.xlsx'
        output = extractSingleReuslt(savePath, errTolerance=3)
        recall = output['recall']
        rot_incr_recall.append(recall)
    
    # Plotting the recall values with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(7, 5))
    
    axs[0, 0].plot(top_n_values, top_n_recall, marker='o', color='tab:blue')
    axs[0, 0].scatter(top_n_values[1], top_n_recall[1], color='k', marker='x', zorder=5, s=75)
    axs[0, 0].set_xlabel('Number of Top Matches (n)', fontsize=14)
    axs[0, 0].grid(True)
    axs[0, 0].set_xlim([0,10.2])
    
    axs[0, 1].plot(unocc_weight_values, unocc_weight_recall, marker='o', color='tab:green')
    axs[0, 1].scatter(unocc_weight_values[3], unocc_weight_recall[3], color='k', marker='x', zorder=5, s=75)
    axs[0, 1].set_xlabel('Unoccupied Cell Weighting (w)', fontsize=14)
    axs[0, 1].grid(True)
    
    axs[1, 0].plot(pool_size_values, pool_size_recall, marker='o', color='tab:purple')
    axs[1, 0].scatter(pool_size_values[1], pool_size_recall[1], color='k', marker='x', zorder=5, s=75)
    axs[1, 0].set_xlabel('Average Pooling Size (u)', fontsize=14)
    axs[1, 0].grid(True)
    
    axs[1, 1].plot(rot_incr_values, rot_incr_recall, marker='o', color='tab:orange')
    axs[1, 1].scatter(rot_incr_values[2], rot_incr_recall[2], color='k', marker='x', zorder=5, s=75)
    axs[1, 1].set_xlabel('Rotation Increments (k)', fontsize=14)
    axs[1, 1].grid(True)
    
    for ax in axs.flat:
        ax.set(ylabel='Recall')
        ax.yaxis.label.set_size(14)
        ax.set_ylim([0,1])
        ax.grid(True, which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.1)
        ax.grid(True, which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=0.1)
        ax.tick_params(axis='both', which='major', labelsize=12)
    
    print('top_n_recall',top_n_values, top_n_recall)
    print('unocc_weight_recall',unocc_weight_values,unocc_weight_recall)
    print('pool_size_recall',pool_size_values,pool_size_recall)
    print('rot_incr_recall',rot_incr_values,rot_incr_recall)
    # plt.suptitle('Recall Values for Different Parameters')
    plt.tight_layout(pad=0.7)
    # plt.show()
    plt.savefig('./paperFigures/ablateParameterSensitivity.png')
    plt.savefig('./paperFigures/ablateParameterSensitivity.pdf')



'''Currently used functions'''
# plotRealtiveShift()
# wildplacesOverTime()
recallNCLT()
recallOxfordRadar()
recallWildplaces()

# output=extractResultNCLT(1,2, refRad=4, queryRad=10, mapRes=0.75)
# print(output['recall'])

''' Pose estimation wihtin Rotation and Trasnaltion threshold '''
# output=extractResultNCLT(1,2, refRad=4, queryRad=10, mapRes=0.75)
# actualYaw=output['queryYaw'][output['correctMatchIds']]%360
# matchedYaw=output['matchYaw'][output['correctMatchIds']]
# yawShift=output['yawShift'][output['correctMatchIds']]

# predYaw=(matchedYaw-yawShift)%360
# diff=(actualYaw-predYaw)
# angleDiff=abs((diff + 180) % 360 - 180)
# print(angleDiff)

# distErr=output['dists'][output['correctMatchIds']]
# recall=output['recall']
# refinedDistErrs=distErr[(distErr < 2) & (angleDiff < 5)]
# refinedAngDiff=angleDiff[(distErr < 2) & (angleDiff < 5)]
# print(refinedDistErrs)
# print(f'DIST- mean: {np.mean(refinedDistErrs)}, std: {np.std(refinedDistErrs)}')
# print(f'ANGLE - mean: {np.mean(refinedAngDiff)}, std: {np.std(refinedAngDiff)}')

# print(len(distErr), len(refinedDistErrs), len(refinedDistErrs)/len(distErr))
# print(f'reacll: {recall}')



'''Ring ++ comparison'''
# output=extractResultNCLT(2,6, refRad=20, queryRad=5, mapRes=0.75, errToler=10, variationToName='_errToler:10')
# print(output['recall'])
# distErr=output['dists'][output['correctMatchIds']]
# refinedDistErrs=distErr[distErr<2]
# print(len(distErr), len(refinedDistErrs), len(refinedDistErrs)/len(distErr))