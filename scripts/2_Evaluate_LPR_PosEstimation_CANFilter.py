import pandas as pd
import matplotlib.patches as patches
from scipy.spatial.distance import cdist
from matplotlib.gridspec import GridSpec
import math
from helperFunctions import activityDecoding, attractorNetwork2D, LiDAR_PlaceRec, extractResultsParam, find_closest_point
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
from matplotlib import rc
rc('text', usetex=True)
from collections import Counter
from matplotlib.colors import Normalize
import matplotlib.cm as cm
'''Plotting'''
def plot(results_filename):
    df = pd.read_excel(results_filename)
    refPositions, queryPositions, matchIds, matchPositions, matchShiftedPositions = df['referencePositions'].dropna(), df['queryPositions'], df['matchID'], df['matchPosition'], df['matchShiftedPosition']
    scanAlignment, imgIds, imgPositions, imgDistances, canPeaks, canVariences = df['scanAlignment'], df['imgID'], df['imgPosition'], df['imgDistance'],df['canPeak'],df['canVarience']
    # canMag=df['canMagnitude']

    refX, refY=zip(*refPositions.apply(lambda x: (ast.literal_eval(x)[0], ast.literal_eval(x)[1])))
    queryX, queryY=zip(*queryPositions.apply(lambda x: (ast.literal_eval(x)[0], ast.literal_eval(x)[1])))
    lprX, lprY=zip(*matchPositions.apply(lambda x: (ast.literal_eval(x)[0], ast.literal_eval(x)[1])))
    lprShiftX, lprShiftY=zip(*matchShiftedPositions.apply(lambda x: (ast.literal_eval(x)[0], ast.literal_eval(x)[1])))
    canX, canY=zip(*canPeaks.apply(lambda x: (ast.literal_eval(x)[0], ast.literal_eval(x)[1])))
    imgY, imgX=zip(*imgPositions.apply(lambda x: (ast.literal_eval(x)[0], ast.literal_eval(x)[1])))

    canError=np.sqrt(    (np.array(canX)-np.array(queryX))**2    +   (np.array(canY)-np.array(queryY))**2   )
    lprShiftError=np.sqrt(   (np.array(lprShiftX)-np.array(queryX))**2   +   (np.array(lprShiftY)-np.array(queryY))**2   )
    imgMatcherror=np.sqrt(   (np.array(imgX)-np.array(queryX))**2    +    (np.array(imgY)-np.array(queryY))**2   )
    

    lprShiftMag=np.sqrt(np.array(np.gradient(lprShiftX))**2+np.array(np.gradient(lprShiftY))**2)
    imgMag=np.sqrt(np.array(np.gradient(imgX))**2+np.array(np.gradient(imgY))**2)
    lprMagPeaks,_=find_peaks(lprShiftMag, threshold=20)

    imgLowErrorIds=np.where(imgMatcherror<20)[0]
    lidarLowShiftErrorIds=np.where(lprShiftError<20)[0]
    lidarLowMagIds=np.where(lprShiftMag<20)[0]
    canLowErrorIds=np.where(canError<20)[0]
    imgLowMagIds=np.where(imgMag<20)[0]

    lidarHighMagIds=np.where(lprShiftMag>20)[0]
    
    
    print(f'SAD- Num errors (<20m radius): {len(imgMatcherror)-len(imgLowErrorIds)}, Recall: {(len(imgLowErrorIds))/len(imgMatcherror)}, Total Distance Error: {np.sum(np.array(imgMatcherror))}')
    print(f'LiDAR shift -Num errors (<20m radius): {len(lprShiftError)-len(lidarLowShiftErrorIds)}, Recall: {(len(lidarLowShiftErrorIds))/len(lprShiftError)}, Total Distance Error: {np.sum(np.array(lprShiftError))}')
    print(f'CAN- Num errors (<20m radius): {len(canError)-len(canLowErrorIds)}, Recall: {(len(canLowErrorIds))/len(canError)}, Total Distance Error: {np.sum(np.array(canError))}')


    # print(np.array(canX)-np.array(queryX))
    fig,((ax7,ax4,ax1))=plt.subplots(1,3)
    # plt.plot(refX, refY, 'c.')
    # plt.plot(queryX, queryY, 'g.')
    # plt.plot(lprX, lprY, 'm.')
    ax1.plot(queryX, queryY, 'g.')
    ax1.plot(canX, canY, 'c.')
    ax1.plot([queryX,canX], [queryY, canY], color='tab:brown', linestyle='-', alpha=0.5)
    ax1.set_title('GT and CAN estimations')


    # ax4.plot(queryX, queryY, 'g.')
    # ax4.plot(lprShiftX, lprShiftY, 'b.')
    # ax4.plot([queryX,lprShiftX], [queryY, lprShiftY], color='tab:brown', linestyle='-', alpha=0.5)
    ax4.plot(np.array(queryX)[lidarLowMagIds], np.array(queryY)[lidarLowMagIds], 'g.')
    ax4.plot(np.array(lprShiftX)[lidarLowMagIds], np.array(lprShiftY)[lidarLowMagIds], 'm.')
    ax4.plot([np.array(queryX)[lidarLowMagIds],np.array(lprShiftX)[lidarLowMagIds]], [np.array(queryY)[lidarLowMagIds], np.array(lprShiftY)[lidarLowMagIds]], color='tab:brown', linestyle='-', alpha=0.5)
    ax4.set_title('GT and velocity filter lidar Match')


    ax7.plot(np.array(queryX)[imgLowMagIds], np.array(queryY)[imgLowMagIds], 'g.')
    ax7.plot(np.array(imgX)[imgLowMagIds], np.array(imgY)[imgLowMagIds], '.', color='tab:pink')
    ax7.plot([np.array(queryX)[imgLowMagIds],np.array(imgX)[imgLowMagIds]], [np.array(queryY)[imgLowMagIds], np.array(imgY)[imgLowMagIds]], color='tab:brown', linestyle='-', alpha=0.5)
    ax7.set_title('GT and velocity filter Img Match')


    
    plt.savefig('./result2.png')


def plotConvolutionMatch(results_filename):
    df = pd.read_excel(results_filename)
    refPositions, queryPositions, matchIds, matchPositions, matchShiftedPositions = df['referencePositions'], df['queryPositions'], df['matchID'], df['matchPosition'], df['matchShiftedPosition']
    scanAlignment, imgIds, imgPositions, imgDistances, canPeaks, canVariences = df['scanAlignment'], df['imgID'], df['imgPosition'], df['imgDistance'],df['canPeak'],df['canVarience']
    # canMag=df['canMagnitude']
    
    refX, refY=zip(*refPositions.apply(lambda x: (ast.literal_eval(x)[0], ast.literal_eval(x)[1])))
    queryX, queryY=zip(*queryPositions.apply(lambda x: (ast.literal_eval(x)[0], ast.literal_eval(x)[1])))
    lprX, lprY=zip(*matchPositions.apply(lambda x: (ast.literal_eval(x)[0], ast.literal_eval(x)[1])))
    lprShiftX, lprShiftY=zip(*matchShiftedPositions.apply(lambda x: (ast.literal_eval(x)[0], ast.literal_eval(x)[1])))
    canX, canY=zip(*canPeaks.apply(lambda x: (ast.literal_eval(x)[0], ast.literal_eval(x)[1])))
    imgY, imgX=zip(*imgPositions.apply(lambda x: (ast.literal_eval(x)[0], ast.literal_eval(x)[1])))

    canError=np.sqrt(    (np.array(canX)-np.array(queryX))**2    +   (np.array(canY)-np.array(queryY))**2   )
    lprShiftError=np.sqrt(   (np.array(lprShiftX)-np.array(queryX))**2   +   (np.array(lprShiftY)-np.array(queryY))**2   )
    lprError=np.sqrt(   (np.array(lprX)-np.array(queryX))**2   +   (np.array(lprY)-np.array(queryY))**2   )
    imgMatcherror=np.sqrt(   (np.array(imgX)-np.array(queryX))**2    +    (np.array(imgY)-np.array(queryY))**2   )
    

    lprMag=np.sqrt(np.array(np.gradient(lprX))**2+np.array(np.gradient(lprY))**2)

    imgLowErrorIds=np.where(imgMatcherror<20)[0]
    lidarLowErrorIds=np.where(lprError<20)[0]
    lidarLowMagIds=np.where(lprMag<20)[0]
    canLowErrorIds=np.where(canError<20)[0]

    lidarHighMagIds=np.where(lprMag>20)[0]

    print(f'Num errors: {len(lprError)-len(lidarLowErrorIds)}, Recall: {(len(lidarLowErrorIds))/len(lprError)}, Total Distance Error: {np.sum(np.array(lprError))}')

    fig,((ax1,ax2),(ax4,ax5))=plt.subplots(2,2)

    ax1.plot(queryX, queryY, 'g.')
    ax1.plot(lprX, lprY, 'm.')
    ax1.plot([queryX,lprX], [queryY, lprY], color='tab:brown', linestyle='-', alpha=0.5)
    ax1.set_title('GT and lidar Matched positions')

    ax2.hist(lprError, bins=5000, color='blue',  histtype='step', stacked=True, fill=False, density=True)
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax2.set_title('Distribution of match error')

    # ax2.plot(lprError, 'y.')
    # ax2.set_title('LPR match error')

    ax4.plot(np.array(queryX)[lidarLowMagIds], np.array(queryY)[lidarLowMagIds], 'g.')
    ax4.plot(np.array(lprX)[lidarLowMagIds], np.array(lprY)[lidarLowMagIds], 'm.')
    ax4.plot([np.array(queryX)[lidarLowMagIds],np.array(lprX)[lidarLowMagIds]], [np.array(queryY)[lidarLowMagIds], np.array(lprY)[lidarLowMagIds]], color='tab:brown', linestyle='-', alpha=0.5)
    ax4.set_title('GT and filtered lidar Matched positions')

    ax5.plot(lprMag, 'y.')
    ax5.plot(lidarHighMagIds, lprMag[lidarHighMagIds], "rx")
    # ax2.plot(canMag, 'y.')
    ax5.set_title('Gradient of matched x y positions')

    plt.show()


def plotRelativeShift(results_filename,toler):
    rslt=extractResultsParam(results_filename, errTolerance=toler)
    queryX,queryY = np.array(rslt['queryX']), np.array(rslt['queryY'])
    
    lprX,lprY = np.array(rslt['lprX']), np.array(rslt['lprY'])
    lidarLowErrorIds, lidarHighErrorIds = np.array(rslt['lidarLowErrorIds']),  np.array(rslt['lidarHighErrorIds'])
    lprError = np.array(rslt['lprError'])/toler
    
    lprShiftX,lprShiftY = np.array(rslt['lprShiftX']),np.array(rslt['lprShiftY'])
    lidarLowShiftErrorIds, lidarHighShiftErrorIds = np.array(rslt['lidarLowShiftErrorIds']), np.array(rslt['lidarHighShiftErrorIds'])
    lprShiftError = np.array(rslt['lprShiftError'])/toler
    
    canX,canY = np.array(rslt['canX']),np.array(rslt['canY'])
    canLowErrorIds, canHighErrorIds = np.array(rslt['canLowErrorIds']), np.array(rslt['canHighErrorIds'])
    canError = np.array(rslt['canError'])/toler

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10,7))
    # ax1.scatter(lprX[lidarLowErrorIds], lprY[lidarLowErrorIds], s=100 * lprError[lidarLowErrorIds], color='skyblue', alpha=0.7)
    # ax4.scatter(lprShiftX[lidarLowShiftErrorIds], lprShiftY[lidarLowShiftErrorIds], s=100 * lprShiftError[lidarLowShiftErrorIds], color='yellowgreen', alpha=0.7)


    
    # ax1.scatter(queryX[lidarLowErrorIds], queryY[lidarLowErrorIds], s=7,c='g')
    ax1.scatter(lprX[lidarLowErrorIds], lprY[lidarLowErrorIds], s=100 * lprError[lidarLowErrorIds], color='skyblue', alpha=0.7)
    # ax1.plot([queryX[lidarLowErrorIds], lprX[lidarLowErrorIds]],[queryY[lidarLowErrorIds], lprY[lidarLowErrorIds]], color='tab:brown', linestyle='-', alpha=0.5)
    ax1.scatter(queryX[lidarHighErrorIds], queryY[lidarHighErrorIds], s=10,c='k',  alpha=0.7)
    ax1.set_title('Lidar Match')

    
    # ax2.scatter(queryX[lidarLowShiftErrorIds], queryY[lidarLowShiftErrorIds], s=7,c='g')
    ax2.scatter(lprShiftX[lidarLowShiftErrorIds], lprShiftY[lidarLowShiftErrorIds], s=100 * lprShiftError[lidarLowShiftErrorIds], color='yellowgreen', alpha=0.7)
    # ax2.plot([queryX[lidarLowShiftErrorIds], lprShiftX[lidarLowShiftErrorIds]], [queryY[lidarLowShiftErrorIds], lprShiftY[lidarLowShiftErrorIds]], color='tab:brown', linestyle='-', alpha=0.5)
    ax2.scatter(queryX[lidarHighShiftErrorIds], queryY[lidarHighShiftErrorIds], s=10,c='k', alpha=0.7)
    ax2.set_title('Shifted lidar Match')

    
    # ax3.scatter(queryX[lidarLowErrorIds], queryY[lidarLowErrorIds], s=7,c='g')
    ax3.scatter(canX[canLowErrorIds], canY[canLowErrorIds], s=100 * canError[canLowErrorIds], color='pink', alpha=0.7)
    # ax3.plot([queryX[canLowErrorIds], canX[canLowErrorIds]], [queryY[canLowErrorIds], canY[canLowErrorIds]], color='tab:brown', linestyle='-', alpha=0.5)
    ax3.scatter(queryX[canHighErrorIds], queryY[canHighErrorIds], s=10,c='k', alpha=0.7)
    ax3.set_title('CAN filter lidar Match')


def plotRelativeShiftStats(results_filename, toler):
    rslt=extractResultsParam(results_filename, errTolerance=toler)
    # Access variables from the dictionary and convert to array
    
    lprShiftError = np.array(rslt['lprShiftError'])
    lidarLowErrorIds = np.array(rslt['lidarLowErrorIds'])
    lprError = np.array(rslt['lprError'])

    recallMatch, recallShift, meanErrorMatch, meanErrorShift=[], [], [], []
    if 'Jackal' in results_filename:
        testRange= np.arange(0,2,0.1)
        numBins=10
    else:
        testRange= np.arange(0,20)
        numBins=20
    for i in testRange:
        rslt=extractResultsParam(results_filename, errTolerance=i)
        lprError = np.array(rslt['lprError'])
        lprShiftError = np.array(rslt['lprShiftError'])
        lidarLowErrorIds = np.array(rslt['lidarLowErrorIds'])
        lidarLowShiftErrorIds = np.array(rslt['lidarLowShiftErrorIds'])

        recallMatch.append((len(lidarLowErrorIds))/len(lprError))
        recallShift.append((len(lidarLowShiftErrorIds))/len(lprError))
        meanErrorMatch.append(np.mean(lprError[lidarLowErrorIds]))
        meanErrorShift.append(np.mean(lprShiftError[lidarLowShiftErrorIds]))


    print(f'Num errors: {len(lprError)-len(lidarLowErrorIds)}, Recall: {(len(lidarLowErrorIds))/len(lprError)}, Total Distance Error: {np.sum(np.array(lprError[lidarLowErrorIds]))}')
    print(f'Num errors (<20m radius): {len(lprShiftError)-len(lidarLowShiftErrorIds)}, Recall: {(len(lidarLowShiftErrorIds))/len(lprShiftError)}, Total Distance Error: {np.sum(lprShiftError[lidarLowErrorIds])}')

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(10,8))
    fig.tight_layout(pad=3)
    
    ax1.plot(testRange, recallMatch, '.-', color='skyblue')
    ax1.plot(testRange, recallShift, '.-', color='yellowgreen')
    ax1.legend(['Match', 'Shifted Match'])
    ax1.set_title('Recall across Error tolerances')
    ax1.set_ylabel('Recall')
    ax1.set_xlabel('Error Tolerence [m]')

    ax2.plot(lprError[lidarLowErrorIds], '.', color='skyblue')
    ax2.set_title('Match errors')
    ax2.set_ylim([0, toler])

    
    # ax3.hist(lprError[lidarLowErrorIds], bins=20, color='skyblue', stacked=True)
    # ax3.hist(lprShiftError[lidarLowShiftErrorIds], bins=20, color='yellowgreen',  alpha=0.5 )
    ax3.hist(lprShiftError[lidarLowErrorIds] - lprError[lidarLowErrorIds], bins=numBins, color='plum', stacked=True)
    ax3.set_title('Delta Error (ShiftedMatch-Match)')

    ax4.plot(testRange, meanErrorMatch, '.-', color='skyblue')
    ax4.plot(testRange, meanErrorShift, '.-', color='yellowgreen')
    ax4.legend(['Match error', 'Shifted Match error'])
    ax4.set_title('Mean Error across Error tolerances')
    ax4.set_ylabel('Mean Error [m]')
    ax4.set_xlabel('Error Tolerence [m]')
    

    ax5.plot(lprShiftError[lidarLowShiftErrorIds], '.', color='yellowgreen')
    ax5.set_title('Shifted Match errors')
    ax5.set_ylim([0, toler])

    colors = ['skyblue', 'yellowgreen']
    bplot= ax6.boxplot([lprError, lprShiftError], showfliers=False, vert=True, patch_artist=True)
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    ax6.legend([bplot["boxes"][0], bplot["boxes"][1]], ['Match error', 'Shifted Match error'])
    ax6.set_title('Error without outliers')


def plotObservingBadShifts(results_filename,toler):
    rslt=extractResultsParam(results_filename, errTolerance=toler)
    # Access variables from the dictionary and convert to array
    queryX,queryY = np.array(rslt['queryX']), np.array(rslt['queryY'])
    lprShiftX,lprShiftY = np.array(rslt['lprShiftX']),np.array(rslt['lprShiftY'])
    lprShiftError = np.array(rslt['lprShiftError'])
    lidarLowErrorIds = np.array(rslt['lidarLowErrorIds'])
    lprError = np.array(rslt['lprError'])
    scanAlignment = np.array(rslt['scanAlignment'])


    deltaErr=(lprShiftError - lprError)
    badShiftIds=np.where(deltaErr > 1)[0]
    
    lowErr_badShiftIds=np.intersect1d(badShiftIds, lidarLowErrorIds)
    print(len(lowErr_badShiftIds), len(badShiftIds), len(lidarLowErrorIds))
    
    fig, ((ax1,ax2),(ax3,ax4))=plt.subplots(2,2)

    ax1.hist(deltaErr, bins=20, color='plum', stacked=True)
    ax1.set_title('Delta Error (ShiftedMatch-Match)')

    ax2.plot(lprShiftX[lowErr_badShiftIds ],lprShiftY[lowErr_badShiftIds], '.')

    ax3.plot(scanAlignment, '.')
    ax3.plot(lowErr_badShiftIds, scanAlignment[lowErr_badShiftIds], 'rx')


def plotCANStats(results_filename, toler):
    
    recallMatch, recallCAN, recallShift, meanErrorMatch, meanErrorCAN, meanErrorShift, stdErrorMatch, stdErrorCAN, stdErrorShift=[], [], [], [], [], [], [], [], []
    if 'Jackal' in results_filename:
        testRange= np.arange(0,2,0.1)
        numBins=10
    else:
        testRange= np.arange(0,25,2)
        testRangeMeanError=np.arange(0,20,1)
        numBins=20
    for i in testRange:
        rslt=extractResultsParam(results_filename, errTolerance=i)
        lprError = np.array(rslt['lprError'])
        canError = np.array(rslt['canError'])
        lprShiftError = np.array(rslt['lprShiftError'])
        canLowErrorIds = np.array(rslt['canLowErrorIds'])
        lidarLowShiftErrorIds = np.array(rslt['lidarLowShiftErrorIds'])
        lidarLowErrorIds = np.array(rslt['lidarLowErrorIds'])

        recallMatch.append((len(lidarLowErrorIds))/len(lprError))
        recallCAN.append((len(canLowErrorIds))/len(lprError))
        recallShift.append((len(lidarLowShiftErrorIds))/len(lprError))

        meanErrorMatch.append(np.mean(lprError[lidarLowErrorIds]))
        meanErrorCAN.append(np.mean(canError[canLowErrorIds]))
        meanErrorShift.append(np.mean(lprShiftError[lidarLowShiftErrorIds]))

        stdErrorMatch.append(np.std(lprError[lidarLowErrorIds]))
        stdErrorCAN.append(np.std(canError[canLowErrorIds]))
        stdErrorShift.append(np.std(lprShiftError[lidarLowShiftErrorIds]))
        
        

    rslt=extractResultsParam(results_filename, errTolerance=toler)
    # Access variables from the dictionary and convert to array
    lprError = np.array(rslt['lprError'])
    lprShiftError = np.array(rslt['lprShiftError'])
    lidarLowErrorIds = np.array(rslt['lidarLowErrorIds'])
    canError = np.array(rslt['canError'])
    deltaError_CAN2Shift= canError - lprShiftError
    deltaError_Shift2Match = lprShiftError - lprError

    print(f'Num errors: {len(lprError)-len(canLowErrorIds)}, Recall: {(len(lidarLowErrorIds))/len(lprError)}, Total Distance Error: {np.sum(np.array(lprError[lidarLowErrorIds]))}')
    print(f'Num errors (<20m radius): {len(lprShiftError)-len(lidarLowShiftErrorIds)}, Recall: {(len(lidarLowShiftErrorIds))/len(lprShiftError)}, Total Distance Error: {np.sum(lprShiftError[lidarLowErrorIds])}')
    print(f'Num errors (<20m radius): {len(canError)-len(canLowErrorIds)}, Recall: {(len(canLowErrorIds))/len(canError)}, Total Distance Error: {np.sum(np.array(canError))}')

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(10,8))
    fig.legend(['Match', 'CAN', 'Shifted Match'])
    fig.tight_layout(pad=4)

    l1=ax1.plot(testRange, recallMatch, '.-', color='skyblue', label='Match')
    l2=ax1.plot(testRange, recallShift, '.-', color='yellowgreen', label='ShiftedMatch')
    l3=ax1.plot(testRange, recallCAN, '.-', color='pink', label='CAN')
    # ax1.legend(['Match', 'CAN', 'Shifted Match'], loc='upper center')
    ax1.set_title('Recall across Error tolerances')
    ax1.set_ylabel('Recall')
    ax1.set_xlabel('Error Tolerence [m]')

    ax2.plot(testRange, meanErrorMatch, '.-', color='skyblue')
    ax2.plot(testRange, meanErrorCAN, '.-', color='pink')
    ax2.plot(testRange, meanErrorShift, '.-', color='yellowgreen')
    # 2x4.legend(['Match Error','CAN error', 'Shifted Match error'])
    ax2.set_title('Mean Error across Error tolerances')
    ax2.set_ylabel('Mean Error [m]')
    ax2.set_xlabel('Error Tolerence [m]')
    
    # ax2.plot(lprError, '.', color='skyblue', alpha=0.5)
    # ax2.plot(canError, '.', color='pink',alpha=0.5)
    # ax2.set_title('Match and CAN errors')
    # ax2.set_ylim([0, toler])
    # ax2.hist(deltaError_Shift2Match, log=True, bins=numBins, color='teal', stacked=True)
    ax3.set_title('Error Distrubtion')
    ax3.hist( lprError, facecolor='skyblue', log=True, histtype='stepfilled', bins=numBins)
    ax3.hist( lprShiftError, facecolor='yellowgreen',   log=True, histtype='stepfilled', bins=numBins)
    ax3.hist( canError, facecolor='pink',  log=True, histtype='stepfilled', bins=numBins)

    
    # ax3.hist(lprError[lidarLowErrorIds], bins=20, color='skyblue', stacked=True)
    # ax3.hist(lprShiftError[lidarLowShiftErrorIds], bins=20, color='yellowgreen',  alpha=0.5 )
    ax4.hist(deltaError_CAN2Shift, log=True, bins=numBins, color='brown', stacked=True)
    ax4.set_title('Delta Error (CANerror-Shifterror)')

    

    
    ax5.plot(lprShiftError, '.', color='yellowgreen', alpha=0.5)
    ax5.plot(canError, '.', color='pink',alpha=0.5)
    # ax5.legend(['Match', 'CAN', 'Shifted Match'], loc='upper center')
    ax5.set_title('Shift and CAN errors')
    # ax5.set_ylim([0, toler])

    
    colors = ['skyblue', 'yellowgreen', 'pink']
    bplot= ax6.boxplot([lprError, lprShiftError, canError], showfliers=False, vert=True, patch_artist=True)
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    # ax6.legend([bplot["boxes"][0], bplot["boxes"][1]], ['CAN error', 'Shifted Match error'])
    ax6.set_title('Error without outliers')

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels,loc = 'upper center', ncols=3)


def plotCANFilter(results_filename):
    rslt=extractResultsParam(results_filename, errTolerance=20)
    # Access variables from the dictionary and convert to array
    queryX, queryY = rslt['queryX'], rslt['queryY']
    canX, canY = rslt['canX'], rslt['canY']
    canError = rslt['canError']
    canLowErrorIds= rslt['canLowErrorIds']

    print(f'Num errors (<20m radius): {len(canError)-len(canLowErrorIds)}, Recall: {(len(canLowErrorIds))/len(canError)}, Total Distance Error: {np.sum(np.array(canError))}')


    fig,((ax1,ax2, ax3))=plt.subplots(1,3)
    ax1.plot(queryX, queryY, 'g.')
    ax1.plot(canX, canY, 'c.')
    ax1.plot([queryX,canX], [queryY, canY], color='tab:brown', linestyle='-', alpha=0.5)
    ax1.set_title('GT and CAN estimations')

    ax2.hist(canError, bins=5000, color='blue',  histtype='step', stacked=True, fill=False, density=True)
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax2.set_title('Distribution of match error')

    ax3.plot(canError,'m.')
    ax3.set_title('CAN and match error')


def plotMetrics(results_filename):
    df = pd.read_excel(results_filename)
    refPositions, queryPositions, matchIds, matchPositions, matchShiftedPositions = df['referencePositions'], df['queryPositions'], df['matchID'], df['matchPosition'], df['matchShiftedPosition']
    scanAlignment, imgIds, imgPositions, imgDistances, canPeaks, canVariences = df['scanAlignment'], df['imgID'], df['imgPosition'], df['imgDistance'],df['canPeak'],df['canVarience']
    # canMag=df['canMagnitude']
    
    refX, refY=zip(*refPositions.apply(lambda x: (ast.literal_eval(x)[0], ast.literal_eval(x)[1])))
    queryX, queryY=zip(*queryPositions.apply(lambda x: (ast.literal_eval(x)[0], ast.literal_eval(x)[1])))
    lprX, lprY=zip(*matchPositions.apply(lambda x: (ast.literal_eval(x)[0], ast.literal_eval(x)[1])))
    lprShiftX, lprShiftY=zip(*matchShiftedPositions.apply(lambda x: (ast.literal_eval(x)[0], ast.literal_eval(x)[1])))
    canX, canY=zip(*canPeaks.apply(lambda x: (ast.literal_eval(x)[0], ast.literal_eval(x)[1])))
    imgY, imgX=zip(*imgPositions.apply(lambda x: (ast.literal_eval(x)[0], ast.literal_eval(x)[1])))

    canError=np.sqrt(    (np.array(canX)-np.array(queryX))**2    +   (np.array(canY)-np.array(queryY))**2   )
    lprShiftError=np.sqrt(   (np.array(lprShiftX)-np.array(queryX))**2   +   (np.array(lprShiftY)-np.array(queryY))**2   )
    lprError=np.sqrt(   (np.array(lprX)-np.array(queryX))**2   +   (np.array(lprY)-np.array(queryY))**2   )
    imgMatcherror=np.sqrt(   (np.array(imgX)-np.array(queryX))**2    +    (np.array(imgY)-np.array(queryY))**2   )
    

    lprMag=np.sqrt(np.array(np.gradient(lprX))**2+np.array(np.gradient(lprY))**2)

    imgLowErrorIds=np.where(imgMatcherror<20)[0]
    lidarLowErrorIds=np.where(lprError<20)[0]
    lidarLowMagIds=np.where(lprMag<20)[0]
    canLowErrorIds=np.where(canError<20)[0]

    lidarHighMagIds=np.where(lprMag>20)[0]
    canHighErrorIds=np.where(canError>20)[0]
    lidarHighErrorIds=np.where(lprError>20)[0]
    imgHighErrorIds=np.where(imgMatcherror>20)[0]

    print(f'Num errors: {len(lprError)-len(lidarLowErrorIds)}, Recall: {(len(lidarLowErrorIds))/len(lprError)}, Total Distance Error: {np.sum(np.array(lprError))}')

    fig,(ax1,ax2, ax3)=plt.subplots(1,3)

    ax1.plot(scanAlignment, '.', color='tab:blue')
    ax1.plot(lidarHighErrorIds, scanAlignment[lidarHighErrorIds], 'rx')
    ax1.set_title('Alignability of Lidar (red: LPR error >20m)')

    ax2.plot(canVariences, '.', color='tab:purple')
    ax2.plot(canHighErrorIds, canVariences[canHighErrorIds], 'rx')
    ax2.set_title('Varience of CAN (red: CAN error>20m)')

    ax3.plot(imgDistances,'.',color='tab:pink')
    ax3.plot(imgHighErrorIds, imgDistances[imgHighErrorIds], "rx")
    ax3.set_title('SAD Distance of Images(red: VPR error>20m)')

    
    plt.show()


def plotSADMatch(results_filename):
    df = pd.read_excel(results_filename)
    refPositions, queryPositions, matchIds, matchPositions, matchShiftedPositions = df['referencePositions'], df['queryPositions'], df['matchID'], df['matchPosition'], df['matchShiftedPosition']
    scanAlignment, imgIds, imgPositions, imgDistances, canPeaks, canVariences = df['scanAlignment'], df['imgID'], df['imgPosition'], df['imgDistance'],df['canPeak'],df['canVarience']
    # canMag=df['canMagnitude']
    
    refX, refY=zip(*refPositions.apply(lambda x: (ast.literal_eval(x)[0], ast.literal_eval(x)[1])))
    queryX, queryY=zip(*queryPositions.apply(lambda x: (ast.literal_eval(x)[0], ast.literal_eval(x)[1])))
    lprX, lprY=zip(*matchPositions.apply(lambda x: (ast.literal_eval(x)[0], ast.literal_eval(x)[1])))
    lprShiftX, lprShiftY=zip(*matchShiftedPositions.apply(lambda x: (ast.literal_eval(x)[0], ast.literal_eval(x)[1])))
    canX, canY=zip(*canPeaks.apply(lambda x: (ast.literal_eval(x)[0], ast.literal_eval(x)[1])))
    imgY, imgX=zip(*imgPositions.apply(lambda x: (ast.literal_eval(x)[0], ast.literal_eval(x)[1])))

    canError=np.sqrt(    (np.array(canX)-np.array(queryX))**2    +   (np.array(canY)-np.array(queryY))**2   )
    lprShiftError=np.sqrt(   (np.array(lprShiftX)-np.array(queryX))**2   +   (np.array(lprShiftY)-np.array(queryY))**2   )
    lprError=np.sqrt(   (np.array(lprX)-np.array(queryX))**2   +   (np.array(lprY)-np.array(queryY))**2   )
    imgMatcherror=np.sqrt(   (np.array(imgX)-np.array(queryX))**2    +    (np.array(imgY)-np.array(queryY))**2   )
    

    lprMag=np.sqrt(np.array(np.gradient(lprX))**2+np.array(np.gradient(lprY))**2)

    imgLowErrorIds=np.where(imgMatcherror<20)[0]
    lidarLowErrorIds=np.where(lprError<20)[0]
    lidarLowMagIds=np.where(lprMag<20)[0]
    canLowErrorIds=np.where(canError<20)[0]

    lidarHighMagIds=np.where(lprMag>20)[0]

    print(f'Num errors: {len(imgMatcherror)-len(imgLowErrorIds)}, Recall: {(len(imgLowErrorIds))/len(imgMatcherror)}, Total Distance Error: {np.sum(np.array(imgMatcherror))}')

    fig,((ax1,ax2, ax3))=plt.subplots(1,3)

    
    ax1.plot(queryX, queryY, 'g.')
    ax1.plot([queryX,imgX], [queryY, imgY], color='tab:brown', linestyle='-', alpha=0.5)
    ax1.plot(imgX, imgY, '.', color='tab:pink')
    ax1.set_title('GT and Img Matched positions')

    ax2.plot(np.array(queryX)[imgLowErrorIds], np.array(queryY)[imgLowErrorIds], 'g.')
    ax2.plot(np.array(imgX)[imgLowErrorIds], np.array(imgY)[imgLowErrorIds], '.', color='tab:pink')
    ax2.plot([np.array(queryX)[imgLowErrorIds],np.array(imgX)[imgLowErrorIds]], [np.array(queryY)[imgLowErrorIds], np.array(imgY)[imgLowErrorIds]], color='tab:brown', linestyle='-', alpha=0.5)
    ax2.set_title('GT and Low Error Img Matched positions')

    ax3.hist(lprError, bins=5000, color='tab:blue',  histtype='step', stacked=True, fill=False, density=True)
    ax3.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax3.set_title('Distribution of match error')


    plt.show()


def plotAllOxford(dataType, toler):
    fig, axs = plt.subplots(5, 5, figsize=(10, 8))
    fig.tight_layout(pad=1)

    for i in range(1,6):
        for j in range(1,6):
            if i!=j:
                print(i,j)
                NetworkNum=6
                increment=20
                refNum, queryNum=i, j
                filename = f'./results/LPR_PosEstimation_CANFilter/Ref:{refNum}_Query:{queryNum}_Incr:{increment}_CANparam:{NetworkNum}_LPRparam:1.xlsx'
                rslt=extractResultsParam(filename, errTolerance=toler)
                queryX, queryY = rslt['queryX'], rslt['queryY']
                if dataType =='CAN':
                    canLowErrorIds=rslt['canLowErrorIds']
                    canError=rslt['canError']
                    X,Y =np.array(rslt['canX'])[canLowErrorIds], np.array(rslt['canY'])[canLowErrorIds]
                    print(canLowErrorIds)
                    axs[i-1, j-1].set_title('GT and CAN estimations')
                    axs[i-1, j-1].scatter(X, Y, s= canError[canLowErrorIds], color='skyblue', alpha=0.7)
                elif dataType == 'LPR':
                    lprError = np.array(rslt['lprError'])
                    lidarLowErrorIds = np.array(rslt['lidarLowErrorIds'])
                    X,Y = np.array(rslt['lprX'])[lidarLowErrorIds], np.array(rslt['lprY'])[lidarLowErrorIds]
                    axs[i-1, j-1].scatter(X, Y, s= lprError[lidarLowErrorIds], color='skyblue', alpha=0.7)
                elif dataType == 'LPRshift': 
                    lprShiftError = np.array(rslt['lprShiftError'])
                    lidarLowShiftErrorIds = np.array(rslt['lidarLowShiftErrorIds'])
                    lidarHighErrorIds = np.array(rslt['lidarHighErrorIds'])
                    X,Y = np.array(rslt['lprShiftX'])[lidarLowShiftErrorIds], np.array(rslt['lprShiftY'])[lidarLowShiftErrorIds]
                    badX,badY = np.array(rslt['lprShiftX'])[lidarHighErrorIds], np.array(rslt['lprShiftY'])[lidarHighErrorIds]
                    axs[i-1, j-1].scatter(X, Y, s=lprShiftError[lidarLowShiftErrorIds], color='skyblue')
                    axs[i-1, j-1].scatter(badX,badY, s=3, marker='*', c='r')
                    axs[i-1, j-1].set_title(f' Recall: {round(len(lidarLowShiftErrorIds)/len(lprShiftError),2)}')
                # axs[i-1, j-1].plot(queryX, queryY, 'g.')
                # axs[i-1, j-1].plot(X, Y, 'c.')
                # axs[i-1, j-1].plot([queryX,X], [queryY, Y], color='tab:brown', linestyle='-', alpha=0.5)
                
                
                
            else:
                axs[i-1, j-1].axis('off')


def plotAllJackal():
    fig, axs = plt.subplots(2, 4, figsize=(12, 10))

    for i in range(1,9,2):
        print(i,i+1)
        NetworkNum=6
        increment=10
        refNum, queryNum=i, i+1
        filename = f'./results/LPR_PosEstimation_CANFilter/Jackal_Ref:{refNum}_Query:{queryNum}_Incr:{increment}_CANparam:{NetworkNum}_LPRparam:1.xlsx'
        rslt=extractResultsParam(filename, errTolerance=2)
        lidarLowErrorIds = np.array(rslt['lidarLowErrorIds'])
        lidarLowShiftErrorIds = np.array(rslt['lidarLowShiftErrorIds'])
        queryX = np.array(rslt['queryX'])
        queryY = np.array(rslt['queryY'])
        lprShiftX = np.array(rslt['lprShiftX'])
        lprShiftY = np.array(rslt['lprShiftY'])
        lprX = np.array(rslt['lprX'])
        lprY = np.array(rslt['lprY'])
        
        axs[0,(i-1)//2].plot(queryX[lidarLowErrorIds], queryY[lidarLowErrorIds], '.', 'g')
        axs[0,(i-1)//2].plot(lprX[lidarLowErrorIds], lprY[lidarLowErrorIds], '.', color='skyblue')
        axs[0,(i-1)//2].plot([queryX[lidarLowErrorIds], lprX[lidarLowErrorIds]],
                [queryY[lidarLowErrorIds], lprY[lidarLowErrorIds]], color='tab:brown', linestyle='-', alpha=0.5)
        axs[0,(i-1)//2].set_title(f'Match Recall: {round(len(lidarLowErrorIds)/len(lprX),2)}') 

        axs[1,(i-1)//2].plot(queryX[lidarLowErrorIds], queryY[lidarLowErrorIds], 'g.')
        axs[1,(i-1)//2].plot(lprShiftX[lidarLowErrorIds], lprShiftY[lidarLowErrorIds],'.', color='yellowgreen')
        axs[1,(i-1)//2].plot([queryX[lidarLowErrorIds], lprShiftX[lidarLowErrorIds]],
                [queryY[lidarLowErrorIds], lprShiftY[lidarLowErrorIds]], color='tab:brown', linestyle='-', alpha=0.5)
        axs[1,(i-1)//2].set_title(f'Shifted Recall:{round(len(lidarLowShiftErrorIds)/len(lprX),2)}')


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
    runTime=np.array(df['timePerQuery'])
    Q1 = np.percentile(runTime, 25)
    Q3 = np.percentile(runTime, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    runTimeIQR = runTime[(runTime >= lower_bound) & (runTime <= upper_bound)]

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
        'incorrectShiftedIds':lidarHighShiftErrorIds,
        'correctMatchIds':lidarLowMatchErrorIds,
        'matchIds': matchIds,
        'closestIds': closestIds,
        'timePerQueryIQR': runTimeIQR,


    }

    return output#recall,  dists, closeDists[lidarLowShiftErrorIds], xr, yr, np.array(queryX)[lidarLowShiftErrorIds], np.array(queryY)[lidarLowShiftErrorIds], np.array(queryX)[lidarHighShiftErrorIds], np.array(queryY)[lidarHighShiftErrorIds]

    # fig, ( ax2) = plt.subplots(1, 1, figsize=(7,7))

    # name=results_filename.rsplit('/',1)[-1].rsplit('_',1)[:-1]   
    # ax2.axis('equal')
    # ax2.plot(xr,yr, 'k-', linewidth=0.6, alpha=0.7)
    # ax2.scatter(np.array(queryX)[lidarLowShiftErrorIds], np.array(queryY)[lidarLowShiftErrorIds], s=2,c='g')
    # # ax2.scatter(lprShiftX[lidarLowShiftErrorIds], lprShiftY[lidarLowShiftErrorIds], s=100 * lprShiftError[lidarLowShiftErrorIds], color='yellowgreen', alpha=0.7)
    # # ax2.plot([queryX[lidarLowShiftErrorIds], lprShiftX[lidarLowShiftErrorIds]], [queryY[lidarLowShiftErrorIds], lprShiftY[lidarLowShiftErrorIds]], color='tab:brown', linestyle='-', alpha=0.5)
    # ax2.scatter(np.array(queryX)[lidarHighShiftErrorIds], np.array(queryY)[lidarHighShiftErrorIds], s=1,c='r', alpha=0.7)
    # ax2.set_title(f'LPR accuracy: {name}')


def updateIncorrectMatches(i, errTolerance=3):
    with open('./scripts/config.json') as f:
        config = json.load(f)
    config=config.get(datasetName, {})

    with open(config.get('evalInfoFile'), 'rb') as f:
        evalInfo = pickle.load(f)[queryNum-1]

    queryFilenames = config.get('details', {}).get(str(queryNum), [])
    all_files = os.listdir(queryFilenames[0])
    framesQuery = []
    for k in range(len(evalInfo)):
        try:
            index = all_files.index(evalInfo[k]['query'].split('/')[-1])
            framesQuery.append(index)
        except ValueError:
            pass

    results_filename = f'./results/LPR_PosEstimation/{envName}_Ref:{refNum}_Query:{queryNum}_Inc:1_Res:0.4_downSampHalf.xlsx'                    # Jackal 
    df = pd.read_excel(results_filename)
    queryPositions, matchIds, matchPositions = df['queryPose'], df['matchID'], df['matchPose'],
    matchShiftedPositions= df['matchShiftedPosition']
    dists=df['dist']
    closestIds=df['closestId']
    heading=df['peakRotation']
    shiftAmount=df['shiftAmount']

    queryX, queryY=zip(*queryPositions.apply(lambda x: (ast.literal_eval(x)[0], ast.literal_eval(x)[1])))
    lprShiftX, lprShiftY=zip(*matchShiftedPositions.apply(lambda x: (ast.literal_eval(x)[0], ast.literal_eval(x)[1])))
    matchXDelta, matchYDelta=zip(*shiftAmount.apply(lambda x: (ast.literal_eval(x)[0], ast.literal_eval(x)[1])))
    lprShiftError=np.sqrt(   (np.array(lprShiftX)-np.array(queryX))**2   +   (np.array(lprShiftY)-np.array(queryY))**2   )
    lidarLowShiftErrorIds=np.where(lprShiftError<errTolerance)[0]
    lidarHighShiftErrorIds=np.where(lprShiftError>errTolerance)[0]

    i=lidarHighShiftErrorIds[i]
    print(i)
    LPR=LiDAR_PlaceRec(datasetName, config, refNum, queryNum, framesQuery ) 
    xq,yq,rollQ, pitchQ, yawQ=LPR.scanPoses('QUERY')
    xr,yr,rollR, pitchR,yawR=LPR.scanPoses('REF')
    closeIds=[find_closest_point(zip(xr,yr), xq[i], yq[i])[2] for i in range(len(xq)) ]
    minDist=[find_closest_point(zip(xr,yr), xq[i], yq[i])[3] for i in range(len(xq))]

    refFilenames = config.get('details', {}).get(str(refNum), [])
    refIncr = config.get('refIncr')
    param = config.get('parameters', {})
    scanDimX = param.get('scanDimX', 0)
    scanDimY = param.get('scanDimY', 0)
    mapRes = param.get('mapRes', 0.0)
    refGridFilePath=refFilenames[0].rsplit('/', 1)[:-1][0] +f'/refGrid_incr{refIncr}_dim{scanDimX}_mapRes{mapRes}_downSampHalf.npy'
    refgrid=np.load(refGridFilePath)
    centerXYs=LPR.scanCenter()
    centerXs,centerYs=zip(*centerXYs)
    centerX, centerY = centerXs[i], centerYs[i]


    queryScan= LPR.loadingCroppingFiltering_2DScan('QUERY', i, rotDeg=heading[i])
    refConv= refgrid[centerY-(scanDimY//2):centerY+(scanDimY//2), centerX-(scanDimX//2):centerX+(scanDimX//2)]
    refClosest= refgrid[centerYs[closeIds[i]]-(scanDimY//2):centerYs[closeIds[i]]+(scanDimY//2), centerXs[closeIds[i]]-(scanDimX//2):centerXs[closeIds[i]]+(scanDimX//2)]
    refShiftMatchScan= LPR.translateScan(refConv, matchXDelta[i], matchYDelta[i])

    ax11.clear(), ax12.clear(), ax13.clear()
            
    ax11.imshow(queryScan, cmap='Blues')
    ax11.set_title(f'Query:{i}')
    ax11.set_ylabel('Top Down View')

    ax12.imshow(queryScan, cmap='Blues', alpha=1)
    ax12.imshow(refClosest ,  cmap='Greens', alpha=0.5)
    ax12.set_title(f'Close Match dist: {round(minDist[i],1)}')

    ax13.imshow(queryScan, cmap='Blues', alpha=1)
    ax13.imshow(refShiftMatchScan, cmap='Reds', alpha=0.5)
    ax13.set_title(f'True Match dist: {round(dists[i],1)}')




'''Oxford radar'''
# refNum, queryNum = 1,4
# increment=20
# filename=f'./results/LPR_PosEstimation_CANFilter/OldRelativeShiftTests/Ref:{refNum}_Query:{queryNum}_Incr:20_CANparam:6_LPRparam:1_ImgUse:False.xlsx'   # Old oxford
# filename=f'./results/LPR_PosEstimation_CANFilter/Ref:{refNum}_Query:{queryNum}_Incr:20_CANparam:6_LPRparam:1.xlsx'                                        # Oxford 

# plotRelativeShift(filename, toler=20)
# plotRelativeShiftStats(filename, toler=20)
# plotObservingBadShifts(filename, toler=20)

# plotCANStats(filename, toler=20)
# plot(filename)
# plotConvolutionMatch(filename)
# plotCANFilter(filename)
# plotMetrics(filename)
# plotSADMatch(filename)

# plotAllOxford(dataType='CAN', toler=20)
# plotAllOxford(dataType='LPR', toler=20)


'''Jackal'''
# refNum, queryNum = 3,4
# filename = f'./results/LPR_PosEstimation_CANFilter/Jackal_Ref:{refNum}_Query:{queryNum}_Incr:10_CANparam:6_LPRparam:1.xlsx'                    # Jackal 
# plotRelativeShift(filename, toler=2)
# plotRelativeShiftStats(filename, toler=2)
# plotAllJackal()

# plt.show()
# plt.savefig('./result2.png', dpi=900)


'''WildPlaces'''
# # Create a figure and axis objects
# fig, axes = plt.subplots(4, 3, figsize=(10, 10))
# envName='Venman'
# # Loop through each subplot
# rowsIds= [[2,3,4], [1,3,4], [1,2,4], [1,2,3]]
# totalRecall=0
# for i in range(4):
#     for j in range(3):
#         ax = axes[i, j]
#         recall, dists, closeDists, xr, yr, corrX, corrY, incorrX, incorrY = plotWildplaces(envName, i+1, rowsIds[i][j], './results', 3)
#         totalRecall += recall

#         '''plot correct and incorrect queries'''
#         # ax.plot(xr,yr, 'k--', linewidth=0.2, alpha=0.7)
#         # ax.plot(incorrX, incorrY, 'r.', markersize=0.2, alpha=1)
#         # ax.plot(corrX, corrY, 'g.', markersize=0.2, alpha=0.5)  
#         # ax.set_title(f'{i+1},{rowsIds[i][j]}, recall: {round(recall,2)}')

#         ax.bar(np.arange(len(dists)), dists)
#         ax.bar(np.arange(len(closeDists)), closeDists)

#         print(i+1, rowsIds[i][j], 'recall:', recall)
#         # print(lidarHighShiftErrorIds)
#         ax.axis('equal')

# print('avgRecall:' , totalRecall/12)
# plt.tight_layout()
# plt.show()
# plt.savefig(f'result_{envName}.png')

def extractAllResults(numRefTraverse=4, numqueryTraverse=4, datasetName='Wildplaces',envName='Venman', mapRes=0.3, scanDimX=120, refIncr=1, queryIncr=10, refRad=2, blockSize=2, NumMatches=2, numPtsMultiplier=2, background=-0.15):
    # fig, (ax1) = plt.subplots(1,1)
    # recall, dists, closeDists, xr, yr, corrX, corrY, incorrX, incorrY = plotWildplaces('Karawatha', 3,1, './results', 3)
    # ax1.plot(dists, '.')
    distsALL, closeDistsALL, totalRecall, count=[],[], 0,0
    for i in range(1,numRefTraverse+1):
        for j in range(1,numqueryTraverse+1):
            if i!=j:
                # try: 
                results_dir=f'./results/LPR_{datasetName}/Ablate_Res:{mapRes}_Dim:{scanDimX}_RInc:{refIncr}_Rrad:{refRad}'
                results_filename=f'/{envName}_R:{i}_Q:{j}_RInc:{refIncr}_Rrad:{refRad}_QInc:{queryIncr}_Res:{mapRes}_Dim:{scanDimX}_blkAvg:{blockSize}_N2ndsearch:{NumMatches}.xlsx'
                refGridFilename=f"/refGrid_incr{refIncr}_dim{scanDimX}_mapRes{mapRes}_refIncr{refIncr}_refRad2_rThresh2_nptsMult2_AvgDownsamp.npy"
                output= extractSingleReuslt(f'{envName}', i,j,results_dir+results_filename, refGridFilename, errTolerance=3)
                recall=output['recall']
                print(i,j, 'recall:', recall )
                totalRecall+=recall
                count+=1
                print('Avg_recall:', totalRecall/count)
                print('')
                # except Exception as e:
                #     continue 
                # distsALL+=list(dists)
                # closeDistsALL+=list(closeDists)
                # print(len(distsALL))


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
    kwargs = dict(histtype='step', alpha=1, bins=60, linewidth=1.5)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(matchErrorsAll, **kwargs, ec='blue', label='Matches before relative shift')
    ax.hist(shiftErrorsAll, **kwargs, ec='magenta', label='Matches after relative shift')
    ax.set_xlabel('Position Error [m]', fontsize=20)
    ax.set_ylabel('Density of Distribution', fontsize=20)
    handles, labels = ax.get_legend_handles_labels()
    new_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles]
    ax.legend(new_handles, labels, fontsize=18)
    # ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=18)
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
    ax1.set_xlabel('Recall@1', fontsize=19)
    ax1.set_xticks(np.arange(10, 110, 10))
    ax1.set_xlim([15, 105])
    ax1.grid(True, which='both', linestyle='--', alpha=0.4)
    ax1.tick_params(axis='both', which='major', labelsize=19)
    ax1.set_title('Venman', fontsize=20, fontweight='bold')

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
    ax2.set_xlabel('Recall@1', fontsize=19)
    ax2.set_xticks(np.arange(10, 110, 10))
    ax2.set_xlim([15, 105])
    ax2.grid(True, which='both', linestyle='--', alpha=0.4)
    ax2.tick_params(axis='both', which='major', labelsize=19)
    ax2.set_title('Karawatha', fontsize=20, fontweight='bold')

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[0], markeredgecolor='k', markersize=14, label='Same Day'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=colors[1], markeredgecolor='k', markersize=14, label='6 Months'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=colors[2], markeredgecolor='k', markersize=14, label='14 Months')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=18, bbox_to_anchor=(0.56, -0.02))
    plt.tight_layout(rect=[0, 0.07, 1, 1])  # Adjust the rect parameter to make space at the bottom
    plt.savefig('./paperFigures/WildplacesRecallOverTime.png')
    plt.savefig('./paperFigures/WildplacesRecallOverTime.pdf')


def plotClosestId_matchId(numRefTraverse=4, numqueryTraverse=4, datasetName='Wildplaces',mapRes=0.3, scanDimX=120, refIncr=1, queryIncr=1, refRad=2, blockSize=2, NumMatches=1):
    match, closest = [], []
    envName, i, j = 'Venman', 3,4
    results_dir=f'./results/LPR_{datasetName}/Ablate_Res:{mapRes}_Dim:{scanDimX}_RInc:{refIncr}_Rrad:{refRad}'
    results_filename=f'/{envName}_R:{i}_Q:{j}_RInc:{refIncr}_Rrad:{refRad}_QInc:{queryIncr}_Res:{mapRes}_Dim:{scanDimX}_blkAvg:{blockSize}.xlsx'
    refGridFilename=f"/refGrid_incr{refIncr}_dim{scanDimX}_mapRes{mapRes}_refIncr{refIncr}_refRad2_rThresh2_nptsMult2_AvgDownsamp.npy"
    output= extractSingleReuslt(f'{envName}', i,j,results_dir+results_filename, refGridFilename, errTolerance=3)
    match+=[output['matchIds']]
    closest+=[output['closestIds']] #[id] for id in output['correctShiftedIds']
    
    
    maxIds=np.argpartition(np.abs(np.array(match)-np.array(closest)), -50)[-50:]
    diffValues=np.abs(np.array(match)-np.array(closest))
    indxLessCond=np.where(diffValues==3)
    print(np.array(closest)[indxLessCond])
    print(np.array(match)[indxLessCond])
    print(indxLessCond)
    print(list(   diffValues[indxLessCond])   )

    # kwargs = dict(histtype='step', alpha=1, bins=60, density=True, linewidth=1.5)
    # fig, ax = plt.subplots(figsize=(7, 5))
    # ax.stem(match,'g', markerfmt='go',label='MatchId')
    # ax.stem(closest,label='ClosestId')
    # ax.set_xlabel('Position Error [m]', fontsize=14)
    # ax.set_ylabel('Density of Distribution', fontsize=14)
    # ax.legend(fontsize=14)
    # # ax.grid(True, linestyle='--', alpha=0.6)
    # ax.tick_params(axis='both', which='major', labelsize=12)
    # plt.tight_layout()
    # # plt.show()
    # plt.savefig('./result.png')
    # plt.savefig('./paperFigures/DistributionOfErrorsRelativeShift.pdf')


def recallOverRotations(numRefTraverse=4, numqueryTraverse=4, datasetName='Wildplaces', mapRes=0.3, scanDimX=120, refIncr=1, queryIncr=10, refrad=2, blockSize=2,  NumMatches=1):
    recallListV, recallListK = [], []
    rotDegs=[1,5,10,20,30]

    envName='Venman' 
    for rot in rotDegs:
        count, currRecall=0, 0
        for i in range(1,numRefTraverse+1):
            for j in range(1,numqueryTraverse+1):
                if i!=j:
                    refNum, queryNum = i,j
                    saveFolder=f'./results/LPR_Wildplaces/Ablate_VariousRot_Res:{mapRes}_Dim:{scanDimX}_RInc:{refIncr}_Rrad:{refrad}'
                    if not os.path.exists(saveFolder):
                        os.makedirs(saveFolder)
                    refNpyName=f"/refGrid_incr{refIncr}_dim{scanDimX}_mapRes{mapRes}_refIncr{refIncr}_refRad{refrad}_rThresh2_nptsMult2_AvgDownsamp.npy"
                    savePath = saveFolder+f'/{envName}_R:{refNum}_Q:{queryNum}_RInc:{refIncr}_Rrad:{refrad}_QInc:{queryIncr}_Res:{mapRes}_Dim:{scanDimX}_blkAvg:{blockSize}_N2ndsearch:{NumMatches}_rotDeg{rot}.xlsx'
                    output= extractSingleReuslt(f'{envName}', i,j,savePath, refNpyName, errTolerance=3)
                    currRecall+=output['recall']
                    count+=1
                    print(output['recall'])
        recallListV.append((currRecall/count)*100)
    print(recallListV)
    

def recallwithNegativeBackground(numRefTraverse=4, numqueryTraverse=4, datasetName='Wildplaces', mapRes=0.3, scanDimX=120, refIncr=1, queryIncr=20, refrad=2, blockSize=2,  NumMatches=1):
    recallListV, recallListK = [], []
    backgroundVals=[0,-0.1,-0.25,-0.5,-1]

    envName='Venman' 
    for val in backgroundVals:
        
        count, currRecall=0, 0
        for i in range(1,numRefTraverse+1):
            for j in range(1,numqueryTraverse+1):
                if i!=j:
                    refNum, queryNum = i,j
                    saveFolder=f'./results/LPR_Wildplaces/Ablate_NegativeBackground_Res:{mapRes}_Dim:{scanDimX}_RInc:{refIncr}_Rrad:{refrad}'
                    if not os.path.exists(saveFolder):
                        os.makedirs(saveFolder)
                    refNpyName=f"/refGrid_incr{refIncr}_dim{scanDimX}_mapRes{mapRes}_refIncr{refIncr}_refRad{refrad}_rThresh2_nptsMult2_AvgDownsamp.npy"
                    savePath = saveFolder+f'/{envName}_R:{refNum}_Q:{queryNum}_RInc:{refIncr}_Rrad:{refrad}_QInc:{queryIncr}_Res:{mapRes}_Dim:{scanDimX}_blkAvg:{blockSize}_N2ndsearch:{NumMatches}_background{val}_nptsMult2.xlsx'
                    output= extractSingleReuslt(f'{envName}', i,j,savePath, refNpyName, errTolerance=3)
                    currRecall+=output['recall']
                    count+=1
                    print(output['recall'])
        recallListV.append((currRecall/count)*100)
        print(recallListV)

def extractResultNCLT(refNum,queryNum,mapRes=1, scanDimX=120, refIncr=1, queryIncr=1, refRad=2, queryRad=5, blockSize=2, NumMatches=2, background=-0.15, refThresh=1, dim_randDwnsmple=10, numPtsMultiplier=2, errToler=25,variationToName=None):
    datasetName='NCLT'
    # results_dir=f'./results/LPR_{datasetName}/Res:{mapRes}_Dim:{scanDimX}_RInc:{refIncr}_Rrad:{refRad}'
    # results_filename=f'/{datasetName}_R:{refNum}_Q:{queryNum}_RInc:{refIncr}_Rrad:{refRad}_QInc:{queryIncr}_Qrad:{queryRad}_Res:{mapRes}_Dim:{scanDimX}_blkAvg:{blockSize}_N2ndsearch:{NumMatches}_background:{background}.xlsx' #
    # refGridFilename=f"/refGrid_incr{refIncr}_dim{scanDimX}_mapRes{mapRes}_refIncr{refIncr}_refRad2_rThresh2_nptsMult2_AvgDownsamp_noReverse.npy"#
    # refGridFilename=f"/refGrid_dim{scanDimX}_mapRes{mapRes}_refIncr{refIncr}_refRad{refRad}_rThresh{refThresh}_patchDim{dim_randDwnsmple}_nptsMult{numPtsMultiplier}_noReverse.npy"
    
    refGridFilename=f"/refGrid_dim{scanDimX}_mapRes{mapRes}_refIncr{refIncr}_refRad{refRad}_rThresh{refThresh}_patchDim{dim_randDwnsmple}_nptsMult{numPtsMultiplier}.npy"
    # saveFolder=f'./results/LPR_{datasetName}/Res:{mapRes}_Dim:{scanDimX}_RInc:{refIncr}_Rrad:{refRad}'
    saveFolder= f'./results/LPR_{datasetName}/SameHyperParam'
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
    
    for j in range(1,6):
        for i in [6,7]:
            count+=1  
            datasetName='OxfordRadar'
            mapRes=1.0
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
            results_dir=f'./results/LPR_{datasetName}/SameHyperParam'
            results_filename=f'/{datasetName}_R:{i}_Q:{j}_RInc:{refIncr}_Rrad:{refRad}_QInc:{queryIncr}_Qrad:{queryRad}_Res:{mapRes}_Dim:{scanDimX}_blkAvg:{blockSize}_N2ndsearch:{NumMatches}_background:{background}.xlsx'
            refGridFilename=refNpyName=f"/refGrid_dim{scanDimX}_mapRes{mapRes}_refIncr{refIncr}_refRad{refRad}_rThresh{refThresh}_patchDim{dim_randDwnsmple}_nptsMult{nMultiplier_randDwnsmpl}.npy"
            output= extractSingleReuslt(results_dir+results_filename, errTolerance=25)
            print(i,j, output['recall'])
            totRecall+=output['recall']
            print(f'average recall: {totRecall/count}')  

def recallWildplaces():
    for envName in ['Venman', 'Karawatha']:
        totRecall,count=0,0
        for i in range(1,5):
            for j in range(1,5):
                if i!=j: 
                    count+=1  
                    datasetName=f'WildPlaces_{envName}'
                    mapRes=0.3
                    scanDimX=120
                    refIncr=1
                    queryIncr=10
                    refRad=2
                    blockSize=2
                    NumMatches=2
                    numPtsMultiplier=2
                    background=-0.15
                    results_dir=f'./results/LPR_Wildplaces/SameHyperParam'
                    results_filename=f'/{envName}_R:{i}_Q:{j}_RInc:{refIncr}_Rrad:{refRad}_QInc:{queryIncr}_Res:{mapRes}_Dim:{scanDimX}_blkAvg:{blockSize}_N2ndsearch:{NumMatches}_nptsMult{numPtsMultiplier}_background{background}.xlsx'
                    
                    output= extractSingleReuslt(results_dir+results_filename, errTolerance=3)
                    print(f"ref:{i}, query:{j}, recall: {output['recall']}")
                    totRecall+=output['recall']
        print(f'{envName} avg recall: {totRecall/count}')  

def ablationNaturalResultsPlot():
   
    saveFolder=f'./results/LPR_Wildplaces/Ablate_TrainingSet_TimeIncl_1'
    refNum=2
    queryNum=3
    refIncr=1
    refRad=2
    queryIncr=10
    mapRes=0.3
    scanDimX=120
    blockSize=2
    n=2
    rotInc=10
    background=-0.15
    numPtsMultiplier=2
    z_max=3
    top_n_recall = []
    top_n_runtime=[]
    top_n_values = [1,2,5,8]
    datasetName="WildPlaces_Venman"
    for top_n in top_n_values:
        savePath= saveFolder+f'/{datasetName}_R:{refNum}_Q:{queryNum}_RInc:{refIncr}_Rrad:{refRad}_QInc:{queryIncr}_Res:{mapRes}_Dim:{scanDimX}_blkAvg:{blockSize}_N2ndsearch:{top_n}_rotInc{rotInc}_UnoccWeight{background}_nptsMult{numPtsMultiplier}_zMax{z_max}.xlsx'

        output = extractSingleReuslt(savePath, errTolerance=3)
        recall = output['recall']
        avgRuntime = round(np.mean(output['timePerQueryIQR']),2)
        top_n_runtime.append(avgRuntime)
        top_n_recall.append(recall)
    
    unocc_weight_recall = []
    unocc_weight_runtime=[]
    unocc_weight_values = [0, -0.05, -0.1, -0.15, -0.2, -0.25, -0.3]
    
    for val in unocc_weight_values:
        savePath= saveFolder+f'/{datasetName}_R:{refNum}_Q:{queryNum}_RInc:{refIncr}_Rrad:{refRad}_QInc:{queryIncr}_Res:{mapRes}_Dim:{scanDimX}_blkAvg:{blockSize}_N2ndsearch:{n}_rotInc{rotInc}_UnoccWeight{val}_nptsMult{numPtsMultiplier}_zMax{z_max}.xlsx'
        output = extractSingleReuslt(savePath, errTolerance=3)
        recall = output['recall']
        avgRuntime = round(np.mean(output['timePerQueryIQR']),2)
        unocc_weight_runtime.append(avgRuntime)
        unocc_weight_recall.append(recall)
    
    pool_size_recall = []
    pool_size_runtime=[]
    pool_size_values = [1, 2, 3, 4, 5]
    
    for blksz in pool_size_values:
        savePath= saveFolder+f'/{datasetName}_R:{refNum}_Q:{queryNum}_RInc:{refIncr}_Rrad:{refRad}_QInc:{queryIncr}_Res:{mapRes}_Dim:{scanDimX}_blkAvg:{blksz}_N2ndsearch:{n}_rotInc{rotInc}_UnoccWeight{background}_nptsMult{numPtsMultiplier}_zMax{z_max}.xlsx'
        output = extractSingleReuslt(savePath, errTolerance=3)
        recall = output['recall']
        avgRuntime = round(np.mean(output['timePerQueryIQR']),2)
        pool_size_runtime.append(avgRuntime)
        pool_size_recall.append(recall)
    
    rot_incr_recall = []
    rot_incr_runtime=[]
    rot_incr_values = [1, 5, 10, 20, 30]
    
    for rot in rot_incr_values:
        savePath= saveFolder+f'/{datasetName}_R:{refNum}_Q:{queryNum}_RInc:{refIncr}_Rrad:{refRad}_QInc:{queryIncr}_Res:{mapRes}_Dim:{scanDimX}_blkAvg:{blockSize}_N2ndsearch:{n}_rotInc{rot}_UnoccWeight{background}_nptsMult{numPtsMultiplier}_zMax{z_max}.xlsx'
        output = extractSingleReuslt(savePath, errTolerance=3)
        recall = output['recall']
        avgRuntime = round(np.mean(output['timePerQueryIQR']),2)
        rot_incr_runtime.append(avgRuntime)
        rot_incr_recall.append(recall)
    

    numPtsMultipliers_recall = []
    numPtsMultipliers_runtime=[]
    numPtsMultipliers=[0.1,0.5,0.75,1.5,2,4,6,8,10]
    
    for n_mult in numPtsMultipliers:
        savePath= saveFolder+f'/{datasetName}_R:{refNum}_Q:{queryNum}_RInc:{refIncr}_Rrad:{refRad}_QInc:{queryIncr}_Res:{mapRes}_Dim:{scanDimX}_blkAvg:{blockSize}_N2ndsearch:{n}_rotInc{rotInc}_UnoccWeight{background}_nptsMult{n_mult}_zMax{z_max}.xlsx'
        output = extractSingleReuslt(savePath, errTolerance=3)
        recall = output['recall']
        avgRuntime = round(np.mean(output['timePerQueryIQR']),2)
        numPtsMultipliers_runtime.append(avgRuntime)
        numPtsMultipliers_recall.append(recall)
    

    z_maxs_recall = []
    z_maxs_runtime=[]
    z_maxs=[2,3,6,8,10,15,20]
    
    for z in z_maxs:
        savePath= saveFolder+f'/{datasetName}_R:{refNum}_Q:{queryNum}_RInc:{refIncr}_Rrad:{refRad}_QInc:{queryIncr}_Res:{mapRes}_Dim:{scanDimX}_blkAvg:{blockSize}_N2ndsearch:{n}_rotInc{rotInc}_UnoccWeight{background}_nptsMult{numPtsMultiplier}_zMax{z}.xlsx'
        output = extractSingleReuslt(savePath, errTolerance=3)
        recall = output['recall']
        avgRuntime = round(np.mean(output['timePerQueryIQR']),2)
        z_maxs_runtime.append(avgRuntime)
        
        z_maxs_recall.append(recall)
    

    # Combine all runtime values into a single array to normalize across all plots
    all_runtimes = np.concatenate([top_n_runtime, unocc_weight_runtime, pool_size_runtime, rot_incr_runtime, numPtsMultipliers_runtime, z_maxs_runtime])

    cmap = plt.cm.jet  # Or another colormap of your choice
    # Create a single ScalarMappable for a global colorbar
    from matplotlib.colors import LogNorm, PowerNorm
    norm = PowerNorm(gamma=0.4, vmin=np.min(all_runtimes), vmax=np.max(all_runtimes))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Required for the ScalarMappable

 

    # Plotting the recall values with 3x2 subplots
    fig, axs = plt.subplots(3, 2, figsize=(10, 8), constrained_layout=True)

    # Number of Top Matches (n)
    arrow_x = top_n_values[1]
    arrow_y = top_n_recall[1]
    axs[0, 0].scatter(arrow_x, arrow_y, marker='*', alpha=1, s=250, edgecolors='black')
    axs[0, 0].plot(top_n_values, top_n_recall, '-',color='grey',alpha=0.5)
    scatter = axs[0, 0].scatter(top_n_values, top_n_recall, c=top_n_runtime, cmap=cmap,norm=norm, marker='o', alpha=1, s=50)
    axs[0, 0].set_xlabel('Number of Top Matches (n)', fontsize=18)
    # annotate('', xy=(arrow_x, arrow_y),xytext=(arrow_x, 0), arrowprops=dict(edgecolor='k', arrowstyle='->', lw=1.5))

    # Unoccupied Cell Weighting (w)
    arrow_x = unocc_weight_values[3]
    arrow_y = unocc_weight_recall[3]
    axs[1, 0].scatter(arrow_x, arrow_y, marker='*', alpha=1, s=250, edgecolors='black')
    axs[1, 0].plot(unocc_weight_values, unocc_weight_recall, '-',color='grey',alpha=0.5)
    scatter = axs[1, 0].scatter(unocc_weight_values, unocc_weight_recall, c=unocc_weight_runtime, cmap=cmap,norm=norm, marker='o', alpha=1, s=50)
    axs[1, 0].set_xlabel('Unoccupied Cell Weighting (w)', fontsize=18)
    # annotate('', xy=(arrow_x, arrow_y),xytext=(arrow_x, 0), arrowprops=dict(edgecolor='k', arrowstyle='->', lw=1.5))

    # Average Pooling Size (u)
    arrow_x = pool_size_values[1]
    arrow_y = pool_size_recall[1]
    axs[0, 1].scatter(arrow_x, arrow_y, marker='*', alpha=1, s=250, edgecolors='black')
    axs[0,1].plot(pool_size_values, pool_size_recall,  '-',color='grey',alpha=0.5)
    scatter = axs[0,1].scatter(pool_size_values, pool_size_recall, c=pool_size_runtime, cmap=cmap,norm=norm,  marker='o', alpha=1, s=50)
    axs[0, 1].set_xlabel('Average Pooling Size (u)', fontsize=18)
    # annotate('', xy=(arrow_x, arrow_y),xytext=(arrow_x, 0), arrowprops=dict(edgecolor='k', arrowstyle='->', lw=1.5))

    # Rotation Increments (k)
    arrow_x = rot_incr_values[2]
    arrow_y = rot_incr_recall[2]
    axs[1, 1].scatter(arrow_x, arrow_y, marker='*', alpha=1, s=250, edgecolors='black')
    axs[1, 1].plot(rot_incr_values, rot_incr_recall,  '-',color='grey',alpha=0.5)
    scatter = axs[1, 1].scatter(rot_incr_values, rot_incr_recall, c=rot_incr_runtime, cmap=cmap,norm=norm, marker='o', alpha=1, s=50)
    axs[1, 1].set_xlabel('Rotation Increments (k)', fontsize=18)
    # annotate('', xy=(arrow_x, arrow_y),xytext=(arrow_x, 0), arrowprops=dict(edgecolor='k', arrowstyle='->', lw=1.5))

    # numPtsMultipliers recall
    arrow_x = np.array(numPtsMultipliers[4])*10
    arrow_y = numPtsMultipliers_recall[4]
    axs[2, 0].scatter(arrow_x, arrow_y, marker='*', alpha=1, s=250, edgecolors='black')
    axs[2, 0].plot(np.array(numPtsMultipliers)*10, numPtsMultipliers_recall, '-',color='grey',alpha=0.5)
    scatter = axs[2, 0].scatter(np.array(numPtsMultipliers)*10, numPtsMultipliers_recall, c=numPtsMultipliers_runtime, cmap=cmap,norm=norm,  marker='o', alpha=1, s=50)
    axs[2, 0].set_xlabel('Maximum 10x10 Patch Occupancy (c)', fontsize=18)
    
    # axs[2, 0].annotate('', xy=(arrow_x, arrow_y),xytext=(arrow_x, 0), arrowprops=dict(edgecolor='k', arrowstyle='->', lw=1.5))
    

    # z_maxs recall
    arrow_x = z_maxs[1]
    arrow_y = z_maxs_recall[1]
    axs[2, 1].scatter(arrow_x, arrow_y, marker='*', alpha=1, s=250, edgecolors='black')
    axs[2, 1].plot(z_maxs, z_maxs_recall, '-',color='grey',alpha=0.5)
    scatter = axs[2, 1].scatter(z_maxs, z_maxs_recall, c=z_maxs_runtime, cmap=cmap,norm=norm,  marker='o', alpha=1, s=50)
    axs[2, 1].set_xlabel('Maximum Z Range (z_max)', fontsize=18)
    # annotate('', xy=(arrow_x, arrow_y),xytext=(arrow_x, 0), arrowprops=dict(edgecolor='k', arrowstyle='->', lw=1.5))
    for ax in axs.flat:
        ax.set(ylabel='Recall')
        ax.yaxis.label.set_size(18)
        ax.set_ylim([0,1])
        ax.grid(True, which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.1)
        ax.grid(True, which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=0.1)
        ax.tick_params(axis='both', which='major', labelsize=18)

    cbar = fig.colorbar(sm, ax=axs, orientation='vertical', fraction=0.04, pad=0.06)
    cbar.set_label('Average Runtime (sec)', fontsize=18)
    cbar.ax.tick_params(labelsize=16)

    # plt.tight_layout(pad=1.3)
    plt.savefig('ablationWildPlacesWithTime.png')
    print('top_n',top_n_values, top_n_runtime, top_n_recall)
    print('')
    print('unocc_weight',unocc_weight_values,unocc_weight_runtime, unocc_weight_recall)
    print('')
    print('pool_size',pool_size_values,pool_size_runtime, pool_size_recall)
    print('')
    print('rot_incr',rot_incr_values,rot_incr_runtime, rot_incr_recall)
    print('')
    print('numMultiplier',numPtsMultipliers,numPtsMultipliers_runtime, numPtsMultipliers_recall)
    print('')
    print('zmaxs',z_maxs,z_maxs_runtime, z_maxs_recall)
    print('')
    # plt.suptitle('Recall Values for Different Parameters')

    plt.savefig('./paperFigures/ablateParameterSensitivity.png',dpi=200)
    plt.savefig('./paperFigures/ablateParameterSensitivity.pdf',dpi=200)
    # plt.show()

def ablationDownSampleResultsPlot():
    numPtsMultipliers_recall = []
    numPtsMultipliers_incorrectIds=[]
    numPtsMultipliers_incorrectMatches=[]
    numPtsMultipliers = [0.1,0.5,0.75,1.5,2,4,6,8,10]
    usedParamIdx, noDownSampIdx=4,8
    for n in numPtsMultipliers:
        savePath = f'./results/LPR_Wildplaces/Ablate_TrainingSet_TimeIncl_1/WildPlaces_Venman_R:2_Q:3_RInc:1_Rrad:2_QInc:10_Res:0.3_Dim:120_blkAvg:2_N2ndsearch:2_rotInc10_UnoccWeight-0.15_nptsMult{n}_zMax3.xlsx'
        output = extractSingleReuslt(savePath, errTolerance=3)
        recall = output['recall']
        numPtsMultipliers_recall.append(recall)
        numPtsMultipliers_incorrectMatches.append(output['matchIds'][output['incorrectShiftedIds']])
        numPtsMultipliers_incorrectIds.append(output['incorrectShiftedIds'])
        print(output['matchIds'][output['incorrectShiftedIds']])
    

    # savePath = f'./results/LPR_Wildplaces/Ablate_TrainingSet/Venman_R:2_Q:3_RInc:1_Rrad:2_QInc:10_Res:0.3_Dim:120_orderedPatchDwnSample.xlsx'
    # orderedPatch = extractSingleReuslt(savePath, errTolerance=3)
    numPtsMultipliers = [1,5,7.5,15,20,40,60,80,100]
    # Plotting the recall values with 2x2 subplots
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8, 5))
    
    ax.plot(numPtsMultipliers, numPtsMultipliers_recall, marker='o', color='tab:blue')
    ax.scatter(numPtsMultipliers[usedParamIdx], numPtsMultipliers_recall[usedParamIdx], color='palevioletred', marker='.', zorder=5, s=100)
    ax.scatter(numPtsMultipliers[noDownSampIdx], numPtsMultipliers_recall[noDownSampIdx], color='seagreen', marker='.', zorder=5, s=100)
    ax.set_xlabel('Maximum Points per Patch (c)', fontsize=14)
    ax.grid(True)
    # ax.set_xlim([0,10.2])
    

    ax.set(ylabel='Recall')
    ax.yaxis.label.set_size(14)
    ax.set_ylim([0,1])
    ax.grid(True, which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.1)
    ax.grid(True, which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=0.1)


    ax.tick_params(axis='both', which='major', labelsize=12)

    print('top_n_recall',numPtsMultipliers, numPtsMultipliers_recall)

    # bar_colors = ['darkseagreen', 'mediumslateblue', 'lightcoral']
    # labels = ['none', 'uniform', 'random']
    # values = [numPtsMultipliers_recall[noDownSampIdx], orderedPatch['recall'], numPtsMultipliers_recall[usedParamIdx]]

    # ax2.bar([0, 1, 2], values, color=bar_colors)
    # ax2.set_xticks([0, 1, 2])
    # ax2.set_xticklabels(labels)
    # ax2.set_ylabel('Recall')
    # ax2.set_ylim([0,1])
    # ax2.set_xlabel('Patch Downsampling Methods', fontsize=14)


    # plt.tight_layout(pad=0.7)
    # plt.savefig('./patchDownsample-ablation.png')
    # plt.show()

    counted_numbers = Counter(numPtsMultipliers_incorrectMatches[noDownSampIdx])
    noneMatches, noneCount = zip(*sorted(counted_numbers.items(), key=lambda x: x[1], reverse=True))
    # counted_numbers = Counter(numPtsMultipliers_incorrectMatches[usedParamIdx])
    # randomMatches, randomCount = zip(*sorted(counted_numbers.items(), key=lambda x: x[1], reverse=True))

    # Sample data (replace with actual data)
    descriptorDensityRef=np.load('./results/LPR_Wildplaces/Ablate_TrainingSet/ptcldDenisty/descriptorDensityRef.npy')   
    descriptorDensityQuery=np.load('./results/LPR_Wildplaces/Ablate_TrainingSet/ptcldDenisty/descriptorDensityQuery.npy' )  
    rawPtcldDensityRef=np.load('./results/LPR_Wildplaces/Ablate_TrainingSet/ptcldDenisty/rawPtcldDensityRef.npy')    
    rawPtcldDensityQuery=np.load('./results/LPR_Wildplaces/Ablate_TrainingSet/ptcldDenisty/rawPtcldDensityQuery.npy')  
    idx=usedParamIdx
    query_ids = numPtsMultipliers_incorrectIds[idx]
    reference_ids = numPtsMultipliers_incorrectMatches[idx]
    query_densities = rawPtcldDensityQuery[numPtsMultipliers_incorrectIds[idx]]
    reference_densities = rawPtcldDensityRef[numPtsMultipliers_incorrectMatches[idx]]
    print(len(reference_densities), len(reference_ids), max(numPtsMultipliers_incorrectMatches[idx]))

    # Sort both lists by density
    sorted_query_indices = np.argsort(query_densities)[::-1]
    sorted_reference_indices = np.argsort(reference_densities)[::-1]

    # Reorder IDs and densities by sorted indices
    sorted_query_ids = [query_ids[i] for i in sorted_query_indices]
    sorted_reference_ids = [reference_ids[i] for i in sorted_reference_indices]
    sorted_query_densities = [query_densities[i] for i in sorted_query_indices]
    sorted_reference_densities = [reference_densities[i] for i in sorted_reference_indices]



    # fig, (ax1, ax2 ) = plt.subplots(1, 2, figsize=(8, 5))
    # kwargs = dict(histtype='step', alpha=1, bins=100, linewidth=1.5)
    # ax1.bar(np.arange(len(noneCount)),noneCount,color='slategray', label='no downsampling' )
    # ax1.bar(np.arange(len(randomCount)),randomCount,color='seagreen', label='random downsampling')
    # ax1.legend()
    # ax1.set_xlabel('Sorted Incorrect Matches')
    # ax1.set_ylabel('Count')

    # Plot the reference and query densities on top and bottom respectively
    ax2.plot(sorted_reference_densities, np.ones_like(sorted_reference_densities), '.-', color='darkslateblue',markersize=10, )#label='References by Density')
    ax2.plot(sorted_query_densities, np.zeros_like(sorted_query_densities), '.-', color='darkslateblue',markersize=10, ) #label='Queries sortedby Density'

    # Draw lines between incorrect query-reference pairs
    ax2.plot([sorted_query_densities[0], sorted_reference_densities[0]], [0, 1], '-', color='k',linewidth=0.5,alpha=0.75, label='Incorrect Matches')
    for i, (query_id, ref_id) in enumerate(zip(sorted_query_ids, sorted_reference_ids)):
        ax2.plot([sorted_query_densities[i], sorted_reference_densities[i]], [0, 1], '-', color='k',linewidth=0.5,alpha=0.75)

    # Customize plot
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Queries', 'References'])
    ax2.set_xlabel("Density (points per scan)")
    ax2.set_ylim([-0.1,1.1])
    # ax2.set_title("Connections between Incorrect Queries and References by Density")
    ax2.legend()

    plt.tight_layout(pad=1)
    plt.show()
    plt.savefig('./patchDownsample-incorrectMatches.png')

    # ax1.plot(numPtsMultipliers_incorrectIds[noDownSampIdx],numPtsMultipliers_incorrectMatches[noDownSampIdx], 'c.')
    # ax1.plot(numPtsMultipliers_incorrectIds[usedParamIdx],numPtsMultipliers_incorrectMatches[usedParamIdx], 'm.')
    # ax1.plot(orderedPatch['incorrectShiftedIds'],orderedPatch['matchIds'][orderedPatch['incorrectShiftedIds']], '.', color='lime')

def ablationDownSampleResultsPlot():
    numPtsMultipliers_recall = []
    numPtsMultipliers_incorrectIds=[]
    numPtsMultipliers_incorrectMatches=[]
    numPtsMultipliers = [0.1,0.5,0.75,1.5,2,4,6,8,10]
    usedParamIdx, noDownSampIdx=4,8
    for n in numPtsMultipliers:
        savePath = f'./results/LPR_Wildplaces/Ablate_TrainingSet_TimeIncl_1/WildPlaces_Venman_R:2_Q:3_RInc:1_Rrad:2_QInc:10_Res:0.3_Dim:120_blkAvg:2_N2ndsearch:2_rotInc10_UnoccWeight-0.15_nptsMult{n}_zMax3.xlsx'
        output = extractSingleReuslt(savePath, errTolerance=3)
        recall = output['recall']
        numPtsMultipliers_recall.append(recall)
        numPtsMultipliers_incorrectMatches.append(output['matchIds'][output['incorrectShiftedIds']])
        numPtsMultipliers_incorrectIds.append(output['incorrectShiftedIds'])
        print(output['matchIds'][output['incorrectShiftedIds']])
    

    # savePath = f'./results/LPR_Wildplaces/Ablate_TrainingSet/Venman_R:2_Q:3_RInc:1_Rrad:2_QInc:10_Res:0.3_Dim:120_orderedPatchDwnSample.xlsx'
    # orderedPatch = extractSingleReuslt(savePath, errTolerance=3)
    numPtsMultipliers = [1,5,7.5,15,20,40,60,80,100]
    # Plotting the recall values with 2x2 subplots
    fig, ((ax, ax1), (ax2,ax3)) = plt.subplots(2, 2, figsize=(9, 6))
    
    ax.plot(numPtsMultipliers, numPtsMultipliers_recall, marker='o', color='k')
    ax.scatter(numPtsMultipliers[usedParamIdx], numPtsMultipliers_recall[usedParamIdx], color='tab:pink', marker='.', zorder=5, s=100)
    ax.scatter(numPtsMultipliers[noDownSampIdx], numPtsMultipliers_recall[noDownSampIdx], color='tab:cyan', marker='.', zorder=5, s=100)
    ax.set_xlabel('Maximum Occupancy per 10x10 Patch (c)', fontsize=12)
    ax.grid(True)
    # ax.set_xlim([0,10.2])
    

    ax.set(ylabel='Recall')
    ax.yaxis.label.set_size(14)
    ax.set_ylim([0,1])
    ax.grid(True, which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.1)
    ax.grid(True, which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=0.1)


    ax.tick_params(axis='both', which='major', labelsize=12)

    print('top_n_recall',numPtsMultipliers, numPtsMultipliers_recall)

    descriptorDensityRef=np.load('./results/LPR_Wildplaces/Ablate_TrainingSet/ptcldDenisty/descriptorDensityRef.npy')   
    descriptorDensityQuery=np.load('./results/LPR_Wildplaces/Ablate_TrainingSet/ptcldDenisty/descriptorDensityQuery.npy' )  
    rawPtcldDensityRef=np.load('./results/LPR_Wildplaces/Ablate_TrainingSet/ptcldDenisty/rawPtcldDensityRef.npy')    
    rawPtcldDensityQuery=np.load('./results/LPR_Wildplaces/Ablate_TrainingSet/ptcldDenisty/rawPtcldDensityQuery.npy')  

    counted_numbers = Counter(rawPtcldDensityRef/np.linalg.norm(rawPtcldDensityRef))
    ptcldDensity, ptcldCount = zip(*sorted(counted_numbers.items(), key=lambda x: x[1], reverse=True))
    counted_numbers = Counter(descriptorDensityRef/np.linalg.norm(descriptorDensityRef))
    descriptorDensity, descriptorCount = zip(*sorted(counted_numbers.items(), key=lambda x: x[1], reverse=True))

    # ax1.bar(np.arange(len(ptcldDensity)),ptcldDensity,color='slategray', label='raw ptcld density' )
    # ax1.bar(np.arange(len(descriptorDensity)),descriptorDensity,color='seagreen', label='downsampled descriptor density')
    
    
    ax1.plot(rawPtcldDensityRef/np.linalg.norm(rawPtcldDensityRef), '-',label='Raw Scan Density', color='tab:purple')
    ax1.plot(descriptorDensityRef/np.linalg.norm(descriptorDensityRef), '-',label='Descriptor Density', color='mediumseagreen')
    ax1.set_xlabel('Scan IDs', fontsize=12)
    ax1.set_ylabel('Normalised Density', fontsize=12)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2, fontsize=10)

    

    # Sample data (replace with actual data)
    
    idx=usedParamIdx
    query_ids = numPtsMultipliers_incorrectIds[idx]
    reference_ids = numPtsMultipliers_incorrectMatches[idx]
    query_densities = rawPtcldDensityQuery[numPtsMultipliers_incorrectIds[idx]]
    reference_densities = rawPtcldDensityRef[numPtsMultipliers_incorrectMatches[idx]]
    print(len(reference_densities), len(reference_ids), max(numPtsMultipliers_incorrectMatches[idx]))

    # Sort both lists by density
    sorted_query_indices = np.argsort(query_densities)[::-1]
    sorted_reference_indices = np.argsort(reference_densities)[::-1]

    # Reorder IDs and densities by sorted indices
    sorted_query_ids = [query_ids[i] for i in sorted_query_indices]
    sorted_reference_ids = [reference_ids[i] for i in sorted_reference_indices]
    sorted_query_densities = [query_densities[i] for i in sorted_query_indices]
    sorted_reference_densities = [reference_densities[i] for i in sorted_reference_indices]

    # Plot the reference and query densities on top and bottom respectively
    ax2.plot(sorted_reference_densities, np.ones_like(sorted_reference_densities), '.-', color='brown',markersize=4, )#label='References by Density')
    ax2.plot(sorted_query_densities, np.zeros_like(sorted_query_densities), '.-', color='brown',markersize=4, ) #label='Queries sortedby Density'

    # Draw lines between incorrect query-reference pairs
    ax2.plot([sorted_query_densities[0], sorted_reference_densities[0]], [0, 1], '-', color='tab:pink',linewidth=0.8,alpha=1, label='Incorrect Matches with Downsampling')
    for i, (query_id, ref_id) in enumerate(zip(sorted_query_ids, sorted_reference_ids)):
        ax2.plot([sorted_query_densities[i], sorted_reference_densities[i]], [0, 1], '-', color='tab:pink',linewidth=0.8,alpha=1)

    # Customize plot
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Queries', 'References'])
    ax2.set_xlabel("Sorted Density (points per scan)", fontsize=12)
    ax2.set_ylim([-0.1,1.1])
    # ax2.set_title("Connections between Incorrect Queries and References by Density")
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=1, fontsize=10)

    idx=noDownSampIdx
    query_ids = numPtsMultipliers_incorrectIds[idx]
    reference_ids = numPtsMultipliers_incorrectMatches[idx]
    query_densities = rawPtcldDensityQuery[numPtsMultipliers_incorrectIds[idx]]
    reference_densities = rawPtcldDensityRef[numPtsMultipliers_incorrectMatches[idx]]
    print(len(reference_densities), len(reference_ids), max(numPtsMultipliers_incorrectMatches[idx]))

    # Sort both lists by density
    sorted_query_indices = np.argsort(query_densities)[::-1]
    sorted_reference_indices = np.argsort(reference_densities)[::-1]

    # Reorder IDs and densities by sorted indices
    sorted_query_ids = [query_ids[i] for i in sorted_query_indices]
    sorted_reference_ids = [reference_ids[i] for i in sorted_reference_indices]
    sorted_query_densities = [query_densities[i] for i in sorted_query_indices]
    sorted_reference_densities = [reference_densities[i] for i in sorted_reference_indices]

    # Plot the reference and query densities on top and bottom respectively
    ax3.plot(sorted_reference_densities, np.ones_like(sorted_reference_densities), '.-', color='brown',markersize=4, )#label='References by Density')
    ax3.plot(sorted_query_densities, np.zeros_like(sorted_query_densities), '.-', color='brown',markersize=4, ) #label='Queries sortedby Density'

    # Draw lines between incorrect query-reference pairs
    ax3.plot([sorted_query_densities[0], sorted_reference_densities[0]], [0, 1], '-', color='tab:cyan',linewidth=0.8,alpha=1, label='Incorrect Matches without Downsampling')
    for i, (query_id, ref_id) in enumerate(zip(sorted_query_ids, sorted_reference_ids)):
        ax3.plot([sorted_query_densities[i], sorted_reference_densities[i]], [0, 1], '-', color='tab:cyan',linewidth=0.8,alpha=1)

    # Customize plot
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['Queries', 'References'])
    ax3.set_xlabel("Sorted Density (points per scan)", fontsize=12)
    ax3.set_ylim([-0.1,1.1])
    # ax2.set_title("Connections between Incorrect Queries and References by Density")
    ax3.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=1, fontsize=10)

    plt.tight_layout(pad=1)
    
    plt.savefig('./patchDownsample-incorrectMatches.png', dpi=500)
    # plt.show()

    # ax1.plot(numPtsMultipliers_incorrectIds[noDownSampIdx],numPtsMultipliers_incorrectMatches[noDownSampIdx], 'c.')
    # ax1.plot(numPtsMultipliers_incorrectIds[usedParamIdx],numPtsMultipliers_incorrectMatches[usedParamIdx], 'm.')
    # ax1.plot(orderedPatch['incorrectShiftedIds'],orderedPatch['matchIds'][orderedPatch['incorrectShiftedIds']], '.', color='lime')

def voxelSizeRelationship():
    # Data
    average_density = [282.1799410029499, 659.3207547169811, 981.3020134228188] #[3024.3748645720475, 18423.7109375, 93127.83900928793]
    voxel_size = [1.3, 0.75, 0.3]
    datasets = ['Oxford Radar', 'NCLT', 'WildPlaces']
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    # Create figure and axis
    fig, ax = plt.subplots( figsize=(5,4))

    # Scatter plot each point with a unique color and label
    for i in range(len(average_density)):
        ax.scatter(average_density[i], voxel_size[i], color=colors[i], label=datasets[i])

    # Fit a linear trend line
    coefficients = np.polyfit(average_density, voxel_size, 1)
    trendline = np.poly1d(coefficients)
    x_values = np.linspace(min(average_density), max(average_density), 100)
    ax.plot(x_values, trendline(x_values), 'b--', label=f"y={coefficients[0]:.3f}x + {coefficients[1]:.2f}")

    # Set labels and title
    ax.set_xlabel("Average BEV Descriptor Density with 1m³ Voxels")
    ax.set_ylabel("Voxel Size (m³)")
    ax.set_title("Voxel Size vs BEV Descriptor Density")

    # Add legend
    ax.legend()

    plt.savefig('./voxelSizeRelationShip.png', dpi=400)
    # plt.show()

def densityDistribution():
    descriptorDensityRef=np.load('./results/LPR_Wildplaces/Ablate_TrainingSet/ptcldDenisty/descriptorDensityRef.npy')   
    rawPtcldDensityRef=np.load('./results/LPR_Wildplaces/Ablate_TrainingSet/ptcldDenisty/rawPtcldDensityRef.npy')   
    descriptorDensityQuery=np.load('./results/LPR_Wildplaces/Ablate_TrainingSet/ptcldDenisty/descriptorDensityQuery.npy' )   
    rawPtcldDensityQuery=np.load('./results/LPR_Wildplaces/Ablate_TrainingSet/ptcldDenisty/rawPtcldDensityQuery.npy')   
    print(np.mean(rawPtcldDensityQuery), np.mean(descriptorDensityQuery))
    # Create a figure with two subplots
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(10, 8))

    # Plot descriptorDensity on the first subplot
    ax1.plot(descriptorDensityQuery, label='queryDescriptorDensity')
    ax1.set_title('Query Descriptor Density')
    ax1.legend()

    # Plot rawPtcldDensity on the second subplot
    ax2.plot(rawPtcldDensityQuery, label='queryRawPtcldDensity', color='orange')
    ax2.set_title('Query Raw Point Cloud Density')
    ax2.legend()
    # Adjust layout to avoid overlap
    plt.tight_layout()
    plt.savefig('ablateQuerydesnityDistrubution.png')


    fig, ((ax4, ax3)) = plt.subplots(2, 1, figsize=(10, 8))
    # Plot reference descriptor density on the third subplot
    ax3.plot(descriptorDensityRef, label='Reference Descriptor Density', color='dodgerblue')
    ax3.set_title('Descriptor Density')
    ax3.set_xlabel('Scan Index')
    ax3.set_ylabel('Number of Points')
    # ax3.legend()

    # Plot reference raw point cloud density on the fourth subplot
    ax4.plot(rawPtcldDensityRef, label='Reference Raw Point Cloud Density', color='orchid')
    ax4.set_title('Raw Point Cloud Density')
    ax4.set_ylabel('Number of Points')
    ax4.set_xlabel('Scan Index')
    # ax4.legend()
    plt.tight_layout()
    plt.savefig('./ablateRefdesnityDistrubution.png')


def ablationUrbanResults(datasetName,refNum, queryNum):
    # ='NCLT',11,1
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
    mapReses=[ 0.3,0.5,0.75,1,1.3,1.5]
    HDM_thresh=1
    for mapRes in mapReses:
        savePath = saveFolder+(f'/{datasetName}_R:{refNum}_Q:{queryNum}_RInc:{refIncr}_Rrad:{refrad}_QInc:'
        f'{queryIncr}_Res:{mapRes}_Dim:{scanDimX}_map_res{mapRes}_HDMthresh{HDM_thresh}.xlsx')  
        output= extractSingleReuslt(savePath, errTolerance=25)
        print('vxlSize',mapRes, 'recall', output['recall'], 'time', np.mean(output['timePerQueryIQR']) )
    print('')       
    
    HDM_threshes=[1,2,3,4]
    mapRes=param.get('mapRes')
    if datasetName == 'OxfordRadar':
        mapRes=1.3
    for HDMthresh in HDM_threshes:
        savePath = saveFolder+(f'/{datasetName}_R:{refNum}_Q:{queryNum}_RInc:{refIncr}_Rrad:{refrad}_QInc:'
        f'{queryIncr}_Res:{mapRes}_Dim:{scanDimX}_map_res{mapRes}_HDMthresh{HDMthresh}.xlsx')  
        output= extractSingleReuslt(savePath, errTolerance=25)
        print('hdmThresh', HDMthresh, 'recall', output['recall'], 'time', np.mean(output['timePerQueryIQR']) )
    print('') 
# plotClosestId_matchId()

# recallOverDistances()
# recallOverRotations()
# recallwithNegativeBackground()

'''Currently used functions'''
# plotRealtiveShift()
# wildplacesOverTime()
# ablationNaturalResultsPlot()
# ablationDownSampleResultsPlot()
# voxelSizeRelationship()
# recallNCLT()
# recallOxfordRadar()
# recallWildplaces()

ablationUrbanResults('NCLT',11,1)
ablationUrbanResults('OxfordRadar',7,1)

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

'''Density evaluation'''
# venmanPtcldDensity=np.load(f'./results/ptcldDensity/WildPlaces_Venman_1_rawPtcldDensityRef.npy')  
# karawathaPtcldDensity=np.load(f'./results/ptcldDensity/WildPlaces_Karawatha_1_rawPtcldDensityRef.npy')  
# ncltPtcldDensity=np.load(f'./results/ptcldDensity/NCLT_11_rawPtcldDensityRef.npy')  
# oxfordPtcldDensity=np.load(f'./results/ptcldDensity/OxfordRadar_6_rawPtcldDensityRef.npy')  

# print(np.mean(oxfordPtcldDensity), np.mean(ncltPtcldDensity), np.mean(venmanPtcldDensity), np.mean(karawathaPtcldDensity))
