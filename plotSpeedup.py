import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import re
from pprint import pprint

if len(sys.argv) not in [2, 3]:
    print 'Usage: python plotSpeedup.py logDir [plotSaveDir]'
    exit()

logDir = sys.argv[1]
plotSaveDir = '' if len(sys.argv) != 3 else sys.argv[2]
maxNumProcs = 0
plotDatas = {}

files = os.listdir(logDir)
for file in files:
    print os.path.join(logDir, file)
    # Get file contents
    f = open(os.path.join(logDir, file), 'r')
    fileDict = {}
    for line in f.readlines():
        # Get the stuff before and after the colon
        regMatch = re.match('(.+):\s(.+)', line)
        fileDict[regMatch.group(1)] = regMatch.group(2)
    # Get string identifying the configuration (everything in this config gets plotted on one graph)
    configId = '%s-%s' % (fileDict['Image'], fileDict['Threshold'])
    if configId not in plotDatas.keys():
        plotDatas[configId] = {}
    method = fileDict['Method']
    if method not in plotDatas[configId].keys():
        plotDatas[configId][method] = {}
    numProcs = int(fileDict.get('Num procs', 1))
    maxNumProcs = max(numProcs, maxNumProcs)
    if numProcs not in plotDatas[configId][method].keys():
        plotDatas[configId][method][numProcs] = []
    time = float(fileDict.get('Total time (s)', 1))
    plotDatas[configId][method][numProcs].append(time)
    # Close the file
    f.close()
# Sort times
for configId in plotDatas.keys():
    for method in plotDatas[configId].keys():
        for numProcs in plotDatas[configId][method].keys():
            plotDatas[configId][method][numProcs].sort()
pprint(plotDatas)

medianDatas = {}
minDatas = {}
for configId in plotDatas.keys():
    medianDatas[configId] = {}
    minDatas[configId] = {}
    for method in plotDatas[configId].keys():
        medianDatas[configId][method] = {'x': [], 'y': []}
        minDatas[configId][method] = {'x': [], 'y': []}
        for numProcs in sorted(plotDatas[configId][method].keys()):
            medianDatas[configId][method]['x'].append(numProcs)
            median = np.median(plotDatas[configId][method][numProcs])
            medianDatas[configId][method]['y'].append(median)
            minDatas[configId][method]['x'].append(numProcs)
            minimum = np.min(plotDatas[configId][method][numProcs])
            minDatas[configId][method]['y'].append(minimum)

pprint(medianDatas)
pprint(minDatas)

for configId in plotDatas.keys():
    plt.figure()
    for method in sorted(plotDatas[configId].keys()):
        serialTime = medianDatas[configId][method]['y'][0]
        plt.plot(medianDatas[configId][method]['x'], [serialTime/medTime for medTime in medianDatas[configId][method]['y']], label=method, linewidth=2)
    # Plot ideal speedup
    plt.plot([0, maxNumProcs], [0, maxNumProcs], label='Ideal')
    # Scale axes
    plt.xlim([0, maxNumProcs])
    plt.ylim([0, maxNumProcs])
    # Labels
    plt.xlabel('# procs')
    plt.ylabel('Speedup')
    plt.title(configId + ' (medians)')
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(plotSaveDir, configId + '-median.png'))

for configId in plotDatas.keys():
    plt.figure()
    for method in sorted(plotDatas[configId].keys()):
        serialTime = minDatas[configId][method]['y'][0]
        plt.plot(minDatas[configId][method]['x'], [serialTime/minTime for minTime in minDatas[configId][method]['y']], label=method, linewidth=2)
    # Plot ideal speedup
    plt.plot([0, maxNumProcs], [0, maxNumProcs], label='Ideal')
    # Scale axes
    plt.xlim([0, maxNumProcs])
    plt.ylim([0, maxNumProcs])
    # Labels
    plt.xlabel('# procs')
    plt.ylabel('Speedup')
    plt.title(configId + ' (min)')
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(plotSaveDir, configId + '-min.png'))
