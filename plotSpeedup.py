import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import re
from pprint import pprint

if len(sys.argv) <= 1:
    exit()

logDirs = sys.argv[1:]
maxNumProcs = 0
for i,logDir in enumerate(logDirs):
    files = os.listdir(logDir)
    plotData = {}
    for file in files:
        if re.match('.+\.o\d+', file):
            # Get file contents
            f = open(os.path.join(logDir, file), 'r')
            # Init numProcs and time
            numProcs = 0
            time = 0
            for line in f.readlines():
                # Get the number of processors
                procMatch = re.match('.*proc.*?(\d+)', line)
                if procMatch is not None:
                    numProcs = int(procMatch.group(1))
                    maxNumProcs = max(numProcs, maxNumProcs)
                # Get the time
                timeMatch = re.match('.*[T|t]ime.*?(\d+.*)', line)
                if timeMatch is not None:
                    time = float(timeMatch.group(1))
            
            # Add to plot data (used to calculate trendline)
            if numProcs not in plotData.keys():
                plotData[numProcs] = []
            plotData[numProcs].append(time)
            
            # Close the file
            f.close()
    # Sort times
    for k in plotData.keys():
        plotData[k].sort()
    # Print info
    print 'Method %d (%s):' % (i, logDir)
    pprint(plotData)
    
    # Calculate mean trend
    medianTimes = [(x, np.median(y)) for x,y in sorted(plotData.iteritems())]
    # Calculate speedup
    oldTime = np.median(plotData[1])
    speedups = [(x, oldTime/y) for x,y in medianTimes]
    # Plot actual speedup
    plt.plot(*zip(*speedups), label='Method '+str(i), linewidth=2)

# Plot ideal speedup
plt.plot([0, maxNumProcs], [0, maxNumProcs], label='Ideal')
# Scale axes
plt.xlim([0, maxNumProcs])
plt.ylim([0, maxNumProcs])
plt.legend(loc='upper left')
plt.show()
