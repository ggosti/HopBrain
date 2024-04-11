#!/usr/bin/env python
# coding: utf-8


#import getCycles
import numpy as np
#import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance
import time

import turboBrainUtils as tb 

runs = 1000#40
passi = 100#200
autapse = True
randomize = True#True #False
parcelsName = 'Schaefer2018_1000Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv'

# # Parcellizzazione
# https://www.sciencedirect.com/science/article/pii/S2211124720314601?via%3Dihub
df = pd.read_csv(parcelsName)
df.head()
X = df['R']
Y = df['A']
Z = df['S']
N=len(X)

coords = np.array([X,Y,Z]).T
dist = distance.cdist(coords, coords, 'euclidean')
uniqDist,iListList,jListList = tb.sortIJbyDist(dist,N)
plt.figure()
h,bins,f=plt.hist(uniqDist,bins=100)
plt.title('unique distance')

lamda = 0.18#0.18
J = tb.makeJ(dist,lamda,autapse,randomize)

#tb.plotInitalJ(X, Y, Z,dist,J)
tb.plotInitalJ(X, Y, Z,dist,J,uniqDist)

np.random.seed(8792)

alphas = []



lambdas = np.arange(0.10,0.27,0.01)#np.arange(0.10,0.27,0.01)

#alphaSrRuns = []
lambdasRuns = []
#alphasSrAggrRun = []
runsList = []
#brsDict = {}
SDict = {}

for lambd in lambdas:
    print('Lambda',lambd)
    lambdasRuns.append(lambd)
    J = tb.makeJ(dist,lambd,autapse,randomize)
    #plt.figure()
    #plt.title('J = np.exp(-0.18*dist) ')
    #plt.imshow(J)
    #
    #plt.figure()
    #plt.title('J>0.04')
    #plt.imshow(J>0.04)
    #plt.show()
    print('runs',runs)
    print('N',N)

    states = np.zeros((runs,passi,N))
    cycle1ConvTime = [] 

    for r in range(runs):
        stasteRun,Cdt1 = tb.run(J, N, passi)
        states[r,:,:] = stasteRun
        cycle1ConvTime.append(np.argmax(Cdt1>=1.))
        #if r<5:    
        #    f,axs=plt.subplots(2)
        #    axs[0].imshow(stasteRun.T)
        #    axs[1].plot(Cdt1)


    #plt.figure()
    #plt.hist(cycle1ConvTime,bins=40)
    #plt.xlabel('convergence time to stationary state')

    numCycle1ConvTime = len(cycle1ConvTime)
    maxConvTime = np.max(cycle1ConvTime)
    print('maxConvTime',maxConvTime)

    # Checks if all runs end in cycle that is not an absorbing state
    assert numCycle1ConvTime == runs, f"not all runs end in absorbing state: {numCycle1ConvTime-runs}"

    #plt.show()

    #Bd = [[] for r in range(runs)]
    if (numCycle1ConvTime == runs ):
        for r in range(runs):
            print('run on stationary state',r)
            #t0 = time.time()
            #BdRun = tb.computeBr(states[r,:,:],uniqDist,iListList,jListList)
            #print(BdRun[:5],len(BdRun))
            #Bd[r] = BdRun
            #t1 = time.time()
            #Bd[r] = BdRun
            SDict['lambd'+str(lambd)+'run'+str(r)] = states[r,-1,:]
    #else:
        #for r in range(runs):
        #    print('run',r)
        #    for d,iList,jList in zip(uniqDist,iListList,jListList):
        #        #print(d,np.sum(dist==d))
        #        cors = []
        #        for i,j in zip(iList,jList):
        #            cor = np.mean(states[r,maxConvTime:,i]*states[r,maxConvTime:,j])
        #            cors.append(cor)
        #        Bd[r].append(np.mean(cors))

        #plt.figure()
        #for r in range(runs):
        #    plt.scatter(uniqDist,Bd[r])
        #plt.loglog()
        #
        #plt.figure()
        #for r in range(runs):
        #    plt.scatter(np.log(uniqDist),np.log(Bd[r]),alpha=0.4)
        #plt.plot([2,3],[-0.1,-0.1 -0.5])


    #plt.show()


df0 = pd.DataFrame({'lambdas':lambdasRuns})
#df1 = pd.DataFrame(brsDict)
df2 = pd.DataFrame(SDict)

strng = '' 
if autapse: strng = strng+'-autapse'
if randomize: strng = strng+'-randomizeJ'
print(strng)

df0.to_csv('lamdaValues'+strng+'.csv', index=False)
#df1.to_csv('BdRuns'+strng+'.csv', index=False)
df2.to_csv('SRuns'+strng+'.csv', index=False)

plt.show()
