#!/usr/bin/env python
# coding: utf-8


#import getCycles
import numpy as np
#import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance
import time
import json

import turboBrainUtils as tb 

runs = 1000#40
passi = 100#200
autapse = True
randomize = False
parcelsName = 'Schaefer2018_1000Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv'

strng = '' 
if autapse: strng = strng+'-autapse'
if randomize: strng = strng+'-randomizeJ'
strng = strng+'-thJ'
print(strng)

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

tb.plotInitalJ(X, Y, Z,dist,J,uniqDist)

np.random.seed(8792)

alphas = []

lambdas = np.arange(0.10,0.30,0.01)
ths = [0.0001,0.0002,0.0004,0.001,0.002,0.004,0.01,0.02,0.04,.1,.2,.4,.8,.9]


#alphaSrRuns = []
lambdasRuns = []
thsRuns = []
dilsRuns = []
#alphasSrAggrRun = []
runsList = []
#SDict = {}
#SList = []

for thJ in ths:
    SDictRunLambda = {}
    for lambd in lambdas:
        print('Lambda',lambd,'thJ',thJ)
        J = tb.makeJ(dist,lambd,autapse,randomize)
        #print(J[:4,:4], (J<thJ)[:4,:4], 'num cut',np.sum(J<thJ), thJ)
        J[J<thJ] = 0.
        
        #print(J[:4,:4])
        #plt.figure()
        #plt.title('J = np.exp(-0.18*dist) ')
        #plt.imshow(J)
        #
        #plt.figure()
        #plt.title('J>0.04')
        #plt.imshow(J>0.04)
        #plt.show()
        #print('runs',runs)
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
        #SDictRun = {}
        if (numCycle1ConvTime == runs ):
            for r in range(runs):
                #print('run on stationary state',r)
                #t0 = time.time()
                #BdRun = tb.computeBr(states[r,:,:],uniqDist,iListList,jListList)
                #print(BdRun[:5],len(BdRun))
                #Bd[r] = BdRun
                #t1 = time.time()
                #Bd[r] = BdRun
                #SDict['lambd'+str(lambd)+'thJ'+str(thJ)+'run'+str(r)] = states[r,-1,:]
                SDictRunLambda['lambd'+str(lambd)+'run'+str(r)] = states[r,-1,:]
                # load data
                lambdasRuns.append(lambd)
                thsRuns.append(thJ)
                runsList.append(r)
                dil = np.sum( (np.sum(J==0) - N)/(N*N - N))
                dilsRuns.append(dil)
                #SList.append({'lambda':lambd,'thJ':thJ,'r':r,'dil':dil,'states':states[r,-1,:].tolist()})
        df2 = pd.DataFrame(SDictRunLambda)
    df2.to_csv('SRuns'+strng+'-'+str(thJ)+'.csv', index=False)
        
df0 = pd.DataFrame({'lambdas':lambdasRuns,'thsRuns':thsRuns,'dilsRuns':dilsRuns,'run':runsList})
#df1 = pd.DataFrame(brsDict)




df0.to_csv('lamdaValues'+strng+'.csv', index=False)
#df1.to_csv('BdRuns'+strng+'.csv', index=False)
#df2.to_csv('SRuns'+strng+'.csv', index=False)

#with open('SRuns'+strng+'.json','w',encoding='utf-8') as f:
#    json.dump(SList,f,ensure_ascii=False,indent=4)

plt.show()
