#!/usr/bin/env python
# coding: utf-8


#import getCycles
import numpy as np
#import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance
import time
import networkx as nx

import turboBrainUtils as tb 

runs = 40#1000#40
passi = 100#200
autapse = True
randomize = False #True #False #True 
parcelsName = 'Centroid_coordinates/Schaefer2018_1000Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv'

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

lamb = 0.18#0.18
J = tb.makeJ(dist,lamb,autapse,randomize)

#tb.plotInitalJ(X, Y, Z,dist,J)
tb.plotInitalJ(X, Y, Z,dist,J,uniqDist)

fig0,ax0 = plt.subplots(1,figsize=(15,5))
degreeCents = []

lambdas = np.arange(0.10,0.27,0.04)#np.arange(0.10,0.27,0.01)
for lamb in lambdas:
    J = tb.makeJ(dist,lamb,autapse,randomize)
    np.fill_diagonal(J, 0)  
    degreeCent = np.sum(J,axis=0)/np.sum(J)
    degreeCents.append(degreeCent)
    
#ax0.hist(degreeCents,bins = 40,histtype = 'stepfilled',alpha = 0.3)
ax0.hist(degreeCents,bins = 40,histtype = 'step')
ax0.legend([str(round(1.0/la, 2)) for la in lambdas],title=r'decay length $\delta$')
ax0.set_title("Weighted degree centrality histogram")
ax0.set_ylabel("Frequence")
ax0.set_xlabel(r'Weighted degree centrality')

#plt.show()

lambdasRuns = []

fig,[ax1,ax2] = plt.subplots(2,figsize=(10,7))
width = 0.2  # the width of the bars
multiplier = 0



lamb = 0.18
ths = [0.0001,0.0003,0.0012,0.003,0.01]
for thJ in ths:
    print('thJ',thJ)
    lambdasRuns.append(lamb)
    J = tb.makeJ(dist,lamb,autapse,randomize)
    
    J[J<thJ] = 0.
    
    numEdges = (np.sum(J==0) - N)/2
    dil = np.sum( (np.sum(J==0) - N)/(N*N - N))
    edgelist = []
    for i in range(N): 
        for j in range(i+1,N):
            if np.abs(J[i,j]) > 0:
                edgelist.append((i,j))
    #print(edgelist)
    
 
    G = nx.from_edgelist(edgelist)

    degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
    dmax = max(degree_sequence)


    ax1.plot(degree_sequence, label = str(round(dil, 2)))
    ax1.set_title("Degree rank plot")
    ax1.set_ylabel("Degree")
    ax1.set_xlabel("Rank")
    ax1.legend(title=r'Dilution $\rho$')

    degree, num = np.unique(degree_sequence, return_counts=True)
    offset = width * multiplier
    #ax2.bar(degree+offset, num , width, label = str(round(dil, 2)))
    ax2.plot(degree, num , label = str(round(dil, 2))) #degree/numEdges
    multiplier += 1
    ax2.set_title("Degree histogram")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("Number of Nodes")
    ax2.legend(title=r'Dilution $\rho$')
    
    
    
dils = [] 
largest_ccs = []

ths = [0.0001,0.0004,0.0012,0.004,0.01,0.02,0.04,0.06,0.1,0.13,0.16,0.2,0.4]

for thJ in ths:
    print('thJ',thJ)
    lambdasRuns.append(lamb)
    J = tb.makeJ(dist,lamb,autapse,randomize)
    J[J<thJ] = 0.
    
    dil = np.sum( (np.sum(J==0) - N)/(N*N - N))
    edgelist = []
    for i in range(N): 
        for j in range(i+1,N):
            if np.abs(J[i,j]) > 0:
                edgelist.append((i,j))
    #print(edgelist)
    
 
    G = nx.from_edgelist(edgelist)
    
    cc = [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
    largest_cc = cc[0]
    print('dil',dil)
    print('largest_cc',largest_cc,cc)
    largest_ccs.append(largest_cc)
    dils.append(dil)

fig.tight_layout()

print('largest_ccs',largest_ccs)
fig2,ax3 = plt.subplots(1)
ax3.plot(dils,largest_ccs,'-o')
#ax3.set_title("Largest Connected Component")
ax3.set_ylabel("Largest connected component")
ax3.set_xlabel(r'Dilution $\delta$')

plt.show()


