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


# # Parcellizzazione
# https://www.sciencedirect.com/science/article/pii/S2211124720314601?via%3Dihub


df = pd.read_csv('Schaefer2018_1000Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv')
df.head()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

X = df['R']
Y = df['A']
Z = df['S']

ax.scatter(X, Y, Z)

ax.set_xlabel('R')
ax.set_ylabel('A')
ax.set_zlabel('S')

coords = np.array([X,Y,Z]).T
dist = distance.cdist(coords, coords, 'euclidean')

f,[ax0,ax1,ax2]=plt.subplots(1,3)
ax0.imshow(dist)

lamda = 0.18#0.18
J = tb.makeJ(dist,lamda)

ax1.set_title('J = np.exp(-0.18*dist) ')
ax1.imshow(J)

ax2.set_title('J>0.04')
ax2.imshow(J>0.04)


np.random.seed(8792)

runs = 5
passi = 200
N=J.shape[0]


#
# Run simulations
#

print('--Run simulations--')
print('runs',runs)
print('N',N)

states = np.zeros((runs,passi,N))
cycle1ConvTime = [] 

for r in range(runs):
    stasteRun,Cdt1 = tb.run(J, N, passi)
    states[r,:,:] = stasteRun
    cycle1ConvTime.append(np.argmax(Cdt1>=1.))

    if r<5:    
        f,axs=plt.subplots(2)
        axs[0].imshow(stasteRun.T)
        axs[1].plot(Cdt1)


plt.figure()
plt.hist(cycle1ConvTime,bins=40)
plt.xlabel('convergence time to stationary state')

numCycle1ConvTime = len(cycle1ConvTime)
maxConvTime = np.max(cycle1ConvTime)
print('maxConvTime',maxConvTime)

# Checks if all runs end in cycle that is not an absorbing state
assert numCycle1ConvTime == runs, f"not all runs end in absorbing state: {numCycle1ConvTime-runs}"

plt.show()

#
# measure correlations
#

uniqDist = np.unique(dist)
ii,jj=np.mgrid[0:N, 0:N]

iListList = []
jListList = []

t0 = time.time()
for d in uniqDist:
    iListList.append(ii[dist==d])
    jListList.append(jj[dist==d])
t1 = time.time()
print('time ij list',t1-t0)

Bd = [[] for r in range(runs)]
if (numCycle1ConvTime == runs ):
    for r in range(runs):
        print('run on stationary state',r)
        t0 = time.time()
        BdRun = tb.computeBr(states[r,:,:],uniqDist,iListList,jListList)
        print(BdRun[:5],len(BdRun))
        #Bd[r] = BdRun
        t1 = time.time()
        Bd[r] = BdRun
        #
        #BdRun2 = tb.computeBrCuda(states[r,:,:],uniqDist,iListList,jListList)
        #print(BdRun2[:5],len(BdRun2))
        #t2 = time.time()
        #print('time comp B(r)',t1-t0,t2-t1)
    #print('len',uniqDist,Bd[r])
    plt.figure()
    for r in range(runs):
        plt.scatter(uniqDist,Bd[r])
    plt.loglog()

    plt.figure()
    for r in range(runs):
        plt.scatter(np.log(uniqDist),np.log(Bd[r]),alpha=0.4)
    plt.plot([2,3],[-0.1,-0.1 -0.5])
else:
    for r in range(runs):
        print('run',r)
        for d,iList,jList in zip(uniqDist,iListList,jListList):
            #print(d,np.sum(dist==d))
            cors = []
            for i,j in zip(iList,jList):
                cor = np.mean(states[r,maxConvTime:,i]*states[r,maxConvTime:,j])
                cors.append(cor)
            Bd[r].append(np.mean(cors))

    plt.figure()
    for r in range(runs):
        plt.scatter(uniqDist,Bd[r])
    plt.loglog()

    plt.figure()
    for r in range(runs):
        plt.scatter(np.log(uniqDist),np.log(Bd[r]),alpha=0.4)
    #plt.plot([2,3],[-0.1,-0.1 -0.5])


plt.show()

plt.figure()
h,bins,f=plt.hist(uniqDist,bins=100)


f,ax=plt.subplots(5,2)

for r in range(5):
    rs = []
    binnedBd = [] 
    for ra,rb in zip(bins[:-1],bins[1:]):
        print(ra,rb,type(Bd[r]),Bd[r][:5])
        gate = np.logical_and(uniqDist>=ra, uniqDist<=rb)
        binnedBd.append(np.mean(np.array(Bd[r])[gate]))
        rs.append(0.5*(ra+rb))
    binnedBd = np.array(binnedBd)
    
    ax[r,1].scatter(np.log(uniqDist),np.log(Bd[r]),alpha=0.4)
    ax[r,1].scatter(np.log(rs),np.log(binnedBd),alpha=0.4)
    ax[r,1].plot([2,3.5],[-0.1,-0.1 -(1.5*0.5)],'r')
    #plt.xlim((2,4))
    x=np.log(rs)
    y=np.log(binnedBd)#[np.logical_and(x>2, x<4)]
    gate= np.logical_and(np.logical_and(x>2, x<3.5),np.isfinite(y))
    x=x[gate]
    y=y[gate]
    

    A = np.vstack([x, np.ones(len(x))]).T
    #print(x,y)
    # Direct least square regression
    alpha = np.dot((np.dot(np.linalg.inv(np.dot(A.T,A)),A.T)),y)
    print('coef log B(r)',alpha)
    ax[r,1].plot(x, alpha[0]*x + alpha[1], '-g')
    ax[r,1].set_ylabel('log( B(r) )')
    ax[r,1].set_xlabel('log( r )')
    ax[r,1].set_xlim((1,5.5))
    ax[r,1].set_ylim((-12,1))
    ax[r,1].text(2, -4, 'slope = '+str(alpha[0]), fontsize=12)


    binnedSr = 2*(binnedBd[0] - binnedBd)
    ax[r,0].scatter(np.log(rs),np.log(binnedSr ),alpha=0.4)
    ax[r,0].plot([2,3.5],[-0.1,-0.1 + 1.5*0.5],'r')
    y=np.log(binnedSr)
    y=y[gate]
    A = np.vstack([x, np.ones(len(x))]).T
    #print(x,y)
    # Direct least square regression
    alpha = np.dot((np.dot(np.linalg.inv(np.dot(A.T,A)),A.T)),y)
    print('coef log S(r)',alpha)

    ax[r,0].plot(x, alpha[0]*x + alpha[1], '-g')
    ax[r,0].set_ylabel('log( S(r) )')
    ax[r,0].set_xlabel('log( r )')
    ax[r,0].set_xlim((1,5.5))
    ax[r,0].set_ylim((-5,2))
    ax[r,0].text(2, -4, 'slope = '+str(alpha[0]), fontsize=12)
    #ax[r,0].text(2, -4, 'slope = '+str(alpha[0]), fontsize=12)
    
f,ax=plt.subplots(1,2)
rs = []
binnedBd = [] 
binnedSr = [] 
ax[0].set_title('lambda'+str(lamda))
for r in range(runs):
    rsTemp = []
    binnedBdTemp = [] 
    for ra,rb in zip(bins[:-1],bins[1:]):
        gate = np.logical_and(uniqDist>=ra, uniqDist<=rb)
        binnedBdTemp.append(np.mean(np.array(Bd[r])[gate]))
        rsTemp.append(0.5*(ra+rb))
    binnedBd = binnedBd + binnedBdTemp
    binnedBdTemp = np.array(binnedBdTemp)

    binnedSrTemp = 2*(binnedBdTemp[0] - binnedBdTemp)
    binnedSr = binnedSr + binnedSrTemp.tolist()
    ax[1].scatter(np.log(rsTemp),np.log(binnedBdTemp),s=1,alpha=0.2)
    ax[0].scatter(np.log(rsTemp),np.log(binnedSrTemp),s=1,alpha=0.2)
    rs = rs + rsTemp
    


#plt.scatter(np.log(uniqDist),np.log(Bd[r]),alpha=0.4)
#plt.scatter(np.log(rs),np.log(binnedBd),alpha=0.4)
ax[1].plot([2,3.5],[-0.1,-0.1 -(1.5*0.5)],'r')
x=np.log(rs)
y=np.log(binnedBd)#[np.logical_and(x>2, x<4)]
gate= np.logical_and(np.logical_and(x>2, x<3.5),np.isfinite(y))
x=x[gate]
y=y[gate]

A = np.vstack([x, np.ones(len(x))]).T
# Direct least square regression
alpha = np.dot((np.dot(np.linalg.inv(np.dot(A.T,A)),A.T)),y)
print('coef log B(r)',alpha)
ax[1].plot(x, alpha[0]*x + alpha[1], '-g')
ax[1].set_ylabel('log( B(r) )')
ax[1].set_xlabel('log( r )')
ax[1].set_xlim((1,5.5))
ax[1].set_ylim((-12,1))
ax[1].text(2, -4, 'slope = '+str(alpha[0]), fontsize=12)



y=np.log(binnedSr)#[np.logical_and(x>2, x<4)]
y=y[gate]

gate2 = np.isfinite(y)
print(np.logical_not(gate2))
print(y)
if np.logical_not(gate2).any():
    print('values that get nan in log ',y[np.logical_not(gate2)])
    y=y[gate2]
    x=x[gate2]

A = np.vstack([x, np.ones(len(x))]).T
# Direct least square regression
try:
    alpha = np.dot((np.dot(np.linalg.inv(np.dot(A.T,A)),A.T)),y)
    print('coef log S(r)',alpha)
    ax[0].plot(x, alpha[0]*x + alpha[1], '-g')
except np.linalg.LinAlgError as err:
    if 'Singular matrix' in str(err):
        print('singular')

ax[0].set_ylabel('log( S(r) )')
ax[0].set_xlabel('log( r )')
ax[0].set_xlim((1,5.5))
ax[0].set_ylim((-5,2))
ax[0].plot([2,3.5],[-0.1,-0.1 + 1.5*0.5],'r')

#ax[0].plot(x, alpha[0]*x + alpha[1], '-g')
ax[0].text(2, -4, 'slope = '+str(alpha[0]), fontsize=12)

plt.savefig('structure'+str(lamda)+'.pdf')
plt.show()
