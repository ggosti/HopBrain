#!/usr/bin/env python
# coding: utf-8


#import getCycles
import numpy as np
#import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

def trans(sigma_path0,net1,N,typ = 0, thr = 0):
    """
    transiton function. net1 is the network that generates the ttransitions
    
    If sigma_path0 is a binary vector it generates the corresponding transtions.
    
    If sigma_path0 is a list of binary vectors it generates a list with the corresponding transtions.
    
    typ determins if the neuron activation state is defined in {-1,1} or {0,1} 
    typ=1 --> {-1,1}    typ=0 --> {0,1} 
    """
    sigma_path1 = net1.dot(sigma_path0.T)
    #print(sigma_path1)
    sigma_path1 [sigma_path1  == 0] = 0.000001
    #print(sigma_path1)
    sigma_path1 = (1-typ+np.sign(sigma_path1 +thr))/(2-typ)
    #print(sigma_path1)
    return sigma_path1.T   


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

def euc(listCoords):
    return np.array([[ np.linalg.norm(i-j) for j in listCoords] for i in listCoords])

def euc2(listCoords):
    return np.array([[np.sum((i-j)**2) for j in listCoords] for i in listCoords])

dist = euc(coords)
dist.shape

dist2=np.sqrt(euc2(coords))



plt.figure()
plt.imshow(dist)

lamda = 0.12#0.18
J = np.exp(-lamda*dist)
np.fill_diagonal(J, 0)
plt.figure()
plt.title('J = np.exp(-0.18*dist) ')
plt.imshow(J)

plt.figure()
plt.title('J>0.04')
plt.imshow(J>0.04)

#plt.show()


np.random.seed(8792)

runs = 5
passi = 200
N=J.shape[0]
print('runs',runs)
print('N',N)
states = np.zeros((runs,passi,N))

cycle1ConvTime = [] 

for r in range(runs):
    s0 = 2*np.random.binomial(1, 0.5, N)-1
    states[r,0,:] = s0

    for t in range(1,passi):
        s1 = trans(s0,J,N,typ = 1, thr = 0) #getCycles.transPy(s0,J,N,typ = 1, thr = 0)
        #print(s1)
        s0=s1.T
        states[r,t,:] = s1.T

    Cdt1 = np.mean(states[r,1:,:]*states[r,:-1,:],axis=1)
    cycle1ConvTime.append(np.argmax(Cdt1>=1.))

    if r<5:    
        f,axs=plt.subplots(2)
        #plt.figure()
        axs[0].imshow(states[r,:,:].T)
        #plt.figure()
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

uniqDist = np.unique(dist)
#plt.figure()
#plt.hist(uniqDist,bins=100)
ii,jj=np.mgrid[0:N, 0:N]
#print(ii)
#print(jj)



dList = []
iListList = []
jListList = []

for d in uniqDist:
    #print(d,np.sum(dist==d))
    #print(ii[dist==d])
    #print(jj[dist==d])
    #dList = dList + np.sum(dist==d)*[d]
    iListList.append(ii[dist==d])
    jListList.append(jj[dist==d])

Bd = [[] for r in range(runs)]
if (numCycle1ConvTime == runs ):
    for r in range(runs):
        print('run on stationary state',r)
        for d,iList,jList in zip(uniqDist,iListList,jListList):
            #print(d,np.sum(dist==d))
            cors = []
            for i,j in zip(iList,jList):
                #cor = np.mean(states[r,:,i]*states[r,:,j])
                #WARNING: given that all
                cor = np.mean(states[r,-1,i]*states[r,-1,j])
                cors.append(cor)
            Bd[r].append(np.mean(cors))

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
    plt.plot([2,3],[-0.1,-0.1 -0.5])


#plt.show()

plt.figure()
h,bins,f=plt.hist(uniqDist,bins=100)


f,ax=plt.subplots(5,2)

for r in range(5):
    rs = []
    binnedBd = [] 
    for ra,rb in zip(bins[:-1],bins[1:]):
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
