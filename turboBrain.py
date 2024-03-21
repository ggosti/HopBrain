#!/usr/bin/env python
# coding: utf-8


#import getCycles
import numpy as np
#import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.ticker

import pandas as pd
from scipy.spatial import distance
from scipy import stats
import time

import turboBrainUtils as tb 


runs = 1000#40#100#1000
passi = 100#200
autapse = True
randomize = False

# # Parcellizzazione
# https://www.sciencedirect.com/science/article/pii/S2211124720314601?via%3Dihub
df = pd.read_csv('Schaefer2018_1000Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv')
df.head()
X = df['R']
Y = df['A']
Z = df['S']
N=len(X)

coords = np.array([X,Y,Z]).T
dist = distance.cdist(coords, coords, 'euclidean')
uniqDist,iListList,jListList = tb.sortIJbyDist(dist,N)
#plt.figure()
#h,bins,f=plt.hist(uniqDist,bins=100)
#plt.title('unique distance')

lamda = 1./5.55 #5.55 fit deco #1./4 # random. #0.18 #1./6.66 (random walk) #5.99 #0.18
J = tb.makeJ(dist,lamda,autapse,randomize)

tb.plotInitalJ(X, Y, Z,dist,J,uniqDist)

plt.show()

np.random.seed(6792)


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
        axs[0].imshow(stasteRun[:80,:].T,cmap='coolwarm')
        axs[1].plot(Cdt1)
        if False: # show final state
            f,ax=plt.subplots(1,figsize=(1.5,9))
            ax.barh(range(N),stasteRun[-1,:]==1,height=1.,color='r')#,cmap='coolwarm')
            ax.barh(range(N),stasteRun[-1,:]==-1,height=1.,color='b')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.axes.get_xaxis().set_visible(False)
            plt.tight_layout()
            #f,ax=plt.subplots(1)
            #ax.imshow(stasteRun[:80,:].T,cmap='coolwarm')
            # show in atlas
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(X, Y, Z, c=stasteRun[-1,:],cmap='coolwarm')
            ax.set_title('$\delta = $'+str(np.round(1./lamda,2)))
            ax.set_xlabel('R (mm)')
            ax.set_ylabel('A (mm)')
            ax.set_zlabel('S (mm)')
 


# convegence time plots
plt.figure()
plt.hist(cycle1ConvTime,bins=40)
plt.xlabel('convergence time to stationary state')

numCycle1ConvTime = len(cycle1ConvTime)
maxConvTime = np.max(cycle1ConvTime)
print('maxConvTime',maxConvTime)

# Checks if all runs end in cycle that is not an absorbing state
assert numCycle1ConvTime == runs, f"not all runs end in absorbing state: {numCycle1ConvTime-runs}"

plt.show()

if False: # measure averages
    mean_r_states = np.mean(states,axis=0)    

    fig = plt.figure(figsize=plt.figaspect(.25))
    ax = fig.add_subplot(1, 4, 1)
    ax.imshow(mean_r_states.T,cmap='coolwarm')  
    ax.set_xlabel('t')
    ax.set_ylabel('i')

    ax = fig.add_subplot(1, 4, 2, projection='3d') 
    sc = ax.scatter(X, Y, Z, c=mean_r_states[0,:],cmap='coolwarm')
    plt.colorbar(sc,ax=ax,fraction=0.046, pad=0.4)
    ax.set_title('average over all \n starting states \n $\delta = $'+str(np.round(1./lamda,2)))
    ax.set_xlabel('R (mm)')
    ax.set_ylabel('A (mm)')
    ax.set_zlabel('S (mm)')

    ax = fig.add_subplot(1, 4, 3, projection='3d') 
    sc = ax.scatter(X, Y, Z, c=mean_r_states[-1,:],cmap='coolwarm')
    plt.colorbar(sc,ax=ax,fraction=0.046, pad=0.4)
    ax.set_title('average over all \n stationary states \n $\delta = $'+str(np.round(1./lamda,2)))
    ax.set_xlabel('R (mm)')
    ax.set_ylabel('A (mm)')
    ax.set_zlabel('S (mm)')

    axH = fig.add_subplot(1, 4, 4) 
    axH.hist(mean_r_states[0,:],bins=40,alpha=0.3,label='start')
    axH.hist(mean_r_states[-1,:],bins=40,alpha=0.3,label='convergence')
    axH.legend()

    fig.text( x= 0.01, y = 0.95 ,s='A', fontsize=14)
    fig.text( x= 0.22, y = 0.95 ,s='B', fontsize=14)
    fig.text( x= 0.52, y = 0.95 ,s='C', fontsize=14)
    fig.text( x= 0.72, y = 0.95 ,s='D', fontsize=14)

    plt.show()

#
# measure correlations
#

fitxlim = 3.5 

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
        #BdRun2 = tb.computeBr2(states[r,:,:],uniqDist,iListList,jListList)
        #print(BdRun2[:5],len(BdRun2))
        #t2 = time.time()
        #print('time comp B(r)',t1-t0,t2-t1,BdRun2==BdRun)
    #print('len',uniqDist,Bd[r])
    plt.figure()
    for r in range(runs):
        plt.scatter(uniqDist,Bd[r])
    plt.loglog()

    plt.figure()
    for r in range(runs):
        plt.scatter(np.log(uniqDist),np.log(Bd[r]),alpha=0.4)
    #plt.plot([2,3],[-0.1,-0.1 -0.5])
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


#plt.show()

plt.figure()
h,bins,f=plt.hist(uniqDist,bins=100)


f,ax=plt.subplots(5,2)

for r in range(5):
    rs = []
    binnedBd = [] 
    for ra,rb in zip(bins[:-1],bins[1:]):
        #print(ra,rb,type(Bd[r]),Bd[r][:5])
        gate = np.logical_and(uniqDist>=ra, uniqDist<=rb)
        binnedBd.append(np.mean(np.array(Bd[r])[gate]))
        rs.append(0.5*(ra+rb))
    binnedBd = np.array(binnedBd)
    
    ax[r,1].scatter(np.log(uniqDist),np.log(Bd[r]),alpha=0.4)
    ax[r,1].scatter(np.log(rs),np.log(binnedBd),alpha=0.4)
    ax[r,1].plot([2,fitxlim],[-0.1,-0.1 -(1.5*0.5)],'r')
    #plt.xlim((2,4))
    x=np.log(rs)
    y=np.log(binnedBd)#[np.logical_and(x>2, x<4)]
    gate= np.logical_and(np.logical_and(x>2, x<fitxlim),np.isfinite(y))
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
    ax[r,0].plot([2,fitxlim],[-0.1,-0.1 + 1.5*0.5],'r')
    y=np.log(binnedSr)
    y=y[gate]
    A = np.vstack([x, np.ones(len(x))]).T
    #print(x,y)
    # Direct least square regression
    alpha = np.dot((np.dot(np.linalg.inv(np.dot(A.T,A)),A.T)),y)
    print('coef log S(r)',alpha)

    ax[r,0].plot(x, alpha[0]*x + alpha[1], '-g')
    ax[r,0].set_ylabel('$\log( S(r) )$')
    ax[r,0].set_xlabel('$\log( r )$')
    ax[r,0].set_xlim((1,5.5))
    ax[r,0].set_ylim((-5,2))
    ax[r,0].text(2, -4, 'slope = '+str(alpha[0]), fontsize=12)
    #ax[r,0].text(2, -4, 'slope = '+str(alpha[0]), fontsize=12)


#plt.show()

if True:   
    f,ax=plt.subplots(2,1,figsize=(4,10))
    f.subplots_adjust(top=0.963, bottom=0.156, left=0.078, right=0.985, hspace=1.0, wspace=0.25)
    f.text( x= 0.01, y = 0.95 ,s='A', fontsize=14)
    f.text( x= 0.01, y = 0.45 ,s='B', fontsize=14)
    
    f2,axs1=plt.subplots(2,1,figsize=(4,10))
    f2.subplots_adjust(top=0.963, bottom=0.156, left=0.078, right=0.985, hspace=1.0, wspace=0.25)
    f2.text( x= 0.01, y = 0.95 ,s='A', fontsize=14)
    f2.text( x= 0.01, y = 0.45 ,s='B', fontsize=14)
    #ax[0].set_title('lambda'+str(lamda))
    #f,axComp=plt.subplots(1,2,figsize=(10,4))
    #f.text( x= 0.01, y = 0.95 ,s='A', fontsize=14)
    #f.text( x= 0.5, y = 0.95 ,s='B', fontsize=14)
    #axComp[0].set_title('A', loc='left')
    #axComp[1].set_title('B', loc='left')
    rs = []
    binnedBd = [] 
    rsKeys = [0.5*(ra+rb) for ra,rb in zip(bins[:-1],bins[1:])]
    binnedBdDict = {rk:[]  for rk in rsKeys}
    binnedSdDict = {rk:[]  for rk in rsKeys}
    binnedBd0 = []
    binnedSr = [] 
    for r in range(runs):
        rsTemp = []
        binnedBdTemp = [] 
        B0 = np.array(Bd[r])[uniqDist==0]
        #print('B0',B0)
        for ra,rb in zip(bins[:-1],bins[1:]):
            gate = np.logical_and(uniqDist>=ra, uniqDist<=rb)
            binnedBdDict[0.5*(ra+rb)] = binnedBdDict[0.5*(ra+rb)] + np.array(Bd[r])[gate].tolist()
            SrTemp = 2* (B0 - np.array(Bd[r])[gate] )
            binnedSdDict[0.5*(ra+rb)] = binnedSdDict[0.5*(ra+rb)] + SrTemp.tolist()
            meanBd_r_binned = np.mean(np.array(Bd[r])[gate])
            binnedBdTemp.append(meanBd_r_binned)
            rsTemp.append(0.5*(ra+rb))
            #print('bin',ra,rb,meanBd_r_binned,stdBd_r_binned,numBd_r_binned),
        binnedBd = binnedBd + binnedBdTemp
        binnedBdTemp = np.array(binnedBdTemp)

        binnedSrTemp = 2*(binnedBdTemp[0] - binnedBdTemp)
        binnedSr = binnedSr + binnedSrTemp.tolist()
        ax[0].plot(np.log(rsTemp),np.log(binnedBdTemp))#,s=4,alpha=0.8)
        ax[1].plot(np.log(rsTemp),np.log(binnedSrTemp))#,s=4,alpha=0.8)
        #axComp[1].plot(np.log(rsTemp),np.log(binnedBdTemp)/np.log(rsTemp))#,s=4,alpha=0.8)
        #axComp[0].plot(np.log(rsTemp),np.log(binnedSrTemp)/np.log(rsTemp))#,s=4,alpha=0.8)
        rs = rs + rsTemp
        
    x=np.log(rs)
    y=np.log(binnedBd)#[np.logical_and(x>2, x<4)]
    gate= np.logical_and(np.logical_and(x>2, x<fitxlim),np.isfinite(y))
    x=x[gate]
    y=y[gate]

    A = np.vstack([x, np.ones(len(x))]).T
    # Direct least square regression
    alpha = np.dot((np.dot(np.linalg.inv(np.dot(A.T,A)),A.T)),y)
    print('alpha[1]',alpha[1])

    #plt.scatter(np.log(uniqDist),np.log(Bd[r]),alpha=0.4)
    #plt.scatter(np.log(rs),np.log(binnedBd),alpha=0.4)
    #ax[1].plot([2,fitxlim],[-0.1,-0.1 -(1.5*0.5)],'k--',alpha=0.6)
    print('binnedBd.shape',np.array(binnedBd).shape)
    rsPlot = [rk for rk in rsKeys]
    rsPlot.sort()
    binnedBdmeanPlot = [np.mean(binnedBdDict[rk]) for rk in rsPlot]
    binnedBdstdPlot = [np.std(binnedBdDict[rk]) for rk in rsPlot]
    binnedBdsePlot = [3*np.std(binnedBdDict[rk]/np.sqrt(runs)) for rk in rsPlot]
    #for rk in rsPlot:
    #    binBd_rk = binnedBdDict[rk]
    #    axs1[0].scatter([rk]*len(binBd_rk),binBd_rk,alpha=0.02)
    #    axs1[0].scatter(rk,np.mean(binBd_rk),color='red')
    axs1[0].errorbar(rsKeys,binnedBdmeanPlot,yerr=binnedBdsePlot,fmt='k.', ecolor='black',  capsize=3, capthick=1)
    axs1[0].loglog()
    xline = np.arange(1,5.5,0.1) #np.array([1,5.5])
    axs1[0].plot(np.exp(xline), np.exp(alpha[0]*xline + alpha[1]), '--',color='tab:blue',label='slope fit '+"%.3f" % alpha[0])
    #axs1[0].plot(np.exp([0,5.5]),np.exp([1.5,1.5 + 5.5*(-0.5)]),'--',color='tab:orange',alpha=0.8,label='slope = -1/2 Deco')
    #axs1[0].plot(np.exp([0,5.5]),np.exp([1.5,1.5 + 5.5*(-0.66)]),'--',color='tab:green',alpha=0.4,label='slope = -2/3 Turbulence')
    axs1[0].set_ylabel(r'$\log( \langle B(d) \rangle)$')
    axs1[0].set_xlabel(r'$\log( d )$')
    axs1[0].set_xticks(np.exp(np.arange(1.5,3.9,0.5)))
    axs1[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axs1[0].minorticks_off()
    axs1[0].set_xticklabels(np.arange(1.5,3.9,0.5))
    axs1[0].set_yticks(np.exp(np.arange(-7.,1.)))
    axs1[0].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axs1[0].set_yticklabels(np.arange(-7.,1.))
    axs1[0].set_xlim((np.exp(1.5),np.exp(3.9)))
    axs1[0].set_ylim((np.exp(-3),np.exp(0.4)))
    #ax[0].text(2.5, 0, 'slope = '+str(alpha[0]), fontsize=8)
    axs1[0].axvline(np.exp(2),color='gray')
    axs1[0].axvline(np.exp(fitxlim),color='gray')
    #axs1[0].legend()
    

    print('coef log B(r)',alpha)
    xline = np.arange(1,5.5,0.1) #np.array([1,5.5])
    ax[0].plot(xline, alpha[0]*xline + alpha[1], 'b--',label='slope fit '+"%.3f" % alpha[0])
    #axs1[0].plot(xline, alpha[0]*xline + alpha[1], 'b--',label='slope fit '+"%.3f" % alpha[0])
    ax[0].plot([0,5.5],[1.5, 1.5 + 5.5*(-0.5)],'k--',alpha=0.8,label='slope = -1/2 Deco')
    ax[0].plot([0,5.5],[1.5, 1.5 + 5.5*(-0.66)],'k--',alpha=0.4,label='slope = -2/3 Turbulence')
    ax[0].set_ylabel(r'$\log( B(d) )$')
    ax[0].set_xlabel(r'$\log( d )$')
    ax[0].set_xlim((1,5.5))
    ax[0].set_ylim((-7,0.4))
    #ax[0].text(2.5, 0, 'slope = '+str(alpha[0]), fontsize=8)
    ax[0].axvline(2,color='gray')
    ax[0].axvline(fitxlim,color='gray')
    #ax[0].legend()

   
    y=np.log(binnedSr)#[np.logical_and(x>2, x<4)]
    y=y[gate]

    gate2 = np.isfinite(y)
    #print(np.logical_not(gate2))
    #print(y)
    if np.logical_not(gate2).any():
        print('values that get nan in log ',y[np.logical_not(gate2)])
        y=y[gate2]
        x=x[gate2]

    A = np.vstack([x, np.ones(len(x))]).T
    # Direct least square regression
    try:
        alpha = np.dot((np.dot(np.linalg.inv(np.dot(A.T,A)),A.T)),y)
        print('coef log S(r)',alpha)
        xline = np.arange(1,5.5,0.1) #np.array([1,5.5])
        ax[1].plot(xline, alpha[0]*xline + alpha[1], 'b--',label='slope fit '+"%.3f" % alpha[0])
        #axComp[0].plot(xline, alpha[0] + alpha[1]/xline, 'k--',label='slope fit '+"%.3f" % alpha[0])
    except np.linalg.LinAlgError as err:
        if 'Singular matrix' in str(err):
            print('singular')

    ax[1].set_ylabel(r'$\log( S_2(d) )$')
    ax[1].set_xlabel(r'$\log( d )$')
    ax[1].set_xlim((1,5.5))
    ax[1].set_ylim((-2,1.77))
    ax[1].plot([0,5.5],[-1.,-1. + 5.5*0.5],'k--',alpha=0.8,label='slope = 1/2 Deco')
    ax[1].plot([0,5.5],[-1.,-1. + 5.5*0.66],'k--',alpha=0.4,label='slope = 2/3 Turbulence')
    ax[1].axvline(2,color='gray')
    ax[1].axvline(fitxlim,color='gray')
    ax[1].legend()

    #print('binnedBdDict[0]',binnedBdDict[0])
    binnedSdmeanPlot = [np.mean(binnedSdDict[rk]) for rk in rsPlot]
    binnedSdstdPlot = [np.std(binnedSdDict[rk]) for rk in rsPlot]
    binnedSdsePlot = [3*np.std(binnedSdDict[rk])/np.sqrt(runs) for rk in rsPlot]
    axs1[1].errorbar(rsKeys,binnedSdmeanPlot,yerr=binnedSdsePlot,fmt='k.', ecolor='black',  capsize=3, capthick=1)
    axs1[1].loglog()
    
    xline = np.arange(1,5.5,0.1) #np.array([1,5.5])
    axs1[1].plot(np.exp(xline), np.exp(alpha[0]*xline + alpha[1]), '--',color='tab:blue',label='slope fit '+"%.3f" % alpha[0])
    axs1[1].plot(np.exp([0,5.5]),np.exp([-.8,-.8 + 5.5*(0.5)]),'--',color='tab:orange',alpha=0.8,label='slope = 1/2 Deco')
    axs1[1].plot(np.exp([0,5.5]),np.exp([-.8,-.8 + 5.5*(0.66)]),'--',color='tab:green',alpha=0.4,label='slope = 2/3 Turbulence')
    
    axs1[1].set_ylabel(r'$\log( \langle S(d) \rangle)$')
    axs1[1].set_xlabel(r'$\log( d )$')
    axs1[1].set_xticks(np.exp(np.arange(1.5,3.9,0.5)))
    axs1[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axs1[1].minorticks_off()
    axs1[1].set_xticklabels(np.arange(1.5,3.9,0.5))
    axs1[1].set_yticks(np.exp(np.arange(-2.,1.77,1.)))
    axs1[1].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axs1[1].set_yticklabels(np.arange(-2.,1.77,1.))
    axs1[1].set_xlim((np.exp(1.5),np.exp(3.9)))
    axs1[1].set_ylim((np.exp(-2),np.exp(1.77)))
    axs1[1].axvline(np.exp(2),color='gray')
    axs1[1].axvline(np.exp(fitxlim),color='gray')
    axs1[1].legend()


    #ax[0].plot(x, alpha[0]*x + alpha[1], '-g')
    #ax[0].text(4, 1, 'slope = '+"%.2f" % alpha[0], fontsize=8)
    #ax[0].legend()
    plt.savefig('structure'+str(lamda)+'.pdf')
plt.show()
