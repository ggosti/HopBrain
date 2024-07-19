import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd
from scipy.spatial import distance
import time


import turboBrainUtils as tb 


def computeBr(statesRun,uniqDist,iListList,jListList):
    BdRun = []
    for d,iList,jList in zip(uniqDist,iListList,jListList):
        #print(d,np.sum(dist==d))
        cors = np.mean(statesRun[np.array(iList)]*statesRun[np.array(jList)])
        BdRun.append(cors)
    return BdRun

runs = 40#1000#40
#passi = 100#200
autapse = True
randomize = False#True #False
parcelsNames = ['Schaefer2018_1000Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv',
                'Schaefer2018_900Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv',
                'Schaefer2018_800Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv',
                'Schaefer2018_700Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv',
                'Schaefer2018_600Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv',
                'Schaefer2018_500Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv',
                'Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv',
                'Schaefer2018_300Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv',
                'Schaefer2018_200Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv',
                'Schaefer2018_100Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv']

fitxlim = 3.5 
nBins = 100

strng = '' 
if autapse: strng = strng+'-autapse'
if randomize: strng = strng+'-randomizeJ'
strng = strng+'-N'
print(strng)

for parcelsName in parcelsNames:
    # # Parcellizzazione
    # https://www.sciencedirect.com/science/article/pii/S2211124720314601?via%3Dihub
    df = pd.read_csv('Centroid_coordinates/'+parcelsName)
    df.head()
    X = df['R']
    Y = df['A']
    Z = df['S']
    N=len(X)

    coords = np.array([X,Y,Z]).T
    dist = distance.cdist(coords, coords, 'euclidean')
    uniqDist,iListList,jListList = tb.sortIJbyDist(dist,N)


    h,bins,f=plt.hist(uniqDist,bins=nBins,histtype="step", density=True)

    df = pd.read_csv('data/lamdaValues'+strng+'.csv')
    print('read lamdaValues'+strng+'.csv')

    dfN = df[df['N']==N]
    lambdas = np.unique(dfN['lambdas'].values)
    #lambdas = np.unique(df['lambdas'].values)
    print('lambdas',len(lambdas))
    print(lambdas)
    
    strngTemp = strng+'-'+str(N)
    print('read SRuns'+strngTemp+'.csv')
    df3 = pd.read_csv('data/SRuns'+strngTemp+'.csv')
    print('read SRuns'+strngTemp+'.csv',df3.shape)

    states = np.zeros((len(lambdas),runs,N))
    SName = list(df3.columns)#[1:]

    t0 = time.time()
    for lindx,lambd in enumerate(lambdas):
        print(lindx,lambd,1./lambd)
        #print(BdName[40*lindx:40*(lindx+1)])
        cols = SName[runs*lindx:runs*(lindx+1)]
        states[lindx,:,:] = df3[cols].values.T
    print(states.shape)
    t1 = time.time()
    print('load states time ',t1-t0)

    rs = [0.5*(ra+rb) for ra,rb in zip(bins[:-1],bins[1:])]

    arBinnedBd_mean = np.empty((len(lambdas),nBins))
    arBinnedBd_std = np.empty((len(lambdas),nBins))
    arBinnedSd_mean = np.empty((len(lambdas),nBins))
    arBinnedSd_std = np.empty((len(lambdas),nBins))

    arBinnedBd_mean[:] = np.nan
    arBinnedBd_std[:] = np.nan
    arBinnedSd_mean[:] = np.nan
    arBinnedSd_std[:] = np.nan

    #linxs = range(len(lambdas))

    t0 = time.time()
    for i in range(len(lambdas)):
        BdRuns = []
        SdRuns = []
        print(i,lambdas[i],1./lambdas[i])
        #binnedBd = {r:[] for r in rs}
        for r in range(runs):
            BdRun = computeBr(states[i,r,:],uniqDist,iListList,jListList)
            BdRuns.append(BdRun)
            SdRuns.append(BdRun[0]-BdRun[2:])
        print('computed Sd')
        binnedBd_mean = []
        binnedBd_std = []
        binnedSd_mean = []
        binnedSd_std = []
        for j,ra,rb in zip(range(nBins), bins[:-1],bins[1:]):
            gate = np.logical_and(uniqDist>=ra, uniqDist<=rb)
            if np.sum(gate)>0:
                binnedBd_mean.append(np.mean(np.array(BdRuns)[:,gate]))
                binnedBd_std.append(np.std(np.array(BdRuns)[:,gate]))
            else:
                binnedBd_mean.append(np.nan)
                binnedBd_std.append(np.nan)
            gate2 = np.logical_and(uniqDist[2:]>=ra, uniqDist[2:]<=rb)
            if np.sum(gate2)>0:    
                binnedSd_mean.append(np.mean(np.array(SdRuns)[:,gate2]))
                binnedSd_std.append(np.std(np.array(SdRuns)[:,gate2]))
            else:
                binnedSd_mean.append(np.nan)
                binnedSd_std.append(np.nan)
                #print(np.mean(np.array(BdRuns)[:,gate]))
        arBinnedBd_mean[i,:] = binnedBd_mean
        arBinnedBd_std[i,:] = binnedBd_std
        arBinnedSd_mean[i,:] = binnedSd_mean
        arBinnedSd_std[i,:] = binnedSd_std
        #axs[i].errorbar(rs,binnedSd_mean,yerr=binnedSd_std)
        #see to believe
        if i in [7,8]:
            f,axs = plt.subplots(2,2,figsize=(5,10))
            axs[0,0].set_title('delta '+str(1./lambdas[i]))
            axs[0,0].scatter(uniqDist,BdRun,alpha=0.01)
            axs[1,0].scatter(uniqDist[2:],BdRun[0]-BdRun[2:],alpha=0.01)
            axs[0,1].errorbar(rs,arBinnedBd_mean[i,:],yerr=arBinnedBd_std[i,:].tolist(),fmt='k.', markersize = 1. , ecolor='black', elinewidth=0.5 , capsize=2, capthick=.5)
            axs[0,1].semilogx()
            axs[0,1].set_xticks([10,20,30,40])
            axs[0,1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            axs[0,1].minorticks_off()
            axs[0,1].set_xticklabels([10,20,30,40])
            axs[0,1].set_xlim((np.exp(1.5),np.exp(4.)))
            axs[0,1].set_ylim((-0.1,1.1))
            axs[0,1].set_xlabel(r'$d$ (mm)')
            axs[1,1].errorbar(rs,arBinnedSd_mean[i,:],yerr=arBinnedSd_std[i,:].tolist(),fmt='k.', markersize = 1. , ecolor='black', elinewidth=0.5 , capsize=2, capthick=.5)
            axs[1,1].axvline(np.exp(2),linestyle='--',color='gray')
            axs[1,1].axvline(np.exp(fitxlim),linestyle='--',color='gray')
            axs[1,1].set_xticks(np.exp(np.arange(1.5,3.9,0.5)))
            axs[1,1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            axs[1,1].minorticks_off()
            axs[1,1].set_xticklabels(np.arange(1.5,3.9,0.5))
            axs[1,1].set_yticks(np.exp(np.arange(-2.,1.77,1.)))
            axs[1,1].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            axs[1,1].set_yticklabels(np.arange(-2.,1.77,1.))
            axs[1,1].set_xlim((np.exp(1.5),np.exp(3.9)))
            axs[1,1].set_ylim((np.exp(-2),np.exp(1.77)))
            #axs[i].set_xlim((np.exp(1),np.exp(4)))
            axs[1,1].set_xlabel(r'$\log(d)$')
            axs[1,1].set_ylabel(r'$\log( \langle S(d) \rangle)$')
            axs[1,1].loglog()
            itc = -1.5
            axs[1,1].plot(np.exp([0,5.5]),np.exp([itc,itc + 5.5*(0.4)]),'--',color='tab:blue',label='slope = 2/5')
            axs[1,1].plot(np.exp([0,5.5]),np.exp([itc,itc + 5.5*(0.5)]),'--',color='tab:orange',label='slope = 1/2 Deco')
            axs[1,1].plot(np.exp([0,5.5]),np.exp([itc,itc + 5.5*(0.66)]),'--',color='tab:green',label='slope = 2/3 Turbulence')
    t1 = time.time()
    print('make sd time',t1-t0)
    np.save('data/binnedBd_mean'+strngTemp,arBinnedBd_mean)
    np.save('data/binnedBd_std'+strngTemp,arBinnedBd_std)
    np.save('data/binnedSd_mean'+strngTemp,arBinnedSd_mean)
    np.save('data/binnedSd_std'+strngTemp,arBinnedSd_std)

plt.show()
