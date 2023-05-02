import numpy as np
import torch
import matplotlib.pyplot as plt

# make J as defined by Deco
def makeJ(dist,lamda):
    J = np.exp(-lamda*dist)
    np.fill_diagonal(J, 0)
    return J

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

def run(J, N, passi):
    statesRun = np.zeros((passi,N))
    s0 = 2*np.random.binomial(1, 0.5, N)-1
    statesRun[0,:] = s0

    for t in range(1,passi):
        s1 = trans(s0,J,N,typ = 1, thr = 0) #getCycles.transPy(s0,J,N,typ = 1, thr = 0)
        #print(s1)
        s0=s1.T
        statesRun[t,:] = s1.T
    Cdt1 = np.mean(statesRun[1:,:]*statesRun[:-1,:],axis=1)
    return statesRun, Cdt1

def computeBr(statesRun,uniqDist,iListList,jListList):
    BdRun = []
    for d,iList,jList in zip(uniqDist,iListList,jListList):
        #print(d,np.sum(dist==d))
        cors = []
        for i,j in zip(iList,jList):
            #cor = np.mean(states[r,:,i]*states[r,:,j])
            #WARNING: given that all runs converge to a stationary state
            cor = np.mean(statesRun[-1,i]*statesRun[-1,j])
            cors.append(cor)
        BdRun.append(np.mean(cors))
    return BdRun

def computeBrCuda(statesRun,uniqDist,iListList,jListList):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BdRun = []
    tStatesRun = torch.from_numpy(statesRun)
    tStatesRun = tStatesRun.to(device)
    print('is in cuda',tStatesRun.is_cuda) 
    for d,iList,jList in zip(uniqDist,iListList,jListList):
        #print(d,np.sum(dist==d))
        cors = []
        for i,j in zip(iList,jList):
            #cor = np.mean(states[r,:,i]*states[r,:,j])
            #WARNING: given that all runs converge to a stationary state
            cor = torch.mean(tStatesRun[-1,i]*tStatesRun[-1,j])
            cors.append(cor.cpu().numpy())
        BdRun.append(np.mean(cors))
    return BdRun


# plot stuff

def plotInitalJ(X, Y, Z,dist,J):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X, Y, Z)

    ax.set_xlabel('R')
    ax.set_ylabel('A')
    ax.set_zlabel('S')

    f,[ax0,ax1,ax2]=plt.subplots(1,3)
    ax0.imshow(dist)

    ax1.set_title('J = np.exp(-0.18*dist) ')
    ax1.imshow(J)

    ax2.set_title('J>0.04')
    ax2.imshow(J>0.04)
