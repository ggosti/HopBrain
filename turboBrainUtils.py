import numpy as np

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
    states = np.zeros((passi,N))
    s0 = 2*np.random.binomial(1, 0.5, N)-1
    states[0,:] = s0

    for t in range(1,passi):
        s1 = trans(s0,J,N,typ = 1, thr = 0) #getCycles.transPy(s0,J,N,typ = 1, thr = 0)
        #print(s1)
        s0=s1.T
        states[t,:] = s1.T
    Cdt1 = np.mean(states[1:,:]*states[:-1,:],axis=1)
    return states, Cdt1
