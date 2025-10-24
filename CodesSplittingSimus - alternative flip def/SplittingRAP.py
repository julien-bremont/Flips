import numpy as np
from numba import jit


@jit(nopython=True)
def X_n(n):
    LX=[0]
    X,V=0,0
    M,m=0,0
    T=0
    while (M-m)<=n+1:
        V+=2*np.random.randint(0,2)-1
        X+=V
        M=max(X,M)
        m=min(X,m)
        LX.append(X)
    return LX

@jit(nopython=True)
def Splitting(N,V,n):
       i=np.where(N>=n)[0][0]
       j=np.where(N>=n+1)[0][0]
       if V[i]*V[j]<=0:
           return 1
       else:
           return 0


listeN=np.int64(np.logspace(1,4,25))

P=np.zeros(len(listeN))
C=np.zeros(len(listeN))
sr=str(np.random.randint(10**5))

for k in range(10000):
    lX=np.array(X_n(max(listeN)))
    M=np.maximum.accumulate(lX)
    m=np.minimum.accumulate(lX)
    N=M-m
    for u in range(len(listeN)):
        n=listeN[u]
        val=Splitting(N,lX,n)
        if val>=0:
            C[u]+=1
            P[u]+=val
    np.save('Splitting/RAP/'+sr,[P,C])


