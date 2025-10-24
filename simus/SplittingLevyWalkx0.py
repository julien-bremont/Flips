import numpy as np
from numba import jit


@jit(nopython=True)
def X_n(n,alpha):
    LX=[0]
    X=0
    M,m=0,0
    T=0
    while (M-m)<=n+1:
        sgn=(2*np.random.randint(0,2)-1)
        Time=np.random.zipf(1+alpha)
        LX+=list(X+sgn*np.arange(1,Time+1))
        X+=sgn*Time
        M=max(X,M)
        m=min(X,m)
    return LX

@jit(nopython=True)
def Splitting(N,V,n):
       i=np.where(N>=n)[0][0]
       j=np.where(N>=n+1)[0][0]
       if np.sign(V[i])*np.sign(V[j])<0:
           return 1,V[i]
       else:
           return 0,V[i]


listeN=np.int64(np.logspace(1,3,25))
X0=[[] for n in range(len(listeN))]


sr=str(np.random.randint(10**5))
alpha=3

for k in range(100001):
    lX=np.array(X_n(max(listeN),alpha))
    M=np.maximum.accumulate(lX)
    m=np.minimum.accumulate(lX)
    N=M-m
    for u in range(len(listeN)):
        n=listeN[u]
        val,x0=Splitting(N,lX,n)
        X0[u].append((np.int64(x0),val))
    if k%100==0:
       np.save('Splitting_x0/LevyWalk/3/'+sr,X0)

