import numpy as np
from numba import jit


@jit(nopython=True)
def X_n(n):
  lX=[0]  
  L=np.zeros(2*n+2)
  m,M=0,0
  c=0
  tau=0
  while M-m<n+1:
     u=np.random.random()
     p1,p2=L[n+c-1],L[n+c+1]
     p1=np.exp(-p1)
     p2=np.exp(-p2)
     p1=p1/(p1+p2)
     if u<p1:
         c-=1
         L[n+c]+=1
         m=min(c,m)
     else:
         c+=1
         L[n+c]+=1
         M=max(c,M)
     lX.append(c)
  return lX


@jit(nopython=True)
def Splitting(N,V,n):
       i=np.argmax(N==n)
       j=np.argmax(V[i:]==0)
       if N[j]==n:
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
    np.save('Splitting/TSAW/alternative-'+sr,[P,C])

