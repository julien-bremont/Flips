import numpy as np
#from numba import jit

beta=0.5
kappa=0.5

#@jit(nopython=True)
def X_n(n):
  lX=[0]
  L=np.zeros(2*n+2)
  m,M=0,0
  c=0
  tau=0
  while M-m<n+1:
     u=np.random.random()
     p1,p2=L[n+c-1],L[n+c+1]
     p1=np.exp(-beta*p1**kappa)
     p2=np.exp(-beta*p2**kappa)
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


#@jit(nopython=True)
def Splitting(N,V,n):
       i=np.searchsorted(N,n)
       j=np.searchsorted(N,n+1)
       if V[i]*V[j]<=0:
           return 1,V[i]
       else:
           return 0,V[i]


listeN=np.int64(np.logspace(1,3,25))

X0=[[] for k in range(len(listeN))]
sr=str(np.random.randint(10**5))

for k in range(100001):
    lX=np.array(X_n(max(listeN)))
    M=np.maximum.accumulate(lX)
    m=np.minimum.accumulate(lX)
    N=M-m
    for u in range(len(listeN)):
        n=listeN[u]
        val,x0=Splitting(N,lX,n)
        if val>=0:
            X0[u].append((x0,val))
    if k%100==0:
       np.save('Splittingx0/SESRW/beta_05_kappa_05/'+sr,X0)

