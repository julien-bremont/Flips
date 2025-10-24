import numpy as np
#from numba import jit

beta=1.
kappa=1.

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
           return 1,V[i],N[i]
       else:
           return 0,V[i],N[i]

Nlis=np.array([25,50,100])
np.random.shuffle(Nlis)
for n in Nlis:
  X0=[]
  sr=str(np.random.randint(10**5))
  for k in range(100001):
    V1=X_n(n)
    M=np.maximum.accumulate(V1)
    m=np.minimum.accumulate(V1)
    S=M-m
    val,x0,Ni=Splitting(S,V1,n)
    X0.append((x0/Ni,val))
    if k%100==0:
        np.save(f'Splitting_x0_n/SESRW/1/{n}/'+sr,X0)

