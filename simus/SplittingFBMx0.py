#from fbm import FBM
import numpy as np
#from numba import jit


#@jit(nopython=True)
def Splitting(N,V,n,H):
    if N[-1]>=n+1:
       i=np.searchsorted(N,n)
       j=np.searchsorted(N,n+1)
       if V[i]*V[j]<=0:
           return 1,V[i],N[i]
       else:
           return 0,V[i],N[i]
    else:
       return -1,0,0

# Generate the initial trajectory
def generate_gaussian(N):
    Vj = np.zeros((2 * N, 2), dtype=np.complex128)
    Vj[0, 0] = np.random.standard_normal()
    Vj[N, 0] = np.random.standard_normal()
    for i in range(1, N):
        Vj1 = np.random.standard_normal()
        Vj2 = np.random.standard_normal()
        Vj[i][0] = Vj1
        Vj[i][1] = Vj2
        Vj[2 * N - i][0] = Vj1
        Vj[2 * N - i][1] = Vj2
    return Vj

def davies_harte(T, N, H,lk):
    '''
    Generates sample paths of fractional Brownian Motion using the Davies Harte method from the list of Gaussian rv

    Args:
        T (float): Length of time
        N (int): Number of time steps within timeframe
        H (float): Hurst parameter
        Vj (2N, 2 array): periodic matrix of Gaussian rv

    Returns:
        numpy.ndarray: Generated path of fractional Brownian Motion
    '''
    Vj=generate_gaussian(N)
    # Step 2: Compute Z
    wk = np.zeros(2 * N, dtype=np.complex128)
    wk[0] = np.sqrt((lk[0] / (2 * N))) * Vj[0][0]
    wk[1:N] = np.sqrt(lk[1:N] / (4 * N)) * ((Vj[1:N].T[0]) + (complex(0, 1) * Vj[1:N].T[1]))
    wk[N] = np.sqrt((lk[0] / (2 * N))) * Vj[N][0]
    wk[N + 1:2 * N] = np.sqrt(lk[N + 1:2 * N] / (4 * N)) * (
                np.flip(Vj[1:N].T[0]) - (complex(0, 1) * np.flip(Vj[1:N].T[1])))

    Z = np.fft.fft(wk)
    fGn = np.real(Z[0:N])
    fBm = np.cumsum(fGn)*(T/N)**H
    path = np.array([0] + list(fBm))

    return path

H=0.4

N=10**4 #H=0.7 ok
#N=10**6

# Step 0: Initialize parameters
gamma = lambda k, H: 0.5 * (np.abs(k - 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H) + np.abs(k + 1) ** (2 * H))
g = [gamma(k, H) for k in range(0, N)]
r = np.array(g + [0] + g[::-1][0:N - 1])

    # Step 1: Compute eigenvalues
j = np.arange(0, 2 * N)
k = 2 * N - 1
lk = np.fft.fft(r * np.exp(2 * np.pi * complex(0, 1) * k * j * (1 / (2 * N))))[::-1]



Nlis=np.array([25,50,100])
np.random.shuffle(Nlis)
for n in Nlis:
  X0=[]
  sr=str(np.random.randint(10**5))
  for k in range(100001):
    V1=davies_harte(N, N, H,lk)
    M=np.maximum.accumulate(V1)
    m=np.minimum.accumulate(V1)
    S=M-m
    val,x0,Ni=Splitting(S,V1,n,H)
    X0.append((x0/Ni,val))
    if k%100==0:
        np.save(f'Splitting_x0_n/FBM/{H}/{n}/'+sr,X0)

