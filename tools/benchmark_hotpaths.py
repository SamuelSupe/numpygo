#!/usr/bin/env python3
"""Same inputs as BenchmarkHotPaths, measured in single-threaded NumPy."""
import sys
import time
import warnings
import numpy as np

assert np.__version__ == "2.5.0"
print("NumPy", np.__version__)
np.show_config()
try:
    np.show_runtime()
except ImportError:
    pass

def matrix(n):
    i,j=np.indices((n,n))
    a=((np.minimum(i,j)*101+np.maximum(i,j)*79)%31-15).astype("f8")/31
    a.flat[::n+1]+=n
    return a

n=64
a=matrix(n)
i,j=np.indices((n,n))
c=a+1j*(i-j)/(n*n)
rhs=np.full((64,n,1),1.)
crhs=np.full((64,n,1),1.+.5j)
mat=matrix(192)
x=((np.arange(128*1024)%137-68)/17.).reshape(128,1024)
xt=x.T
prime=x.reshape(-1)[:4*4093].reshape(4,4093)
other=np.full((128,1),1.25)
dst=np.empty_like(xt,order="K")
src=np.full((1024,128),1.25)
cases=[
    ("solve_broadcast_f64",lambda:np.linalg.solve(a,rhs)),
    ("solve_broadcast_c128",lambda:np.linalg.solve(c,crhs)),
    ("cholesky_f64",lambda:np.linalg.cholesky(mat)),
    ("det_f64",lambda:np.linalg.det(mat)),
    ("rfft_dense",lambda:np.fft.rfft(x)),
    ("rfft_strided",lambda:np.fft.rfft(xt,axis=0)),
    ("fft_prime4093",lambda:np.fft.fft(prime)),
    ("rfft_prime4093",lambda:np.fft.rfft(prime)),
    ("add_transposed",lambda:np.add(xt,.5,order="C")),
    ("mul_broadcast",lambda:np.multiply(x,other)),
    ("assign_strided",lambda:np.copyto(dst,src)),
]
for cls in (np.random.PCG64,np.random.MT19937,np.random.Philox,np.random.SFC64):
    bit=cls(123)
    cases.append(("raw_"+cls.__name__.lower(),lambda bit=bit:bit.random_raw(1<<18)))
g=np.random.default_rng(123)
mu=x[0:1,:]
sigma=np.full((128,1),1.)
cases+= [
    ("integers_i8",lambda:g.integers(-50,50,size=1<<18,dtype="i1")),
    ("integers_i64",lambda:g.integers(-1000000,1000000,size=1<<18,dtype="i8")),
    ("normal_broadcast",lambda:g.normal(mu,sigma)),
]
signal=x.reshape(-1)
spectrum=(signal[:4*4093]+1j*signal[17:4*4093+17]).reshape(4,4093)
half=spectrum[:,:2047]
dx=np.random.PCG64DXSM(123)
legacy=np.random.RandomState(123)
lows=(-np.arange(128)-10).astype("i2").reshape(128,1)
hi=np.int16(100)
extra=[
    ("ifft_prime4093",lambda:np.fft.ifft(spectrum)),
    ("irfft_prime4093",lambda:np.fft.irfft(half,n=4093)),
    ("raw_pcg64dxsm",lambda:dx.random_raw(1<<18)),
    ("legacy_i16",lambda:legacy.randint(-3000,3000,size=1<<18,dtype="i2")),
    ("integers_array",lambda:g.integers(lows,hi,size=(128,1024),dtype="i2")),
    ("astype_strided",lambda:xt.astype("f8",order="C")),
]
if "--additional" in sys.argv:
    cases=extra
elif "--all" in sys.argv:
    cases+=extra

with warnings.catch_warnings():
    warnings.simplefilter("ignore",RuntimeWarning)
    for name,fn in cases:
        start=time.perf_counter_ns()
        result=fn()
        elapsed=max(1,time.perf_counter_ns()-start)
        loops=max(1,min(10000,round(100_000_000/elapsed)))
        for _ in range(5):
            start=time.perf_counter_ns()
            for _ in range(loops):
                result=fn()
            ns=(time.perf_counter_ns()-start)/loops
            print(f"NumPyHotPaths/{name} {loops} {ns:.0f} ns/op",flush=True)
