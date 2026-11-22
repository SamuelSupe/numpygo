#!/usr/bin/env python3
"""NumPy counterpart of BenchmarkNativeKernels (single-CPU comparison)."""
import time
import numpy as np

assert np.__version__ == "2.5.0"
print("NumPy",np.__version__)
np.show_config()
np.show_runtime()

v=(np.arange(128*1024)%137-68)/17.
z=v+1j*(np.arange(v.size)%97-48)/31.
a=v.reshape(128,1024)
c=z.reshape(128,1024)
at=a.T
half=c[:,:513]
big=((np.arange(1024*1024)%763)/13.).reshape(1024,1024)
big_t=big.T
dst=np.empty_like(big_t,order="K")
src=np.full((1024,1024),1.25)
g=np.random.default_rng(123)
dx=np.random.Generator(np.random.PCG64DXSM(123))
p=np.random.Philox(123)
n=192
i,j=np.indices((n,n))
mat=((np.minimum(i,j)*101+np.maximum(i,j)*79)%31-15)/31.+1j*(i-j)/(n*n)
mat.flat[::n+1]+=n
rhs=np.full((n,16),1.+.25j)
mat32=mat.astype("complex64")
small=mat[:64,:64]
small32=small.astype("complex64")
cases=[
    ("rfft_1024",lambda:np.fft.rfft(a)),
    ("rfft_transposed_1024",lambda:np.fft.rfft(at,axis=0)),
    ("irfft_1024",lambda:np.fft.irfft(half,n=1024)),
    ("fft_1024",lambda:np.fft.fft(c)),
    ("ifft_1024",lambda:np.fft.ifft(c)),
    ("rfft_512",lambda:np.fft.rfft(a,n=512)),
    ("copy_transposed_1m",lambda:big_t.astype("f8",order="C")),
    ("assign_transposed_1m",lambda:np.copyto(dst,src)),
    ("assign_same_order_1m",lambda:np.copyto(dst,big_t)),
    ("add_transposed_1m",lambda:np.add(big_t,.5,order="C")),
    ("pcg64_int64_small_range",lambda:g.integers(-1000000,1000000,size=1<<18)),
    ("pcg64_int64_large_range",lambda:g.integers(-1<<55,1<<55,size=1<<18)),
    ("pcg64_double",lambda:g.random(1<<18)),
    ("pcg64_raw",lambda:g.bit_generator.random_raw(1<<18)),
    ("pcg64dxsm_int64",lambda:dx.integers(-1000000,1000000,size=1<<18)),
    ("philox_raw",lambda:p.random_raw(1<<18)),
    ("complex_solve_192x16",lambda:np.linalg.solve(mat,rhs)),
    ("complex_cholesky_192",lambda:np.linalg.cholesky(mat)),
    ("complex_matmul_192",lambda:mat@mat),
    ("complex64_matmul_192",lambda:mat32@mat32),
    ("rfft_single_64",lambda:np.fft.rfft(v[:64])),
    ("rfft_single_1024",lambda:np.fft.rfft(v[:1024])),
    ("rfft_single_65536",lambda:np.fft.rfft(v[:65536])),
    ("fft_single_65536",lambda:np.fft.fft(z[:65536])),
    ("complex_matmul_64",lambda:small@small),
    ("complex64_matmul_64",lambda:small32@small32),
]
for name,fn in cases:
    start=time.perf_counter_ns()
    result=fn()
    loops=max(1,min(10000,round(100_000_000/max(1,time.perf_counter_ns()-start))))
    for _ in range(5):
        start=time.perf_counter_ns()
        for _ in range(loops):
            result=fn()
        print(f"NumPyNativeKernels/{name} {loops} {(time.perf_counter_ns()-start)/loops:.0f} ns/op",flush=True)
