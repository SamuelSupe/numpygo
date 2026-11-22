#!/usr/bin/env python3
"""NumPy 2.5.0 counterpart of BenchmarkDecompositionKernels."""
import time
import numpy as np

assert np.__version__=="2.5.0"
print("NumPy",np.__version__)
np.show_config()
np.show_runtime()

def matrix(m,n):
    state=123
    def sample():
        nonlocal state
        state=(state*6364136223846793005+1442695040888963407)&((1<<64)-1)
        return (state>>11)*2.**-53-.5
    r=np.empty(m*n,dtype="f8")
    c=np.empty(m*n,dtype="c16")
    for i in range(r.size):
        r[i]=sample()
        c[i]=complex(r[i],sample())
    return r.reshape(m,n),c.reshape(m,n)

real_qr,complex_qr=matrix(512,64)
_,svd_a=matrix(96,64)
real_eig,complex_eig=matrix(96,96)
herm=np.tril(complex_eig)+np.tril(complex_eig,-1).conj().T
herm.flat[::97]=herm.diagonal().real
sym=np.tril(real_eig)+np.tril(real_eig,-1).T
v=(np.arange(1<<20)%137-68)/17.
a=v.reshape(1024,1024)
t=a.T
prime=v[:4*4093].reshape(4,4093)
cases=[
    ("qr_real_tall",lambda:np.linalg.qr(real_qr,mode="reduced")),
    ("qr_real_r",lambda:np.linalg.qr(real_qr,mode="r")),
    ("qr_complex_tall",lambda:np.linalg.qr(complex_qr,mode="reduced")),
    ("qr_complex_r",lambda:np.linalg.qr(complex_qr,mode="r")),
    ("svd_complex_reduced",lambda:np.linalg.svd(svd_a,full_matrices=False)),
    ("svd_complex_values",lambda:np.linalg.svd(svd_a,compute_uv=False)),
    ("svd_complex_full",lambda:np.linalg.svd(svd_a,full_matrices=True)),
    ("eig_complex",lambda:np.linalg.eig(complex_eig)),
    ("eigvals_complex",lambda:np.linalg.eigvals(complex_eig)),
    ("eigvals_real",lambda:np.linalg.eigvals(real_eig)),
    ("eigh_complex",lambda:np.linalg.eigh(herm)),
    ("eigvalsh_complex",lambda:np.linalg.eigvalsh(herm)),
    ("eigvalsh_real",lambda:np.linalg.eigvalsh(sym)),
    ("svd_hermitian_values",lambda:np.linalg.svd(herm,compute_uv=False,hermitian=True)),
    ("sum_all_1m",lambda:np.sum(a)),
    ("sum_axis0_1m",lambda:np.sum(a,axis=0)),
    ("sum_transposed_axis1_1m",lambda:np.sum(t,axis=1)),
    ("mean_axis1_1m",lambda:np.mean(a,axis=1)),
    ("fft_prime4093",lambda:np.fft.fft(prime)),
    ("rfft_prime4093",lambda:np.fft.rfft(prime)),
]
for name,fn in cases:
    start=time.perf_counter_ns()
    result=fn()
    loops=max(1,min(10000,round(100_000_000/max(1,time.perf_counter_ns()-start))))
    for _ in range(5):
        start=time.perf_counter_ns()
        for _ in range(loops):
            result=fn()
        print(f"NumPyDecompositionKernels/{name} {loops} {(time.perf_counter_ns()-start)/loops:.0f} ns/op",flush=True)
