"""Equivalent single-CPU workloads for BenchmarkSpectralKernels."""
import time
import numpy as np

assert np.__version__ == "2.5.0"
print("NumPy",np.__version__)
np.show_config()
np.show_runtime()

def matrix(m,n,hermitian=False):
    state=123
    def nextval():
        nonlocal state
        state=(state*6364136223846793005+1442695040888963407) & ((1<<64)-1)
        return (state>>11)*2.**-53-.5
    a=np.array([complex(nextval(),nextval()) for _ in range(m*n)]).reshape(m,n)
    if hermitian:
        a=np.tril(a,-1)+np.tril(a,-1).conj().T+np.diag(a.diagonal().real)
    return a

cases=[]
for n in (32,96,192):
    a=matrix(n,n,True)
    cases.extend([(f"eigh_{n}",lambda a=a:np.linalg.eigh(a)),
                  (f"eigvalsh_{n}",lambda a=a:np.linalg.eigvalsh(a))])
for m,n in ((32,32),(96,64),(256,48)):
    a=matrix(m,n)
    cases.extend([(f"svd_{m}x{n}",lambda a=a:np.linalg.svd(a,full_matrices=False)),
                  (f"svdvals_{m}x{n}",lambda a=a:np.linalg.svd(a,compute_uv=False))])
v=(np.arange(128*1024)%137-68)/17.
z=v+1j*(np.arange(v.size)%97-48)/31.
a=v.reshape(128,1024)
c=z.reshape(128,1024)
half=c[:,:513]
prime=v[:4*4093].reshape(4,4093)
cases.extend([("fft_128x1024",lambda:np.fft.fft(c)),
              ("ifft_128x1024",lambda:np.fft.ifft(c)),
              ("rfft_128x1024",lambda:np.fft.rfft(a)),
              ("irfft_128x1024",lambda:np.fft.irfft(half,n=1024)),
              ("fft_4x4093",lambda:np.fft.fft(prime)),
              ("rfft_4x4093",lambda:np.fft.rfft(prime)),
              ("fft_65536",lambda:np.fft.fft(z[:65536]))])
for dtype in ("float32","float64"):
    x=a.astype(dtype)
    y=x+np.array(5,dtype=dtype)
    for name,fn in (("add",np.add),("sub",np.subtract),("mul",np.multiply),("div",np.divide)):
        cases.append((f"{name}_{dtype}_128k",lambda x=x,y=y,fn=fn:fn(x,y)))
for name,fn in cases:
    start=time.perf_counter_ns()
    result=fn()
    loops=max(1,min(10000,round(100_000_000/max(1,time.perf_counter_ns()-start))))
    for _ in range(5):
        start=time.perf_counter_ns()
        for _ in range(loops):
            result=fn()
        print(f"NumPySpectralKernels/{name} {loops} {(time.perf_counter_ns()-start)/loops:.0f} ns/op",flush=True)
