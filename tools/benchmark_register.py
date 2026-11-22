"""NumPy counterpart of BenchmarkRegisterKernels."""
import time
import numpy as np

assert np.__version__ == "2.5.0"
print("NumPy",np.__version__)
np.show_runtime()
cases=[]
for n in (32,257,128*1024):
    i=np.arange(n)
    a=((i%137-68)/17.+1j*((i%97-48)/31.)).astype("complex64")
    b=((i%149-74)/23.+.51+1j*((i%101-50)/27.+.31)).astype("complex64")
    out=np.zeros_like(a)
    for name in ("multiply","divide"):
        f=getattr(np,name)
        cases += [(f"{name}_c64_{n}_out",lambda f=f,a=a,b=b,out=out:f(a,b,out=out)),
                  (f"{name}_c64_{n}_alloc",lambda f=f,a=a,b=b:f(a,b))]
fft_cases=((81,128,True,False,True),(125,128,True,False,True),(729,128,True,True,True),
           (1125,128,True,False,True),(2025,64,True,False,True),(3125,32,True,True,True),
           (2025,1,True,False,False),(3125,1,True,True,False),
           (150,128,True,True,False),(1000,128,True,True,True),(1536,128,True,True,True),
           (1500,128,True,True,False),(1536,1,False,True,True))
for n,rows,real,cplx,both in fft_cases:
    i=np.arange(rows*n).reshape(rows,n)
    for dtype in (("float32","float64") if both else ("float32",)):
        cd="complex64" if dtype=="float32" else "complex128"
        a=((i%137-68)/17.).astype(dtype)
        c=((i%137-68)/17.+1j*((i%97-48)/31.)).astype(cd)
        tag=f"{dtype}_{rows}x{n}"
        if cplx:
            co=np.zeros_like(c)
            cases.append(("fft_"+tag,lambda c=c,co=co:np.fft.fft(c,out=co)))
        if real:
            half=c[:,:n//2+1].copy()
            ho=np.zeros_like(half)
            ro=np.zeros_like(a)
            cases += [("rfft_"+tag,lambda a=a,ho=ho:np.fft.rfft(a,out=ho)),
                      ("irfft_"+tag,lambda half=half,ro=ro,n=n:np.fft.irfft(half,n=n,out=ro))]
with np.errstate(all="ignore"):
    for name,fn in cases:
        t=time.perf_counter_ns()
        result=fn()
        loops=max(1,min(20000,round(100_000_000/max(1,time.perf_counter_ns()-t))))
        for _ in range(5):
            t=time.perf_counter_ns()
            for _ in range(loops):result=fn()
            print(f"NumPyRegisterKernels/{name} {loops} {(time.perf_counter_ns()-t)/loops:.0f} ns/op",flush=True)
