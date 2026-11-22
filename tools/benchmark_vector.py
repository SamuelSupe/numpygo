"""NumPy counterpart of BenchmarkVectorKernels."""
import time
import numpy as np

assert np.__version__ == "2.5.0"
print("NumPy", np.__version__)
np.show_runtime()
i = np.arange(128*1024)
x, y = (i % 137-68)/17., (i % 149-74)/23.
c, z = x+1j*((i % 97-48)/31.), (y+.51)+1j*((i % 101-50)/27.+.31)
cases = []
for dtype in ("float32", "float64", "complex64", "complex128"):
    complex_type = dtype.startswith("complex")
    a, b = (c.astype(dtype), z.astype(dtype)) if complex_type else (x.astype(dtype), y.astype(dtype))
    names = ("add", "subtract", "multiply", "divide") if complex_type else ("exp", "minimum", "maximum")
    out = np.zeros_like(a)
    for name in names:
        f = getattr(np, name)
        args = (a,) if name == "exp" else (a,b)
        cases.append((f"{name}_{dtype}_out",lambda f=f,args=args,out=out:f(*args,out=out)))
        cases.append((f"{name}_{dtype}_alloc",lambda f=f,args=args:f(*args)))
for n in (1000,1536,150,1500,729,4093,1024):
    rows = 4 if n==4093 else 128
    i = np.arange(rows*n).reshape(rows,n)
    for dtype in ("float32","float64"):
        cd = "complex64" if dtype=="float32" else "complex128"
        a = ((i%137-68)/17.).astype(dtype)
        c = ((i%137-68)/17.+1j*((i%97-48)/31.)).astype(cd)
        co = np.zeros_like(c)
        half = c[:,:n//2+1].copy()
        ho = np.zeros_like(half)
        ro = np.zeros_like(a)
        tag = f"{dtype}_{n}"
        cases += [("fft_"+tag,lambda c=c,co=co:np.fft.fft(c,out=co)),
                  ("rfft_"+tag,lambda a=a,ho=ho:np.fft.rfft(a,out=ho)),
                  ("irfft_"+tag,lambda half=half,n=n,ro=ro:np.fft.irfft(half,n=n,out=ro))]
with np.errstate(all="ignore"):
    for name, fn in cases:
        t = time.perf_counter_ns()
        result = fn()
        loops = max(1,min(10000,round(100_000_000/max(1,time.perf_counter_ns()-t))))
        for _ in range(5):
            t = time.perf_counter_ns()
            for _ in range(loops): result=fn()
            print(f"NumPyVectorKernels/{name} {loops} {(time.perf_counter_ns()-t)/loops:.0f} ns/op",flush=True)
