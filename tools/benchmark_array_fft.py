"""NumPy counterpart of BenchmarkArrayFFTKernels."""
import time
import numpy as np
assert np.__version__ == "2.5.0"
print("NumPy",np.__version__)
np.show_config()
np.show_runtime()
v=(np.arange(128*1024)%137-68)/17.
c=v+1j*(np.arange(v.size)%97-48)/31.
a=v.reshape(128,1024)
z=c.reshape(128,1024)
a32=a.astype("float32")
z32=z.astype("complex64")
cases=[]
for tag,x,q in (("64",a,z),("32",a32,z32)):
    cases.extend([(f"fft_batch_{tag}",lambda q=q:np.fft.fft(q)),
                  (f"ifft_batch_{tag}",lambda q=q:np.fft.ifft(q)),
                  (f"rfft_batch_{tag}",lambda x=x:np.fft.rfft(x)),
                  (f"irfft_batch_{tag}",lambda q=q:np.fft.irfft(q[:,:513],n=1024))])
for n in (4096,65536):
    cases.extend([(f"rfft_single_{n}",lambda n=n:np.fft.rfft(v[:n])),
                  (f"irfft_single_{n}",lambda n=n:np.fft.irfft(c[:n//2+1],n=n))])
cases.extend([("fft_single_65536",lambda:np.fft.fft(c[:65536])),
              ("rfft_prime_single_4093",lambda:np.fft.rfft(v[:4093])),
              ("fft_prime_4x4093",lambda:np.fft.fft(v[:4*4093].reshape(4,4093))),
              ("rfft_prime_4x4093",lambda:np.fft.rfft(v[:4*4093].reshape(4,4093)))])
for tag,x,d in (("f32_f64",a32,"float64"),("f64_f32",a,"float32"),
                ("c64_c128",z32,"complex128"),("c128_c64",z,"complex64"),
                ("f32_c128",a32,"complex128"),("transpose_f32_f64",a32.T,"float64")):
    cases.append(("cast_"+tag,lambda x=x,d=d:x.astype(d,order="C")))
for tag,x,d in (("64",a,"float64"),("32",a32,"float32")):
    scalar=np.array(1.25,dtype=d)
    cases.extend([(f"scalar_add_{tag}",lambda x=x,scalar=scalar:x+scalar),
                  (f"scalar_div_{tag}",lambda x=x,scalar=scalar:scalar/x),
                  (f"row_add_{tag}",lambda x=x:x+x[0]),
                  (f"dense_mul_{tag}",lambda x=x:x*x)])
with np.errstate(all="ignore"):
    for name,fn in cases:
        start=time.perf_counter_ns()
        result=fn()
        loops=max(1,min(10000,round(100_000_000/max(1,time.perf_counter_ns()-start))))
        for _ in range(5):
            start=time.perf_counter_ns()
            for _ in range(loops):result=fn()
            print(f"NumPyArrayFFTKernels/{name} {loops} {(time.perf_counter_ns()-start)/loops:.0f} ns/op",flush=True)
