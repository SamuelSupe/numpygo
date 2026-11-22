"""NumPy counterpart of BenchmarkDirectKernels."""
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
a32=a.astype("float32")
c32=c.astype("complex64")
mask=(np.arange(v.size)%256<128).reshape(a.shape)
alt=(np.arange(v.size)%2==0).reshape(a.shape)
cases=[]
for tag,x,d in (("f32_f64",a32,"float64"),("f64_f32",a,"float32"),
                ("c64_c128",c32,"complex128"),("c128_c64",c,"complex64"),
                ("reverse_rows_f32_f64",a32[::-1],"float64")):
    cases.append(("cast_"+tag,lambda x=x,d=d:x.astype(d,order="C")))
for tag,x in (("64",a),("32",a32)):
    out=np.zeros_like(x)
    alias=x.copy()
    scalar=np.array(1.25,dtype=x.dtype)
    one=np.array(1,dtype=x.dtype)
    cases.extend([(f"out_add_{tag}",lambda x=x,out=out:np.add(x,x,out=out)),
                  (f"out_scalar_div_{tag}",lambda x=x,out=out,scalar=scalar:np.divide(scalar,x,out=out)),
                  (f"out_row_add_{tag}",lambda x=x,out=out:np.add(x,x[0],out=out)),
                  (f"out_mask_blocks_{tag}",lambda x=x,out=out:np.add(x,x,where=mask,out=out)),
                  (f"out_mask_alternating_{tag}",lambda x=x,out=out:np.add(x,x,where=alt,out=out)),
                  (f"out_alias_mul_{tag}",lambda alias=alias,one=one:np.multiply(alias,one,out=alias))])
half=c[:,:513].copy()
co=np.zeros_like(c)
c_alias=c.copy()
ro=np.zeros_like(a)
ho=np.zeros_like(half)
co32=np.zeros_like(c32)
ro32=np.zeros_like(a32)
half32=half.astype("complex64")
ho32=np.zeros_like(half32)
small=v[:4*4093].reshape(4,4093)
prime_out=np.zeros((4,2047),dtype="complex128")
mixed_out=np.zeros((128,501),dtype="complex128")
cases.extend([("fft_alloc",lambda:np.fft.fft(c)),("rfft_alloc",lambda:np.fft.rfft(a)),
              ("irfft_alloc",lambda:np.fft.irfft(half,n=1024)),
              ("fft_out",lambda:np.fft.fft(c,out=co)),("ifft_out",lambda:np.fft.ifft(c,out=co)),
              ("fft_alias",lambda:np.fft.fft(c_alias,norm="ortho",out=c_alias)),
              ("rfft_out",lambda:np.fft.rfft(a,out=ho)),("irfft_out",lambda:np.fft.irfft(half,n=1024,out=ro)),
              ("fft32_alloc",lambda:np.fft.fft(c32)),("rfft32_alloc",lambda:np.fft.rfft(a32)),
              ("fft32_out",lambda:np.fft.fft(c32,out=co32)),
              ("rfft32_out",lambda:np.fft.rfft(a32,out=ho32)),
              ("irfft32_out",lambda:np.fft.irfft(half32,n=1024,out=ro32)),
              ("prime_rfft_alloc",lambda:np.fft.rfft(small)),("prime_rfft_out",lambda:np.fft.rfft(small,out=prime_out)),
              ("fftpack_rfft_out",lambda:np.fft.rfft(a,n=1000,out=mixed_out))])
with np.errstate(all="ignore"):
    for name,fn in cases:
        start=time.perf_counter_ns()
        result=fn()
        loops=max(1,min(10000,round(100_000_000/max(1,time.perf_counter_ns()-start))))
        for _ in range(5):
            start=time.perf_counter_ns()
            for _ in range(loops):result=fn()
            print(f"NumPyDirectKernels/{name} {loops} {(time.perf_counter_ns()-start)/loops:.0f} ns/op",flush=True)
