"""Direct casts, vector lane broadcasts and fused FFT references."""
import numpy as np


def populate_array_fft(record, fails):
    def r(name, x):
        record("arrayfft_"+name,x)
    rng=np.random.default_rng(319)
    for dtype in ("float32","float64","complex64","complex128"):
        rd="float32" if dtype in ("float32","complex64") else "float64"
        f=np.finfo(rd)
        vals=np.array([0.,-0.,.1,-.2,np.inf,-np.inf,np.nan,
                       f.max,-f.max,f.tiny,f.smallest_subnormal],dtype=rd)
        a=np.resize(vals,(35,37))
        if "complex" in dtype:
            q=np.empty(a.shape,dtype=dtype);q.real=a;q.imag=a[:,::-1];a=q
        r("cast_"+dtype,a)
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for dest in ("float32","float64","complex64","complex128"):
                    if dest==dtype:continue
                    for tag,x in (("packed",a),("view",a[::-1,::-1].T)):
                        r(f"cast_{dtype}_{dest}_{tag}",x.astype(dest,order="C"))
                if dtype!="complex128":
                    r("cast_"+dtype+"_complex128_broadcast",
                      np.broadcast_to(a[:1,:],(4,35,37)).astype("complex128",order="C"))
    for dtype in ("float32","float64"):
        x=np.resize(np.array([0.,-0.,.1,-.2,np.inf,-np.inf,np.nan,
                               np.finfo(dtype).smallest_subnormal,
                               np.finfo(dtype).max,3.7],dtype=dtype),(3,69))
        a=x.reshape(-1)[1:205].reshape(3,68)
        r("lane_"+dtype,x)
        r("lane_matrix_"+dtype,a)
        b=np.array([[1.25],[-0.],[np.inf]],dtype=dtype)
        r("lane_col_"+dtype,b)
        with np.errstate(all="ignore"):
            for name,fn in (("add",np.add),("sub",np.subtract),("mul",np.multiply),("div",np.divide)):
                s=np.array(-1.25,dtype=dtype)
                # Offset and tail, both scalar orders and row/column broadcasts.
                y=x.reshape(-1)[1:70]
                r(f"lane_{name}_right_{dtype}",fn(y,s))
                r(f"lane_{name}_left_{dtype}",fn(s,y))
                r(f"lane_{name}_col_{dtype}",fn(a,b))
                r(f"lane_{name}_row_{dtype}",fn(x,x[0]))
    # Single-lane real power-of-two and large-prime paths, including a
    # length where prior code used the FFTPACK fallback.
    x=rng.uniform(-1,1,65536)
    r("real_long",x)
    for n in (4096,65536,4093):
        a=x[:n]
        z=np.fft.rfft(a,norm="ortho")
        r(f"rfft_{n}",z)
        r(f"irfft_{n}",np.fft.irfft(z,n=n,norm="ortho"))
    a=(rng.uniform(-1,1,(3,4096))+1j*rng.uniform(-1,1,(3,4096))).astype("complex128")
    r("complex_input",a)
    r("fft_transposed",np.fft.fft(a.T,axis=0,norm="ortho"))
    r("ifft_reverse",np.fft.ifft(a[:,::-1],norm="forward"))
    a=a.astype("complex64")
    r("complex32_input",a)
    r("fft32",np.fft.fft(a))
    r("fft32_oracle",np.fft.fft(a.astype("complex128")).astype("complex64"))
    r("ifft32",np.fft.ifft(a))
    r("ifft32_oracle",(np.fft.ifft(a.astype("complex128"),norm="forward")*np.float32(1/4096)).astype("complex64"))
    x=a.real.copy()
    r("real32_input",x)
    z=np.fft.rfft(x)
    r("rfft32",z)
    r("rfft32_oracle",np.fft.rfft(x.astype("float64")).astype("complex64"))
    r("irfft32",np.fft.irfft(z,n=4096))
    r("irfft32_oracle",(np.fft.irfft(z.astype("complex128"),n=4096,norm="forward")*np.float32(1/4096)).astype("float32"))
