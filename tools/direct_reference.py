"""NumPy references for direct output kernels, precision casts and FFT borrows."""
import numpy as np


def populate_direct(record, fails):
    def r(name,x):
        record("direct_"+name,x)
    for dtype in ("float32","float64","complex64","complex128"):
        rd="float32" if dtype in ("float32","complex64") else "float64"
        f=np.finfo(rd)
        x=np.resize(np.array([0.,-0.,.1,-.2,np.inf,-np.inf,np.nan,f.max,
                             -f.max,f.tiny,f.smallest_subnormal],dtype=rd),263)
        if "complex" in dtype:
            z=np.empty(x.shape,dtype=dtype);z.real=x;z.imag=x[::-1];x=z
        r("cast_input_"+dtype,x)
        dest={"float32":"float64","float64":"float32",
              "complex64":"complex128","complex128":"complex64"}[dtype]
        with np.errstate(all="ignore"):
            r("cast_offset_"+dtype,x[1:262].astype(dest))
            r("cast_reverse_rows_"+dtype,x[1:261].reshape(4,65)[::-1].astype(dest,order="C"))
    rng=np.random.default_rng(750)
    for dtype in ("float32","float64"):
        a=np.resize(np.array([.1,-.2,3.7,-0.,0.,np.inf,-np.inf,np.nan,
                             np.finfo(dtype).smallest_subnormal,np.finfo(dtype).max],dtype=dtype),(3,69))
        b=np.resize(np.array([2.3,.7,-1.1,0.,-0.,0.,np.inf,1.,.25,2.],dtype=dtype),69)
        mask=((np.arange(a.size)%71)<39).reshape(a.shape)
        alt=(np.arange(a.size)%2==0).reshape(a.shape)
        r("ufunc_a_"+dtype,a);r("ufunc_b_"+dtype,b);r("ufunc_mask_"+dtype,mask)
        r("ufunc_altmask_"+dtype,alt)
        scalar=np.array(-1.25,dtype=dtype)
        for name,fn in (("add",np.add),("sub",np.subtract),("mul",np.multiply),("div",np.divide)):
            with np.errstate(all="ignore"):
                base=np.full(a.size+2,-7,dtype=dtype)
                fn(a,b,out=base[1:-1].reshape(a.shape),where=mask)
                r(f"ufunc_{name}_masked_{dtype}",base)
                base.fill(-7)
                fn(a,b,out=base[1:-1].reshape(a.shape),where=alt)
                r(f"ufunc_{name}_alt_{dtype}",base)
                base.fill(-7)
                fn(scalar,a,out=base[1:-1].reshape(a.shape))
                r(f"ufunc_{name}_left_{dtype}",base)
                x=a.copy()
                fn(x,scalar,out=x,where=mask)
                r(f"ufunc_{name}_alias_{dtype}",x)
        base=np.arange(73,dtype=dtype)/7
        r("overlap_input_"+dtype,base)
        np.subtract(base[:-1],scalar,out=base[1:])
        r("overlap_shift_"+dtype,base)
        base=np.arange(73,dtype=dtype)/7+2
        r("broadcast_alias_input_"+dtype,base)
        np.add(base,base[:1],out=base)
        r("broadcast_alias_"+dtype,base)
        out=np.full((69,3),-7,dtype=dtype).T
        with np.errstate(all="ignore"):
            np.add(a,b,out=out,where=mask[:,::-1])
        r("noncontig_out_"+dtype,out)
    x=np.ones(69,dtype="float64")
    y=np.ones(69,dtype="float64");y[4]=0
    def fp_alias():
        with np.errstate(divide="raise"):
            np.divide(x,y,out=x)
    fails("direct_fp_alias_raise",fp_alias)
    r("fp_alias_result",x)

    a=rng.uniform(-1,1,(3,1025))
    c=(a+1j*rng.uniform(-1,1,a.shape)).astype("complex128")
    r("fft_real_input",a)
    r("fft_complex_input",c)
    for n in (1024,1000,509):
        z=np.fft.rfft(a,n=n,norm="ortho")
        r("rfft_"+str(n),z)
        r("irfft_"+str(n),np.fft.irfft(z,n=n,norm="ortho"))
        r("fft_"+str(n),np.fft.fft(c,n=n,norm="forward"))
        r("ifft_"+str(n),np.fft.ifft(c,n=n,norm="ortho"))
    a=np.resize(np.array([np.inf,1.,np.nan,-np.inf,0.,-0.]),(2,64))
    r("fft_nonfinite_input",a)
    with np.errstate(all="ignore"):
        r("rfft_nonfinite",np.fft.rfft(a))
    r("fft_alias_input",c[0,:1024])
    r("fft_alias_result",np.fft.fft(c[0,:1024]))
    for n in (1000,509):
        r("fft_alias_"+str(n),np.fft.fft(c[0,:n]))
    r("fft_axis0_alias",np.fft.fft(c[:,:1024].T,axis=0,norm="ortho"))
