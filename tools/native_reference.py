"""Differential cases for tiled copies, block RNG fills and native kernels."""
import numpy as np


def populate_native(record, fails):
    def r(name,val):
        record("native_"+name,val)

    # Odd tile boundaries, negative strides and all native storage types.
    for dtype in ("bool","int8","int16","int32","int64","uint8","uint16","uint32","uint64",
                  "float16","float32","float64","complex64","complex128"):
        raw=np.arange(35*37,dtype="i8").reshape(35,37)
        d=np.dtype(dtype)
        if d.kind=="b":
            a=(raw%3==0)
        elif d.kind=="u":
            a=(raw%127).astype(d)
            a[0,0]=np.iinfo(d).max
        elif d.kind=="i":
            a=(raw%127-63).astype(d)
            a[0,0]=np.iinfo(d).min
        elif d.kind=="c":
            a=((raw%127-63)/4+1j*(raw%37-18)/8).astype(d)
        else:
            a=((raw%127-63)/4).astype(d)
            a[1,2]=-0.
            a[2,3]=np.inf
            a[3,4]=np.nan
        v=a[::-1,::-1].T
        r("copy_input_"+dtype,a)
        r("copy_"+dtype,v.copy(order="C"))
        r("assign_f_"+dtype,np.array(v,order="F"))

    a=np.arange(41*41,dtype="i8").reshape(41,41)
    r("overlap_input",a)
    a.T[:]=a
    r("overlap_transpose",a)
    for dtype in ("float32","float64"):
        raw=np.arange(37*35,dtype="i8").reshape(37,35)
        a=((raw%43-21)/4).astype(dtype)
        a[0,0]=np.nan;a[1,1]=np.inf;a[2,2]=-0.
        b=np.array((np.arange(37)%7-3)/2,dtype=dtype).reshape(1,37)
        r("arithmetic_input_"+dtype,a)
        r("arithmetic_rhs_"+dtype,b)
        with np.errstate(all="ignore"):
            for name,fn in (("add",np.add),("sub",np.subtract),("mul",np.multiply),("div",np.divide)):
                r(name+"_"+dtype,fn(a[::-1,::-1].T,b))

    # Interleave full uint32, rejection-heavy 32-bit and 64-bit paths,
    # raw and doubles with a pending cached high word.
    for alg in ("PCG64","PCG64DXSM"):
        g=np.random.Generator(getattr(np.random,alg)(642))
        r("rng_first_"+alg,g.integers(0,2**32,size=1,dtype="u4"))
        r("rng_reject_"+alg,g.integers(-17,2**31-16,size=2051,dtype="i8"))
        r("rng_double_"+alg,g.random(17))
        r("rng_full32_"+alg,g.integers(0,2**32,size=5,dtype="u8"))
        r("rng_wide_"+alg,g.integers(0,2**32+1,size=9,dtype="u8"))
        r("rng_full_i64_"+alg,g.integers(-2**63,2**63-1,size=11,dtype="i8",endpoint=True))
        r("rng_full_u64_"+alg,g.integers(0,2**64-1,size=11,dtype="u8",endpoint=True))
        r("rng_fixed_"+alg,g.integers(4,4,size=3,dtype="i8",endpoint=True))
        r("rng_raw_"+alg,g.bit_generator.random_raw(13))
        r("rng_float32_"+alg,g.random(13,dtype="f4"))
        r("rng_tail_"+alg,g.random(11))
    u=np.iinfo(np.uint64).max
    p=np.random.Philox(counter=np.array([u,u,u,u],dtype="u8"),key=np.array([17,88],dtype="u8"))
    g=np.random.Generator(p)
    r("philox_carry",p.random_raw(13))
    r("philox_cache",g.integers(0,2**32,size=3,dtype="u4"))
    r("philox_tail",p.random_raw(9))
    r("philox_cache_tail",g.integers(0,2**32,size=3,dtype="u4"))

    # Power-of-two transforms, both directions, crop/pad, F-order and
    # negative axes, plus single precision backed by a double oracle.
    def transform(name,fn,a,n,axis=-1,norm="backward"):
        out=fn(a,n=n,axis=axis,norm=norm)
        r(name,out)
        if a.dtype in (np.dtype("float32"),np.dtype("complex64")):
            inverse=fn in (np.fft.ifft,np.fft.irfft)
            wide=a.astype("complex128" if a.dtype.kind=="c" else "float64")
            v=fn(wide,n=n,axis=axis,norm="forward" if inverse else "backward")
            scale=np.float32(1)
            if norm=="ortho":
                scale/=np.sqrt(np.float32(n))
            if norm=="forward" and not inverse or norm=="backward" and inverse:
                scale/=np.float32(n)
            r(name+"_oracle",(v*float(scale)).astype(out.dtype))
    raw=(np.arange(2*519,dtype="i8").reshape(2,519)%197-98)/31
    for dtype in ("float32","float64"):
        a=raw.astype(dtype)
        r("fft_real_input_"+dtype,a)
        transform("rfft_512_"+dtype,np.fft.rfft,a.T,512,0,"ortho")
        transform("rfft_1024_"+dtype,np.fft.rfft,a[:,::-1],1024,-1,"forward")
        z=np.fft.rfft(a,n=512)
        r("ifft_real_input_"+dtype,z)
        transform("irfft_512_"+dtype,np.fft.irfft,z,512)
        transform("irfft_1024_"+dtype,np.fft.irfft,z[:,::-1].T,1024,0,"ortho")
    for dtype in ("complex64","complex128"):
        a=(raw+1j*raw[:,::-1]/7).astype(dtype)
        r("fft_complex_input_"+dtype,a)
        transform("fft_512_"+dtype,np.fft.fft,a[:,::-1],512,-1,"backward")
        transform("ifft_1024_"+dtype,np.fft.ifft,a.T,1024,0,"ortho")
    a=np.zeros(64,dtype="f8");a[0]=np.inf
    with np.errstate(all="ignore"):
        r("fft_nonfinite_input",a)
        r("rfft_nonfinite",np.fft.rfft(a))

    # Wider complex products exercise the optimized GEMM threshold and
    # shared contraction kernel while keeping exact shape and dtype.
    x=np.arange(65*71).reshape(65,71)
    y=np.arange(71*67).reshape(71,67)
    aa=(x%53-26)/16+1j*(x%31-15)/16
    bb=(y%47-23)/32+1j*(y%17-8)/16
    for dtype in ("complex64","complex128"):
        a,b=aa.astype(dtype),bb.astype(dtype)
        r("matmul_a_"+dtype,a)
        r("matmul_b_"+dtype,b)
        r("matmul_"+dtype,a[::-1,::-1]@b[::-1,::-1])
        r("tensordot_"+dtype,np.tensordot(a,b,axes=1))
    # Non-dyadic values and unequal dimensions exercise rounding and odd
    # vector tails; the exact-power-of-two fixtures above exercise indexing.
    a=((x%43-21)/19+1j*(x%31-15)/23).astype("complex128")
    b=((y%37-18)/29+1j*(y%17-8)/31).astype("complex128")
    r("matmul_mixed_a",a);r("matmul_mixed_b",b)
    r("matmul_mixed",a@b)
    a=np.eye(37,dtype="complex128")*complex(1e200,1e200)
    b=np.eye(37,dtype="complex128")*complex(1e-200,-1e-200)
    r("matmul_scale_a",a);r("matmul_scale_b",b)
    r("matmul_scale",a@b)
    n=65
    i,j=np.indices((n,n))
    a=((np.minimum(i,j)*101+np.maximum(i,j)*79)%31-15)/31+1j*(i-j)/(n*n)
    a=a.astype("complex128")
    a.flat[::n+1]+=n
    b=((np.arange(n*3).reshape(n,3)%41-20)/7+1j/3)
    r("solve_matrix",a);r("solve_rhs",b)
    r("solve",np.linalg.solve(a[::-1],b[::-1]))
    s,l=np.linalg.slogdet(a[::-1])
    r("sign",s);r("logdet",l)
    r("chol_lower",np.linalg.cholesky(a))
    r("chol_upper",np.linalg.cholesky(a,upper=True))
