"""Reference cases guarding the optimized numerical and random stream paths."""
import numpy as np


def populate_hotpaths(record, fails):
    def r(name, val):
        record("hot_" + name, val)

    for dtype in ("float32", "float64", "complex64", "complex128"):
        a = np.array([[[0, 2, 1], [3, 1, -1], [1, .5, 4]],
                      [[2, -1, 0], [2, 3, 1], [1, 2, 5]]], dtype=dtype)
        b = ((np.arange(24).reshape(4, 3, 2) - 10) / 7).astype(dtype)
        if "complex" in dtype:
            a += 1j * a.swapaxes(-1, -2) / 7
            b += 1j * b[::-1] / 3
        r("solve_a_" + dtype, a); r("solve_b_" + dtype, b)
        for name, aa, bb in (("repeat", a[:, None, ::-1, ::-1], b[None, :, ::-1, :]),
                             ("alternate", a[None, :, :, :], b[:, None, :, :])):
            r(f"solve_{name}_{dtype}", np.linalg.solve(aa, bb))
    for dtype in ("float32", "float64", "int64"):
        # Only the specified triangle is used, not the deliberately
        # incompatible values in the other half of the matrix.
        a = np.array([[[16, 9, -7], [4, 10, 13], [2, -1, 8]],
                      [[9, 7, 8], [1, 12, 15], [-2, 3, 14]]], dtype=dtype)
        r("chol_input_" + dtype, a)
        r("chol_lower_" + dtype, np.linalg.cholesky(a[::-1]))
        r("chol_upper_" + dtype, np.linalg.cholesky(a[::-1].swapaxes(-1, -2), upper=True))
        d = np.array([[[0, 2, 0], [3, 1, 1], [0, 1, 4]],
                      [[1, 2, 3], [2, 4, 6], [1, 2, 3]],
                      [[2, 0, 1], [0, 3, 0], [1, 0, 4]]], dtype=dtype)
        r("det_input_" + dtype, d)
        s, l = np.linalg.slogdet(d[::-1])
        r("det_sign_" + dtype, s); r("det_log_" + dtype, l)
        r("det_" + dtype, np.linalg.det(d[::-1]))
    s,l = np.linalg.slogdet(np.empty((2,0,0)))
    r("det_empty_sign",s);r("det_empty_log",l)
    fails("hot_chol_not_pd",lambda: np.linalg.cholesky([[1.,2.],[2.,1.]]))

    # Prime and composite lengths with a large prime factor, odd/even
    # spectra, padded/cropped lengths, multiple norms and non-C-order axes.
    def transform(name, fn, a, n, axis=-1, norm="backward"):
        result=fn(a,n=n,axis=axis,norm=norm)
        r(name,result)
        if a.dtype in (np.dtype("float32"),np.dtype("complex64")):
            # PocketFFT's single-precision intermediate rounding may be
            # several ULP near cancellation. Independently compare the
            # Go binary64-work kernel to the binary64 NumPy transform,
            # with the single-precision normalization factor and cast.
            inverse=fn in (np.fft.ifft,np.fft.irfft)
            wide=a.astype("complex128" if a.dtype.kind=="c" else "float64")
            raw=fn(wide,n=n,axis=axis,norm="forward" if inverse else "backward")
            scale=np.float32(1)
            if (norm=="backward" and inverse) or (norm=="forward" and not inverse):
                scale=scale/np.float32(n)
            if norm=="ortho":
                scale=scale/np.sqrt(np.float32(n))
            r(name+"_double_oracle",(raw*float(scale)).astype(result.dtype))
    x = (np.arange(2*271).reshape(2,271)%79-39)/7
    for dtype in ("float32", "float64"):
        a = x.astype(dtype)
        r("fft_input_"+dtype,a)
        transform("rfft_crop_"+dtype,np.fft.rfft,a.T,257,axis=0,norm="ortho")
        transform("rfft_pad_"+dtype,np.fft.rfft,a[:,::-1],514,axis=1,norm="forward")
        z = np.fft.rfft(a,n=257)
        r("irfft_input_"+dtype,z)
        transform("irfft_odd_"+dtype,np.fft.irfft,z,257)
        transform("irfft_even_"+dtype,np.fft.irfft,z,514,norm="ortho")
        transform("fft_real_"+dtype,np.fft.fft,a,1009)
    for dtype in ("complex64","complex128"):
        z=(x+1j*x[:,::-1]/5).astype(dtype)
        r("fft_input_"+dtype,z)
        transform("fft_complex_"+dtype,np.fft.fft,z[:,::-1].T,257,axis=0,norm="forward")
        transform("ifft_complex_"+dtype,np.fft.ifft,z,257,norm="ortho")
    a=(np.arange(4093,dtype="f8")%137-68)/17
    r("fft_long_input",a)
    r("fft_long",np.fft.fft(a))
    r("rfft_long",np.fft.rfft(a))
    z=np.fft.rfft(a)
    r("irfft_long_input",z)
    r("irfft_long",np.fft.irfft(z,n=len(a)))
    a=np.arange(32.,dtype="f8").reshape(4,8)
    r("rfft_layout_input",a)
    for axis in (0,1):
        r("rfft_layout_"+str(axis),np.fft.rfft(a.T,axis=axis))

    for dtype in ("float32","float64"):
        a=np.array([0.,-0.,1.,-2.,np.inf,-np.inf,np.nan,3.,4.,-5.,6.,-7.],dtype=dtype).reshape(3,4)
        b=np.array([0.,-0.,2.],dtype=dtype).reshape(1,3)
        r("arithmetic_input_"+dtype,a);r("arithmetic_rhs_"+dtype,b)
        with np.errstate(all="ignore"):
            for name,op in (("add",np.add),("subtract",np.subtract),("multiply",np.multiply),("divide",np.divide)):
                r(name+"_"+dtype,op(a[::-1,::-1].T,b))
        r("astype_"+dtype,a[::-1,::-1].T.copy())
        c=a.copy()
        c.T[::-1,:]=a.T
        r("assign_"+dtype,c)
    a=np.arange(12.,dtype="f8")
    a[2:10]=a[:8]
    r("assign_overlap",a)

    for name in ("PCG64","PCG64DXSM","MT19937","Philox","SFC64"):
        g=np.random.Generator(getattr(np.random,name)(123))
        r("raw_"+name,g.bit_generator.random_raw(1251))
        r("uint32_"+name,g.integers(0,2**32,dtype="u4",size=3))
        r("raw_cached_"+name,g.bit_generator.random_raw(9))
        r("random_after_"+name,g.random(11))
        r("uint32_after_"+name,g.integers(0,2**32,dtype="u4",size=5))
    for dtype in ("bool","int8","int16","int32","int64","uint8","uint16","uint32","uint64"):
        if dtype=="bool": lo,hi=0,1
        else: lo,hi=np.iinfo(dtype).min,np.iinfo(dtype).max
        g=np.random.default_rng(456)
        r("integers_full_"+dtype,g.integers(lo,hi,1029,dtype=dtype,endpoint=True))
        r("integers_after_"+dtype,g.bit_generator.random_raw(7))
        g=np.random.default_rng(456)
        low=-11 if dtype.startswith("int") else 0
        high=1 if dtype=="bool" else 53
        r("integers_bounded_"+dtype,g.integers(low,high,1029,dtype=dtype))
        r("bounded_after_"+dtype,g.bit_generator.random_raw(7))
    rs=np.random.RandomState(456)
    r("legacy_integers",rs.randint(-55,73,size=1705,dtype="int16"))
    r("legacy_after",rs.random_sample(7))
    g=np.random.default_rng(789)
    lo=np.array([[-9],[-3]],dtype="i2")
    hi=np.array([[15,20,45,99]],dtype="i2")
    r("integers_array",g.integers(lo,hi,size=(3,2,4),dtype="i2"))
    r("integer_array_after",g.bit_generator.random_raw(9))
    g=np.random.default_rng(123)
    mu=np.arange(12.,dtype="f8").reshape(3,4)/5
    sig=np.array([[.1],[1.],[3.]],dtype="f8")
    r("normal_broadcast_input",mu);r("normal_broadcast_sigma",sig)
    r("normal_broadcast",g.normal(mu[:,::-1],sig,size=(5,3,4)))
    r("poisson_broadcast",g.poisson(mu[:,::-1],size=(2,3,4)))
    trials=np.array([[2],[5],[9]],dtype="i8")
    prob=np.array([[.2,.4,.6,.8]])
    r("binomial_broadcast",g.binomial(trials,prob,size=(2,3,4)))
    fails("hot_normal_invalid",lambda: g.normal(mu,-sig))
    r("broadcast_stream_after",g.bit_generator.random_raw(9))
