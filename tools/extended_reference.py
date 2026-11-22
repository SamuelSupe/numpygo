"""Additional NumPy 2.5.0 fixtures; consumed by generate_reference.py."""
import numpy as np
import warnings


def populate(record, fails):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dtypes = ["bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
                  "float16", "float32", "float64", "complex64", "complex128"]
        for a in dtypes:
            for b in dtypes:
                record("promote_" + a + "_" + b, np.add(np.array(1, dtype=a), np.array(1, dtype=b)))
        for dtype in dtypes[1:9]:
            info = np.iinfo(dtype)
            a = np.array([info.max, info.min, 1, 2], dtype=dtype)
            b = np.array([1, 1, info.max, 3], dtype=dtype)
            record("integer_input_" + dtype, a)
            for fn in (np.add, np.subtract, np.multiply, np.floor_divide, np.remainder, np.bitwise_and,
                       np.bitwise_or, np.bitwise_xor, np.true_divide):
                record("integer_" + fn.__name__ + "_" + dtype, fn(a,b))
            record("integer_other_" + dtype, b)
            record("integer_sum_" + dtype, np.sum(a))
            record("integer_prod_" + dtype, np.prod(a))
            reciprocal=np.array([0,-2,-1,1,2] if dtype.startswith("int") else [0,1,2],dtype=dtype)
            record("integer_reciprocal_input_"+dtype,reciprocal);record("integer_reciprocal_"+dtype,np.reciprocal(reciprocal))
        big = np.array([2**63+123, 2**64-1, 2**53+1, 0], dtype=np.uint64)
        record("uint64_input", big)
        record("uint64_cast_int64", big.astype(np.int64))
        record("uint64_cast_float64", big.astype(np.float64))
        record("uint64_negative_slice", big[::-1])
        record("integer_floor_negative", np.floor_divide([-8,-7,7,8], [3,-3,-3,3]))
        record("integer_mod_negative", np.remainder([-8,-7,7,8], [3,-3,-3,3]))
        record("integer_pow", np.power(np.array([-3, 3, 100], dtype="int8"), np.array([3,4,2], dtype="int8")))
        fails("integer_negative_power", lambda: np.power(np.array(2), np.array(-1)))
        fails("bool_subtract", lambda: np.subtract(np.array(True), np.array(False)))
        h = np.array([0., -0., 1., -1., 2**-24, 2**-25, 65504., 65520., np.inf, np.nan, 1.00048828125])
        record("half_input", h)
        record("half_cast", h.astype(np.float16))
        record("half_sum", np.sum(np.array([0.1,0.2,-0.3,5], dtype=np.float16)))
        record("complex_cast", np.array([1+2j,3-4j], dtype=np.complex64))
        record("bool_input", np.array([True, False, True, False]))
        record("bool_sum", np.sum(np.array([True, False, True])))
        record("bool_prod", np.prod(np.array([True, False, True])))
        record("bool_logic", np.logical_xor([True,False], [True,True]))
        record("bool_square",np.square(np.array([True,False])))
        record("bool_reciprocal",np.reciprocal(np.array([True,False])))
        record("bool_conjugate",np.conjugate(np.array([True,False])))
        signed=np.array([2**63-1,-1,7],dtype=np.int64);unsigned=np.array([2**63,2**64-1,7],dtype=np.uint64)
        record("integer_cross_equal",np.equal(signed,unsigned));record("integer_cross_less",np.less(signed,unsigned));record("integer_cross_greater",np.greater(unsigned,signed))
        record("integer_cross_isin",np.isin(signed,unsigned))

        a = np.arange(120, dtype=np.int64).reshape(2,3,4,5)
        i = np.array([[0],[2]])
        j = np.array([[1,3,0]])
        record("advanced_input", a)
        record("advanced_i", i)
        record("advanced_j", j)
        record("advanced_adjacent", a[:,i,j,:])
        record("advanced_separated", a[:,i,:,j])
        record("advanced_scalar", a[1,:,j,:])
        record("advanced_ellipsis", a[np.array([1,0]), ..., np.array([4,1])])
        record("advanced_newaxis", a[:,i,None,j,:])
        mask = np.array([[True,False,True],[False,True,False]])
        record("advanced_mask_input", mask)
        record("advanced_mask", a[mask])
        record("advanced_mask_true", a[np.array(True)])
        record("advanced_mask_false", a[np.array(False)])
        record("advanced_negative", a[np.array([-1,0])])
        record("advanced_take", np.take(a,np.array([[3,1],[0,-1]]),axis=2))
        for k, coord in enumerate(np.nonzero(mask)):
            record("advanced_nonzero_"+str(k),coord)
        record("advanced_flatnonzero", np.flatnonzero(mask))
        record("advanced_where", np.where(mask, np.arange(3), -1))
        target=np.arange(12,dtype=np.int64).reshape(3,4)
        target[np.array([0,0,2]),np.array([1,1,0])]=[100,200,300]
        record("advanced_assignment",target)
        target=np.arange(6,dtype=np.int64);target[1:]=target[:-1]
        record("advanced_assign_alias",target)
        fails("advanced_bad_mask", lambda:a[np.ones((3,3),dtype=bool)])
        fails("advanced_index_oob", lambda:a[np.array([2])])
        fails("advanced_index_type", lambda:a[np.array([0.])])
        fails("advanced_broadcast_error",lambda:a[:,np.array([0,1]),np.array([0,1,2]),:])

        s=np.array([[1.5,np.nan,-2.,4.],[0.,3.,3.,9.]])
        record("stats_input",s)
        for fn in (np.mean,np.nanmean,np.var,np.nanvar,np.std,np.nanstd,np.sum,np.nansum,np.prod,np.nanprod,
                   np.min,np.max,np.nanmin,np.nanmax,np.median,np.nanmedian,np.ptp,np.all,np.any,np.count_nonzero):
            record("stats_"+fn.__name__, fn(s,axis=1,keepdims=True))
        record("stats_var_ddof",np.nanvar(s,axis=0,ddof=1))
        record("stats_std_ddof",np.nanstd(s,axis=1,ddof=2))
        record("stats_quantile",np.quantile(s,.2,axis=1))
        record("stats_nanquantile",np.nanquantile(s,.75,axis=1))
        record("stats_percentile",np.percentile(s,35,axis=1))
        record("stats_nanpercentile",np.nanpercentile(s,60,axis=1))
        record("stats_argmin",np.argmin(s,axis=1))
        record("stats_argmax",np.argmax(s,axis=1))
        record("stats_nanargmin",np.nanargmin(s,axis=1))
        record("stats_nanargmax",np.nanargmax(s,axis=1))
        record("stats_arg_flat_transpose",np.argmax(s.T))
        record("stats_argsort",np.argsort(s,axis=1,stable=True))
        record("stats_sort",np.sort(s,axis=-1,stable=True))
        for fn in (np.cumsum,np.nancumsum,np.cumprod,np.nancumprod):
            record("stats_"+fn.__name__,fn(s,axis=1))
        record("stats_empty_mean",np.mean(np.empty((2,0)),axis=1))
        record("stats_empty_sum",np.sum(np.empty((2,0),dtype=np.int32),axis=1))
        fails("stats_empty_min",lambda:np.min(np.empty((2,0)),axis=1))
        fails("stats_all_nan_arg",lambda:np.nanargmax([np.nan,np.nan]))
        fails("stats_invalid_quantile",lambda:np.quantile(s,1.1))
        record("stats_low_precision_quantile",np.quantile(np.array([1.,2.,3.],dtype=np.float32),.7))
        c=np.array([[1+2j,3-1j,2+0j],[3-2j,1+4j,-2+1j]],dtype=np.complex64)
        record("stats_complex_input",c)
        record("stats_complex_mean",np.mean(c,axis=1))
        record("stats_complex_var",np.var(c,axis=1))
        record("stats_complex_std",np.std(c,axis=1))
        record("stats_cov",np.cov(c))
        record("stats_corrcoef",np.corrcoef(c))
        record("stats_complex_corrcoef_scalar",np.corrcoef(np.array([1+2j,3-1j])))

        x=np.array([-.5,.25,1.5,2.5],dtype=np.float64)
        z=np.array([-1+.5j,.5+.2j,2-1j],dtype=np.complex128)
        record("math_input",x)
        record("math_complex_input",z)
        unary=["sqrt","exp","exp2","expm1","log","log2","log10","log1p","sin","cos","tan","arcsin","arccos","arctan",
               "sinh","cosh","tanh","arcsinh","arccosh","arctanh","cbrt","degrees","radians","rint","floor","ceil","trunc",
               "negative","positive","square","reciprocal","absolute","conjugate","sign"]
        for name in unary:
            record("math_"+name,getattr(np,name)(x))
        for name in ["sqrt","exp","log","log1p","sin","cos","tan","arcsin","arccos","arctan","sinh","cosh","tanh",
                     "arcsinh","arccosh","arctanh","absolute","conjugate","sign","rint","reciprocal","square"]:
            record("math_complex_"+name,getattr(np,name)(z))
        values=np.array([-np.inf,-0.,0.,np.nan,np.inf,2.])
        record("math_special",values)
        for name in ["isnan","isinf","isfinite","signbit","logical_not"]:
            record("math_"+name,getattr(np,name)(values))
        record("math_invert",np.invert(np.array([-3,0,3],dtype=np.int8)))
        record("math_sqrt_int",np.sqrt(np.array([1,4,9],dtype=np.int16)))
        record("math_clip",np.clip(x,-.25,1.))
        for name in ["equal","not_equal","less","less_equal","greater","greater_equal","minimum","maximum","fmin","fmax",
                     "logical_and","logical_or","logical_xor","hypot","arctan2","copysign","logaddexp"]:
            record("math_"+name,getattr(np,name)(x,.25))
        record("math_isclose",np.isclose(values,values,equal_nan=True))
        record("math_allclose",np.array(np.allclose(values,values,equal_nan=True)))

        record("create_ones",np.ones((2,3),dtype=np.int16))
        record("create_full",np.full((2,2),3+2j,dtype=np.complex64))
        record("create_eye",np.eye(3,4,k=1,dtype=np.uint8))
        record("create_identity",np.identity(3,dtype=np.float32))
        record("create_arange_int",np.arange(9,-4,-3,dtype=np.int64))
        record("create_arange_float",np.arange(0.5,2.6,.3,dtype=np.float32))
        record("create_linspace",np.linspace(-2.5,3.5,7,dtype=np.int64))
        record("create_logspace",np.logspace(-2,2,5))
        a=np.arange(6,dtype=np.int32).reshape(2,3)
        record("manip_input",a)
        record("manip_zeros_like",np.zeros_like(a))
        record("manip_ones_like",np.ones_like(a))
        record("manip_full_like",np.full_like(a,7))
        record("manip_ravel",np.ravel(a.T))
        record("manip_flatten",a.T.flatten())
        record("manip_expand",np.expand_dims(a,1))
        record("manip_squeeze",np.squeeze(np.expand_dims(a,1),axis=1))
        record("manip_swap",np.swapaxes(a,0,-1))
        record("manip_move",np.moveaxis(a,0,1))
        record("manip_flip",np.flip(a,axis=1))
        record("manip_roll",np.roll(a,2,axis=1))
        record("manip_repeat",np.repeat(a,2,axis=0))
        record("manip_tile",np.tile(a,(2,1,2)))
        record("manip_diff",np.diff(a,n=2,axis=1))
        record("manip_bool_diff",np.diff(np.array([True,False,False,True])))
        record("manip_diagonal",np.diagonal(a,offset=1))
        record("manip_trace",np.trace(a,offset=1))
        record("manip_concat",np.concatenate([a,a],axis=1))
        record("manip_stack",np.stack([a,a],axis=-1))

        a=np.array([[[1+2j,3-1j],[2j,4+1j]],[[0j,1+1j],[2-1j,3+0j]]],dtype=np.complex128)
        b=np.array([2+3j,5-2j],dtype=np.complex64)
        record("complex_linalg_a",a)
        record("complex_linalg_b",b)
        record("complex_matmul",a@b)
        record("complex_matmul_mixed",a@np.array([[1,2],[3,4]],dtype=np.float32))
        record("complex_solve",np.linalg.solve(a,b))
        record("complex_solve_matrix",np.linalg.solve(a,np.array([[1+1j,3-1j],[2j,5+2j]],dtype=np.complex64)))
        record("complex_solve_single",np.linalg.solve(a.astype(np.complex64),b))
        record("complex_inv",np.linalg.inv(a))
        record("complex_det",np.linalg.det(a))
        sign,logdet=np.linalg.slogdet(a);record("complex_slogdet_sign",sign);record("complex_slogdet_log",logdet)
        record("integer_matmul",np.array([[120,2],[-3,5]],dtype=np.int8)@np.array([[3,2],[5,6]],dtype=np.int8))
        record("bool_matmul",np.array([[True,False],[True,True]])@np.array([[False,True],[True,False]]))
        record("real_float32_solve",np.linalg.solve(np.array([[3,1],[2,4]],dtype=np.float32),np.array([5,6],dtype=np.float32)))
        record("real_integer_solve",np.linalg.solve(np.array([[3,1],[2,4]],dtype=np.int8),np.array([5,6],dtype=np.int8)))
        c=np.array([[[1+2j,3+1j],[2-1j,4j],[5+2j,-1+2j]],
                    [[1+1j,2+2j],[2-1j,4-2j],[3j,6j]]],dtype=np.complex128)
        record("complex_svd_input",c)
        for name,x,full in [("complex_svd_tall",c,True),("complex_svd_wide",c.swapaxes(-1,-2),False),
                            ("complex_svd_single",c.astype(np.complex64),False),("complex_svd_empty",np.empty((0,3),dtype=np.complex128),True)]:
            u,s,vh=np.linalg.svd(x,full_matrices=full)
            record(name+"_u",u);record(name+"_s",s);record(name+"_vh",vh)
        record("complex_svd_values",np.linalg.svd(c,compute_uv=False))
        record("complex_pinv",np.linalg.pinv(c))
        record("complex_rank",np.linalg.matrix_rank(c))
        h=a.conj().swapaxes(-1,-2)@a+np.eye(2)
        record("complex_cholesky_input",h)
        record("complex_cholesky_lower",np.linalg.cholesky(h))
        record("complex_cholesky_upper",np.linalg.cholesky(h,upper=True))
        for mode in ["reduced","complete"]:
            q,r=np.linalg.qr(c,mode=mode);record("complex_qr_"+mode+"_q",q);record("complex_qr_"+mode+"_r",r)
        fails("complex_singular",lambda:np.linalg.solve(np.array([[1+1j,2+2j],[2+2j,4+4j]]),np.ones(2)))
        fails("complex_not_pd",lambda:np.linalg.cholesky(np.array([[1+0j,2j],[-2j,1+0j]])))

        x=np.array([1.,2.,-3.,4.,5.],dtype=np.float32)
        for dtype in ["int16","float16","float32"]:
            y=x.astype(dtype);record("fft_dtype_"+dtype,np.fft.fft(y))
            r=np.fft.rfft(y);record("rfft_dtype_"+dtype,r);record("irfft_dtype_"+dtype,np.fft.irfft(r,n=5))
        record("fft_dtype_complex64",np.fft.fft(np.array([1+2j,3-1j,2+0j],dtype=np.complex64)))

        seed=12345
        def rng():return np.random.default_rng(seed)
        methods={
            "random":lambda g:g.random(32),
            "random32":lambda g:g.random(32,dtype=np.float32),
            "normal":lambda g:g.normal(2.,.5,64),
            "standard_normal":lambda g:g.standard_normal(1024),
            "standard_normal32":lambda g:g.standard_normal(512,dtype=np.float32),
            "standard_exponential":lambda g:g.standard_exponential(1024),
            "standard_exponential32":lambda g:g.standard_exponential(512,dtype=np.float32),
            "standard_exponential_inv":lambda g:g.standard_exponential(64,method="inv"),
            "uniform":lambda g:g.uniform(-4,8,32),
            "exponential":lambda g:g.exponential(2.5,64),
            "gamma_small":lambda g:g.gamma(.4,2.,64),
            "gamma_big":lambda g:g.gamma(5.,.25,64),
            "standard_gamma":lambda g:g.standard_gamma(2.5,64),
            "beta_small":lambda g:g.beta(.4,.7,64),
            "beta_big":lambda g:g.beta(2.,4.,64),
            "chisquare":lambda g:g.chisquare(3.,64),
            "f":lambda g:g.f(3.,5.,64),
            "standard_cauchy":lambda g:g.standard_cauchy(64),
            "standard_t":lambda g:g.standard_t(4.,64),
            "pareto":lambda g:g.pareto(3.,64),
            "weibull":lambda g:g.weibull(2.5,64),
            "power":lambda g:g.power(3.,64),
            "laplace":lambda g:g.laplace(2.,3.,64),
            "gumbel":lambda g:g.gumbel(-1.,2.,64),
            "logistic":lambda g:g.logistic(2.,1.5,64),
            "lognormal":lambda g:g.lognormal(.5,.25,64),
            "rayleigh":lambda g:g.rayleigh(2.,64),
            "poisson_small":lambda g:g.poisson(3.,64),
            "poisson_big":lambda g:g.poisson(20.,64),
            "negative_binomial":lambda g:g.negative_binomial(3.,.4,64),
            "binomial_small":lambda g:g.binomial(10,.2,64),
            "binomial_big":lambda g:g.binomial(200,.7,64),
            "choice_replace":lambda g:g.choice(10,size=20),
            "choice_no_replace":lambda g:g.choice(9,5,replace=False),
            "choice_no_shuffle":lambda g:g.choice(12,8,replace=False,shuffle=False),
            "choice_weighted":lambda g:g.choice(4,20,p=[.1,.2,0,.7]),
            "choice_weighted_no_replace":lambda g:g.choice(4,3,replace=False,p=[.1,.2,0,.7]),
            "choice_large":lambda g:g.choice(12000,300,replace=False),
            "permutation":lambda g:g.permutation(20),
            "permutation_array":lambda g:g.permutation(np.arange(12,dtype=np.int32).reshape(3,4),axis=1),
        }
        for name,fn in methods.items():record("rng_"+name,fn(rng()))
        for dtype in dtypes[:9]:
            if dtype=="bool":low,high=0,2
            elif dtype.startswith("uint"):low,high=5,230
            else:low,high=-55,100
            record("rng_integer_"+dtype,rng().integers(low,high,size=64,dtype=dtype))
        record("rng_uint64_full",rng().integers(0,np.iinfo(np.uint64).max,endpoint=True,size=32,dtype=np.uint64))
        record("rng_int64_full",rng().integers(np.iinfo(np.int64).min,np.iinfo(np.int64).max,endpoint=True,size=32,dtype=np.int64))
        record("rng_seed32",np.random.SeedSequence(seed).generate_state(12,np.uint32))
        record("rng_seed64",np.random.SeedSequence(seed).generate_state(8,np.uint64))
        record("rng_seed_large",np.random.SeedSequence(2**60+123).generate_state(8,np.uint64))
        record("rng_seed_words",np.random.SeedSequence([5,7,9,11,13,15]).generate_state(8,np.uint64))
        for k,ss in enumerate(np.random.SeedSequence(seed).spawn(3)):
            record("rng_spawn_"+str(k),ss.generate_state(8,np.uint64))
        record("rng_raw",np.random.PCG64(seed).random_raw(32))
        record("rng_dxsm_raw",np.random.PCG64DXSM(seed).random_raw(32))
        for cls,name in [(np.random.PCG64,"pcg"),(np.random.PCG64DXSM,"dxsm")]:
            bg=cls(seed);bg.advance(2**64+123);record("rng_advance_"+name,bg.random_raw(16))
        g=rng();record("rng_cache_first",g.integers(0,100,size=3));record("rng_cache_double",g.random(2))
        record("rng_cache_last",g.integers(0,100,size=5))
        v=np.arange(12,dtype=np.int32).reshape(3,4);rng().shuffle(v,axis=1);record("rng_shuffle",v)
        fails("rng_invalid_normal",lambda:rng().normal(0,-1))
        fails("rng_invalid_integer",lambda:rng().integers(3,2))
        fails("rng_invalid_probability",lambda:rng().choice(2,3,p=[.2,.2]))
        fails("rng_invalid_choice",lambda:rng().choice(2,3,replace=False))
        fails("rng_invalid_poisson",lambda:rng().poisson(-1))
        fails("rng_invalid_binomial",lambda:rng().binomial(5,1.2))

        x=np.arange(30.,dtype=np.float64).reshape(2,3,5)
        record("fft_nd_input",x)
        record("fft_nd",np.fft.fftn(x))
        record("ifft_nd",np.fft.ifftn(np.fft.fftn(x)))
        record("fft_2",np.fft.fft2(x,s=(4,7),norm="ortho"))
        record("ifft_2",np.fft.ifft2(np.fft.fft2(x,norm="ortho"),norm="ortho"))
        record("rfft_nd",np.fft.rfftn(x))
        record("irfft_nd",np.fft.irfftn(np.fft.rfftn(x),s=x.shape,axes=(0,1,2)))
        record("rfft_2",np.fft.rfft2(x,axes=(0,2),norm="forward"))
        record("irfft_2",np.fft.irfft2(np.fft.rfft2(x,axes=(0,2),norm="forward"),s=(2,5),axes=(0,2),norm="forward"))
        record("fft_repeated_axes",np.fft.fftn(x,s=(4,7),axes=(1,1)))
        h=np.array([1+0j,2+3j,4-2j],dtype=np.complex128)
        record("fft_hfft",np.fft.hfft(h,n=5))
        record("fft_ihfft",np.fft.ihfft(np.fft.hfft(h,n=5),n=5))
        record("fft_fftfreq_odd",np.fft.fftfreq(5,.25))
        record("fft_fftfreq_even",np.fft.fftfreq(6,.25))
        record("fft_rfftfreq",np.fft.rfftfreq(5,.25))
        record("fft_shift",np.fft.fftshift(x,axes=(1,2)))
        record("fft_ishift",np.fft.ifftshift(x,axes=(1,2)))

        a=np.array([[[1+2j,3-1j],[2j,4+1j]],[[0j,1+1j],[2-1j,3+0j]]])
        h=a.conj().swapaxes(-1,-2)@a+np.eye(2)
        for name,hdata,upper in [("eigh_complex",h,False),("eigh_single",h.astype(np.complex64),True),
                                  ("eigh_repeated",np.eye(3,dtype=np.complex128),False)]:
            e,v=np.linalg.eigh(hdata,UPLO="U" if upper else "L")
            record(name+"_input",hdata);record(name+"_values",e);record(name+"_vectors",v)
        record("eigvalsh",np.linalg.eigvalsh(h))
        for name,adata in [("eig_complex",a),("eig_real",np.array([[0.,-1.],[1.,0.]],dtype=np.float32)),
                           ("eig_repeated",np.eye(3,dtype=np.complex128)),
                           ("eig_complex_real_pair",np.array([[1+2j,2j],[0,1-2j]],dtype=np.complex128))]:
            e,v=np.linalg.eig(adata);record(name+"_input",adata);record(name+"_values",e);record(name+"_vectors",v)
        record("eigvals",np.linalg.eigvals(a))
        record("matrix_power_pos",np.linalg.matrix_power(a,3))
        record("matrix_power_neg",np.linalg.matrix_power(a,-2))
        record("matrix_power_zero",np.linalg.matrix_power(a,0))
        c=np.array([[1+2j,3+1j],[2-1j,4j],[5+2j,-1+2j]])
        rhs=np.array([1+0j,2-3j,-1+2j])
        for name,data in [("lstsq_full",c),("lstsq_rank_deficient",np.column_stack([c[:,0],2*c[:,0]]))]:
            x,r,k,s=np.linalg.lstsq(data,rhs,rcond=None)
            record(name+"_input",data);record(name+"_rhs",rhs);record(name+"_x",x);record(name+"_residuals",r);record(name+"_rank",np.array(k,dtype=np.int64));record(name+"_s",s)
        for name,ord in [("default",None),("fro","fro"),("nuc","nuc"),("2",2),("-2",-2),("1",1),("-1",-1),("inf",np.inf),("-inf",-np.inf)]:
            record("norm_matrix_"+name,np.linalg.norm(a,ord=ord,axis=(-2,-1),keepdims=True))
        for name,ord in [("default",None),("0",0),("1",1),("2",2),("3",3),("-2",-2),("inf",np.inf),("-inf",-np.inf)]:
            record("norm_vector_"+name,np.linalg.norm(rhs,ord=ord))
        record("norm_all",np.linalg.norm(a))

        a=np.arange(24,dtype=np.int16).reshape(2,3,4)
        b=np.arange(20,dtype=np.int16).reshape(4,5)
        record("contract_a",a);record("contract_b",b)
        record("contract_dot",np.dot(a,b))
        record("contract_tensordot",np.tensordot(a,b,axes=([2],[0])))
        record("contract_inner",np.inner(a,a))
        record("contract_outer",np.outer(a[0,0],b[:,0]))
        record("contract_vdot",np.vdot(np.array([1+2j,3-1j]),np.array([4+0j,1+2j])))
        record("contract_kron",np.kron(np.array([[1,2],[3,4]],dtype=np.int8),np.array([2,3],dtype=np.int8)))
        record("contract_einsum",np.einsum("...ij,jk->...ik",a,b))
        record("contract_einsum_reduce",np.einsum("ijk,ijk->j",a,a))
        record("contract_einsum_implicit",np.einsum("ji",b))
        diag=np.arange(9,dtype=np.int64).reshape(3,3)
        record("contract_einsum_diagonal",np.einsum("ii->i",diag))
        record("contract_einsum_trace",np.einsum("ii",diag))
        np.einsum("ii->i",diag)[1]=55;record("contract_einsum_writeback",diag)
        fails("contract_bad_einsum",lambda:np.einsum("ii",np.ones((2,3))))
        fails("contract_bad_tensordot",lambda:np.tensordot(a,b,axes=([1],[0])))

        x=np.array([[3.,np.nan,1.],[3.,-2.,np.nan]])
        record("unique_input",x)
        u,ind,inv,counts=np.unique(x,return_index=True,return_inverse=True,return_counts=True)
        record("unique_values",u);record("unique_indices",ind);record("unique_inverse",inv);record("unique_counts",counts)
        record("unique_values_separate",np.unique_values(x))
        a=np.array([5,1,3,1,2],dtype=np.int64);b=np.array([3,6,1],dtype=np.int32)
        record("set_input_a",a);record("set_input_b",b)
        for fn in (np.union1d,np.intersect1d,np.setdiff1d,np.setxor1d):record("set_"+fn.__name__,fn(a,b))
        record("set_isin",np.isin(a,b));record("set_isin_invert",np.isin(a,b,invert=True))
        grid=np.array([1.,2.,2.,4.,np.nan]);query=np.array([0.,2.,3.,5.,np.nan])
        record("search_grid",grid);record("search_query",query)
        record("search_left",np.searchsorted(grid,query,side="left"));record("search_right",np.searchsorted(grid,query,side="right"))
        for reverse in (False,True):
            for right in (False,True):record("digitize_"+str(int(reverse))+str(int(right)),np.digitize(query,np.array([1.,2.,4.])[::-1] if reverse else [1.,2.,4.],right=right))
        c=np.array([complex(1,np.nan),complex(np.nan,0),2+1j,complex(0,np.nan),-3-2j,complex(np.nan,np.nan)])
        record("complex_sort_input",c);record("complex_sort",np.sort(c,stable=True))
        a=np.arange(12,dtype=np.int64).reshape(3,4);idx=np.array([[3,0],[2,1],[0,0]])
        record("along_input",a);record("along_index",idx)
        record("along_take",np.take_along_axis(a,idx,axis=1));record("along_take_broadcast",np.take_along_axis(a,idx[:1],axis=1))
        np.put_along_axis(a,idx,np.array([[99],[88],[77]]),axis=1);record("along_put",a)

        s=np.array([[1.,2.,3.],[5.,7.,9.]])
        record("sample_input",s)
        avg,w=np.average(s,axis=1,weights=[1.,3.,2.],returned=True)
        record("sample_average",avg);record("sample_average_weights",w)
        record("sample_average_unweighted",np.average(s))
        hist,edges=np.histogram(s,bins=4,range=(1,9));record("sample_histogram",hist);record("sample_histogram_edges",edges)
        hist,edges=np.histogram(s,bins=[0.,2.,5.,10.],weights=np.array([[1.,2.,3.],[.5,1.5,2.5]]),density=True)
        record("sample_hist_density",hist);record("sample_hist_density_edges",edges)
        record("sample_bincount",np.bincount([0,1,1,3,3,3],minlength=5))
        record("sample_bincount_weights",np.bincount([0,1,1,3,3,3],weights=[1.,2.,.5,3.,1.,.5],minlength=5))
        record("sample_gradient",np.gradient(s,.25,axis=1,edge_order=2))
        record("sample_trapezoid",np.trapezoid(s,dx=.5,axis=1));record("sample_trapezoid_x",np.trapezoid(s,x=[0.,.5,2.],axis=1))
        record("sample_interp",np.interp([-1.,.5,1.5,3.],[0.,1.,2.],[1+2j,2-1j,4+0j],left=-3+1j,right=9+2j))

        a=np.array([[[1+2j,3-1j],[2j,4+1j]],[[0j,1+1j],[2-1j,3+0j]]])
        for name,order in [("default",None),("1",1),("-2",-2),("fro","fro")]:record("linalg_cond_"+name,np.linalg.cond(a,p=order))
        record("linalg_svdvals",np.linalg.svdvals(a));record("linalg_matrix_transpose",np.linalg.matrix_transpose(a))
        record("linalg_matrix_norm",np.linalg.matrix_norm(a,ord=1,keepdims=True))
        record("linalg_vector_norm",np.linalg.vector_norm(a,ord=3,axis=(0,2),keepdims=True))
        v=np.array([[1+2j,3-1j,2+0j],[3+1j,2j,4-1j]]);w=np.array([2+1j,3+0j,4-2j])
        record("linalg_cross_input",v);record("linalg_cross_other",w)
        record("linalg_cross",np.linalg.cross(v,w));record("linalg_vecdot",np.linalg.vecdot(v,w))
        fails("core_cross2",lambda:np.cross([2,3],[4,5]))
        record("linalg_multidot",np.linalg.multi_dot([np.array([1.,2.]),np.arange(6.).reshape(2,3),np.array([2.,1.,-1.])]))
        tensor=np.eye(6).reshape(2,3,3,2)
        record("linalg_tensorinv",np.linalg.tensorinv(tensor,ind=2));record("linalg_tensorsolve",np.linalg.tensorsolve(tensor,np.arange(6.).reshape(2,3)))

        for name,fn in {
            "geometric_search":lambda g:g.geometric(.6,40),"geometric_inv":lambda g:g.geometric(.1,40),
            "triangular":lambda g:g.triangular(-2,1,5,40),"wald":lambda g:g.wald(2.,3.,40),
            "logseries":lambda g:g.logseries(.8,40),"zipf":lambda g:g.zipf(2.,40),
            "noncentral_chisquare":lambda g:g.noncentral_chisquare(3.,2.,40),"noncentral_chisquare_small":lambda g:g.noncentral_chisquare(.5,2.,40),
            "noncentral_f":lambda g:g.noncentral_f(3.,4.,2.,40),"vonmises":lambda g:g.vonmises(.3,2.,40),
            "multinomial":lambda g:g.multinomial(10,[.1,.2,.7],20),"dirichlet":lambda g:g.dirichlet([1.,2.,3.],20),
            "dirichlet_small":lambda g:g.dirichlet([.03,.04,.05],20),
            "multivariate_normal":lambda g:g.multivariate_normal([1.,2.],[[4.,1.],[1.,3.]],size=20,method="cholesky"),
            "permuted":lambda g:g.permuted(np.arange(12,dtype=np.int32).reshape(3,4),axis=1),
            "permuted_flat":lambda g:g.permuted(np.arange(12,dtype=np.int32).reshape(3,4),axis=None),
        }.items():record("rng_"+name,fn(rng()))
        record("rng_bytes",np.frombuffer(rng().bytes(17),dtype=np.uint8))
        for i,child in enumerate(rng().spawn(2)):record("rng_generator_spawn_"+str(i),child.random(16))
        record("rng_hypergeometric_small",rng().hypergeometric(7,13,5,64))
        record("rng_hypergeometric_big",rng().hypergeometric(40,60,30,64))
        record("rng_hypergeometric_complement",rng().hypergeometric(60,40,95,64))
        record("rng_multivariate_hypergeometric_marginals",rng().multivariate_hypergeometric([12,7,5],9,size=24,method="marginals"))
        record("rng_multivariate_hypergeometric_count",rng().multivariate_hypergeometric([12,7,5],9,size=24,method="count"))
        record("rng_multivariate_hypergeometric_complement",rng().multivariate_hypergeometric([12,7,5],19,size=24,method="marginals"))
