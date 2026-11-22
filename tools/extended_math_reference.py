"""x87 reference values for transcendentals, ufuncs and non-binary64 FFT."""
import warnings
import numpy as np


def populate_extended_math(record, fails):
    assert np.finfo(np.longdouble).nmant == 63
    def put(name, value):
        a = np.asarray(value)
        assert a.dtype in (np.dtype(np.longdouble), np.dtype(np.clongdouble)), (name, a.dtype)
        slots = np.frombuffer(a.tobytes(order="C"), dtype=np.uint8).reshape(-1,16)
        record(name, slots[:,:10].copy())
        record(name+"_shape", np.array(a.shape,dtype=np.int64))
    def real(values,shape=None):
        a=np.array(values,dtype=np.longdouble)
        return a if shape is None else a.reshape(shape)
    def cplx(values,shape=None):
        a=np.empty(len(values),dtype=np.clongdouble)
        a.real=real([r for r,i in values])
        a.imag=real([i for r,i in values])
        return a if shape is None else a.reshape(shape)

    # Exercise both extended exponent range and bits beyond float64.
    x=real(["-0","0","1e-4000","-1e-4000","0.125","-0.125",
            "0.9999999999999999999","1.0000000000000000001",
            "2.5","-2.5","1e4000","-1e4000","nan","inf","-inf"])
    put("exmath_real_input",x)
    z=cplx([("0","0"),("-0","-0"),("1","0"),("-1","-0"),
            ("0","1"),("0","-1"),("0.25","-0.5"),("2","3"),
            ("-3","1e-30"),("1","1e-40"),
            ("1e-2000","-1e-2000"),("1e4000","2e4000")])
    put("exmath_complex_input",z)
    real_names=("sqrt exp exp2 expm1 log log2 log10 log1p sin cos tan arcsin arccos arctan "
                "sinh cosh tanh arcsinh arccosh arctanh cbrt degrees radians rint floor ceil trunc fabs sign spacing sinc").split()
    complex_names=("sqrt exp exp2 expm1 log log2 log10 log1p sin cos tan arcsin arccos arctan "
                   "sinh cosh tanh arcsinh arccosh arctanh rint sign sinc").split()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in real_names:
            put("exmath_real_"+name,getattr(np,name)(x[::-1]))
        for name in complex_names:
            put("exmath_complex_"+name,getattr(np,name)(z[::-1]))
        far=real(["1e4932","-1e4932","11356.52","-11398.8","-11399.5","0.0000000000000000001",
                  "-0.0000000000000000001","1e-4950","-1e-4950"])
        put("exmath_far_input",far)
        for name in ("exp","expm1","exp2","log","log1p","sin","cos","tan"):
            put("exmath_far_"+name,getattr(np,name)(far))
        nonfinite=cplx([("inf","0"),("-inf","-0"),("0","inf"),("0","-inf"),
                        ("inf","inf"),("nan","0"),("0","nan"),("1","inf")])
        put("exmath_nonfinite_input",nonfinite)
        for name in ("sqrt","exp","log","arcsin","arctan","arccosh","sign"):
            put("exmath_nonfinite_"+name,getattr(np,name)(nonfinite))

        a=real(["-5.5","-0","2.0000000000000000002","1e4000","nan","inf"],(6,1))
        b=real(["-2","3","0","inf","0.5"],(1,5))
        put("exmath_binary_a",a);put("exmath_binary_b",b)
        for name in ("power","float_power","floor_divide","remainder","fmod","hypot","arctan2","copysign","logaddexp","logaddexp2","heaviside"):
            put("exmath_binary_"+name,getattr(np,name)(a,b))
        ca=cplx([("2","3"),("-1","-0"),("1.0000000000000000001","1e-20")],(3,1))
        cb=cplx([("2","0"),("-2","0"),("0.5","0.25"),("0","0")],(1,4))
        put("exmath_cpower_a",ca);put("exmath_cpower_b",cb)
        put("exmath_cpower",np.power(ca,cb))
        put("exmath_cfloat_power",np.float_power(ca,cb))
        put("exmath_angle",np.angle(z))
        put("exmath_angle_degree",np.angle(z,deg=True))
        record("exmath_isreal",np.isreal(z))
        record("exmath_iscomplex",np.iscomplex(z))
        record("exmath_isposinf",np.isposinf(x))
        record("exmath_isneginf",np.isneginf(x))
        put("exmath_nan_to_num",np.nan_to_num(x))
        put("exmath_cnan_to_num",np.nan_to_num(nonfinite))
        put("exmath_round",np.round(x,2))
        put("exmath_cround",np.round(z,2))
        frac,integ=np.modf(x)
        put("exmath_modf_frac",frac);put("exmath_modf_integ",integ)
        m,e=np.frexp(x);put("exmath_frexp",m);record("exmath_frexp_exponent",e)
        ex=np.array([-16380,-4,0,4,16380],dtype=np.int64).reshape(1,5)
        v=real(["1.0000000000000000001","1e4000","-1e-4000","0"],(4,1))
        put("exmath_ldexp_input",v);record("exmath_ldexp_exponents",ex)
        put("exmath_ldexp",np.ldexp(v,ex))
        record("exmath_isclose",np.isclose(real(["1","1e4000","nan","0"]),real(["1.0000000000000000001","1.01e4000","nan","1e-4000"]),rtol=0,atol=0,equal_nan=True))
        i0=real(["0","1","-2","8","10","1000","11356","11360","inf","nan"])
        put("exmath_i0_input",i0);put("exmath_i0",np.i0(i0))
        near=cplx([("1.0000000000000000001","1e-20"),("2","0")])
        put("exmath_near_real_input",near);put("exmath_near_real",np.real_if_close(near))
        put("exmath_not_near_real",np.real_if_close(z))
        for name in ("floor","ceil","trunc","cbrt","degrees","radians","spacing","fabs"):
            fails("exmath_bad_complex_"+name,lambda: getattr(np,name)(z))

    a=real(["1.0000000000000000001","2","3","4","5","6"],(2,3))
    mask=np.array([[True,False,True],[False,True,False]])
    put("exmath_ufunc_input",a);record("exmath_ufunc_mask",mask)
    for name in ("sin","log","exp"):
        out=real(["91"]*6,(2,3))[:,::-1]
        getattr(np,name)(a[:,::-1],where=mask,out=out)
        put("exmath_ufunc_"+name,out)
    out=real(["91"]*6,(2,3))[:,::-1]
    np.power(a,real(["2","3","4"]),where=mask,out=out)
    put("exmath_ufunc_power",out)
    put("exmath_ufunc_reduce",np.add.reduce(a,axis=1,initial=np.longdouble("0.0000000000000000001"),where=mask,keepdims=True))
    put("exmath_ufunc_hypot",np.hypot.reduce(a,axis=1))
    put("exmath_ufunc_accumulate",np.maximum.accumulate(a[:,::-1],axis=1))
    put("exmath_ufunc_reduceat",np.subtract.reduceat(a,[0,2,1],axis=1))
    put("exmath_ufunc_outer",np.power.outer(real(["2","3"]),real(["0.5","2"])))
    at=real(["1","2","3"])
    np.add.at(at,[1,1,0],real(["0.0000000000000000001","1","2"]))
    put("exmath_ufunc_at",at)
    for label,op,args in (
        ("finite","exp",["10000"]),
        ("overflow","exp",["12000"]),
        ("underflow","exp",["-12000"]),
        ("divide","divide",["1e4000","0"]),
        ("invalid","divide",["0","0"]),
        ("sqrt_invalid","sqrt",["-1e4000"]),
        ("exact_tiny","multiply",[str(np.finfo(np.longdouble).tiny),"0.5"]),
        ("inexact_tiny","multiply",["1e-4950","0.3"]),
    ):
        observed=[]
        old_call=np.seterrcall(lambda message,flag:observed.append(flag))
        try:
            with np.errstate(all="call"):
                getattr(np,op)(*map(np.longdouble,args))
        finally:
            np.seterrcall(old_call)
        record("exmath_fp_"+label,np.int64(observed[0] if observed else 0))

    a=real([str((i%9)-4)+".125" for i in range(42)],(2,3,7))
    a[0,0,0]=np.longdouble("1.0000000000000000001")
    z=cplx([(str((i%11)-5)+".25",str(i%5-2)+".125") for i in range(42)],(2,3,7))
    z.real[0,0,0]=np.longdouble("1.0000000000000000001")
    put("exfft_real_input",a);put("exfft_complex_input",z)
    for name in ("fft","ifft","rfft","irfft","hfft","ihfft"):
        fn=getattr(np.fft,name)
        source=a if name in ("rfft","ihfft") else z
        for norm in ("backward","forward","ortho"):
            for axis,n in ((-1,7),(-1,8),(-1,17),(1,5),(0,1),(0,4)):
                v=source[::-1,:,::-1]
                k=f"exfft_{name}_{norm}_{axis+2}_{n}"
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    put(k,fn(v,n=n,axis=axis,norm=norm))
    for name in ("fftn","ifftn","rfftn","irfftn","fft2","ifft2","rfft2","irfft2"):
        src=a if name.startswith("r") else z
        put("exfft_"+name,getattr(np.fft,name)(src.transpose(2,0,1),s=(5,8),axes=(0,2),norm="ortho"))
    for label,source in (("small",a[0,0]*np.longdouble("1e-2000")),("large",a[0,0]*np.longdouble("1e2000"))):
        put("exfft_"+label+"_input",source)
        put("exfft_"+label,np.fft.fft(source,n=31))
    prime=real([str(i%17-8) for i in range(127)])
    put("exfft_prime_input",prime);put("exfft_prime",np.fft.fft(prime))
    precision=real(["1.0000000000000000001","-1","1","-1"])
    put("exfft_precision_input",precision)
    put("exfft_precision",np.fft.fft(precision))
    out=np.zeros((2,3,14),dtype=np.clongdouble)[:,:,::2]
    np.fft.fft(z,n=7,out=out)
    put("exfft_out",out)
    for name in ("fft","ifft","rfft","irfft"):
        source=np.array([1.25,2.5,-3,4,5],dtype=np.float32)
        fn=getattr(np.fft,name)
        out=np.zeros(fn(source,n=7).shape,dtype=np.longdouble if name=="irfft" else np.clongdouble)
        fn(source,n=7,out=out)
        put("exfft_native_out_"+name,out)
    fails("exfft_bad_real",lambda: np.fft.rfft(z))
    fails("exfft_bad_length",lambda: np.fft.fft(a,n=0))
    for name,fn in (("fftfreq",np.fft.fftfreq),("rfftfreq",np.fft.rfftfreq)):
        for n in (1,6,7):
            for i,d in enumerate(map(np.longdouble,["1.0000000000000000001","-1e4000","1e-4000","0","inf"])):
                with np.errstate(all="ignore"):
                    put(f"exfft_{name}_{n}_{i}",fn(n,d))
    put("exfft_shift",np.fft.fftshift(z,axes=(0,2)))
    put("exfft_ishift",np.fft.ifftshift(z,axes=(0,2)))
