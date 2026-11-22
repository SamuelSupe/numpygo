"""Differential contracts for the six numpy.polynomial series classes."""
import numpy as np
from numpy.polynomial import Polynomial, Chebyshev, Legendre, Hermite, HermiteE, Laguerre


def populate_series(record, fails):
    specs = [
        ("polynomial", Polynomial), ("chebyshev", Chebyshev),
        ("legendre", Legendre), ("hermite", Hermite),
        ("hermite_e", HermiteE), ("laguerre", Laguerre),
    ]
    domain, window = [1., 5.], [-1., 2.]
    coeff = np.array([1.5, -.5, .75, 1.2, 0.])
    other = np.array([-.3, 2., 1.])
    x = np.array([-1., .2, 1.7, 3.])
    fit_x = np.linspace(2., 5., 10)
    fit_y = 2 + 3 * fit_x + .5 * fit_x**2 + .01 * np.sin(fit_x)
    weights = np.linspace(.8, 1.2, 10)
    roots = np.array([1.5, 2.75, 4.])
    record("series_input_coeff", coeff)
    record("series_input_other", other)
    record("series_input_x", x)
    record("series_input_fit_x", fit_x)
    record("series_input_fit_y", fit_y)
    record("series_input_weights", weights)
    record("series_input_roots", roots)
    for name, cls in specs:
        tag = "series_" + name + "_"

        def num(n, a):
            record(tag+n, a)

        def ints(n, a):
            num(n, np.asarray(a, dtype=np.int64))

        def text(n, s):
            num(n, np.frombuffer(s.encode(), dtype=np.uint8))

        def series(n, value):
            num(n+"_coef", value.coef)
            num(n+"_domain", value.domain)
            num(n+"_window", value.window)
            text(n+"_symbol", value.symbol)

        a = coeff.copy()
        p = cls(a, domain=domain, window=window, symbol="t")
        q = cls(other, domain=domain, window=window, symbol="t")
        num("coef", p.coef)
        num("domain", p.domain)
        num("window", p.window)
        num("default_domain", cls.domain)
        num("default_window", cls.window)
        ints("metadata", [len(p), p.degree(), cls.maxpower])
        text("symbol", p.symbol)
        text("basis_name", cls.basis_name or "")
        num("iter", list(p))
        num("mapparms", p.mapparms())
        num("call", p(x))
        text("repr", repr(p))
        text("ascii", format(p, "ascii"))
        text("unicode", format(p, "unicode"))
        text("str", str(p))
        a[0] = 100.
        num("constructor_copy", p.coef)
        cp = p.copy()
        series("copy", cp)
        ints("copy_equal", [p == cp, p is cp, +p is p])
        cp.coef[0] = -9.
        num("copy_mutated", cp.coef)
        num("copy_original", p.coef)
        series("coef_view_mutation", cp)
        for n, v in [
            ("cutdeg", p.cutdeg(2)), ("cutdeg_large", p.cutdeg(100)),
            ("truncate", p.truncate(2)), ("truncate_large", p.truncate(100)),
            ("trim", p.trim()), ("trim_tol", cls([1.,2.,.001,0.], symbol="t").trim(.01)),
            ("trim_all", cls([.01,-.01], symbol="t").trim(.02)),
        ]:
            series(n,v)
        others = [
            p.copy(), q, cls(coeff, domain=[-1.,1.], window=window, symbol="t"),
            cls(coeff, domain=domain, window=[0.,1.], symbol="t"),
            cls(coeff, domain=domain, window=window, symbol="s"),
            (Chebyshev if cls is Polynomial else Polynomial)(coeff, domain=domain, window=window, symbol="t"),
        ]
        ints("comparisons", [[p.has_samecoef(v), p.has_samedomain(v), p.has_samewindow(v),
                              p.has_sametype(v), p == v, p != v] for v in others])
        series("basis", cls.basis(3, domain=domain, window=window, symbol="t"))
        series("basis_default", cls.basis(0))
        series("identity", cls.identity(domain=domain, window=window, symbol="t"))
        series("identity_default", cls.identity())
        num("identity_eval", cls.identity(domain=domain, window=window)(x))
        for n, v in [
            ("add",p+q), ("sub",p-q), ("mul",p*q),
            ("neg",-p), ("pos",+p), ("pow",p**3), ("pow_zero",p**0),
            ("add_coef",p+[2.,3.]), ("radd",2.+p), ("rsub",2.-p),
            ("rmul",2.*p), ("true_div",p/2.), ("floor_div",p//q), ("mod",p%q),
        ]:
            series(n, v)
        quo, rem = divmod(p, q)
        series("divmod_q", quo); series("divmod_r", rem)
        quo, rem = p.__rdivmod__([2.,3.,4.,5.,6.,7.])
        series("rdivmod_q", quo); series("rdivmod_r", rem)
        series("rfloordiv", p.__rfloordiv__([2.,3.,4.,5.,6.,7.]))
        series("rmod", p.__rmod__([2.,3.,4.,5.,6.,7.]))
        series("pow_17", cls([2.])**17)
        series("deriv", p.deriv(2))
        series("integ", p.integ(2, k=[2.,-1.], lbnd=2.))
        series("integ_default", p.integ(1, k=[.5]))
        for n, v in [("linspace_default",p.linspace(5)), ("linspace_reverse",p.linspace(6, [5., 1.])),
                     ("linspace_empty",p.linspace(0))]:
            num(n+"_x",v[0]); num(n+"_y",v[1])
        series("convert_default", p.convert())
        series("convert_same", p.convert(domain=domain, window=window))
        for other_name, target in specs:
            v = target([.4,-.3,.2], domain=[-2.,2.], window=[.5,1.5], symbol="u")
            composed = p(v)
            series("compose_"+other_name, composed)
            num("compose_"+other_name+"_eval", composed(x))
            converted = p.convert(domain=[-2.,2.], kind=target, window=[-1.,1.])
            series("convert_"+other_name, converted)
            num("convert_"+other_name+"_eval", converted(x))
            cast = target.cast(p, domain=[-2.,2.], window=[-1.,1.])
            series("cast_"+other_name, cast)
        fromroots = cls.fromroots(roots, domain=domain, window=window, symbol="t")
        series("fromroots", fromroots)
        num("roots", fromroots.roots())
        series("fromroots_default", cls.fromroots(roots))
        series("fromroots_infer", cls.fromroots(roots, domain=None))
        complex_roots = np.array([2+1j, 3-.5j])
        series("fromroots_complex", cls.fromroots(complex_roots, domain=domain, window=window, symbol="t"))
        for n, options in [
            ("fit",dict(domain=domain,window=window,w=weights,symbol="t")),
            ("fit_infer",dict(symbol="t")),
            ("fit_default",dict(domain=[],symbol="t")),
        ]:
            fit, status = cls.fit(fit_x, fit_y, [0,2,3], full=True, **options)
            series(n, fit)
            num(n+"_residual", status[0]); ints(n+"_rank", status[1])
            num(n+"_singular", status[2]); num(n+"_rcond", status[3])
            num(n+"_eval", fit(fit_x))
        fit = cls.fit(np.full(4,2.), np.array([1.,2.,3.,4.]), 0)
        series("fit_constant_domain",fit)
        for dtype in ["int32","float32"]:
            xv=np.arange(1,6,dtype=dtype)
            fit=cls.fit(xv,1.+2.*xv.astype("f8"),1)
            series("fit_"+dtype,fit)
            series("fromroots_"+dtype,cls.fromroots(xv[:2],domain=None))
        f32 = cls(np.array([1.,-.5,0.], dtype=np.float32), domain=domain,window=window,symbol="θ2")
        series("float32", f32)
        series("float32_neg", -f32)
        series("float32_pow", f32**2)
        series("float32_convert", f32.convert())
        text("float32_repr",repr(f32))
        text("float32_format",format(f32,"unicode"))
        cplx = cls(np.array([1+.5j, -.3+2j, .5],dtype="c16"), domain=domain,window=window,symbol="t")
        series("complex_mul", cplx*cplx)
        series("complex_convert",cplx.convert())
        num("complex_eval",cplx(x+1j))
        for dtype in ["int8","uint8","int16","uint16","int32","uint32","int64","uint64",
                      "float16","float32","float64","complex64","complex128"]:
            arr=np.array([1,2,0],dtype=dtype)
            if arr.dtype.kind=="c":
                arr[1]+=1j
            v=cls(arr)
            series("dtype_"+dtype, v)
            num("dtype_"+dtype+"_neg",(-v).coef)
            num("dtype_"+dtype+"_trim",v.trim().coef)
            num("dtype_"+dtype+"_pow",(v**2).coef)
            num("dtype_"+dtype+"_eval",v(np.array([.25,.5],dtype="f4")))
        fails(tag+"bad_bool",lambda: cls(np.array([True,False])))
        fails(tag+"bad_symbol", lambda: cls([1.,2.], symbol="a+b"))
        fails(tag+"bad_type", lambda: p + (Chebyshev if cls is Polynomial else Polynomial)(coeff,domain=domain,window=window,symbol="t"))
        fails(tag+"bad_domain", lambda: p + cls(coeff, domain=[-1,1], window=window, symbol="t"))
        fails(tag+"bad_window", lambda: p + cls(coeff, domain=domain, window=[0,1], symbol="t"))
        fails(tag+"different_symbol", lambda: p + cls(coeff, domain=domain, window=window, symbol="s"))
        fails(tag+"bad_basis", lambda: cls.basis(-1))
        fails(tag+"bad_cutdeg", lambda: p.cutdeg(-1))
        fails(tag+"bad_truncate", lambda: p.truncate(0))
        fails(tag+"bad_trim", lambda: p.trim(-.25))
        fails(tag+"bad_power", lambda: p**101)
        fails(tag+"bad_power_negative", lambda: p**-1)
        fails(tag+"bad_division", lambda: p // cls([0.],domain=domain,window=window,symbol="t"))
        fails(tag+"bad_true_div", lambda: p/q)
        fails(tag+"bad_format", lambda: format(p,"unknown"))
        fails(tag+"bad_empty", lambda: cls([]))
    p = Chebyshev.interpolate(np.exp, 6, domain=[1.,3.])
    record("series_chebyshev_interpolate_coef", p.coef)
    record("series_chebyshev_interpolate_domain", p.domain)
    record("series_chebyshev_interpolate_eval", p(np.array([1.,2.,3.])))

    from numpy.polynomial import polyutils as pu
    def util(n, value):
        record("polyutils_"+n,value)
    def util_text(n, value):
        util(n, np.frombuffer(value.encode(),dtype=np.uint8))
    a=np.array([1,2,0,0],dtype="i2")
    v=pu.trimseq(a)
    util("trimseq",v)
    v[0]=9
    util("trimseq_shared",a)
    util("trimseq_all",pu.trimseq(np.zeros(4,dtype="f4")))
    util("trimseq_empty",pu.trimseq(np.empty(0,dtype="i8")))
    util("trimseq_complex",pu.trimseq(np.array([1+2j,0,0],dtype="c8")))
    input_arrays=[np.array([2],dtype="i2"), np.array([1.,2.,0.], dtype="f4"),np.array([1+2j,0],dtype="c8")]
    for trim in [True,False]:
        values=pu.as_series(input_arrays,trim=trim)
        for i, v in enumerate(values):
            util("as_series_"+str(int(trim))+"_"+str(i),v)
        values[1][0]=44
        util("as_series_original_"+str(int(trim)),input_arrays[1])
    util("as_series_half",pu.as_series([np.array([1.,2.],dtype="f2")])[0])
    util("as_series_scalar",pu.as_series([np.float32(2.5)])[0])
    util("trimcoef_complex",pu.trimcoef(np.array([1+1j,.2j,0],dtype="c8"),.5))
    util("trimcoef_integer",pu.trimcoef(np.array([1,2,0,0],dtype="i4")))
    util("trimcoef_nan",pu.trimcoef(np.array([1.,2.,np.nan])))
    util("trimcoef_nan_tol",pu.trimcoef([1.,2.],np.nan))
    util("trimcoef_all",pu.trimcoef(np.array([-0.25,0.125],dtype="f4"),.5))
    for dtype in ["int32","float16","float32","float64","complex64","complex128"]:
        x=np.array([-2.,.5,3.],dtype=dtype)
        if x.dtype.kind=="c":
            x += np.array([1j,2j,-3j],dtype=dtype)
        util("getdomain_"+dtype,pu.getdomain(x))
        old=np.array([-2.,2.],dtype=dtype)
        new=np.array([1.,5.],dtype=dtype)
        if x.dtype.kind=="c":
            old+=np.array([-1j,1j],dtype=dtype)
            new+=np.array([2j,-2j],dtype=dtype)
        off,scale=pu.mapparms(old,new)
        util("mapparms_"+dtype+"_off",off)
        util("mapparms_"+dtype+"_scale",scale)
        util("mapdomain_"+dtype,pu.mapdomain(x,old,new))
    util("getdomain_nan",pu.getdomain([1.,np.nan,3.]))
    util("mapdomain_2d",pu.mapdomain(np.arange(6.,dtype="f4").reshape(2,3),np.array([0.,5.]),np.array([10.,-2.])))
    with np.printoptions(precision=5, floatmode="fixed",sign="+", nanstr="NaN",infstr="Infinity"):
        for i,x in enumerate([0.,-0.,1.25,-2.5,1e-8,3e9,np.nan,np.inf,-np.inf]):
            util_text("format_float_"+str(i),pu.format_float(np.float64(x),True))
    fails("polyutils_bad_trim",lambda: pu.trimcoef([1,2],-1))
    fails("polyutils_bad_empty",lambda: pu.as_series([np.empty(0)]))
    fails("polyutils_bad_rank",lambda: pu.as_series([np.ones((2,2))]))
    fails("polyutils_bad_bool",lambda: pu.as_series([np.array([True,False])]))
