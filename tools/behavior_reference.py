"""Systematic dtype/parameter behavior cases, beyond API-name coverage."""
import numpy as np


DTYPES = (
    "bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
    "uint64", "float16", "float32", "float64", "complex64", "complex128",
)
BINARY = (
    "add", "subtract", "multiply", "divide", "floor_divide", "remainder",
    "fmod", "power", "float_power", "bitwise_and", "bitwise_or", "bitwise_xor",
    "left_shift", "right_shift", "gcd", "lcm", "equal", "not_equal", "less",
    "less_equal", "greater", "greater_equal", "minimum", "maximum", "fmin",
    "fmax", "logical_and", "logical_or", "logical_xor", "hypot", "arctan2",
    "copysign", "logaddexp", "logaddexp2", "heaviside", "nextafter", "ldexp",
)
UNARY = (
    "sqrt", "exp", "exp2", "expm1", "log", "log2", "log10", "log1p", "sin",
    "cos", "tan", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh",
    "arcsinh", "arccosh", "arctanh", "cbrt", "degrees", "radians", "rint",
    "floor", "ceil", "trunc", "fix", "negative", "positive", "square",
    "reciprocal", "absolute", "fabs", "conjugate", "sign", "logical_not",
    "invert", "isnan", "isinf", "isfinite", "signbit", "spacing",
)


def populate_behavior(record, fails):
    def case(name, fn):
        try:
            value = fn()
        except (TypeError, ValueError, OverflowError):
            fails(name, fn)
        else:
            record(name, value)

    left, right = {}, {}
    with np.errstate(all="ignore"):
        for dtype in DTYPES:
            kind = np.dtype(dtype).kind
            x = [2, 0, -3] if kind == "i" else [2, 0, 3]
            y = [1, 2, 0]
            if kind == "b":
                x, y = [True, False, True], [True, False, True]
            elif kind == "f":
                x, y = [2.25, -0., -3.5], [1.5, 2., 0.]
            elif kind == "c":
                x, y = [2+1j, 0j, -3-1j], [1+0j, 2-.25j, 0j]
            a = np.array(x, dtype=dtype).reshape(3, 1)
            b = np.array(y, dtype=dtype).reshape(1, 3)
            record("behavior_left_" + dtype, a)
            record("behavior_right_" + dtype, b)
            left[dtype], right[dtype] = a[::-1], b[:, ::-1]
            for name in UNARY:
                case(f"behavior_unary_{name}_{dtype}", lambda: getattr(np, name)(a[::-1]))

            for name in ("sum", "prod", "mean", "var", "std", "min", "max",
                         "nansum", "nanprod", "nanmean", "nanvar", "nanstd", "nanmin",
                         "nanmax", "all", "any", "count_nonzero"):
                # Reduction on a negative-stride, rank-three view with a
                # singleton dimension and a nontrivial tuple of axes.
                v = np.concatenate((a, b.T), axis=1).reshape(3, 1, 2)[::-1]
                case(f"behavior_reduce_{name}_{dtype}", lambda: getattr(np, name)(
                    v, axis=(0, -1), keepdims=True))
            for name in ("argmin", "argmax", "nanargmin", "nanargmax"):
                case(f"behavior_reduce_{name}_{dtype}", lambda: getattr(np, name)(
                    a[::-1], axis=0, keepdims=True))

        for name in BINARY:
            fn = getattr(np, name)
            for a in DTYPES:
                for b in DTYPES:
                    case(f"behavior_binary_{name}_{a}_{b}", lambda: fn(left[a], right[b]))

        # Accumulator dtype, overflow before division, and precision loss
        # matter even when the final result fits the dtype.
        samples = [
            ("f32_cancel", [16777216., 1., -16777216.], "float32", None),
            ("f64_as_f32", [16777216., 1., -16777216.], "float64", "float32"),
            ("f32_overflow", [3e38, 3e38], "float32", None),
            ("f16_square_sum", [300., -300., 0., 0.], "float16", None),
            ("f32_product", [1e20, 1e20, 1e-20], "float32", None),
            ("explicit_int", [1.9, 2.9], "float64", "int64"),
            ("c64_cancel", [16777216+16777216j, 1+1j, -16777216-16777216j], "complex64", None),
            ("nan_f32", [16777216., np.nan, 1., -16777216.], "float32", None),
        ]
        for label, values, dtype, out in samples:
            a = np.array(values, dtype=dtype)
            record("behavior_precision_input_" + label, a)
            for name in ("sum", "prod", "mean", "var", "std", "nansum", "nanprod",
                         "nanmean", "nanvar", "nanstd"):
                case(f"behavior_precision_{name}_{label}", lambda: getattr(np, name)(a, dtype=out))

        # Exact signs of zero and NaN/Inf branches, tested separately from
        # tolerance-based comparisons.
        a = np.array([-0., 0., 1., -1., np.inf, -np.inf, np.nan])
        b = np.array([0., -0., -0., 0., np.inf, np.inf, 2.])
        record("behavior_special_a", a)
        record("behavior_special_b", b)
        for name in ("add", "subtract", "multiply", "divide", "floor_divide", "remainder",
                     "minimum", "maximum", "fmin", "fmax", "copysign", "arctan2", "nextafter"):
            out = getattr(np, name)(a, b)
            record("behavior_special_" + name, out)
            record("behavior_special_sign_" + name, np.signbit(out[np.equal(out, 0)]))
        for name in ("negative", "absolute", "rint", "sqrt", "sign", "spacing"):
            out = getattr(np, name)(a)
            record("behavior_special_" + name, out)
            record("behavior_special_sign_" + name, np.signbit(out[np.equal(out, 0)]))

        for dtype in ("float32", "float64"):
            a = np.array([-3., 1.5, -0., np.inf, -np.inf, np.nan], dtype=dtype)
            b = np.array([2., -0., 0., 4., np.inf, 5.], dtype=dtype)
            scalar = np.array(-.75, dtype=dtype)
            record("behavior_dense_a_" + dtype, a)
            record("behavior_dense_b_" + dtype, b)
            record("behavior_dense_scalar_" + dtype, scalar)
            for name in ("add","subtract","multiply","divide"):
                f = getattr(np, name)
                for label, x, y in (("array",a,b),("right",a,scalar),("left",scalar,b),("scalar",scalar,scalar)):
                    record(f"behavior_dense_{name}_{dtype}_{label}",f(x,y))

        # Ufunc call parameters: selected-only evaluation, broadcast mask,
        # strided/aliased out, dtype resolution and numpy casting categories.
        xa = np.arange(6,dtype=np.int16).reshape(2,3)
        xb = np.array([.5,1.5,2.5],dtype=np.float32)
        mask = np.array([True,False,True])
        base = np.full((2,3),-99.,dtype=np.float64)
        out = base[:,::-1]
        np.add(xa,xb,out=out,where=mask)
        record("behavior_ufunc_mask_out",out)
        record("behavior_ufunc_mask_base",base)
        overlap=np.arange(1,7,dtype=np.int64)
        np.add(overlap[:-1],10,out=overlap[1:])
        record("behavior_ufunc_overlap",overlap)
        one=np.array([2,2,2],dtype=np.int16)
        ex=np.array([-1,3,-2],dtype=np.int16)
        out=np.array([77,77,77],dtype=np.int16)
        np.power(one,ex,out=out,where=[False,True,False])
        record("behavior_ufunc_mask_power",out)
        record("behavior_ufunc_explicit_dtype",np.add(xa,xb,dtype=np.float32))
        record("behavior_ufunc_cast_unsafe",np.add(np.array([127,128,300],dtype=np.int16),np.int16(1),dtype=np.int8,casting="unsafe"))
        record("behavior_ufunc_complex_sqrt",np.sqrt(np.array([-1.,4.],dtype=np.float32),dtype=np.complex64))
        fails("behavior_ufunc_abs_complex_cast",lambda:np.absolute(np.array([3+4j,1+2j],dtype=np.complex128),dtype=np.float32))
        record("behavior_ufunc_bool_compare",np.equal(xa,np.int16(2),dtype=np.bool_))
        out=np.zeros((2,3),dtype=np.int64)
        np.add(np.array([1,2,3]),np.int64(5),out=out)
        record("behavior_ufunc_out_broadcast",out)
        out=np.full((2,3),-1.,dtype=np.float64)
        np.add(np.array([1.,2.,3.]),np.float64(2.),out=out,where=np.array([[True],[False]]))
        record("behavior_ufunc_where_broadcast",out)
        alias=np.array([True,False,True])
        np.logical_not(alias,out=alias,where=alias)
        record("behavior_ufunc_where_alias",alias)
        fails("behavior_ufunc_cast_no",lambda:np.add(xa,xb,casting="no"))
        fails("behavior_ufunc_cast_safe",lambda:np.add(xa,xb,dtype=np.int8,casting="safe"))
        fails("behavior_ufunc_cast_out",lambda:np.add(xa,xb,out=np.zeros((2,3),dtype=np.int8)))
        fails("behavior_ufunc_bad_mask_dtype",lambda:np.add(xa,xb,where=np.array([1,0,1],dtype=np.int64)))
        fails("behavior_ufunc_bad_out_shape",lambda:np.add(xa,xb,out=np.zeros((3,2))))
        fails("behavior_ufunc_readonly_out",lambda:np.add(xa,xb,out=np.broadcast_to(np.float64(0),(2,3))))

        # Quantile method/typed-q grid; dtype and the leading q shape both
        # belong to the API contract, not just the interpolated values.
        methods=("inverted_cdf","averaged_inverted_cdf","closest_observation",
                 "interpolated_inverted_cdf","hazen","weibull","linear",
                 "median_unbiased","normal_unbiased","lower","higher","midpoint","nearest")
        data=np.array([0,7,3,11,2,10,5,8,1,9,4,6,19,13,21,15,18,12,20,14,23,16,22,17]).reshape(2,3,4)
        for dtype in ("bool","int16","int64","float16","float32","float64"):
            a=data.astype(dtype)
            record("behavior_quantile_input_"+dtype,a)
            for qdtype in ("float16","float32","float64"):
                q=np.array([[0.,.125,.25],[.5,.75,1.]],dtype=qdtype)
                record("behavior_quantile_q_"+qdtype,q)
                for meth in methods:
                    case(f"behavior_quantile_{meth}_{dtype}_{qdtype}",lambda:np.quantile(a[::-1,:,::-1],q,axis=(0,2),keepdims=True,method=meth))
        a=data.astype(np.float64);a[0,1,1]=np.nan;a[1,2,3]=np.nan
        q=np.array([0.,.25,.5,.75,1.],dtype=np.float32)
        record("behavior_quantile_nan_input",a);record("behavior_quantile_nan_q",q)
        for meth in methods:
            record("behavior_quantile_nan_"+meth,np.nanquantile(a[::-1,:,::-1],q,axis=(0,2),method=meth))
        weights=np.array([[0.,1.,2.,3.],[4.,5.,6.,7.]])
        record("behavior_quantile_weights",weights)
        record("behavior_quantile_weighted",np.quantile(data[::-1,:,::-1],q,axis=(0,2),method="inverted_cdf",weights=weights))
        record("behavior_quantile_weighted_reversed_axes",np.quantile(data,q,axis=(2,0),method="inverted_cdf",weights=weights.T))
        fullweights=np.arange(1.,25.).reshape(2,3,4)
        record("behavior_quantile_fullweights",fullweights)
        record("behavior_quantile_nan_weighted",np.nanquantile(a,q,axis=(0,2),method="inverted_cdf",weights=fullweights))
        exact=np.array([2**63+1,2**63+3,2**63+9,2**63+17],dtype=np.uint64)
        record("behavior_quantile_exact_input",exact)
        record("behavior_quantile_exact",np.quantile(exact,q,method="nearest"))
        record("behavior_quantile_integer_q",np.quantile(exact,np.array([0,1],dtype=np.int32)))
        record("behavior_quantile_percentile",np.percentile(data,np.array([20,70],dtype=np.float32),axis=1,method="hazen"))
        outbase=np.full((5,3),-99.,dtype=np.float64)
        np.quantile(data,q,axis=(0,2),out=outbase[:,::-1])
        record("behavior_quantile_out_base",outbase)
        record("behavior_quantile_infinity",np.quantile(np.array([-np.inf,0.,np.inf]),np.array([0.,.5,1.])))
        record("behavior_quantile_nan_empty",np.nanquantile(np.array([],dtype=np.float64),q))
        record("behavior_quantile_nan_all",np.nanquantile(np.array([np.nan,np.nan],dtype=np.float32),np.array([.25,.75],dtype=np.float64)))
        record("behavior_quantile_nan_first_lane",np.nanquantile(np.array([[np.nan,np.nan],[1.,2.]],dtype=np.float32),np.array([.25,.75],dtype=np.float64),axis=1))
        record("behavior_quantile_nan_scalar",np.quantile(np.array([np.nan,2.],dtype=np.float32),np.float64(.25)))
        record("behavior_quantile_nanpercentile",np.nanpercentile(a,np.array([20,70],dtype=np.float32),axis=1,method="weibull"))
        for dtype in ("float16","float32"):
            for method in methods:
                record(f"behavior_quantile_scalar_{dtype}_{method}",np.quantile(data.astype(dtype),.33,axis=1,method=method))
        record("behavior_quantile_bool_median",np.median(np.array([True,False,True,False])))
        fails("behavior_quantile_empty",lambda:np.quantile(np.array([]),q))
        fails("behavior_quantile_bad_q",lambda:np.quantile(data,[-.1,.5]))
        fails("behavior_quantile_bad_method",lambda:np.quantile(data,q,method="foo"))
        fails("behavior_quantile_bad_weights_method",lambda:np.quantile(data,q,axis=(0,2),weights=weights))
        fails("behavior_quantile_bad_weights_shape",lambda:np.quantile(data,q,axis=(0,2),method="inverted_cdf",weights=weights.T))
        fails("behavior_quantile_negative_weights",lambda:np.quantile(data,q,axis=(0,2),method="inverted_cdf",weights=weights-2))
        fails("behavior_quantile_zero_weights",lambda:np.quantile(data,q,axis=(0,2),method="inverted_cdf",weights=np.zeros_like(weights)))

        a=np.arange(1,25,dtype=np.int16).reshape(2,3,4).transpose(2,1,0)[::-1]
        mask=np.array([[[True,False]],[[False,True]],[[True,True]],[[False,False]]])
        record("behavior_reduceopt_input",np.arange(1,25,dtype=np.int16).reshape(2,3,4))
        record("behavior_reduceopt_mask",mask)
        for name,initial in (("add",3),("multiply",2),("maximum",-100),("minimum",100),
                             ("bitwise_xor",1),("gcd",6),("logical_or",False)):
            record("behavior_reduceopt_"+name,getattr(np,name).reduce(a,axis=(0,2),keepdims=True,initial=initial,where=mask))
        record("behavior_reduceopt_subtract",np.subtract.reduce(a,axis=0,initial=100,where=mask))
        record("behavior_reduceopt_divide",np.divide.reduce(np.array([2,4,8],dtype=np.int32)))
        outbase=np.full((3,),-1.,dtype=np.float32)
        np.add.reduce(a,axis=(0,2),out=outbase[::-1],where=mask)
        record("behavior_reduceopt_out_base",outbase)
        record("behavior_reduceopt_scalar",np.add.reduce(np.array(7,dtype=np.int16),axis=0,keepdims=True,initial=3))
        record("behavior_reduceopt_empty",np.maximum.reduce(np.empty((0,3)),axis=0,initial=-2.))
        record("behavior_reduceopt_empty_mask",np.add.reduce(a,axis=2,where=False,initial=5))
        fails("behavior_reduceopt_missing_initial",lambda:np.maximum.reduce(a,axis=0,where=mask))
        fails("behavior_reduceopt_nonreorderable",lambda:np.subtract.reduce(a,axis=(0,2)))
        fails("behavior_reduceopt_empty_no_identity",lambda:np.minimum.reduce(np.empty((0,3)),axis=0))
        for name in ("maximum","minimum","bitwise_xor","gcd","subtract","logical_xor"):
            record("behavior_accumulate_"+name,getattr(np,name).accumulate(a,axis=-2))
        for name in ("add","multiply","maximum","subtract"):
            record("behavior_reduceat_"+name,getattr(np,name).reduceat(a,[0,2,2,1],axis=1))
        fails("behavior_reduceat_index",lambda:np.add.reduceat(a,[-1],axis=1))

        weak_values=(("bool",True),("small",3),("negative",-1),("wide",256),
                     ("u63",2**63),("big",2**64),("float",.5),("complex",1+2j))
        weak_ops=("add","subtract","multiply","divide","floor_divide","power","hypot","minimum",
                  "equal","less","greater_equal","logical_and")
        for dtype in DTYPES:
            a=np.array([1,2],dtype=dtype)
            record("behavior_weak_input_"+dtype,a)
            for tag,v in weak_values:
                for name in weak_ops:
                    fn=getattr(np,name)
                    case(f"behavior_weak_{name}_{dtype}_{tag}_right",lambda:fn(a,v))
                    case(f"behavior_weak_{name}_{dtype}_{tag}_left",lambda:fn(v,a))
        a=np.array([127,1],dtype=np.int8)
        record("behavior_weak_dtype",np.add(a,256,dtype=np.int16))
        record("behavior_weak_float32",np.add(np.array([1,2],dtype=np.float32),.25))
        record("behavior_weak_strong",np.add(np.array([1,2],dtype=np.float32),np.float64(.25)))
        out=np.array([True,False])
        np.less(a,100,out=out,where=[False,True])
        record("behavior_weak_compare_mask",out)
