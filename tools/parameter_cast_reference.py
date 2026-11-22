"""Fixed-width cast values, including exact rounding and memory order."""
import warnings
import numpy as np

DTYPES = ("bool","int8","int16","int32","int64","uint8","uint16","uint32","uint64",
          "float16","float32","float64","complex64","complex128")

def populate_cast(record, fails):
    with warnings.catch_warnings(),np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        for source in DTYPES:
            dt = np.dtype(source)
            if dt.kind=="b":
                a=np.array([False,True,False],dtype=dt)
            elif dt.kind in "iu":
                info=np.iinfo(dt)
                vals=[info.min,info.min+1,0,1,info.max-1,info.max]
                if dt.kind=="i":
                    vals.extend([-2,-1])
                vals.extend(v for v in (255,256,65519,65520,2**24+1,2**53+1,2**54+2**30+1) if info.min<=v<=info.max)
                a=np.array(vals,dtype=dt)
            else:
                rd=np.empty((),dtype=dt).real.dtype
                f=np.finfo(rd)
                vals=[-257.9,-128.5,-1.9,-.9,-0.,0.,.9,1.5,127.9,128.1,255.9,256.1,
                      f.smallest_subnormal,-f.smallest_subnormal,f.tiny,1+2**-11,1+2**-24]
                a=np.array(vals,dtype=rd)
                if dt.kind=="c":
                    z=np.empty(a.shape,dtype=dt)
                    z.real=a; z.imag=a[::-1]; a=z
            record("param_cast_input_"+source,a)
            for target in DTYPES:
                record(f"param_cast_{source}_{target}",a[::-1].astype(target))
            if dt.kind in "fc":
                a=np.array([-np.inf,-f.max,-2.**63,-2.**31,65519.,65520.,2.**31,2.**63,2.**64,f.max,np.inf,np.nan],dtype=rd)
                if dt.kind=="c":
                    z=np.empty(a.shape,dtype=dt)
                    z.real=a; z.imag=a[::-1]; a=z
                record("param_cast_extreme_"+source,a)
                for target in ("bool","int8","int16","int32","int64","uint8","uint16","uint64",
                               "float16","float32","float64","complex64","complex128"):
                    record(f"param_cast_extreme_{source}_{target}",a.astype(target))
                # Out-of-range uint32 conversion differs between NumPy's
                # scalar and SIMD paths. Verify the scalar contract explicitly.
                record(f"param_cast_extreme_{source}_uint32_scalar",
                       np.array([np.array(v).astype(np.uint32) for v in a],dtype=np.uint32))
        base=np.arange(24,dtype=np.int64).reshape(2,3,4)
        for kind,x in (("c",base),("f",base.transpose(2,1,0)),
                       ("perm",base.transpose(1,2,0)),("reverse",base[::-1,:,::-1])):
            for order in "CFAK":
                record(f"param_cast_layout_{kind}_{order}",x.astype(np.float32,order=order))
        fails("param_cast_safe",lambda:base.astype(np.int8,casting="safe"))
        fails("param_cast_order",lambda:base.astype(np.int64,order="invalid"))
        fails("param_cast_rule",lambda:base.astype(np.int64,casting="invalid"))
        x=np.array([1,-2,300],dtype=">i2")
        record("param_cast_endian_equiv",x.astype(np.int16,casting="equiv",copy=False))
        fails("param_cast_endian_no",lambda:x.astype(np.int16,casting="no"))
