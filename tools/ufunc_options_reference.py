"""Core-axis remapping, explicit loops and output allocation for NumPy ufuncs."""
import numpy as np


def populate_ufunc_options(record, fails):
    def r(name, a):
        record("ufopt_" + name, a)

    def f(name, op):
        fails("ufopt_" + name, op)

    dtypes = ("bool", "int8", "uint8", "int16", "uint16", "int32", "uint32",
              "int64", "uint64", "float16", "float32", "float64", "float128",
              "complex64", "complex128", "complex256")
    # Canonical batch dimensions (2,1) and (1,5) broadcast. Permutations
    # put core dimensions in nontrailing positions in both inputs and output.
    for name in ("matmul", "matvec", "vecmat", "vecdot"):
        u = getattr(np, name)
        lm, rm = name in ("matmul", "matvec"), name in ("matmul", "vecmat")
        sha = (2, 1, 3, 4) if lm else (2, 1, 4)
        shb = (1, 5, 4, 2) if rm else (1, 5, 4)
        pa = (2, 0, 3, 1) if lm else (2, 0, 1)
        pb = (0, 3, 1, 2) if rm else (0, 2, 1)
        axes = [(0, 2) if lm else (0,), (3, 1) if rm else (1,),
                (0, 3) if name == "matmul" else () if name == "vecdot" else (1,)]
        for dtype in dtypes:
            a = np.arange(np.prod(sha)).reshape(sha) % 13
            b = np.arange(np.prod(shb)).reshape(shb) % 7 + 1
            if np.dtype(dtype).kind == "c":
                a = a + (a % 3 - 1)*1j
                b = b + (b % 4 - 2)*1j
            a, b = a.astype(dtype), b.astype(dtype)
            prefix = name + "_" + dtype
            r(prefix + "_a", a)
            r(prefix + "_b", b)
            a = (a[..., ::-1]).transpose(pa)
            b = (b[..., ::-1, :] if rm else b[..., ::-1]).transpose(pb)
            for order in "KACF":
                r(prefix + "_" + order, u(a, b, axes=axes, order=order))

    a = np.arange(24., dtype="f8").reshape(2,3,4)
    b = np.arange(20., dtype="f8").reshape(4,5)
    v = np.arange(4., dtype="f8")
    r("mv_optional", np.matmul(a, v, axes=[(1,2),(0,),(0,)]))
    r("vm_optional", np.matmul(v, a.swapaxes(-1,-2), axes=[(0,),(1,2),(0,)]))
    r("vv_optional", np.matmul(v, v, axes=[(0,),(0,),()]))
    p = np.arange(24., dtype="f8").reshape(2,4,3)
    q = p[:, :, ::-1]
    r("vd_axis", np.vecdot(p,q,axis=1))
    r("vd_axis_keep", np.vecdot(p,q,axis=1,keepdims=True))
    r("vd_axes_keep", np.vecdot(p,q,axes=[(1,),(1,)],keepdims=True))
    r("vd_axes_keep_out", np.vecdot(p,q,axes=[(1,),(1,),(0,)],keepdims=True))
    r("vd_axis_neg", np.vecdot(p,q,axis=-2,keepdims=True))
    r("vd_axis_false", np.vecdot(p,q,axis=1,keepdims=False))
    r("vd_mixed_ranks", np.vecdot(p,v.reshape(4,1),axis=-2,keepdims=True))
    f("vd_axis_both", lambda: np.vecdot(p,q,axis=1,axes=[(1,),(1,)]))
    f("vd_axis_bad", lambda: np.vecdot(p,q,axis=3))
    f("vd_keep_wrong_axes", lambda: np.vecdot(p,q,axes=[(1,),(1,),()],keepdims=True))
    f("mm_axis", lambda: np.matmul(a,b,axis=-1))
    f("mm_keep", lambda: np.matmul(a,b,keepdims=False))
    f("mm_axes_short", lambda: np.matmul(a,b,axes=[(1,2),(0,1)]))
    f("mm_axes_duplicate", lambda: np.matmul(a,b,axes=[(1,1),(0,1),(1,2)]))
    f("mm_axes_wrong_rank", lambda: np.matmul(a,b,axes=[(1,2),(0,),(1,2)]))
    f("mm_axes_bad_output", lambda: np.matmul(a,b,axes=[(1,2),(0,1),(0,4)]))
    f("mm_where", lambda: np.matmul(a,b,where=True))
    f("mm_bad_order", lambda: np.matmul(a,b,order="Q"))
    f("mm_core_mismatch", lambda: np.matmul(a,np.zeros((1,5))))

    # Output contributes to outer broadcasting, but never core broadcasting.
    for name, u, lhs, rhs, sh in [
        ("mm",np.matmul,a,b,(7,2,3,5)),
        ("mv",np.matvec,a,v,(7,2,3)),
        ("vm",np.vecmat,v,a.swapaxes(-1,-2),(7,2,3)),
        ("vd",np.vecdot,a,v,(7,2,3)),
    ]:
        out = np.full(sh,-99.,dtype="f4")[...,::-1]
        u(lhs,rhs,out=out)
        r(name+"_broadcast_out",out)
    out = np.full((3,7,2,5),-99.,dtype="f8")[:,:,::-1,:]
    np.matmul(a,b,axes=[(1,2),(0,1),(0,3)],out=out)
    r("mm_mapped_out",out)
    out = np.full((2,1,3),-99.,dtype="f8")[:, :, ::-1]
    np.vecdot(p,q,axis=1,keepdims=True,out=out)
    r("vd_keep_out",out)
    f("mm_bad_core_out",lambda: np.matmul(np.ones((2,1)),np.ones((1,1)),out=np.zeros((2,3))))
    f("mm_bad_batch_out",lambda: np.matmul(a,b,out=np.zeros((1,3,5))))
    f("vd_bad_keep_out",lambda: np.vecdot(p,q,axis=1,keepdims=True,out=np.zeros((2,2,3))))
    # Repeated output overlaps an input: computation uses a snapshot.
    x=np.arange(12.,dtype="f8").reshape(2,2,3)
    np.matmul(x, np.eye(3)*2, out=x)
    r("mm_alias_out",x)

    # K preserves compatible batch stride order, C/F refer to canonical axes.
    x=np.arange(120.,dtype="f8").reshape(2,3,4,5).swapaxes(0,1)
    y=np.arange(30.,dtype="f8").reshape(5,6)
    for order in "KACF":
        r("mm_batch_order_"+order,np.matmul(x,y,order=order))
        r("mm_f_order_"+order,np.matmul(np.asfortranarray(x),np.asfortranarray(y),order=order))
        r("mm_empty_order_"+order,np.matmul(np.empty((2,0,5)),y,order=order))
        r("mm_empty_contract_"+order,np.matmul(np.empty((2,3,0)),np.empty((0,4)),order=order))
        r("mm_mapped_f_order_"+order,np.matmul(np.asfortranarray(a.transpose(1,0,2)),
            np.asfortranarray(b),axes=[(0,2),(0,1),(0,2)],order=order))

    x=np.arange(24.,dtype="f8").reshape(2,3,4).transpose(1,0,2)[:,::-1,::2]
    y=np.arange(4.,dtype="f8").reshape(2,2)
    for order in "KACF":
        r("add_order_"+order,np.add(x,y,order=order))
        r("add_f_order_"+order,np.add(np.asfortranarray(x),1.,order=order))
        r("add_f_conflict_"+order,np.add(np.asfortranarray(x),np.ascontiguousarray(x),order=order))
        for i, z in enumerate(np.modf(x/3.,order=order)):
            r("modf_order_"+order+"_"+str(i),z)
        for i, z in enumerate(np.frexp(x/3.,signature=("f4","f4","i4"),order=order)):
            r("frexp_signature_"+order+"_"+str(i),z)
    # A singleton axis must not block comparing the two nonsingleton axes.
    # Fast contiguous allocation differs from masked/iterator allocation.
    for sh in [(2,1,3),(1,2,3),(2,3,1),(2,1,1,3)]:
        a1=np.asfortranarray(np.arange(6.).reshape(sh))
        name="_".join(map(str,sh))
        for order in "KACF":
            for label,rhs in [("scalar",np.array(1.)),("row",np.ones(sh[-1])),
                              ("cast",np.ones(sh,dtype="f4",order="F"))]:
                r("singleton_"+name+"_"+label+"_"+order,np.add(a1,rhs,order=order))
            r("singleton_"+name+"_mask_"+order,np.add(a1,1.,where=np.array(True),order=order))
            for i,z in enumerate(np.modf(a1/3.,order=order)):
                r("singleton_"+name+"_modf_"+order+"_"+str(i),z)
    a1=np.arange(6.,dtype="f8").reshape(2,3).T / 3
    for order in "KACF":
        for arr_order in "CF":
            out=np.empty(a1.shape,order=arr_order)
            v,w=np.modf(a1,order=order,out=(out,None))
            r("partial_"+arr_order+"_"+order+"_0",v)
            r("partial_"+arr_order+"_"+order+"_1",w)
        r("lower_"+order,np.add(a1,1.,order=order.lower()))
    x=np.array([1.1,1.2,1.0000000001],dtype="f8")
    y=np.array([1.2,1.1,1.0000000002],dtype="f8")
    r("less_dtype",np.less(x,y,dtype="bool"))
    r("less_signature",np.less(x,y,signature=("f8","f8","?")))
    z=np.array([3+4j,-5+12j],dtype="c16")
    r("abs_dtype",np.absolute(z,dtype="f8"))
    r("abs_dtype_unsafe",np.absolute(z,dtype="f8",casting="unsafe"))
    r("abs_dtype_widen",np.absolute(z.astype("c8"),dtype="f8"))
    r("abs_signature",np.absolute(z,signature=("c8","f4")))
    f("abs_narrow_dtype",lambda: np.absolute(z,dtype="f4"))
    f("dtype_signature",lambda: np.add(x,y,dtype="f8",signature=("f8","f8","f8")))
    f("invalid_signature",lambda: np.add(x,y,signature=("f8","f4","f8")))
    f("signature_safe",lambda: np.add(x,y,signature=("f4","f4","f4"),casting="safe"))
    f("elementwise_axes",lambda: np.add(x,y,axes=[(),(),()]))
    f("frexp_dtype",lambda: np.frexp(x,dtype="f8"))
    f("elementwise_order",lambda: np.add(x,y,order="Q"))
    r("signature_add",np.add(x,y,signature=("f4","f4","f4")))
    r("signature_matmul",np.matmul(a,b,signature=("f4","f4","f4")))
    r("signature_ldexp",np.ldexp(np.array([1,2],dtype="i2"),np.array([2,3],dtype="i1"),signature=("f4","i4","f4")))
    r("signature_weak",np.add(np.array([1,2],dtype="i1"),300,signature=("i2","i2","i2")))
    f("signature_weak_overflow",lambda: np.add(np.array([1,2],dtype="i1"),300,signature=("i1","i1","i1")))
