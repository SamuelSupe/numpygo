"""Stateful class contracts, with numerical values captured before advancing."""
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory


def populate_classes(record, fails):
    def integers(name, values):
        record("class_" + name, np.asarray(values, dtype=np.int64))

    def text(name, value):
        record("class_" + name, np.frombuffer(value.encode(), dtype=np.uint8))

    def num(name, value):
        record("class_" + name, value)

    a = np.arange(6, dtype=np.int64).reshape(2, 3).T
    f = a.flat
    state, values = [], []
    for _ in range(a.size):
        state.append((f.index, *f.coords))
        values.append(next(f))
    state.append((f.index, *f.coords))
    integers("flat_state", state)
    integers("flat_values", values)
    integers("flat_base", f.base.shape)
    f = a.flat
    next(f)
    num("flat_copy", f.copy())
    num("flat_array", np.asarray(f))
    integers("flat_copy_cursor", [f.index, len(f)])
    num("flat_slice", f[1:5:2])
    integers("flat_get_cursor", [f.index])
    num("flat_scalar", f[-1])
    num("flat_gather", f[np.array([[0, -1], [2, 2]])])
    f[1:5] = [10, 20]
    num("flat_assign", a)
    f[[0, 0, 5]] = [9, 8, 7]
    num("flat_repeated", a)
    f[np.array([True, False, True, False, True, False])] = [99, 88]
    num("flat_mask", a)
    f[:] = []
    num("flat_empty_rhs", a)
    fails("class_flat_oob", lambda: f[6])
    fails("class_flat_bad_mask", lambda: f[np.array([True, False])])
    c = np.arange(4, dtype=np.int64)
    np.asarray(c.flat)[0] = 99
    num("flat_shared", c)

    a = np.arange(3, dtype=np.int64).reshape(1, 3)
    b = np.arange(2, dtype=np.int64).reshape(2, 1)
    it = np.broadcast(a, b)
    integers("broadcast_meta", [it.nd, it.ndim, it.numiter, it.size, *it.shape])
    state, values = [], []
    for _ in range(it.size):
        state.append([it.index, *[x.index for x in it.iters]])
        values.append(next(it))
    integers("broadcast_state", state)
    integers("broadcast_values", values)
    it.reset()
    integers("broadcast_reset", [it.index, *[x.index for x in it.iters], *next(it)])
    empty = np.broadcast()
    integers("broadcast_zero", [empty.ndim, empty.size, empty.numiter, len(next(empty))])
    fails("class_broadcast_shape", lambda: np.broadcast(np.ones(2), np.ones(3)))
    integers("ndindex", list(np.ndindex(2, 3)))
    integers("ndindex_empty", [len(list(np.ndindex(2, 0))), len(list(np.ndindex()))])
    en = np.ndenumerate(np.arange(6).reshape(2,3).T)
    first = next(en)
    integers("ndenumerate_iter", [en.iter.index, *en.iter.coords])
    integers("ndenumerate", [[*i, v] for i, v in [first, *en]])

    a = np.arange(12, dtype=np.int64).reshape(3, 4)
    for name, source, order, flags in [
        ("c", a, "C", []), ("f", a, "F", []),
        ("transpose", a.T, "K", []), ("reverse", a[::-1, ::-1], "K", []),
        ("reverse_c", a[::-1, ::-1], "C", ["multi_index"]),
        ("multi", a.T, "K", ["multi_index"]),
        ("c_index", a.T, "K", ["c_index"]),
        ("f_index", a, "C", ["f_index"]),
        ("external", a, "F", ["external_loop"]),
        ("gap", a[:, ::2], "C", []),
        ("broadcast", [a, np.arange(4, dtype=np.int64)], "C", ["external_loop"]),
        ("scalar", np.array(9, dtype=np.int64), "K", []),
    ]:
        with np.nditer(source, order=order, flags=flags) as it:
            prefix = "nditer_" + name
            integers(prefix+"_shape", it.shape)
            integers(prefix+"_meta", [it.ndim, it.nop, it.itersize, it.has_index,
                                          it.has_multi_index, it.iterationneedsapi])
            text(prefix+"_dtypes", ",".join(d.name for d in it.dtypes))
            for j, view in enumerate(it.itviews):
                integers(prefix+"_view"+str(j), [*view.shape, *view.strides])
            rows, steps, tracked = [], [], []
            for v in it:
                values = v if isinstance(v, tuple) else (v,)
                rows.extend([list(x) for x in zip(*(np.ravel(x).tolist() for x in values))])
                steps.append([it.iterindex, values[0].size])
                if it.has_multi_index:
                    tracked.append(it.multi_index)
                elif it.has_index:
                    tracked.append([it.index])
            integers(prefix+"_values", rows)
            integers(prefix+"_steps", steps)
            if tracked:
                integers(prefix+"_tracked", tracked)
    with np.nditer(a, flags=["multi_index", "c_index", "ranged"]) as it:
        it.multi_index = (1, 2)
        integers("nditer_seek", [it.index, it.iterindex, int(it.value), *it.multi_index])
        cp = it.copy()
        it.index = 2
        integers("nditer_copy", [cp.iterindex, int(cp.value), it.iterindex, int(it.value)])
        cp.close()
    # NumPy 2.5 has inconsistent traversal after combining both tracked
    # index flags, an explicit seek and an iterrange change. Test ranged
    # traversal independently of that upstream flag interaction.
    with np.nditer(a, flags=["ranged"]) as it:
        it.iterrange = (3, 7)
        integers("nditer_range_meta", it.iterrange)
        integers("nditer_range", [int(v) for v in it])
        it.reset()
        integers("nditer_reset", [it.iterindex, int(it.value)])
        it.iterindex = 5
        integers("nditer_seek_iterindex", [it.iterindex, int(it.value)])
    with np.nditer(a, flags=["multi_index"]) as it:
        it.remove_axis(1)
        integers("nditer_remove_axis", [int(x) for x in it])
    with np.nditer(a, flags=["multi_index"]) as it:
        it.remove_multi_index()
        it.enable_external_loop()
        integers("nditer_external_enable", [x.copy().tolist() for x in it])
    with np.nditer(a, flags=["c_index"]) as it:
        values = [int(it.value)]
        while it.iternext():
            values.append(int(it[0]))
        integers("nditer_iternext", values)
    with np.nditer([a, None], op_flags=[["readonly"], ["writeonly", "allocate"]],
                   op_dtypes=["i8", "f8"]) as it:
        for x, y in it:
            y[...] = x * 2
        num("nditer_allocate", it.operands[1])
    with np.nditer(a, flags=["buffered", "external_loop", "delay_bufalloc"],
                   op_flags=["readwrite"], op_dtypes=["f8"], casting="unsafe", buffersize=3) as it:
        integers("nditer_delayed", [it.has_delayed_bufalloc])
        it.reset()
        for x in it:
            x[...] = x * 2 + 1
    num("nditer_writeback", a)
    with np.nditer(np.empty((0,2)), flags=["zerosize_ok"]) as it:
        integers("nditer_empty", [it.itersize, it.finished])
    fails("class_nditer_zero_error", lambda: np.nditer(np.empty(0)))
    fails("class_nditer_flags_error", lambda: np.nditer(a, flags=["multi_index", "external_loop"]))

    names = sorted(n for n in np.__all__ if isinstance(getattr(np, n, None), np.ufunc))
    meta = []
    for n in names:
        u = getattr(np, n)
        meta.append("|".join((n, u.__name__, str(u.nin), str(u.nout), str(u.nargs),
                              str(u.ntypes), str(u.identity), u.signature or "", ",".join(u.types))))
    text("ufunc_metadata", "\n".join(meta))
    # Strong type resolution covers representative homogeneous and mixed
    # loops, including complex-to-real and integer-to-floating results.
    cases = {
        "add": [("int8","int16"), ("float32","int16"), ("complex64","float64"),
                ("int64","uint64"), ("bool","bool"), ("float16","float16")],
        "divide": [("int32","int32"), ("float32","float32"), ("complex64","float32")],
        "sqrt": [("int8",), ("int64",), ("float32",), ("complex64",)],
        "absolute": [("complex64",), ("int32",), ("float16",)],
        "less": [("int64","uint64"), ("float32","int16"), ("complex128","complex128")],
        "ldexp": [("float32","int16"), ("float64","int64")],
        "modf": [("float16",), ("int32",), ("float64",)],
        "frexp": [("float32",), ("int16",), ("float64",)],
        "divmod": [("int16","int8"), ("float32","float16")],
        "logical_and": [("int64","float32"), ("bool","complex64")],
        "matmul": [("float32","float64"), ("int32","int32")],
    }
    rows = []
    for name, inputs in cases.items():
        u = getattr(np, name)
        for inp in inputs:
            result = u.resolve_dtypes(tuple(np.dtype(x) for x in inp)+(None,)*u.nout)
            rows.append("|".join((name, ",".join(inp), ",".join(x.name for x in result))))
    text("ufunc_resolve", "\n".join(rows))
    for name, u, args in [
        ("weak_int",np.add,(np.dtype("i1"),int,None)),
        ("weak_float",np.add,(np.dtype("f4"),float,None)),
        ("weak_complex",np.add,(np.dtype("f4"),complex,None)),
    ]:
        text("ufunc_resolve_"+name, ",".join(x.name for x in u.resolve_dtypes(args)))
    text("ufunc_resolve_reduction", ",".join(x.name for x in np.add.resolve_dtypes((None,np.dtype("i1"),None),reduction=True)))
    text("ufunc_resolve_signature", ",".join(x.name for x in np.add.resolve_dtypes((np.dtype("i2"),np.dtype("i2"),None),signature=(None,None,np.dtype("f4")))))
    x = np.array([-2.5, 3.25, 0.0], dtype="f4")
    for name in ["frexp", "modf"]:
        for i, v in enumerate(getattr(np, name)(x)):
            num("ufunc_"+name+"_"+str(i), v)
    q,r = np.divmod(np.array([-7,7,9],dtype="i2"), np.array(3,dtype="i2"))
    num("ufunc_divmod_0",q); num("ufunc_divmod_1", r)
    out0, out1 = np.full(3, 19., dtype="f4")[::-1], np.full(3, 29., dtype="f4")[::-1]
    np.modf(x, out=(out0,out1), where=[True,False,True])
    num("ufunc_multi_out0",out0); num("ufunc_multi_out1",out1)
    a = np.arange(6, dtype=np.int64).reshape(2,3)
    np.add.at(a, ([0,0,1],[1,1,0]), [10,20,30])
    num("ufunc_at_multi",a)
    np.negative.at(a, ([0,0,1],[1,1,0]))
    num("ufunc_at_unary",a)
    out=np.zeros((2,3),dtype="f4")[:,::-1]
    np.add.accumulate(np.arange(6,dtype="i2").reshape(2,3),axis=1,out=out,dtype="i8")
    num("ufunc_accumulate_out",out)
    out=np.full((2,3),-9,dtype="i8")[:,::-1]
    np.add.outer(np.array([1,2],dtype="i8"),np.array([10,20,30],dtype="i8"),where=[True,False,True],out=out)
    num("ufunc_outer_out",out)
    m = np.arange(12,dtype="f8").reshape(2,2,3)
    v = np.array([1,2,3],dtype="f8")
    num("ufunc_matmul",np.matmul(m,v))
    num("ufunc_matvec",np.matvec(m,v))
    num("ufunc_vecmat",np.vecmat(v,m.swapaxes(-1,-2)))
    num("ufunc_vecdot",np.vecdot(v+1j, m))

    a = np.array([0,1,-2,3], dtype="i2")
    p = np.poly1d(a, variable="z")
    num("poly_coeff",p.coeffs)
    integers("poly_meta",[p.order,len(p),p[0],p[2],p[-1],p[8]])
    text("poly_variable",p.variable)
    text("poly_repr",repr(p))
    text("poly_str",str(p))
    a[2] = -4
    num("poly_shared",p.c)
    p[3] = 2
    num("poly_expand",p.c)
    p[3] = 0
    num("poly_leading_zero",p.c)
    fails("class_poly_bad_power",lambda: p.__setitem__(-1,2))
    p=np.poly1d([1.,-2.,3.])
    q=np.poly1d([2.,1.])
    for name, v in [("add",p+q),("sub",p-q),("mul",p*q),("pow",p**3),
                    ("compose",p(q)),("negative",-p),("positive",+p),
                    ("scale",p*2),("scalar_div",p/2),("deriv",p.deriv(2)),("integ",p.integ(2,[3,4]))]:
        num("poly_"+name,v.c)
    quotient, remainder = p/q
    num("poly_quotient",quotient.c); num("poly_remainder",remainder.c)
    num("poly_eval",p(np.array([-2.,0.,3.])))
    num("poly_fromroots",np.poly1d([1.,-1.],r=True).c)
    num("poly_roots",np.sort(np.poly1d([1.,0.,-1.]).roots))
    integers("poly_equal",[p==np.poly1d(p),p==q])

    with TemporaryDirectory() as directory:
        path = Path(directory) / "mapped.bin"
        a = np.memmap(path, dtype=">i4", mode="w+", offset=3, shape=(2,3), order="F")
        a[:] = np.arange(6).reshape(2,3)
        v = a[:, ::-1].T
        text("memmap_filename", Path(a.filename).name)
        text("memmap_mode", a.mode)
        integers("memmap_metadata", [a.offset, *a.shape, *a.strides, v.offset, *v.shape, *v.strides])
        text("memmap_view_filename", Path(v.filename).name)
        text("memmap_view_mode", v.mode)
        integers("memmap_copy", [a.copy().filename is None, a.copy().offset is None, a.copy().mode is None])
        v[0, 1] = 123
        v.flush()
        num("memmap_view", v.astype("i4"))
        num("memmap_bytes", np.frombuffer(path.read_bytes(),dtype="u1"))
