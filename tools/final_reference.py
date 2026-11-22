"""Reference observations for extended/structured/runtime/foreign-boundary APIs."""
import contextlib
import ctypes
import io
import json
import pathlib
import pickle
import tempfile
import warnings

import numpy as np


def populate_final(record, fails):
    def raw(name, data):
        record(name, np.frombuffer(data, dtype=np.uint8).copy())

    def text(name, data):
        raw(name, data.encode("utf-8"))

    def ext(name, value):
        a = np.asarray(value)
        assert np.finfo(np.longdouble).nmant == 63 and np.dtype(np.longdouble).itemsize == 16
        # x87 has six unspecified padding bytes in each 16-byte storage slot.
        slots = np.frombuffer(a.tobytes(), dtype=np.uint8).reshape(-1, 16)
        record(name, slots[:, :10].copy())
        record(name + "_shape", np.asarray(a.shape, dtype=np.int64))

    vals = ["1.0000000000000000001", "9223372036854775809", "1e4000", "1e-4000",
            "-0", "1e-4951", "-1.25", "3.5"]
    a = np.array(vals, dtype=np.longdouble)
    ext("final_extended_parse", a)
    ext("final_extended_alias", np.array(vals, dtype=np.float128))
    info = np.finfo(np.longdouble)
    ext("final_extended_info", np.array([info.eps, info.tiny, info.max, info.smallest_subnormal]))
    x = np.array(["1.0000000000000000001", "3.125", "-7", "1e3000"], dtype=np.longdouble).reshape(2, 2)
    y = np.array(["0.0000000000000000001", "2.25"], dtype=np.longdouble)
    for name, fn in [("add", np.add), ("sub", np.subtract), ("mul", np.multiply), ("div", np.divide)]:
        ext("final_extended_" + name, fn(x, y))
    ext("final_extended_next", np.nextafter(np.array([0, 1, -1], dtype=np.longdouble), np.longdouble(2)))
    ext("final_extended_sqrt", np.sqrt(np.array(["2", "9", "1e3000"], dtype=np.longdouble)))
    ext("final_extended_sum", np.sum(x, axis=0))
    ext("final_extended_matmul", x @ np.array([[1., 2.], [3., 4.]], dtype=np.longdouble))
    t = x.T.copy().T
    t[1, 0] = np.longdouble("11.000000000000000001")
    ext("final_extended_view", t)
    z = np.zeros((2,), dtype=np.clongdouble)
    z.real = np.array(["1.0000000000000000001", "-3.125"], dtype=np.longdouble)
    z.imag = np.array(["2.25", "4.5"], dtype=np.longdouble)
    w = np.zeros((2,), dtype=np.complex256)
    w.real = [2, 3]
    w.imag = [-1, 4]
    ext("final_extended_complex", z)
    ext("final_extended_complex_alias", w)
    for name, fn in [("cadd", np.add), ("cmul", np.multiply), ("cdiv", np.divide)]:
        ext("final_extended_" + name, fn(z, w))
    ext("final_extended_conj", z.conj())
    ext("final_extended_abs", np.abs(z))
    special = np.array(["nan", "inf", "-inf", "-0"], dtype=np.longdouble)
    record("final_extended_special", np.array([np.isnan(special), np.isinf(special), np.signbit(special)]))

    # Scalar flexible dtypes and native record storage (including explicit endian).
    b = np.bytes_(b"a\0b\0")
    s = np.str_("中é\0")
    raw("final_record_bytes", b.tobytes())
    record("final_record_bytes_len", np.int64(len(b)))
    text("final_record_unicode", str(s))
    record("final_record_unicode_len", np.int64(len(s)))
    raw("final_record_void", np.void(b"\x01\xff\0").tobytes())
    record("final_record_hierarchy", np.array([
        np.issubdtype(t, family) for t in (np.bytes_, np.str_, np.void, np.int64)
        for family in (np.character, np.flexible)]))
    dt = np.dtype([("tag", "S3"), ("n", ">i4"), ("vec", "<f8", (2,))])
    rec = np.zeros(2, dtype=dt).view(np.recarray)
    rec.tag = [b"one", b"two"]; rec.n = [10, -20]; rec.vec = [[1.5, 2.5], [3.5, 4.5]]
    rec[::-1].n[0] = 70
    raw("final_record_layout_bytes", rec.tobytes())
    record("final_record_layout_offsets", np.array([dt.fields[n][1] for n in dt.names] + [dt.itemsize, dt.alignment], dtype=np.int64))
    record("final_record_column", rec[0].vec)
    record("final_record_scalar", np.int32(rec[1].n))
    text("final_record_scalar_tag", rec[1].tag.decode())
    record("final_record_is_record", np.bool_(isinstance(rec[0], np.record)))
    nested = np.dtype([("id", "i4"), ("pos", "f8", (2,)), ("name", "U3"),
                       ("nested", [("x", "i2"), ("tag", "S2")])], align=True)
    record("final_record_aligned_offsets", np.array([nested.fields[n][1] for n in nested.names]+[nested.itemsize,nested.alignment], dtype=np.int64))
    maskdt = np.ma.make_mask_descr(nested)
    record("final_record_mask_offsets", np.array([maskdt.fields[n][1] for n in maskdt.names]+[maskdt.itemsize], dtype=np.int64))
    r = np.fromregex(io.StringIO("item=alpha count=17\nignored\nitem=beta count=-3\n"),
                     r"item=(\w+) count=(-?\d+)", dtype=[("name","S5"),("count","i4")])
    raw("final_record_fromregex", r.tobytes())
    num = np.array([(1,[2.,3.]),(4,[5.,6.])],dtype=[("a","i4"),("v","f8",(2,))])
    record("final_record_flat", np.ma.flatten_structured_array(num))
    m = np.ma.array(num,mask=[(False,[True,False]),(True,[False,True])])
    flex=m.toflex(); recovered=np.ma.fromflex(flex); mf=np.ma.flatten_structured_array(recovered)
    record("final_record_fromflex_data", mf.data)
    record("final_record_fromflex_mask", mf.mask)
    record("final_record_mvoid", np.array([recovered[0]["a"],recovered[0]["v"][1]],dtype=np.float64))
    record("final_record_mvoid_mask",np.array([np.ma.is_masked(recovered[0]["a"]),np.ma.is_masked(recovered[0]["v"][0])]))
    ma = np.ma.array([1,2,3],mask=[False,True,False]); same=ma.view()
    record("final_record_ids", np.array([np.ma.ids(ma)[0]==np.ma.ids(same)[0],np.ma.ids(ma)[1]==np.ma.ids(same)[1],np.ma.ids(ma)[0]!=0,np.ma.ids(ma)[1]!=0]))
    obj=np.array(["red","missing","blue","missing"],dtype=object)
    masked=np.ma.masked_object(obj,"missing",copy=False)
    record("final_object_mask",masked.mask)
    text("final_object_filled","|".join(masked.filled("NA")))
    f=np.frompyfunc(lambda a,b:(a+b,a*b),2,2)
    ox=np.array([1,2],dtype=object).reshape(2,1);oy=np.array([10,20,30],dtype=object)
    outputs=f(ox,oy)
    record("final_object_callback_sum",outputs[0].astype(np.int64))
    record("final_object_callback_product",outputs[1].astype(np.int64))
    record("final_object_callback_meta",np.array([f.nin,f.nout],dtype=np.int64))
    g=np.frompyfunc(lambda a,b:a+b,2,1); base=np.array([1,2,3],dtype=object)
    record("final_object_reduce",np.int64(g.reduce(base)))
    record("final_object_accumulate",g.accumulate(base).astype(np.int64))
    record("final_object_outer",g.outer(base,np.array([10,20],dtype=object)).astype(np.int64))
    g.at(base,[0,0,2],np.array([4,5,6],dtype=object));record("final_object_at",base.astype(np.int64))
    text("final_object_type",str(np.object_(obj[0])))
    u=np.add; vals=np.array([1,2,3],dtype=np.int64)
    record("final_ufunc_meta",np.array([u.nin,u.nout,u.identity],dtype=np.int64))
    record("final_ufunc_call",u(vals,10))
    record("final_ufunc_reduce",u.reduce(vals));record("final_ufunc_accumulate",u.accumulate(vals))
    record("final_ufunc_outer",u.outer(vals,[4,5]))
    u.at(vals,[0,0,2],[4,5,6]);record("final_ufunc_at",vals)

    # Floating-point event policy, including event aggregation and restoration.
    olderr=np.geterr(); oldcall=np.geterrcall(); oldbuf=np.getbufsize()
    def policy(name):
        text(name,"|".join(np.geterr()[k] for k in ("divide","over","under","invalid")))
    policy("final_fp_default");record("final_fp_buf_default",np.int64(oldbuf))
    try:
        events=[]
        def cb(kind,flag):events.append(f"{kind}:{flag}")
        previous=np.seterr(all="call")
        text("final_fp_seterr_old","|".join(previous[k] for k in ("divide","over","under","invalid")))
        record("final_fp_seterrcall", np.bool_(np.seterrcall(cb) is oldcall and np.geterrcall() is cb))
        record("final_fp_divide",np.divide([1.,0.],[0.,0.]))
        np.exp(1000.);np.multiply(1e-300,1e-300);np.sqrt(-1.);np.multiply(np.float64(2.**-1022),0.5)
        np.floor_divide(np.array([1],dtype=np.int64),np.array([0],dtype=np.int64))
        text("final_fp_callback","|".join(events))
        events.clear()
        np.multiply(np.float32(1e30),np.float32(1e30))
        np.remainder(np.float64(1),np.float64(0))
        text("final_fp_callback_extra","|".join(events))
        with np.errstate(all="ignore",divide="raise"):
            policy("final_fp_scope");fails("final_fp_raises",lambda:np.divide(1.,0.))
        policy("final_fp_restore")
        record("final_fp_buf_old",np.int64(np.setbufsize(64)))
        record("final_fp_buf_new",np.int64(np.getbufsize()))
        fails("final_fp_invalid_buf",lambda:np.setbufsize(17))
        np.seterr(all="ignore",invalid="log");log=io.StringIO();np.seterrcall(log);np.sqrt(-1.)
        text("final_fp_log",log.getvalue())
        np.seterr(all="ignore",invalid="warn")
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always");np.sqrt(-1.)
        text("final_fp_warn",str(ws[0].message))
    finally:
        np.seterr(**olderr);np.seterrcall(oldcall);np.setbufsize(oldbuf)

    # Byte-for-byte formatted text, with both explicit and scoped settings.
    opts=np.get_printoptions()
    text("final_format_default","|".join(map(str,[opts[k] for k in ("precision","threshold","edgeitems","linewidth","floatmode","sign","nanstr","infstr")])))
    for name, value in [("int",np.array([1,-20,300],dtype=np.int16)),("bool",np.array([True,False])),
                        ("float",np.array([1.,2.345,-12.3,np.nan])),("sci",np.array([1e-10,12345.4])),
                        ("complex",np.array([1+2j,-3-4.5j])),("matrix",np.arange(14).reshape(2,7)),
                        ("scalar",np.array(1.25)),("empty",np.zeros((0,2),dtype="f8"))]:
        text("final_format_"+name+"_2string",np.array2string(value))
        text("final_format_"+name+"_repr",np.array_repr(value))
        text("final_format_"+name+"_str",np.array_str(value))
    with np.printoptions(precision=2,floatmode="fixed",sign="+",nanstr="NaN",infstr="Inf"):
        text("final_format_scope",np.array2string(np.array([1.,-2.345,np.nan,np.inf])))
        text("final_format_scope_options","|".join(map(str,[np.get_printoptions()[k] for k in ("precision","floatmode","sign","nanstr","infstr")])))
    text("final_format_summary",np.array2string(np.arange(10),threshold=5,edgeitems=2))
    text("final_format_line",np.array2string(np.arange(14),max_line_width=18))
    text("final_format_custom",np.array2string(np.arange(3),separator=", ",formatter={"int":lambda x:f"<{x}>"}))
    try:
        np.set_printoptions(precision=3,floatmode="maxprec_equal")
        text("final_format_set",np.array_str(np.array([1.,2.34567])))
    finally:
        np.set_printoptions(**opts)
    for style in ("ascii","unicode"):
        np.polynomial.set_default_printstyle(style)
        for name in ("Polynomial","Chebyshev","Legendre","Hermite","HermiteE","Laguerre"):
            cls=getattr(np.polynomial,name)
            text("final_poly_"+name+"_"+style,str(cls([1.,-2.,0.,3.])))
        text("final_poly_domain_"+style,str(np.polynomial.Polynomial([1.,-2.,3.],domain=[0,2])))
    np.polynomial.set_default_printstyle("unicode")
    fails("final_poly_invalid_style",lambda:np.polynomial.set_default_printstyle("bad"))

    # Real disk-backed mapping: offset/Fortran order, persistence and private writes.
    with tempfile.TemporaryDirectory() as td:
        path=pathlib.Path(td)/"mapping.bin"
        mm=np.memmap(path,dtype="<i4",mode="w+",offset=7,shape=(2,3),order="F")
        mm[:]=[[1,2,3],[4,5,6]]
        mm[:,::-1][0,0]=99;mm.flush()
        record("final_mmap_layout",mm);record("final_mmap_strides",np.array(mm.strides,dtype=np.int64))
        raw("final_mmap_file",path.read_bytes())
        del mm
        ro=np.memmap(path,dtype="<i4",mode="r",offset=7)
        record("final_mmap_read",ro);record("final_mmap_readonly",np.bool_(not ro.flags.writeable))
        fails("final_mmap_write_readonly",lambda:ro.__setitem__(0,0));del ro
        private=np.memmap(path,dtype="<i4",mode="c",offset=7);private[0]=42;private.flush()
        record("final_mmap_private",private)
        raw("final_mmap_private_file",path.read_bytes());del private
        rw=np.memmap(path,dtype="<i4",mode="r+",offset=7);rw[0]=-12;rw.flush();del rw
        raw("final_mmap_update_file",path.read_bytes())

    # Go equivalents: compare observable metadata/aliasing/stream behavior.
    arr=np.arange(6,dtype=np.int32).reshape(2,3); v=arr.T
    imported=np.from_dlpack(v)
    record("final_abi_dlpack_meta",np.array([np.shares_memory(v,imported),not imported.flags.writeable]))
    record("final_abi_dlpack_strides",np.array(imported.strides,dtype=np.int64))
    arr[1,2]=77;record("final_abi_dlpack_data",imported)
    imported[1,0]=123
    record("final_abi_dlpack_writeback",arr)
    ptr=arr.ctypes
    record("final_abi_ctypes_meta",np.array([ptr.data==arr.__array_interface__["data"][0],tuple(ptr.shape)==arr.shape,tuple(ptr.strides)==arr.strides]))
    ctypes.cast(ptr.data,ctypes.POINTER(ctypes.c_int32))[1]=-13
    record("final_abi_ctypes_data",arr)
    record("final_abi_dumps",pickle.loads(arr.dumps()))
    with tempfile.TemporaryDirectory() as td:
        path=pathlib.Path(td)/"array.pkl";arr.dump(path)
        record("final_abi_dump",pickle.loads(path.read_bytes()))
    ns=np.__array_namespace_info__();caps=ns.capabilities()
    record("final_abi_namespace_capabilities",np.array([caps["boolean indexing"],caps["data-dependent shapes"],caps["max dimensions"]],dtype=np.int64))
    text("final_abi_namespace_devices","|".join(ns.devices()))
    text("final_abi_namespace_default","|".join(str(ns.default_dtypes()[k]) for k in ("real floating","complex floating","integral","indexing")))
    text("final_abi_namespace_dtypes","|".join(sorted(ns.dtypes())))
    text("final_abi_namespace_real","|".join(sorted(ns.dtypes(kind="real floating"))))
    record("final_abi_include",np.bool_((pathlib.Path(np.get_include())/"numpy/arrayobject.h").is_file()))
    report=io.StringIO();np.info(arr,output=report)
    info=report.getvalue()
    record("final_abi_info",np.array([x in info for x in ("shape:  (2, 3)","strides:  (12, 4)","itemsize:  4","type: int32")]))
    cfg=np.show_config(mode="dicts")
    record("final_abi_config",np.bool_("Machine Information" in cfg and "Build Dependencies" in cfg))
    with contextlib.redirect_stdout(io.StringIO()) as report:np.show_runtime()
    record("final_abi_runtime",np.bool_("num_py" in report.getvalue().lower() or "numpy_version" in report.getvalue().lower()))
    record("final_abi_test",np.bool_(np.test(tests=["numpy._core.tests.test_arrayprint"],extra_argv=["-k","test_format_function"],verbose=0)))
    # Each specific class and interface gets its own raw-stream observation.
    isvalid=ctypes.pythonapi.PyCapsule_IsValid
    isvalid.argtypes=[ctypes.py_object,ctypes.c_char_p];isvalid.restype=ctypes.c_int
    for name in ("MT19937","PCG64","PCG64DXSM","Philox","SFC64"):
        cls=getattr(np.random,name)
        b=cls(123)
        record("final_rng_"+name+"_base", np.bool_(isinstance(b,np.random.BitGenerator)))
        record("final_rng_"+name+"_capsule_valid",np.bool_(isvalid(b.capsule,b"BitGenerator")))
        record("final_rng_"+name+"_capsule_raw",np.random.Generator(b).bit_generator.random_raw(4))
        for variant in ("ctypes","cffi"):
            b=cls(123);iface=getattr(b,variant)
            with b.lock:
                u64=[iface.next_uint64(iface.state) for _ in range(2)]
                u32=[iface.next_uint32(iface.state) for _ in range(3)]
                double=[iface.next_double(iface.state) for _ in range(2)]
            prefix="final_rng_"+name+"_"+variant
            record(prefix+"_u64",np.array(u64,dtype=np.uint64))
            record(prefix+"_u32",np.array(u32,dtype=np.uint32))
            record(prefix+"_double",np.array(double,dtype=np.float64))
            record(prefix+"_following",b.random_raw(4))
