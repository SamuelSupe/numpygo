"""Reference cases for the unified ndarray storage, shared streams and eig."""
import io
import numpy as np


def populate_unified(record, fails):
    def ext(name, value):
        a = np.asarray(value)
        assert np.finfo(np.longdouble).nmant == 63
        slots = np.frombuffer(a.tobytes(), dtype=np.uint8).reshape(-1, 16)
        record("unified_ext_" + name, slots[:, :10].copy())

    a = np.array(["1.0000000000000000001", "2.25", "-3.5", "4",
                  "1e4000", "9223372036854775809"], dtype=np.longdouble).reshape(2, 3)
    ext("broadcast", a[:, ::-1] + np.array([1, 2, 3], dtype=np.int64))
    ext("index", a[[1, 0], [2, 0]])
    ext("transpose_reshape", a.T.reshape(6))
    ext("where", np.where([[True], [False]], a, np.longdouble("3.0000000000000000002")))
    ext("sum", a.sum(axis=0))
    ext("dot", a[:, :2] @ np.array([[1, 2], [3, 4]], dtype=np.longdouble))
    ext("integer", np.array([2**63+1, 2**64-1], dtype=np.uint64).astype(np.longdouble))
    a.T[1, 0] = np.longdouble("1.0000000000000000002")
    ext("writeback", a)
    z = np.zeros((2, 2), dtype=np.clongdouble)
    z.real = [[1, 2], [3, 4]]
    z.imag = [[-1, 0], [1, 2]]
    z.imag.T[1, 0] = np.longdouble("0.0000000000000000001")
    ext("complex_writeback", z)
    ext("complex_sum", z.sum(axis=0))
    ext("complex_dot", z @ z)
    record("unified_ext_compare", z.real > np.longdouble(2))
    stat = np.array(["1.0000000000000000001", "2", "3", "4"], dtype=np.longdouble).reshape(2,2)
    ext("cumsum", np.cumsum(stat, axis=1))
    ext("cumprod", np.cumprod(stat, axis=0))
    ext("mean", stat.mean(axis=0))
    ext("var", stat.var(axis=0))
    ext("std", stat.std())
    ext("complex_var", np.array([1+2j,3+4j], dtype=np.clongdouble).var())
    ext("nextafter", np.nextafter(np.array([0,1,-1],dtype=np.longdouble),np.longdouble(2)))
    ext("info", np.array([np.finfo(np.longdouble).eps,np.finfo(np.longdouble).tiny,np.finfo(np.longdouble).max]))
    ext("norm", np.linalg.norm(stat))
    ext("complex_norm", np.linalg.norm(z,axis=1))

    def npy(name,v):
        buf=io.BytesIO()
        np.save(buf,v,allow_pickle=False)
        raw=bytearray(buf.getvalue())
        if v.dtype.kind in "fc" and v.dtype.itemsize>=16:
            start=10+int.from_bytes(raw[8:10],"little")
            for i in range(start,len(raw),16):
                raw[i+10:i+16]=b"\0"*6
        record("unified_npy_"+name,np.frombuffer(raw,dtype=np.uint8))
    npy("extended",stat.T)
    npy("complex",z)

    dt = np.dtype([("id", "i1"), ("value", ">f8"), ("vec", "<i2", (2,)),
                   ("nested", [("x", "<f4"), ("tag", "S3")])])
    a = np.zeros((2, 3), dtype=dt)
    a["id"] = np.arange(6).reshape(2, 3)
    a["value"] = np.arange(6).reshape(2, 3) + .5
    a["vec"] = np.arange(12).reshape(2, 3, 2)
    a["nested"]["x"] = 2.5
    a["nested"]["tag"] = b"abc"
    record("unified_record_stride", np.array(a["value"].strides, dtype=np.int64))
    record("unified_record_nested", a["nested"]["x"])
    a.T[::-1]["value"][0, 1] = 90.25
    a["vec"][0, 1, 1] = -11
    a["nested"]["x"][1, 0] = 7.5
    def raw(name, v):
        record("unified_record_" + name, np.frombuffer(v.tobytes(), dtype=np.uint8))
    raw("writeback", a)
    raw("index", a[[1, 0], [2, 1]])
    raw("reshape", a.T.reshape(6))
    raw("concat", np.concatenate([a, a[::-1]], axis=0))
    raw("where", np.where([[True], [False]], a, a[:, ::-1]))
    b = a.copy()
    b["id"][1, 1] = 99
    record("unified_record_equal", a == b)
    fails("unified_record_readonly", lambda: np.broadcast_to(a, (2, 2, 3))["id"].__setitem__((0, 0, 0), 1))

    a = np.array(["1969-12-31", "1970-01-01", "2000-02-29", "NaT"], dtype="datetime64[D]").reshape(2, 2)
    b = np.array([3, -2], dtype="timedelta64[h]")
    def ticks(name, v):
        record("unified_date_" + name, np.asarray(v).view("i8"))
    ticks("plus", a + b)
    ticks("minus", a - np.datetime64("1970-01-01T03", "h"))
    ticks("cast", a.astype("datetime64[M]"))
    ticks("sort", np.sort(a, axis=None))
    ticks("index", a[[1, 0], [0, 1]])
    ticks("concat", np.concatenate([a, a.astype("datetime64[h]")], axis=1))
    ticks("where", np.where([[True], [False]], a, a.astype("datetime64[h]")))
    record("unified_date_nat", np.isnat(a))
    record("unified_date_eq", a == a.astype("datetime64[h]"))
    a.T[0, 1] = np.datetime64("2001-01-01")
    ticks("writeback", a)
    d = np.array([-5, 6, 9], dtype="timedelta64[3h]")
    e = np.array([1, 2, 3], dtype="timedelta64[2h]")
    ticks("duration_add", d + e)
    ticks("duration_mod", d % e)
    ticks("duration_scale", d * 2)
    ticks("duration_sum", d.sum())
    record("unified_date_duration_ratio", d / e)
    ticks("duration_cumsum",np.cumsum(d))
    npy("datetime",a.T)
    npy("timedelta",d)

    a = np.array([1, 2, 3, 4, 5, 6], dtype=object).reshape(2, 3)
    record("unified_obj_add", (a[:, ::-1] + np.arange(3)).astype("i8"))
    record("unified_obj_index", a[[1, 0], [2, 1]].astype("i8"))
    record("unified_obj_sum", a.sum(axis=0).astype("i8"))
    record("unified_obj_compare", a > 3)
    a.T[1, 0] = 90
    record("unified_obj_writeback", a.astype("i8"))
    record("unified_obj_where", np.where([[True], [False]], a, a[:, ::-1]).astype("i8"))
    mixed = np.array([None, "", "abc", 0, 2], dtype=object)
    record("unified_obj_truth", mixed.astype(bool))

    for name in ("MT19937", "PCG64", "PCG64DXSM", "Philox", "SFC64"):
        bit = getattr(np.random, name)(12345)
        g = np.random.Generator(bit)
        r = np.random.RandomState(bit)
        r2 = np.random.RandomState(bit)
        def put(suffix, v):
            record("unified_random_" + name + "_" + suffix, v)
        put("uint32", g.integers(0, 123456, 3, dtype=np.uint32))
        put("legacy_normal", r.standard_normal(1))
        saved = r.get_state(legacy=False)
        put("cache_flag", np.bool_(saved["has_gauss"]))
        put("cache_value", np.float64(saved["gauss"]))
        put("modern", g.random(3))
        put("raw", bit.random_raw(5))
        put("second_legacy", r2.standard_normal(1))
        put("cached", r.standard_normal(1))
        put("randint", r.randint(-15, 21, 7, dtype=np.int8))
        put("modern_normal", g.standard_normal(4))
        put("second_cached", r2.standard_normal(1))
        r.set_state(saved)
        put("replay_modern", g.random(3))
        put("replay_raw", bit.random_raw(5))
        put("replay_cached", r.standard_normal(1))
        if name == "MT19937":
            r.seed(42)
            put("reseed_raw", bit.random_raw(4))
            put("reseed_legacy", r.random_sample(3))
        else:
            fails("unified_random_" + name + "_seed", lambda: r.seed(42))
            put("after_rejected_seed", bit.random_raw(4))

    rng = np.random.default_rng(118)
    matrices = {"jordan": np.array([[2+3j, 1, 0], [0, 2+3j, 1], [0, 0, 2+3j]]),
                "repeated": np.diag([1+2j, 1+2j, -3+1j, -3+1j]),
                "conjugate": np.array([[2+1j, 4], [1, 2-1j]]),
                "nonnormal": np.array([[1+1j, 1000, 10], [0, 2-2j, -200], [0, 0, -2+3j]])}
    for n in (2, 3, 4, 6, 8, 12, 16, 24, 32, 48):
        matrices[f"random{n}"] = rng.normal(size=(n,n)) + 1j*rng.normal(size=(n,n))
    d=np.geomspace(1,1e3,24)
    matrices["graded24"] = matrices["random24"] * d[None,:] / d[:,None]
    q,_=np.linalg.qr(matrices["random16"])
    matrices["repeated16"] = (q*np.repeat([1+2j,-3+1j,4-2j,-1-3j],4))@q.conj().T
    matrices["small"] = matrices["random4"] * 1e-200
    matrices["large"] = matrices["random4"] * 1e200
    matrices["batch32"] = np.stack([matrices["random3"], matrices["jordan"]]).astype(np.complex64)
    for name, a in matrices.items():
        record("unified_eigen_" + name + "_input", a)
        record("unified_eigen_" + name + "_values", np.linalg.eig(a).eigenvalues)
        record("unified_eigen_" + name + "_only", np.linalg.eigvals(a))
    fails("unified_eigen_nan", lambda: np.linalg.eig(np.array([[complex(np.nan, 0)]])))
