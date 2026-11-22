"""Fixed-width character/opaque ndarray behavior and mixed NPZ archives."""
import io
import pathlib
import numpy as np


def populate_flexible(record,fails):
    def bad(name,fn):fails("flex_"+name,fn)
    def put(name,a,layout=False):
        a=np.asarray(a)
        record("flex_"+name,np.frombuffer(a.tobytes(order="C"),dtype=np.uint8))
        record("flex_"+name+"_shape",np.array(a.shape,dtype=np.int64))
        record("flex_"+name+"_dtype",np.frombuffer(a.dtype.str.encode("ascii"),dtype=np.uint8))
        if layout:
            record("flex_"+name+"_strides",np.array(a.strides,dtype=np.int64))
    def check(name,a):
        record("flex_"+name,a)
    b=np.array([b"a",b"b\0c",b"\xff",b"",b"012345",b"z"],dtype="S5").reshape(2,3)
    u=np.array(["中🙂","a\0b","","z","12345","é"],dtype=">U4").reshape(2,3)
    for label,a in (("s",b),("u",u)):
        put(label+"_input",a,True)
        put(label+"_reverse",a[::-1,::-1],True)
        put(label+"_transpose",a.T,True)
        put(label+"_reshape",a.T.reshape(6))
        put(label+"_index",a[[1,0],[2,1]])
        put(label+"_take",np.take(a,[2,0],axis=1))
        put(label+"_broadcast",np.broadcast_to(a[:1],(2,3)),True)
        put(label+"_copy",a[::-1,::-1].copy())
        put(label+"_concat",np.concatenate([a,a[::-1]],axis=0))
        put(label+"_where",np.where([[True],[False]],a,a[:,::-1]))
        put(label+"_zeros",np.zeros_like(a))
        put(label+"_ones",np.ones_like(a))
        put(label+"_sort",np.sort(a,axis=1))
        check(label+"_argsort",np.argsort(a,axis=1,kind="stable"))
        check(label+"_argmin",np.argmin(a,axis=1))
        check(label+"_argmax",np.argmax(a,axis=1))
        check(label+"_bool",a.astype(bool))
        check(label+"_any",np.any(a,axis=0))
        check(label+"_all",np.all(a,axis=0))
        check(label+"_count",np.count_nonzero(a))
        coords=np.nonzero(a)
        check(label+"_nonzero0",coords[0]);check(label+"_nonzero1",coords[1])
        vals,idx,inv,cnt=np.unique(a,return_index=True,return_inverse=True,return_counts=True)
        put(label+"_unique",vals);check(label+"_unique_indices",idx);check(label+"_unique_inverse",inv);check(label+"_unique_counts",cnt)
        changed=a.copy()
        changed[:,1:]=changed[:,::-1][:,:2]
        put(label+"_assign",changed)
        put(label+"_byteswap",a.byteswap())
        bad(label+"_readonly",lambda:np.broadcast_to(a[:1],(2,3)).__setitem__((1,0),a[0,1]))
        for name,op in (("min",np.min),("sum",np.sum),("logical_not",np.logical_not)):
            bad(label+"_"+name+"_error",lambda:op(a))
    for label,a,c in (
        ("s",np.array([b"abc",b"",b"a",b"aa",b"z"],dtype="S3"),np.array([b"a",b"z",b"zz"],dtype="S2")),
        ("u",np.array(["abc","","a","aa","中"],dtype="U3"),np.array(["a","中","🙂"],dtype="U2")),
    ):
        put(label+"_set_a",a);put(label+"_set_b",c)
        check(label+"_equal",np.equal(a[:,None],c))
        check(label+"_less",np.less(a[:,None],c))
        put(label+"_add",np.add(a[:,None],c))
        check(label+"_isin",np.isin(a,c))
        put(label+"_union",np.union1d(a,c))
        put(label+"_intersect",np.intersect1d(a,c))
        put(label+"_setdiff",np.setdiff1d(a,c))
        put(label+"_setxor",np.setxor1d(a,c))
        check(label+"_search",np.searchsorted(np.sort(a),c))
        check(label+"_searchright",np.searchsorted(np.sort(a),c,side="right"))
    put("mixed_char",np.concatenate([np.array([b"abc"],"S3"),np.array(["中文"],"U2")]))
    put("mixed_num",np.concatenate([np.array([b"abc"],"S3"),np.array([2**63-1],dtype="i8")]))
    put("mixed_where",np.where([False,True],np.array(["a","b"],"U1"),np.array([-1,22],dtype="i4")))
    nums=np.array([-2**63,2**63-1,0],dtype="i8")
    put("cast_int",nums.astype("S"))
    put("cast_float",np.array([0.,-0.,1.,1e15,1e16,1e-4,1e-5,1.23456789]).astype("U"))
    put("cast_complex",np.array([1+2j,0j,complex(-0.,-0.),complex(float("nan"),-float("inf"))]).astype("U"))
    put("cast_bool",np.array([True,False]).astype("S"))
    check("cast_back",np.array([b"12",b"-3",b" 7"],dtype="S4").astype("i8"))
    put("cast_text",np.array([b"ab",b"xyz"],dtype="S3").astype("U4"))
    pairs=[("S3","S3"),("S3","S2"),("S3","S5"),("S3","U3"),("U3","S3"),(">U3","<U3"),
           ("V3","V3"),("V3","V2"),("i8","S20"),("i8","S21"),("f4","U32"),("bool","S4")]
    check("can_cast",np.array([[np.can_cast(a,b,casting=r) for r in ("no","equiv","safe","same_kind","unsafe")]for a,b in pairs]))
    put("cast_fortran",u.astype("<U4",order="F"),True)
    src=np.array([b"abc",b"a",b"z"],dtype="S3")
    rhs=np.array([b"xy",b"b",b"q"],dtype="S2")
    out=np.array([b"old",b"old",b"old"],dtype="S4")[::-1]
    np.add(src,rhs,where=[True,False,True],out=out)
    put("ufunc_add_out",out,True)
    flags=np.array([False,True,False])[::-1]
    np.equal(src,rhs,where=[True,False,True],out=flags)
    check("ufunc_equal_out",flags)
    bad("ufunc_bad_safe",lambda:np.add(src,rhs,out=np.empty(3,dtype="S4"),casting="safe"))
    bad("bad_encode",lambda:np.array(["中"],dtype="U1").astype("S3"))
    bad("bad_decode",lambda:np.array([b"\xff"],dtype="S1").astype("U1"))
    bad("bad_mixed_equal",lambda:np.equal(np.array(["a"],"U1"),np.array([b"a"],"S1")))
    for label,a in (
        ("s",np.array([b"a",b"a\0b",b"'",b"\xff"],"S4")),
        ("u",np.array(["中🙂","a\0b","'","é"],"U3")),
        ("v",np.array([np.void(b"abc"),np.void(b"\0\1\xff")],"V3")),
    ):
        put("fmt_"+label+"_input",a)
        for name,result in (("array2string",np.array2string(a)),("repr",np.array_repr(a)),("str",np.array_str(a)),("scalar",np.array_str(a[:1].reshape(())))):
            check("fmt_"+label+"_"+name,np.frombuffer(result.encode("utf-8"),dtype=np.uint8))

    v=np.array([np.void(x) for x in (b"\0\0\0",b"\x01\0\x02",b"\xff\0\0",b"\x01\0\x02")],dtype="V3")
    put("void_input",v,True);put("void_sort",np.sort(v));put("void_unique",np.unique(v))
    check("void_bool",v.astype(bool));check("void_equal",v==v[::-1])
    check("void_isin",np.isin(v,v[:2]))
    put("void_concat",np.concatenate([v,v[::-1]]))
    bad("void_bad_width",lambda:np.concatenate([v,np.zeros(1,dtype="V2")]))
    bad("void_bad_order",lambda:np.less(v,v))
    zero=np.zeros((2,3),dtype="V0")
    put("void_zero",zero,True);put("void_zero_unique",np.unique(zero))
    check("void_zero_bool",zero.astype(bool))
    raw=np.arange(12,dtype=np.uint8).reshape(2,6)
    view=raw.view("V3");put("void_view",view,True)
    view[1,0]=np.void(b"\xff\xaa\x00")
    check("void_view_backing",raw)

    dt=np.dtype([("id","i1"),("s","S3"),("u",">U2",(2,)),("v","V2"),("zero","V0")])
    rec=np.zeros((2,2),dt)
    rec["id"]=np.arange(4).reshape(2,2)
    rec["s"]=[[b"abc",b"de"],[b"xy",b""]]
    rec["u"]=[[["中","a"],["b","🙂"]],[["cd","é"],["", "z"]]]
    rec["v"]=np.array([[np.void(b"ab"),np.void(b"cd")],[np.void(b"ef"),np.void(b"gh")]])
    put("field_s",rec["s"],True);put("field_u",rec["u"],True);put("field_v",rec["v"],True);put("field_zero",rec["zero"],True)
    rec.T[::-1]["s"][0,1]=b"12345"
    rec["u"][0,1,1]="中文x"
    rec["v"][1,0]=np.void(b"\xff\0")
    check("field_backing",np.frombuffer(rec.tobytes(),dtype=np.uint8))
    bad("field_readonly",lambda:np.broadcast_to(rec,(2,2,2))["s"].__setitem__((1,0,0),b"x"))
    # A mix of numeric/character scalar fields promotes with the full numeric cast capacity.
    r1=np.array([(1,b"x")],dtype=[("key","i8"),("value","S2")])
    r2=np.array([("abc",7)],dtype=[("key","U3"),("value","i2")])
    p=np.concatenate([r1,r2])
    check("field_mixed_promotion",np.frombuffer(p.tobytes(),dtype=np.uint8))
    check("field_mixed_offsets",np.array([p.dtype.itemsize]+[p.dtype.fields[n][1] for n in p.dtype.names],dtype=np.int64))

    dest=pathlib.Path(__file__).resolve().parents[1]/"testdata/numpy"
    for name,a in (("flex_bytes",b),("flex_unicode",u.T.copy(order="F")),("flex_void",v)):
        buf=io.BytesIO();np.save(buf,a,allow_pickle=False)
        data=buf.getvalue();check(name+"_file",np.frombuffer(data,dtype=np.uint8))
        put(name+"_payload",a,True)
        (dest/(name+".npy")).write_bytes(data)
    members={"record":rec,"words":u.T.copy(order="F"),"raw":v,"numeric":np.arange(6,dtype=np.int64).reshape(2,3),"dates":np.array(["2024-01-01","NaT"],dtype="datetime64[D]")}
    for name,save in (("flex_mixed",np.savez),("flex_mixed_compressed",np.savez_compressed)):
        buf=io.BytesIO();save(buf,**members)
        data=buf.getvalue();check(name+"_file",np.frombuffer(data,dtype=np.uint8))
        (dest/(name+".npz")).write_bytes(data)

    # The external algorithm consumes the exact independently generated
    # NumPy PCG64 tape through its own four callbacks and cached state.
    seed=322
    record("external_rng_tape",np.random.PCG64(seed).random_raw(512))
    bit=np.random.PCG64(seed);g=np.random.Generator(bit);r=np.random.RandomState(bit);r2=np.random.RandomState(bit)
    def rng(name,value):record("external_rng_"+name,value)
    rng("uint32",g.integers(0,123456,3,dtype=np.uint32))
    rng("legacy",r.standard_normal(1))
    state=r.get_state(legacy=False)
    rng("modern",g.random(3));rng("raw",bit.random_raw(5))
    rng("other_legacy",r2.standard_normal(1));rng("cached",r.standard_normal(1))
    rng("integer",r.randint(-15,21,7,dtype=np.int8));rng("normal",g.standard_normal(4))
    rng("other_cache",r2.standard_normal(1))
    r.set_state(state)
    rng("replay_modern",g.random(3));rng("replay_raw",bit.random_raw(5));rng("replay_cache",r.standard_normal(1))
