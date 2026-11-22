"""Mixed advanced indexing, result storage, and parametric dtype casts."""
import numpy as np


def populate_index_descriptor(record, fails):
    def r(name, value):
        record("ixdeep_" + name, value)

    def index(name, a, key):
        result = a[key]
        r(name, result)
        b = result.base if isinstance(result.base, np.ndarray) else None
        r(name + "_base", np.array([b is not None, *(b.shape if b is not None else ())], dtype="i8"))

    a = np.arange(120,dtype="i4").reshape(2,3,4,5)
    i = np.array([[0],[1]],dtype="i8")
    j = np.array([[0,1,0]],dtype="i8")
    f = np.asfortranarray([[0,1,0],[1,0,1]],dtype="i8")
    for name, src in [("c",a),("f",np.asfortranarray(a)),("perm",a.transpose(1,0,3,2)),
                      ("reverse",a[::-1,::-1,::2,::-1])]:
        for label,key in [
            ("first",(i,)),("middle",(slice(None),i)),("last",(Ellipsis,i)),
            ("adjacent",(slice(None),i,j)),("separated",(slice(None),i,slice(None),j)),
            ("newaxis",(slice(None),i,None,j)),("prefix_new",(None,i)),
            ("middle_f",(slice(None),f)),("single_f",(f,)),("zero_idx",(np.array(1),)),
            ("middle_zero",(slice(None),np.array(1))),("bool0",(np.array(True),)),
            ("boolfalse",(np.array(False),)),("suffix_bool",(slice(None),np.array(True))),
            ("slice_step",(i,slice(None,None,-2),slice(None,None,-1))),
            ("mask",(np.arange(src.shape[0])%2==0,)),
            ("mask2",(np.arange(src.shape[0]*src.shape[1]).reshape(src.shape[:2])%3==0,)),
            ("empty_adv",(np.array([],dtype="i8").reshape(0,1),j)),
        ]:
            index(name+"_"+label,src,key)
    a=np.arange(12,dtype="i4").reshape(3,4)
    for label,key in [
        ("ff",(f,np.asfortranarray(1-f))),("fc",(f,np.ascontiguousarray(1-f))),
        ("fb",(f,np.array([0,1,2]))),("single0",(np.array(1),np.array(2))),
        ("ellipsis0",(f,Ellipsis,np.asfortranarray(1-f))),
        ("scalar_and_f",(np.array(1),f)),
        ("empty_oob",(np.array([99]),np.array([],dtype="i8"))),
        ("maskempty",(np.empty((3,0),dtype=bool),)),
    ]:
        index(label,a,key)
    for label, src in [("singleton",np.asfortranarray(np.arange(6,dtype="i4").reshape(2,1,3))),
                       ("broadcast",np.broadcast_to(np.arange(3,dtype="i4"),(2,1,3)))]:
        index(label+"_bool0",src,(np.array(True),))
        index(label+"_prefix",src,(None,np.array([0,1])))
        index(label+"_suffix",src,(np.array([0,1]),None))
    for name,ix in [("only",f),("gap",f[:,::-1]),("singleton",np.asfortranarray(np.zeros((2,1,3),dtype="i8")))]:
        src=np.arange(4,dtype="i4")
        index("index_"+name,src,(ix,))
        index("index_"+name+"_newaxis",src,(ix,None))
    fails("ixdeep_oob_empty_slice",lambda:a[np.array([99]),:0])
    fails("ixdeep_oob_scalar",lambda:a[99,np.array([],dtype="i8")])
    fails("ixdeep_bad_empty_mask",lambda:a[np.empty((0,5),dtype=bool)])
    index("uint_wrap",a,(np.array([2**64-1],dtype="u8"),))
    fails("ixdeep_uint_oob",lambda:a[np.array([2**63],dtype="u8")])
    fails("ixdeep_bad_index_type",lambda:a[np.array([1.0])])
    fails("ixdeep_two_ellipsis",lambda:a[...,np.array([0]),...])

    def raw(name, result):
        r(name+"_bytes",np.frombuffer(result.tobytes(),dtype="u1"))
        r(name+"_strides",np.array(result.strides,dtype="i8"))
        r(name+"_dtype",np.frombuffer(result.dtype.str.encode(),dtype="u1"))
    a=np.arange(12,dtype=">i4").reshape(3,4)
    y=a[:,[3,1,3]]
    raw("endian",y);r("endian_values",y.astype("i4"))
    a=np.arange(6,dtype=">i8").view(">M8[3h]").reshape(2,3)
    y=a[:,[2,0]]
    raw("temporal",y);r("temporal_ticks",y.view(">i8").astype("i8"))
    a=np.array([["a","b","😀"],["d","é","f"]],dtype=">U2")
    raw("unicode",a[:,[2,0]])
    a=np.zeros((2,3),dtype=[("code",">i2"),("value","<f4")])
    a["code"]=np.arange(6).reshape(2,3)
    a["value"]=np.arange(6).reshape(2,3)+.25
    raw("record",a[:,[2,0]])
    p=np.array([0x7fa12345,0x80000000,0x7fcabcde,0x3f800000],dtype="u4").view("f4")
    r("nan_payload",p[[2,0,1]].view("u4"))

    a=np.arange(24,dtype="i8").reshape(2,3,4)
    ix=np.array([[2,0],[2,2]])
    a[:,ix,1]=np.array([[[100,101],[102,103]]])
    r("assign_repeat",a)
    a=np.arange(6,dtype="i8")
    a[[2,1,3,0]]=a[:4]
    r("assign_alias",a)
    a=np.arange(24,dtype="i8").reshape(2,3,4).transpose(1,0,2)
    np.add.at(a,(ix,slice(None),2),np.arange(8,dtype="i8").reshape(2,2,2)+1)
    r("at_repeat",a)
    a=np.arange(12,dtype="i8").reshape(3,4)
    fails("ixdeep_assign_bad_rhs",lambda:a.__setitem__((ix,slice(None)),np.ones((3,3))))
    r("assign_unchanged",a)

    def d(name,value):
        record("subdeep_"+name,value)
    base = [np.dtype(s) for s in ("i2",">i2","i8","f4","f8","S2","U3","V0","V4","V8","O")]
    sub = [np.dtype((s,sh)) for s,sh in
           [("i2",(2,)),(">i2",(2,)),("f4",(2,)),("i8",(2,)),("S2",(2,)),
            ("U3",(2,)),("V4",(2,)),("i2",(1,2)),("i2",(0,)),("i2",(3,))]]
    types = base + sub
    rules = ("no","equiv","safe","same_kind","unsafe")
    flags, promoted, result_types = [], [], []
    def desc(x):
        if x.subdtype:
            return desc(x.base)+"("+",".join(map(str,x.shape))+")"
        return x.str
    for x in types:
        for y in types:
            if x.subdtype is None and y.subdtype is None:
                continue
            flags.extend(np.can_cast(x,y,casting=rule) for rule in rules)
            try: promoted.append(desc(np.promote_types(x,y)))
            except Exception: promoted.append("!")
            try: result_types.append(desc(np.result_type(x,y)))
            except Exception: result_types.append("!")
    d("can_cast",np.array(flags,dtype=bool))
    d("promote",np.frombuffer("\n".join(promoted).encode(),dtype="u1"))
    d("result_type",np.frombuffer("\n".join(result_types).encode(),dtype="u1"))

    for i, a in enumerate([np.array(7,dtype="i2"),np.arange(3,dtype="i2"),
                            np.arange(6,dtype="i2").reshape(2,3)[:,::-1]]):
        for j, (dt,shape) in enumerate([("i4",(2,)),("f4",(2,3)),(">i4",(1,2)),("i8",(1,)),("i2",(0,))]):
            v=a.astype(np.dtype((dt,shape)),casting="safe")
            name=f"cast_{i}_{j}"
            d(name+"_bytes",np.frombuffer(v.tobytes(),dtype="u1"))
            d(name+"_shape",np.array(v.shape,dtype="i8"))
            d(name+"_dtype",np.frombuffer(v.dtype.str.encode(),dtype="u1"))
            if v.dtype.kind!="V":
                d(name+"_values",v.astype(v.dtype.newbyteorder("=")))
    a=np.array([1.8,-2.8],dtype="f8")
    d("unsafe_cast",a.astype(np.dtype(("i2",(2,)))))
    fails("subdeep_safe_reject",lambda:a.astype(np.dtype(("i2",(2,))),casting="safe"))

    def rec(name,dt,shape=()):
        return np.dtype([(name,dt,shape)])
    structs = [rec("a","i2"),rec("b","i4"),rec("a",">i2"),rec("a","i2",(2,)),
               rec("a","i4",(3,)),rec("a","i4",(1,2)),rec("a","i2",(0,)),
               np.dtype([("a","i2"),("b","f4")]),np.dtype([("x","i4"),("y","f8")]),
               np.dtype("i2"),np.dtype("V0"),np.dtype("V4"),np.dtype("O")]
    d("record_cast",np.array([[[np.can_cast(x,y,casting=rule) for rule in rules] for y in structs] for x in structs],dtype=bool))
    for i,(sa,sb) in enumerate([((2,),(3,)),((3,),(2,)),((1,),(3,)),((2,1),(2,3)),
                         ((2,),(2,3)),((3,),(2,3)),((2,3),(3,)),((2,3),(3,2)),
                         ((2,3),(1,2,3)),((2,3),(2,1,3)),((2,3),()),((),(2,3)),
                         ((1,),()),((0,),(2,)),((2,),(0,)),((1,2),(2,1))]):
        # NumPy 2.5.0 does not safely execute every zero-width
        # structured destination conversion. Do not run that path.
        if i == 14:
            continue
        a=np.zeros(2,dtype=rec("a",">i2",sa))
        a["a"]=np.arange(a["a"].size,dtype="i2").reshape(a["a"].shape)+1
        name=f"record_shape_{i}"
        try: v=a.astype(rec("b","i4",sb))
        except Exception:
            fails("subdeep_"+name,lambda:a.astype(rec("b","i4",sb)))
        else: d(name,v["b"])
    a=np.zeros(3,dtype=[("a",">i2"),("b","f4")])
    a["a"]=[3,-9,7];a["b"]=[1.5,-2.25,9.0]
    v=a[::-1].astype([("x","i4"),("y",">f8")],casting="safe")
    d("record_pos_x",v["x"]);d("record_pos_y",v["y"].astype("f8"))
    v=np.arange(3,dtype="i2").astype([("x","i4"),("y","f8",(2,))])
    d("scalar_record_x",v["x"]);d("scalar_record_y",v["y"])
    n=np.array([(3,),(-7,)],dtype=[("a",">i2")])
    d("record_scalar",n.astype("f8"))
    plain=a.astype("V",casting="no")
    d("record_auto_void_bytes",np.frombuffer(plain.tobytes(),dtype="u1"))
    d("record_auto_void_fields",np.frombuffer(",".join(plain.dtype.names).encode(),dtype="u1"))
    n=np.zeros(2,dtype=[("a",">i2",(2,))]);n["a"]=[[256,17],[3,-7]]
    for w in (1,2,4,6):
        d(f"record_void_{w}",np.frombuffer(n.astype(f"V{w}").tobytes(),dtype="u1"))
    outer=np.dtype([("point",[("coord",">i2",(2,)),("time",">M8[D]")],(2,)),("tag","S3")])
    target=np.dtype([("node",[("q","f4",(3,)),("ts","M8[h]")],(2,)),("label","U4")])
    n=np.zeros(2,dtype=outer)
    n["point"]["coord"]=np.arange(8).reshape(2,2,2)+1
    n["point"]["time"]=np.arange(4).reshape(2,2).astype("M8[D]")
    n["tag"]=[b"abc",b"xy"]
    v=n[::-1].astype(target)
    d("record_nested_q",v["node"]["q"])
    d("record_nested_time",v["node"]["ts"].view("i8"))
    d("record_nested_label",np.frombuffer(v["label"].tobytes(),dtype="u1"))
    n=np.zeros(3,dtype=[("x","i4"),("y","f8",(2,))])
    n[[0,0,2]]=np.array([2,7,9],dtype="i2")
    d("record_assign_x",n["x"])
    d("record_assign_y",n["y"])
    fails("subdeep_multi_record_scalar",lambda:a.astype("f8"))
    fails("subdeep_bad_field_count",lambda:a.astype([("x","i4")]))
