"""Remaining runtime-equivalent metadata, managed fields and layout boundaries."""
import hashlib
import json
import sys
import types
import warnings
from pathlib import Path
import numpy as np


def populate_final_closure(record, fails):
    def r(name, value):
        record("closure_" + name, value)

    def text(name, value):
        r(name, np.frombuffer(value.encode(), dtype="u1"))

    def dump(name, value):
        text(name, json.dumps(value, ensure_ascii=False, separators=(",", ":")))

    def raw(name, a):
        r(name+"_bytes", np.frombuffer(a.tobytes(),dtype="u1"))
        r(name+"_shape",np.array(a.shape,dtype="i8"))
        r(name+"_strides",np.array(a.strides,dtype="i8"))
        text(name+"_dtype",a.dtype.str)

    def subraw(name,x,d,order):
        v=x.astype(d,order=order)
        if order=="F" or order in ("K","A") and x.flags.f_contiguous and not x.flags.c_contiguous:
            # The expanded F-order astype path in 2.5.0 overlaps
            # subarray writes and can leave unwritten values. Never
            # capture its unspecified bytes. Compare actual metadata
            # and the intended scalar replication using NumPy broadcast.
            leaf=d
            trailing=()
            while leaf.subdtype:
                trailing+=leaf.shape
                leaf=leaf.base
            logical=np.broadcast_to(x.reshape(x.shape+(1,)*len(trailing)),v.shape).astype(leaf)
            r(name+"_logical_bytes",np.frombuffer(logical.tobytes(),dtype="u1"))
            r(name+"_shape",np.array(v.shape,dtype="i8"))
            r(name+"_strides",np.array(v.strides,dtype="i8"))
            text(name+"_dtype",v.dtype.str)
        else:
            raw(name,v)

    f=lambda:None
    doc="  direct\n\tindent"
    np.lib.add_docstring(f,doc)
    np.lib.add_docstring(f,doc)
    text("doc_direct",f.__doc__)
    fails("closure_doc_conflict",lambda:np.lib.add_docstring(f,"different"))
    fails("closure_doc_unsupported",lambda:np.lib.add_docstring(type("Locked",(),{"__slots__":()})(),"x"))
    mod=types.ModuleType("_numpygo_doc_fixture")
    mod.fn=lambda:None
    mod.fn.first=lambda:None
    mod.fn.second=lambda:None
    mod.fn.last=lambda:None
    sys.modules[mod.__name__]=mod
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        np.lib.add_newdoc(mod.__name__,"fn","\n    Summary\n      detail\n    final\n")
        np.lib.add_newdoc(mod.__name__,"fn",("first","\tTitle\n\t  detail"))
        np.lib.add_newdoc(mod.__name__,"fn",[("first","ignored conflict"),("second","  Second")])
        np.lib.add_newdoc(mod.__name__,"fn",("last","Last"),warn_on_python=False)
        r("doc_warn_count",np.array(len(w),dtype="i8"))
    text("doc_main",mod.fn.__doc__)
    text("doc_first",mod.fn.first.__doc__)
    text("doc_second",mod.fn.second.__doc__)
    text("doc_last",mod.fn.last.__doc__)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fails("closure_doc_missing_member",lambda:np.lib.add_newdoc(mod.__name__,"fn",("missing","x")))
        fails("closure_doc_missing_target",lambda:np.lib.add_newdoc(mod.__name__,"missing","x"))
    del sys.modules[mod.__name__]

    def describe(d):
        if d.subdtype is not None:
            return [describe(d.base),list(d.shape),d.itemsize,d.flags,d.hasobject]
        return [d.str,d.itemsize,d.flags,d.hasobject]
    d=np.dtype((np.dtype((">i2",(2,))),(3,)))
    dump("nested_dtype",describe(d))
    meta=np.dtype((np.dtype("i2",metadata={"unit":"ms"}),(2,)),metadata={"name":"x"})
    cleaned=np.lib.format.drop_metadata(meta)
    dump("drop_metadata", [meta.metadata is not None,meta.base.metadata is not None,cleaned.metadata is None,cleaned.base.metadata is None,describe(cleaned)])
    raw("nested_zero",np.zeros((2,3),dtype=d))
    a=np.arange(6,dtype="i2").reshape(2,3).T
    for o in "KACF":
        subraw("nested_cast_"+o,a,d,o)
    e=np.dtype((np.dtype(("f4",(2,))),(3,)))
    promoted=np.promote_types(d,e)
    dump("nested_promote",describe(promoted))
    # The 2.5.0 promotion may retain stale outer item sizes after
    # widening a nested base. Compare its shape/base independently of
    # that inconsistent allocation metadata; Go computes correct widths.
    def semantic(d):
        if d.subdtype is not None:
            return [semantic(d.base),list(d.shape),d.flags,d.hasobject]
        return [d.str,d.itemsize,d.flags,d.hasobject]
    dump("nested_promote_semantic",semantic(promoted))
    r("nested_cast_predicate",np.array([np.can_cast(d,e,casting=c) for c in ("no","equiv","safe","same_kind","unsafe")],dtype=bool))
    rd=np.dtype([("x",d)])
    flat=np.dtype([("x",">i2",(3,2))])
    dump("nested_record_descr",rd.descr)
    dump("nested_record_swapped",rd.newbyteorder().descr)
    dump("nested_record_field",describe(rd.fields["x"][0]))
    r("nested_record_predicate",np.array([np.can_cast(rd,flat,casting=c) for c in ("no","equiv","safe","same_kind","unsafe")],dtype=bool))
    rows=np.zeros(2,dtype=rd)
    rows["x"]=np.arange(12).reshape(2,3,2)
    r("nested_record_values",rows["x"].astype("i2"))
    np.save(Path(__file__).resolve().parents[1]/"testdata/numpy/closure_nested.npy",rows,allow_pickle=False)
    od=np.dtype(("O",(2,)))
    dump("object_subdtype",describe(od))
    r("object_sub_zero",np.zeros((2,3),dtype=od).astype("i8"))
    r("object_sub_cast",np.arange(3,dtype="i2").astype(od).astype("i8"))
    td=np.dtype((np.dtypes.StringDType(),(2,)))
    dump("string_sub_dtype",describe(td))
    dump("string_sub_zero",np.zeros(2,dtype=td).tolist())
    dump("string_sub_cast",np.arange(3,dtype="i2").astype(td).tolist())
    fails("closure_string_field",lambda:np.dtype([("x",np.dtypes.StringDType())]))

    a=np.arange(24,dtype=">i2").reshape(2,3,4)
    for n,x in [("c",a),("f",np.asfortranarray(a)),("perm",a.transpose(2,0,1)),
                ("rev",a[::-1,:,::2])]:
        for o in "KACF":
            raw("copy_"+n+"_"+o,x.copy(order=o))
            raw("cast_"+n+"_"+o,x.astype(">i4",order=o))
            subraw("subcast_"+n+"_"+o,x,np.dtype((">i4",(2,3))),o)
            y=x.astype(x.dtype,order=o,copy=False)
            r("nocopy_"+n+"_"+o,np.array(y is x,dtype=bool))
    raw("copy_default",np.copy(a.transpose(2,0,1)))

    inner=np.dtype([("value","f4"),("obj","O")])
    dt=np.dtype([("id",">i2"),("payload","O"),("tags","O",(2,)),("inner",inner)])
    dump("managed_descr",dt.descr)
    r("managed_meta",np.array([dt.itemsize,dt.alignment,dt.flags,dt.hasobject,*[dt.fields[n][1] for n in dt.names]],dtype="i8"))
    a=np.zeros((2,3),dtype=dt)
    dump("managed_zero_payload",a["payload"].tolist())
    dump("managed_zero_tags",a["tags"].tolist())
    a["id"]=np.arange(6).reshape(2,3)
    shared=["seed"]
    a["payload"][0,0]=shared
    a["payload"][0,1]="hello"
    a["payload"][0,2]=None
    a["payload"][1,0]=7
    a["payload"][1,1]="猫"
    a["payload"][1,2]=False
    a["tags"]=np.array([["L","R"]],dtype=object)
    a["inner"]["value"]=np.arange(6).reshape(2,3)+.25
    a["inner"]["obj"]="inner"
    a["inner"]["obj"][1,2]=None
    def check_managed(name,x):
        r(name+"_id",x["id"].astype("i2"))
        dump(name+"_payload",x["payload"].tolist())
        dump(name+"_tags",x["tags"].tolist())
        r(name+"_inner_value",x["inner"]["value"])
        dump(name+"_inner_obj",x["inner"]["obj"].tolist())
        r(name+"_strides",np.array(x.strides,dtype="i8"))
    check_managed("managed_initial",a)
    r("managed_field_strides",np.array(a["tags"].strides,dtype="i8"))
    v=a.T[::-1]
    v["payload"][0,1]="view"
    v["tags"][1,0,1]="changed"
    check_managed("managed_view_write",a)
    b=a.copy(order="F")
    b["payload"][0,0].append("shared")
    check_managed("managed_copy_shared",a)
    b["payload"][0,0]="replacement"
    check_managed("managed_copy_detached",b)
    check_managed("managed_fancy",a[:,[2,0]])
    c=a[["id","payload"]]
    c["payload"][0,2]="projection"
    dump("managed_projection",a["payload"].tolist())
    r("managed_projection_strides",np.array(c.strides,dtype="i8"))
    c=np.concatenate([a[:1],a[1:]],axis=0)
    check_managed("managed_concat",c)
    r("managed_equal",np.asarray(a==c,dtype=bool))
    c["payload"][1,2]="different"
    r("managed_neq",np.asarray(a!=c,dtype=bool))
    target=np.dtype([("code","i4"),("data","O"),("names","O",(2,)),("node",[("v","f8"),("o","O")])])
    x=a[::-1].astype(target)
    r("managed_cast_code",x["code"])
    dump("managed_cast_data",x["data"].tolist())
    dump("managed_cast_names",x["names"].tolist())
    r("managed_cast_value",x["node"]["v"])
    dump("managed_cast_object",x["node"]["o"].tolist())
    p=np.array([(2,),(7,)],dtype=[("x","i2")]).astype([("y","O")])
    r("managed_promote_numeric",p["y"].astype("i8"))
    r("managed_cast_numeric",p.astype([("q","i4")])["q"])
    fails("closure_managed_overlap",lambda:np.dtype({"names":["o","x"],"formats":["O","i4"],"offsets":[0,2],"itemsize":8}))
    import io
    fails("closure_managed_pickle",lambda:np.save(io.BytesIO(),a,allow_pickle=False))

    # Compact exhaustive binary16 checks. Canonicalize only NaN payloads,
    # whose binary representation is not portable across scalar kernels.
    bits=np.arange(65536,dtype="u2")
    half=bits.view("f2")
    def digest16(name,x):
        b=np.asarray(x,dtype="f2").view("u2").copy()
        b[(b&0x7c00==0x7c00)&(b&0x03ff!=0)]=0x7e00
        r(name,np.frombuffer(hashlib.sha256(b.astype("<u2").tobytes()).digest(),dtype="u1"))
    def digest64(name,x):
        b=np.asarray(x,dtype="f8").view("u8").copy()
        b[np.isnan(x)]=0x7ff8000000000000
        r(name,np.frombuffer(hashlib.sha256(b.astype("<u8").tobytes()).digest(),dtype="u1"))
    with np.errstate(all="ignore"):
        digest64("half_decode",half.astype("f8"))
        digest16("half_roundtrip",half.astype("f8").astype("f2"))
        digest16("half_next_pos",np.nextafter(half,np.float16(np.inf)))
        digest16("half_next_neg",np.nextafter(half,np.float16(-np.inf)))
        digest16("half_next_zero",np.nextafter(half,np.float16(0)))
        digest16("half_spacing",np.spacing(half))
        finite=np.arange(0x7c00,dtype="u2").view("f2").astype("f8")
        midpoint=np.concatenate([(finite[:-1]+finite[1:])/2,[65520.]])
        around=np.stack((np.nextafter(midpoint,-np.inf),midpoint,np.nextafter(midpoint,np.inf)),axis=1).ravel()
        digest16("half_round_ties",np.concatenate((around,-around)).astype("f2"))
