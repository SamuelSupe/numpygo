"""NumPy dtype classes, descriptor-level casting/promotion and records."""
import io
import pathlib
import warnings
import numpy as np


def populate_descriptors(record, fails):
    def text(name, v):
        record(name,np.frombuffer(v.encode("utf-8"),dtype=np.uint8))
    def raw(name, v):
        record(name,np.frombuffer(v,dtype=np.uint8))
    def meta(name,d):
        text(name+"_meta","|".join([type(d).__name__,d.name,d.str,d.kind,d.char,d.byteorder]))
        record(name+"_sizes",np.array([d.itemsize,d.alignment,d.num,int(d.hasobject)],dtype=np.int64))
    def array(name,a,layout=False):
        a=np.asarray(a)
        if a.dtype.char=="T":
            words=[v.encode("utf-8") for v in a.flat]
            raw(name,b"".join(words))
            record(name+"_offsets",np.cumsum([0]+[len(v) for v in words],dtype=np.int64))
            text(name+"_dtype",str(a.dtype))
        else:
            raw(name,a.tobytes(order="C"));text(name+"_dtype",a.dtype.str)
        record(name+"_shape",np.array(a.shape,dtype=np.int64))
        if layout:record(name+"_strides",np.array(a.strides,dtype=np.int64))
    for name in np.dtypes.__all__:
        if name=="register_dlpack_dtype":continue
        k=getattr(np.dtypes,name)
        if name=="BytesDType":d=k(3)
        elif name=="StrDType":d=k(3)
        elif name=="VoidDType":d=np.dtype("V3")
        elif name=="DateTime64DType":d=np.dtype("M8[3h]")
        elif name=="TimeDelta64DType":d=np.dtype("m8[2D]")
        else:d=k()
        meta("dtype_class_"+name,d)
        a=np.zeros(2,dtype=d)
        if d.kind=="O":record("dtype_class_"+name+"_zeros",np.asarray(a,dtype=np.int64))
        elif d.kind in "Mm":record("dtype_class_"+name+"_zeros",a.view("i8"))
        elif d.kind in "SUV" or d.char=="T":array("dtype_class_"+name+"_zeros",a)
        else:record("dtype_class_"+name+"_zeros",a)
    for i,s in enumerate(["S0","U0","V0",">i4",">U3",">M8[3h]",">m8[D]","M8","m8"]):
        meta("dtype_extra_"+str(i),np.dtype(s))
    meta("dtype_nullable",np.dtypes.StringDType(na_object=None,coerce=False))
    meta("dtype_record",np.dtype([(("title","a"),">i2"),("b","M8[D]")],align=True))

    specs=["bool","i1","i2","i4","i8","q","u1","u2","u4","u8","Q","f2","f4","f8","g","c8","c16","G",
           "S0","S3","S12","U0","U3","U12","V0","V3","V8","M8","M8[D]","M8[3h]","M8[Y]","M8[3D]","M8[W]",
           "m8","m8[D]","m8[3h]","m8[Y]","m8[M]","m8[2D]","m8[W]","m8[2h]","O","T",">i2",">f8",">U3",">M8[D]",">m8[3h]"]
    types=[np.dtype(s) for s in specs]
    rules=["no","equiv","safe","same_kind","unsafe"]
    codes=[];promote=np.empty((len(types),len(types)),dtype=np.int64)
    cast=np.empty((len(types),len(types),len(rules)),dtype=bool)
    for i,a in enumerate(types):
        for j,b in enumerate(types):
            try:
                x=np.promote_types(a,b)
                key=x.str+"|"+x.char
                if key not in codes:codes.append(key)
                promote[i,j]=codes.index(key)
            except np.exceptions.DTypePromotionError:promote[i,j]=-1
            except TypeError:promote[i,j]=-2
            cast[i,j]=[np.can_cast(a,b,casting=r) for r in rules]
    text("dtype_matrix_input","\n".join(specs));text("dtype_matrix_codes","\n".join(codes))
    record("dtype_matrix_promote",promote);record("dtype_matrix_cast",cast)
    # Descriptor-level record casts, including positional field names.
    structs=[np.dtype([("a","i2")]),np.dtype([("b","i4")]),np.dtype([("a",">i2")]),np.dtype([("a","i2",(2,))]),np.dtype("V2"),np.dtype("i2")]
    record("dtype_struct_cast",np.array([[[np.can_cast(a,b,casting=r) for r in rules] for b in structs] for a in structs],dtype=bool))
    t=[np.dtypes.StringDType(),np.dtypes.StringDType(coerce=False),np.dtypes.StringDType(na_object=None),np.dtypes.StringDType(na_object=np.nan),np.dtypes.StringDType(na_object="NA")]
    record("dtype_string_cast",np.array([[[np.can_cast(a,b,casting=r) for r in rules] for b in t] for a in t],dtype=bool))
    for k,s in enumerate(["M8[D]","M8[3h]","m8[D]","m8[3h]","m8[Y]"]):
        a=np.array([0,1,-2,np.iinfo(np.int64).min],dtype=s)
        p="temporal_cast_"+str(k)+"_"
        for n,d in [("float","f8"),("complex","c16"),("int","i8")]:record(p+n,a.astype(d))
        for n,d in [("string","T"),("unicode","U"),("bytes","S32")]:array(p+n,a.astype(d))
        for n,d in [("date_h","M8[h]"),("delta_h","m8[h]")]:
            record(p+n,a.astype(d).view("i8"))
    for k,s in enumerate(["f8","c16"]):
        a=np.array([1.8,-2.8,np.nan,np.inf],dtype=s)
        for n,d in [("date","M8[D]"),("delta","m8[3h]")]:
            record("temporal_from_"+str(k)+"_"+n,a.astype(d).view("i8"))
    a=np.array([1,-7,256],dtype=">i2")
    out=a.astype(">i4")
    raw("dtype_cast_endian",out.tobytes())
    record("dtype_cast_endian_values",out.astype("i4"))
    array("dtype_cast_void",a.astype("V5"))
    array("dtype_cast_void_auto",a.astype("V"))
    dates=np.array([0,1,-1],dtype="M8[D]")
    delta=np.array([2,3,-4],dtype="m8[3h]")
    record("temporal_mixed_concat",np.concatenate([delta,np.array([5,-2],dtype="i2")]).view("i8"))
    record("temporal_mixed_where",np.where([True,False,True],delta,np.array([7,8,9],dtype="i2")).view("i8"))
    fails("temporal_mixed_concat_date",lambda:np.concatenate([dates,delta]))
    a=np.array([-2,1,0],dtype="M8[D]")[::-1]
    b=np.array([2,3,4],dtype="m8[h]")
    mask=np.array([True,False,True])
    for unit in ["h","D","m"]:
        for rule in rules:
            name="temporal_ufunc_add_"+unit+"_"+rule
            out=np.array([99,99,99],dtype="M8["+unit+"]")[::-1]
            f=lambda:np.add(a,b,out=out,where=mask,casting=rule)
            if rule in ("no","equiv") or unit=="D" and rule=="safe":fails(name,f)
            else:
                assert f() is out
                record(name,out.view("i8"))
    a=np.array([-2,np.iinfo("i8").min,3],dtype="m8[3h]")
    for n in ("negative","absolute","isnan","isfinite","isinf","isnat"):
        f=getattr(np,n)
        out=np.array([9,9,9],dtype="m8[h]")[::-1] if n in ("negative","absolute") else np.array([True,True,True])[::-1]
        f(a,out=out,where=mask)
        record("temporal_ufunc_"+n,out.view("i8") if n in ("negative","absolute") else out)

    # Structured temporal fields: packed odd offsets, subarray and nested
    # date descriptors, both endiannesses, views and NPY interchange.
    dt=np.dtype([("tag","i1"),("when",">M8[D]"),("delta","<m8[3h]",(2,)),("sub",[("stamp",">M8[ns]")])])
    a=np.zeros((2,2),dtype=dt)
    a["tag"][:]=[[1,2],[3,4]]
    a["when"].view(">i8")[:]=[[0,2],[-4,np.iinfo(np.int64).min]]
    a["delta"].view("<i8")[:]=np.arange(8).reshape(2,2,2)-3
    a["sub"]["stamp"].view(">i8")[:]=[[1234,5678],[-9000,0]]
    a["when"][:,::-1][0,0]=np.datetime64("1970-01-04","D")
    a["delta"][1,0]=np.array(["NaT","2"],dtype="m8[3h]")
    record("rectime_when",a["when"].view(">i8").astype("i8"))
    record("rectime_delta",a["delta"].view("<i8"))
    record("rectime_stamp",a["sub"]["stamp"].view(">i8").astype("i8"))
    record("rectime_stride",np.array(a["when"].strides,dtype=np.int64))
    record("rectime_delta_stride",np.array(a["delta"].strides,dtype=np.int64))
    raw("rectime_bytes",a.tobytes())
    root=pathlib.Path(__file__).resolve().parents[1]/"testdata"/"numpy"
    np.save(root/"structured_temporal.npy",a,allow_pickle=False)
    raw("rectime_file",(root/"structured_temporal.npy").read_bytes())
    x=np.array([("1970-01-02",2),("1970-01-03",-1)],dtype=[("when",">M8[D]"),("delta","m8[3h]")])
    y=np.array([("1970-01-02T00",6),("1970-01-03T00",-3)],dtype=[("when","M8[3h]"),("delta","m8[h]")])
    for n,z in [("concat",np.concatenate([x,y])),("where",np.where([True,False],x,y))]:
        raw("rectime_"+n,z.tobytes());record("rectime_"+n+"_when",z["when"].view("i8"));record("rectime_"+n+"_delta",z["delta"].view("i8"))
    record("rectime_equal",x==y)
    fails("rectime_bad_promotion",lambda:np.promote_types(np.dtype([("x","m8[Y]")]),np.dtype([("x","m8[D]")])))

    # Record-array API constructors and buffer ownership.
    def rec_layout(n,a):
        a=np.asarray(a);raw(n,a.tobytes(order="C"))
        record(n+"_shape",np.array(a.shape,dtype=np.int64))
        record(n+"_strides",np.array(a.strides,dtype=np.int64))
        record(n+"_layout",np.array([a.dtype.itemsize,a.dtype.alignment,int(a.dtype.isalignedstruct)]+[a.dtype.fields[name][1] for name in a.dtype.names],dtype=np.int64))
        text(n+"_names","|".join(a.dtype.names))
        text(n+"_descr","|".join(a.dtype.fields[name][0].base.str+":"+str(a.dtype.fields[name][0].shape) for name in a.dtype.names))
    parser=np.rec.format_parser(["i2","U3","(2,)f8","M8[D]"],[" id ","label","vec","date"],["ident","标签","vector","time"],aligned=True,byteorder=">")
    rec_layout("rec_parser",np.zeros(2,dtype=parser.dtype))
    fails("rec_parser_duplicate",lambda:np.rec.format_parser(["i2","i2"],["id","id"],None))
    a=np.rec.fromarrays([np.arange(6,dtype="i2").reshape(2,3),np.array(["a","bc","中文","d","e","f"],dtype="U3").reshape(2,3),np.arange(6,dtype="i8").astype("M8[D]").reshape(2,3)],names=["id","label","date"])
    rec_layout("rec_fromarrays",a)
    s=np.rec.fromarrays([np.arange(12,dtype="f8").reshape(2,3,2),np.arange(6,dtype="i2").reshape(2,3)],formats=["2f8","i2"],names=["vec","id"])
    rec_layout("rec_subarrays",s)
    fails("rec_bad_arrays",lambda:np.rec.fromarrays([np.arange(2),np.arange(3)]))
    r=np.rec.fromrecords([(1,"中",1.25),(-7,"ab",-3.5)],names=["id","tag","value"])
    rec_layout("rec_fromrecords",r)
    raw("rec_record",r[0].tobytes())
    a=np.rec.recarray((2,3),formats=["i2",">f8"],names=["id","value"],order="F")
    a["id"][:]=np.arange(6,dtype="i2").reshape(2,3)
    a["value"][:]=np.arange(6).reshape(2,3)/4
    rec_layout("rec_recarray",a)
    buf=bytearray(range(40))
    b=np.rec.recarray((2,2),dtype=[("x","<i2"),("y",">i2")],buf=buf,offset=9,strides=(11,-4))
    b["x"][1,0]=-55
    rec_layout("rec_strided",b);raw("rec_strided_buffer",bytes(buf))
    fails("rec_bad_strides",lambda:np.rec.recarray((2,2),dtype=[("x","i2")],buf=bytearray(8),offset=0,strides=(8,8)))
    data=bytes(range(23))
    for n,fun in [("fromstring",lambda:np.rec.fromstring(data,formats="i2,i2",offset=4)),("fromfile",lambda:np.rec.fromfile(io.BytesIO(data),formats="i2,i2",offset=4))]:
        v=fun();rec_layout("rec_"+n,v);record("rec_"+n+"_writable",np.array(v.flags.writeable))
    fails("rec_buffer_short",lambda:np.rec.fromstring(bytes(7),formats="i2,i2",shape=2))
    fails("rec_file_short",lambda:np.rec.fromfile(io.BytesIO(bytes(7)),formats="i2,i2",shape=2))
    for copy in [False,True]:
        b=np.rec.array(r,copy=copy)
        rec_layout("rec_array_"+str(int(copy)),b)
        record("rec_share_"+str(int(copy)),np.array(np.shares_memory(r,b)))
    text("rec_duplicate","|".join(np.rec.find_duplicate(["z","a","b","a","z","b"])))

    # Registration is tested last: NumPy mutates a process-global table.
    reg=np.dtypes.register_dlpack_dtype
    reg((42,32),np.dtype("i4"));reg((42,32),np.dtype("i4"))
    record("dtype_dlpack_registered",np.array(True))
    fails("dtype_dlpack_conflict",lambda:reg((42,32),np.dtype("f4")))
    fails("dtype_dlpack_export_conflict",lambda:reg((43,32),np.dtype("f4")))
    fails("dtype_dlpack_bad_code",lambda:reg((-1,32),np.dtype("i4")))
    fails("dtype_dlpack_bad_width",lambda:reg((44,16),np.dtype("i4")))
