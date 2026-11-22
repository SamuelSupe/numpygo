"""NumPy 2.5 dtype descriptors, legacy matrix and numeric scalar contracts."""
import json
import warnings
import numpy as np


def populate_numeric_classes(record, fails):
    def put(n, v):
        record("numeric_class_" + n, v)
    def text(n, s):
        put(n, np.frombuffer(s.encode(), dtype=np.uint8))
    def ints(n, v):
        put(n, np.asarray(v, dtype=np.int64))
    def dtype(n, d):
        d = np.dtype(d)
        text(n+"_meta","|".join((d.name,d.str,d.char,d.kind,d.byteorder,d.type.__name__)))
        ints(n+"_flags",[d.itemsize,d.alignment,d.num,d.flags,d.isbuiltin,d.isnative,d.isalignedstruct,d.hasobject,d.ndim])
        ints(n+"_shape",d.shape)
        text(n+"_base",d.base.str)
        text(n+"_descr",json.dumps(d.descr,ensure_ascii=False,separators=(",",":")))
    specs=["?","b","B","i2","u2","i4","u4","i8","u8","q","Q","f2","f4","f8","g","c8","c16","G",
           "S0","S3","U0","U3","V0","V3","O","M8","M8[D]","m8[3h]","T",">i4",">U2"]
    text("dtype_specs","\n".join(specs))
    for i,s in enumerate(specs):
        d=np.dtype(s);dtype("dtype_"+str(i),d)
        if s=="T":
            fails("numeric_class_dtype_string_order",lambda:d.newbyteorder("S"))
        else:
            for j,o in enumerate(["S","<",">","=","|"]):
                dtype("dtype_order_"+str(i)+"_"+str(j),d.newbyteorder(o))
    struct=np.dtype([(("ident","id"),">i2"),("name","U2"),("vec","<f4",(2,)),("nested",[("x",">i4"),("flag","?")])],align=True)
    dtype("dtype_struct",struct)
    text("dtype_names","|".join(struct.names))
    for i,n in enumerate(struct.names):
        d,off,*title=struct.fields[n]
        dtype("dtype_field_"+str(i),d)
        ints("dtype_field_offset_"+str(i),[off])
        text("dtype_field_title_"+str(i),str(title[0]) if title else "")
    dtype("dtype_title",struct["ident"])
    dtype("dtype_last",struct[-1])
    for j,o in enumerate(["S","<",">","=","|"]):
        d=struct.newbyteorder(o);dtype("dtype_struct_order_"+str(j),d)
        for k,n in enumerate(d.names):
            text("dtype_struct_order_field_"+str(j)+"_"+str(k),json.dumps(d[n].descr,separators=(",",":")))
    renamed=np.dtype(struct,copy=True)
    renamed.names=("code","label","samples","sub")
    dtype("dtype_renamed",renamed)
    text("dtype_renamed_names","|".join(renamed.names))
    for i,(s,shape) in enumerate([("i2",(2,3)),(">f4",(2,)),("U2",(3,)),("i2",(0,)),("i2",())]):
        d=np.dtype((np.dtype(s),shape));dtype("dtype_sub_"+str(i),d)
        z=np.zeros((2,),dtype=d)
        ints("dtype_sub_zeros_shape_"+str(i),z.shape)
        text("dtype_sub_zeros_type_"+str(i),z.dtype.str)
    meta=np.dtype("i4",metadata={"unit":"ms","scale":2})
    dtype("dtype_metadata",meta)
    text("dtype_metadata_value",json.dumps(dict(meta.metadata),sort_keys=True,separators=(",",":")))
    t=np.dtypes.StringDType(na_object=None,coerce=False)
    dtype("dtype_nullable",t)
    ints("dtype_nullable_options",[t.coerce,t.na_object is None,hasattr(np.dtypes.StringDType(),"na_object")])
    pairs=[("S0","S3"),("l","q"),("i4",">i4"),("i4","i8"),("f8","i8"),("U2","U3"),("M8[D]","M8[h]"),("V2","V3")]
    for i,(a,b) in enumerate(pairs):
        a,b=np.dtype(a),np.dtype(b)
        put("dtype_compare_"+str(i),np.array([a==b,a!=b,a<b,a<=b,a>b,a>=b]))
    fails("numeric_class_dtype_bad_order",lambda:np.dtype("i4").newbyteorder("unknown"))
    fails("numeric_class_dtype_bad_field",lambda:struct["absent"])
    fails("numeric_class_dtype_bad_shape",lambda:np.dtype(("i2",(-1,))))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore",PendingDeprecationWarning)
        m=np.matrix([[1,2,3],[4,5,6]],dtype="i8")
        a=np.matrix([[2.,1.],[1.,3.]])
        z=np.matrix([[1+2j,3-4j],[5+6j,-1j]])
        text("matrix_tolist",json.dumps(m.tolist(),separators=(",",":")))
        put("matrix_transpose",m.transpose(1,0))
        for n,v in [("A",m.A),("A1",m.A1),("T",m.T),("H",z.H),("I",a.I),("rect_I",m.I),
                    ("row",m[0]),("col",m[:,1]),("scalar",m[1,1]),("fancy",m[[0,1],[2,0]]),
                    ("bool",m[m>3]),("rev",m[::-1,::-1]),("reshape",m.reshape(3,2)),("reshape1",m.reshape(-1)),
                    ("ravel",m.ravel()),("flatten",m.flatten()),("row_squeeze",m[:1].squeeze()),
                    ("col_squeeze",m[:,:1].squeeze()),("scalar_squeeze",np.matrix([[2]]).squeeze()),
                    ("take",m.take([0,3])),("diagonal",m.diagonal()),("copy",m.copy()),("astype",m.astype("f4")),
                    ("real",z.real),("imag",z.imag),("mul",m*np.matrix([[1,2],[3,4],[5,6]])),
                    ("scale",m*2),("rmul",np.arange(2)*m),("matmul",m@np.arange(3)),
                    ("rmatmul",np.arange(2)@m),("add",m+np.array([1,2,3])),("sub",m-2),
                    ("rsub",2-m),("div",m/2),("rdiv",2/m),("pow0",a**0),("pow3",a**3),("powneg",a**-2),
                    ("parse_int",np.matrix("1,2;3,4")),("parse_complex",np.matrix("1+2j 3;4 -2j")),
                    ("block",np.bmat([[a,a],[a,a]]))]:
            put("matrix_"+n,v)
        for n in ["sum","prod","mean","min","max","ptp","all","any","var","std"]:
            for label,axis in [("none",None),("0",0),("1",1),("both",(0,1))]:
                if n=="ptp" and label=="both":
                    fails("numeric_class_matrix_ptp_both",lambda:m.ptp(axis=(0,1)))
                    continue
                kw={"ddof":1} if n in ("var","std") and axis in (0,1) else {}
                put("matrix_"+n+"_"+label,getattr(m,n)(axis=axis,**kw))
        for n in ("argmin","argmax"):
            for label,axis in [("none",None),("0",0),("1",1)]:
                put("matrix_"+n+"_"+label,getattr(m,n)(axis=axis))
        src=np.arange(6).reshape(2,3)
        view=np.asmatrix(src);view.A[0,1]=99
        put("matrix_shared",src)
        copy=np.matrix(src,copy=True);copy[0,1]=77
        put("matrix_copy_original",src)
        m[:,1]=np.matrix([[90],[91]]);put("matrix_assign",m)
        fails("numeric_class_matrix_bad_rank",lambda:np.matrix(np.zeros((2,2,2))))
        fails("numeric_class_matrix_bad_multiply",lambda:m*np.arange(3))
        fails("numeric_class_matrix_bad_parse",lambda:np.matrix("1 2;3"))
        fails("numeric_class_matrix_singular",lambda:np.matrix([[1.,2.],[2.,4.]]).I)

    # Generic scalar members are tested once across concrete representative
    # types; integer and floating-specific operations get exact text or bit tests.
    for i,v in enumerate([np.int16(-7),np.uint64(2**64-1),np.float32(3.25),np.complex64(1+2j),np.bool_(True)]):
        p="scalar_"+str(i)+"_"
        put(p+"value",v)
        ints(p+"meta",[v.ndim,v.size,v.itemsize,v.nbytes,*v.shape,*v.strides])
        put(p+"real",v.real);put(p+"imag",v.imag);put(p+"conj",v.conjugate())
        put(p+"copy",v.copy());put(p+"T",v.T)
        put(p+"transpose",v.transpose());put(p+"squeeze",v.squeeze())
        put(p+"item",np.asarray(v.item(),dtype=v.dtype))
        text(p+"device",v.device)
        put(p+"reshape",v.reshape(1,1));put(p+"ravel",v.ravel());put(p+"flatten",v.flatten())
        put(p+"bytes",np.frombuffer(v.tobytes(),dtype=np.uint8))
        put(p+"byteswap",v.byteswap())
        if v.dtype.kind!="c":put(p+"cast",v.astype("f8"))
    put("scalar_view",np.int16(-7).view("u2"))
    put("scalar_getfield",np.float32(3.25).getfield("i4"))
    v=np.int16(-7);detached=np.array(v,copy=True);detached[()]=88
    put("scalar_detached",v)
    for i,d in enumerate([np.int8,np.int16,np.int32,np.int64,np.uint8,np.uint16,np.uint32,np.uint64,np.longlong,np.ulonglong]):
        for j,v in enumerate([d(0),d(np.iinfo(d).max),d(np.iinfo(d).min)]):
            p="scalar_integer_"+str(i)+"_"+str(j)
            put(p+"_numerator",v.numerator)
            ints(p+"_properties",[v.denominator,v.bit_count(),v.is_integer()])
    cases=["0","-0","0.1","-3.25","4","1.0000000000000000001"]
    for i,d in enumerate([np.float16,np.float32,np.float64,np.longdouble]):
        for j,s in enumerate(cases):
            v=d(s);n,den=v.as_integer_ratio()
            text("scalar_ratio_"+str(i)+"_"+str(j),str(n)+"/"+str(den))
            put("scalar_is_integer_"+str(i)+"_"+str(j),v.is_integer())
        for s in ["nan","inf","-inf"]:
            v=d(s)
            fails("numeric_class_scalar_ratio_"+str(i)+"_"+s,lambda:v.as_integer_ratio())
            put("scalar_is_integer_"+str(i)+"_"+s,v.is_integer())
    hexvalues=[0.,-0.,.1,-3.25,np.nextafter(0.,1.),np.finfo("f8").tiny,np.finfo("f8").max,float("inf"),float("-inf"),float("nan")]
    for i,x in enumerate(hexvalues):
        v=np.float64(x)
        text("scalar_hex_"+str(i),v.hex())
        put("scalar_fromhex_"+str(i),np.float64.fromhex(v.hex()))
    for i,s in enumerate(["  -0x1.8p+2  ","a.8","0x1","-0x0p0","0X1.FP3"]):
        put("scalar_hex_parse_"+str(i),np.float64.fromhex(s))
    fails("numeric_class_scalar_hex_bad",lambda:np.float64.fromhex("xzz"))
    fails("numeric_class_scalar_hex_over",lambda:np.float64.fromhex("0x1p2000"))
