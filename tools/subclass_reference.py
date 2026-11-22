"""Class-specific contracts for ndarray subclasses, pinned to NumPy 2.5."""
import json
import warnings
import numpy as np


def populate_subclasses(record, fails):
    M = np.ma
    def put(n, a):
        a=np.asarray(a)
        if n.startswith("rec_") and a.dtype.kind in "biufc" and not a.dtype.isnative:
            record("subclass_"+n+"_dtype",np.frombuffer(a.dtype.str.encode(),dtype=np.uint8))
            a=a.astype(a.dtype.newbyteorder("="))
        record("subclass_"+n, a)
    def text(n, s):
        put(n, np.frombuffer(s.encode(), dtype=np.uint8))
    def ints(n, values):
        put(n, np.asarray(values, dtype=np.int64))
    def ma(n,a,raw=False,meta=False):
        put("ma_"+n+"_mask",M.getmaskarray(a))
        put("ma_"+n+"_values",M.filled(a,0))
        if raw:
            put("ma_"+n+"_raw",M.getdata(a))
        if meta:
            ints("ma_"+n+"_flags",[a.hardmask,a.sharedmask,a.mask is M.nomask])
            put("ma_"+n+"_fill", np.asarray(a.fill_value,dtype=np.complex128))
    def masked():
        return M.array(np.arange(6,dtype="i2").reshape(2,3),
                       mask=np.array([[False,True,False],[True,False,False]]),
                       fill_value=-7, hard_mask=True)
    a=masked()
    a[0,:]=M.array([80,90,100],mask=[True,False,False])
    ma("hard_set_masked_rhs",a,True)
    a=masked()
    ma("input",a,True,True)
    ints("ma_meta",[a.ndim,a.size,a.itemsize,a.nbytes,len(a),a.iscontiguous(),*a.shape,*a.strides])
    text("ma_baseclass",a.baseclass.__name__)
    put("ma_recordmask",a.recordmask)
    put("ma_filled",a.filled())
    put("ma_filled_explicit",a.filled(-99))
    put("ma_compressed",a.compressed())
    put("ma_item",np.asarray(a.item(1),dtype=np.int64))
    text("ma_tolist",json.dumps(a.tolist(),separators=(",",":")))
    text("ma_tolist_fill",json.dumps(a.tolist(fill_value=-99),separators=(",",":")))
    put("ma_tobytes",np.frombuffer(a.tobytes(),dtype=np.uint8))
    put("ma_tobytes_f",np.frombuffer(a.tobytes(order="F"),dtype=np.uint8))
    ma("copy",a.copy(),True,True)
    ma("astype",a.astype("f4"),True,True)
    ma("byteswap",a.byteswap(),True,True)
    ma("real",a.real,True,True)
    ma("imag",a.imag,True,True)
    c=M.array(np.array([1+2j,3-4j,5+6j],dtype="c8"),mask=[False,True,False],hard_mask=True,fill_value=9+2j)
    ma("complex_real",c.get_real(),True,True)
    ma("complex_imag",c.get_imag(),True,True)
    ma("complex_conj",c.conjugate())
    ma("view",a.view(),True,True)
    ma("slice",a[:,1:],True,True)
    ma("advanced",a[[0,0]],True,True)
    ma("flatten",a.flatten(),True,True)
    ma("ravel",a.ravel(),True,True)
    ma("reshape",a.reshape(3,2))
    ma("transpose",a.transpose())
    ma("mT",a.mT)
    ma("swapped",a.swapaxes(0,1))
    ma("squeeze",a[None,...].squeeze(axis=0))
    ma("diagonal",a.diagonal())
    ma("repeat",a.repeat(2,axis=1))
    ma("take",a.take([2,0],axis=1))
    ma("compress",a.compress([True,False,True],axis=1))
    ma("trace",a.trace())
    for i,v in enumerate(a):
        ma("iter_"+str(i),v)
    # Pointer equality is checked, never machine addresses.
    vv=a[:]
    av=a[[0,1]]
    ints("ma_ids",[a.ids()[0]==vv.ids()[0],a.ids()[1]==vv.ids()[1],
                   a.ids()[0]==av.ids()[0],a.ids()[1]==av.ids()[1]])
    v=a.view()
    v.data[0,0]=77
    put("ma_data_view_write",a.data)
    a=masked()
    a.mask=False
    ma("hard_mask_clear",a,True,True)
    a.mask=np.array([[True,False,True],[False,False,False]])
    ma("hard_mask_or",a,True,True)
    fails("subclass_ma_hard_mask_bad_dtype",lambda:setattr(a,"mask",np.ones(a.shape,dtype="i8")))
    a.soften_mask()
    a.mask=None
    ma("soft_mask_clear",a,True,True)
    a.shrink_mask()
    ints("ma_shrink_nomask",[a.mask is M.nomask])
    put("ma_shrink_recordmask",a.recordmask)
    a=masked()
    a.soften_mask()
    v=a[:]
    v.unshare_mask()
    v.mask=False
    ma("unshare_view",v,True,True)
    ma("unshare_base",a,True,True)
    v.shrink_mask()
    ints("ma_unshare_shrink",[v.mask is M.nomask,v.sharedmask])
    nm=M.array([1,2])
    ints("ma_nomask",[nm.mask is M.nomask,nm.sharedmask])
    nm.mask=False
    ints("ma_nomask_explicit",[nm.mask is M.nomask,nm.sharedmask])
    a=masked()
    a.set_fill_value(-21)
    put("ma_get_fill",np.asarray(a.get_fill_value(),dtype=np.complex128))
    a.fill(9)
    ma("fill_data",a,True,True)
    a=masked()
    # Use a 1-D hard-mask view: NumPy 2.5's multidimensional hard-mask
    # put path indexes the first axis before flattening.
    # A masked RHS additionally encounters the upstream resize rejection.
    # The ordinary RHS exercises the observable hard-mask preservation.
    a.ravel().put([1,2], np.array([80,90]))
    ma("put",a,True)
    a.soften_mask()
    a[0,1]=24
    ma("setitem",a,True)
    a.harden_mask()
    ints("ma_harden",[a.hardmask])
    for hard in (False,True):
        b=M.array([3,4,1,2],mask=[False,True,False,True],hard_mask=hard)
        b.sort()
        ma("sort_"+str(int(hard)),b,True)
    b=M.array([3,4,1,2],mask=[False,True,False,True])
    put("ma_argsort",b.argsort())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        put("ma_argpartition",b.argpartition([0,1,2,3]))
        b.partition([0,1,2,3])
    ma("partition",b,True)
    put("ma_searchsorted",M.array([1,3,5,7],mask=[False,True,False,False]).searchsorted([2,5,8]))
    a=masked()
    for n,v in [("toflex",a.toflex()),("torecords",a.torecords())]:
        put("ma_"+n+"_data",v["_data"])
        put("ma_"+n+"_mask",v["_mask"])
        ints("ma_"+n+"_layout",[v.dtype.itemsize,*[v.dtype.fields[k][1] for k in v.dtype.names]])
        text("ma_"+n+"_names","|".join(v.dtype.names))
        put("ma_"+n+"_bytes",np.frombuffer(v.tobytes(),dtype=np.uint8))
    fails("subclass_ma_resize",lambda:a.resize((6,)))
    import io
    fails("subclass_ma_tofile",lambda:a.tofile(io.BytesIO()))
    g=M.array([[3.,-2.,1.],[5.,4.,0.]],mask=[[False,True,False],[True,False,False]])
    for name in ("all","any","sum","prod","product","min","max","mean","ptp"):
        ma("reduce_"+name,getattr(g,name)(axis=1,keepdims=True))
    for name in ("var","std"):
        ma("reduce_"+name,getattr(g,name)(axis=1,ddof=1,keepdims=True))
    for name in ("argmin","argmax"):
        put("ma_"+name,getattr(g,name)(axis=1))
    put("ma_count",g.count(axis=1,keepdims=True))
    for n in ("cumsum","cumprod","anom"):
        ma(n,getattr(g,n)(axis=1))
    ma("round",g.round(1))
    ma("clip",g.clip(1.,3.))
    ma("dot",g.dot(M.array([1.,2.,3.],mask=[False,True,False])))
    nz=g.nonzero()
    for i,v in enumerate(nz):put("ma_nonzero_"+str(i),v)
    a=M.array([[1,2,3],[4,5,6]],mask=[[False,True,False],[False,False,True]])
    f=a.flat
    ints("ma_flat_iterators",[f.ma is a, f.dataiter.index, f.maskiter.index])
    vals,mask=[],[]
    for v in f:
        mask.append(v is M.masked)
        vals.append(0 if v is M.masked else v)
    ints("ma_flat_values",vals);put("ma_flat_mask",np.asarray(mask,dtype=bool))
    f=a.flat
    ma("flat_slice",f[1:5:2],True,True)
    ma("flat_gather",f[[0,3,5]],True,True)
    f[1:5]=[9,8]
    ma("flat_assign",a,True)
    f[[0,0,5]]=M.array([1,2,3],mask=[False,False,True])
    ma("flat_repeat",a,True)

    def chars(n,v):
        a=np.asarray(v)
        put("char_"+n,np.frombuffer(a.tobytes(order="C"),dtype=np.uint8))
        text("char_"+n+"_dtype",a.dtype.str)
        ints("char_"+n+"_shape",a.shape)
        put("char_"+n+"_subclass",np.asarray(isinstance(v,np.char.chararray)))
    def lists(n,a):
        words,counts=[],[]
        for p in a.flat:
            counts.append(len(p))
            words.extend([v if isinstance(v,bytes) else v.encode("utf-8") for v in p])
        put("char_"+n+"_counts",np.asarray(counts,dtype=np.int64).reshape(a.shape))
        put("char_"+n+"_offsets",np.cumsum([0]+[len(s) for s in words],dtype=np.int64))
        put("char_"+n,np.frombuffer(b"".join(words),dtype=np.uint8))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for kind,dt in [("u","U5"),("s","S5")]:
            a=np.char.array(np.array([" a ","Bb  ","ab","a  ","xyz","12","\t",""],dtype=dt).reshape(2,4))
            n=lambda k:kind+"_"+k
            arg=lambda s:s if kind=="u" else s.encode()
            for op in ("capitalize","upper","lower","title","swapcase","strip","lstrip","rstrip"):
                chars(n(op),getattr(a,op)())
            for op in ("center","ljust","rjust"):
                chars(n(op),getattr(a,op)(8,arg("-")))
            chars(n("zfill"),a.zfill(8))
            chars(n("expandtabs"),a.expandtabs(4))
            chars(n("replace"),a.replace(arg("a"),arg("Q"),1))
            chars(n("partition"),a.partition(arg("b")));chars(n("rpartition"),a.rpartition(arg("b")))
            for op in ("isalpha","isalnum","isdigit","islower","isupper","istitle","isspace"):
                put("char_"+n(op),getattr(a,op)())
            for op in ("isdecimal","isnumeric"):
                if kind=="u":put("char_"+n(op),getattr(a,op)())
                else:fails("subclass_char_"+n(op),lambda op=op:getattr(a,op)())
            for op in ("find","rfind","count","startswith","endswith"):
                put("char_"+n(op),getattr(a,op)(arg("a"),0,5))
            put("char_"+n("index"),a.index(arg("")))
            put("char_"+n("rindex"),a.rindex(arg("")))
            chars(n("add"),a+("!" if kind=="u" else b"!"))
            chars(n("radd"),("!" if kind=="u" else b"!")+a)
            chars(n("multiply"),a*2);chars(n("rmul"),2*a)
            fmt=np.char.array(np.array(["%03d","%d%%"],dtype=dt))
            chars(n("mod"),fmt%np.array([7,-4]))
            b=np.array("a",dtype=dt)
            for op,fn in [("eq",np.equal),("ne",np.not_equal),("lt",np.less),("le",np.less_equal),("gt",np.greater),("ge",np.greater_equal)]:
                # Python operators, not the direct ufunc, exercise the
                # legacy whitespace stripping override.
                import operator
                put("char_"+n(op),getattr(operator,op)(a,b))
            put("char_"+n("argsort"),a.argsort(axis=1))
            trailing=np.char.array(np.array(["a ","a","a  "],dtype=dt))
            put("char_"+n("argsort_trailing"),trailing.argsort())
            for op in ("split","rsplit"):
                lists(n(op),getattr(a,op)(None,1))
            lists(n("splitlines"),a.splitlines(True))
            if kind=="u":
                chars(n("encode"),a.encode("utf-8"))
                chars(n("join"),np.char.array(["-"]).join(np.array(["ab"])))
                chars(n("translate"),a.translate({ord("a"):"XY",ord("b"):None}))
            else:
                chars(n("decode"),a.decode("utf-8"))
                chars(n("translate"),a.translate(bytes.maketrans(b"ab",b"XY")))
            chars(n("slice"),a[:,::-1])
            chars(n("gather"),a[:,[2,0]])
            chars(n("copy"),a.copy())
            chars(n("reshape"),a.reshape(4,2))
            chars(n("transpose"),a.T)
            chars(n("ravel"),a.ravel())
            chars(n("flatten"),a.flatten())
            chars(n("take"),a.take([2,0],axis=1))
            chars(n("repeat"),a.repeat(2,axis=1))
            chars(n("squeeze"),a[None,...].squeeze(axis=0))
            chars(n("swapaxes"),a.swapaxes(0,1))
            chars(n("compress"),a.compress([True,False,True,False],axis=1))
            chars(n("diagonal"),a.diagonal())
            chars(n("cast"),a.astype("U3" if kind=="u" else "S3"))
            chars(n("view"),a.view())
            text("char_"+n("scalar"),a[0,0].decode() if kind=="s" else str(a[0,0]))
            v=a[:,::-1]
            v[0,0]="Go" if kind=="u" else b"Go"
            chars(n("view_write"),a)

    # Structured class behavior is compared by actual field values and
    # descriptor layout, not Python attribute lookup or uninitialized bytes.
    dt=np.dtype([(("ident","id"),">i2"),("shape","<f8"),("vec","<i2",(2,)),("sub",[("x","<f4"),("flag","?")])])
    a=np.zeros((2,3),dtype=dt).view(np.recarray)
    a.id=np.arange(6).reshape(2,3)
    a.field("shape")[:]=np.arange(6).reshape(2,3)+.5
    a.vec=np.arange(12).reshape(2,3,2)
    a.sub.x=np.arange(6).reshape(2,3)+.25
    a.sub.flag=[[True,False,True],[False,True,False]]
    put("rec_field_name",a.field("id"))
    put("rec_field_title",a.field("ident"))
    put("rec_field_index",a.field(0))
    put("rec_field_collision",a.field("shape"))
    put("rec_field_vector",a.field("vec"))
    put("rec_nested_x",a.sub.x)
    put("rec_nested_flag",a.sub.flag)
    rec=a[1,0]
    put("rec_scalar_index",rec[0])
    put("rec_scalar_title",rec["ident"])
    put("rec_scalar_last",rec[-1]["x"])
    rec[0]=91
    rec["shape"]=-7.25
    rec.sub.x=99.5
    put("rec_scalar_write_id",a.id)
    put("rec_scalar_write_shape",a.field("shape"))
    put("rec_scalar_write_nested",a.sub.x)
    view=a[::-1,::-1]
    put("rec_reverse",view.id)
    ints("rec_reverse_strides",view.strides)
    v=view[["shape","id"]]
    ints("rec_multifield_layout",[v.dtype.itemsize,*[v.dtype.fields[k][1] for k in v.dtype.names],*v.strides])
    text("rec_multifield_names","|".join(v.dtype.names))
    v.field("id")[0,0]=123
    put("rec_multifield_write",a.id)
    v=a[[0,1],[1,0]]
    put("rec_gather_id",v.id)
    v.id[0]=42
    put("rec_gather_original",a.id)
    a[[0,1],[0,1]]=v
    put("rec_assign_id",a.id)
    put("rec_assign_shape",a.field("shape"))
    put("rec_assign_vec",a.vec)
    fails("subclass_rec_field_oob",lambda:a.field(99))
    fails("subclass_rec_field_unknown",lambda:a.field("unknown"))
    fails("subclass_rec_multifield_duplicate",lambda:a[["id","id"]])
    plain=np.array([(7,1.25,True),(3,2.,False)],dtype=[("id","i2"),("value","f8"),("flag","?")]).view(np.recarray)
    text("rec_pprint",plain[0].pprint())
    text("rec_pprint_integral",plain[1].pprint())
