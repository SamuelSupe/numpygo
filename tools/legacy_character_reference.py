"""Legacy char namespace, nullable StringDType and titled record descriptors."""
import pathlib
import numpy as np


def populate_legacy_character(record, fails):
    def put(name, a, layout=False):
        a = np.asarray(a)
        dtype = str(a.dtype) if a.dtype.char == "T" else a.dtype.str
        if a.dtype.char == "T":
            words = [v.encode("utf-8") for v in a.flat]
            raw = b"".join(words)
            record(name+"_offsets",np.cumsum([0]+[len(v) for v in words],dtype=np.int64))
        else:
            raw=a.tobytes(order="C")
        record(name,np.frombuffer(raw,dtype=np.uint8))
        record(name+"_dtype",np.frombuffer(dtype.encode("utf-8"),dtype=np.uint8))
        record(name+"_shape",np.array(a.shape,dtype=np.int64))
        if layout:record(name+"_strides",np.array(a.strides,dtype=np.int64))

    def lists(name,a):
        all_words=[]
        counts=[]
        for p in a.flat:
            counts.append(len(p))
            all_words.extend([v if isinstance(v,bytes) else v.encode("utf-8") for v in p])
        record(name+"_counts",np.array(counts,dtype=np.int64).reshape(a.shape))
        record(name+"_offsets",np.cumsum([0]+[len(s) for s in all_words],dtype=np.int64))
        record(name,np.frombuffer(b"".join(all_words),dtype=np.uint8))

    # The legacy namespace is independently exercised, even for delegated kernels.
    for kind,dt in [("u","U12"),("s","S12"),("t","T")]:
        a=np.array([" a ","a  ","A","a b","xy\r\nz"," \t","-7",""],dtype=dt).reshape(2,4)[:,::-1]
        arg_dt="S3" if kind=="s" else "U3"
        b=np.array(["a","A ","a",""],dtype=arg_dt)
        new=np.array(["Q","XY","","+ "],dtype=arg_dt)
        empty=np.array("",dtype=arg_dt)
        sep=np.array(" ",dtype=arg_dt)
        base="char_"+kind+"_"
        q=lambda n,x:put(base+n,x)
        r=lambda n,x:record(base+n,x)
        bad=lambda n,f:fails(base+n,f)
        for name in ("equal","not_equal","less","less_equal","greater","greater_equal"):
            f=getattr(np.char,name)
            if kind=="t":bad(name,lambda:f(a,b))
            else:r(name,f(a,b))
        if kind!="t":
            r("compare_chararrays",np.char.compare_chararrays(a,b,"==",False))
            q("array",np.asarray(np.char.array(a)))
            q("asarray",np.asarray(np.char.asarray(a)))
            r("array_copy",np.shares_memory(a,np.char.array(a)))
            r("asarray_share",np.shares_memory(a,np.char.asarray(a)))
            scalar=np.char.asarray(a)[0,3]
            r("scalar",np.frombuffer(scalar if isinstance(scalar,bytes) else scalar.encode("utf-8"),dtype=np.uint8))
            c=np.char.chararray((2,2),itemsize=4,unicode=kind=="u")
            c[:]=[["abc","ab"],["x","1234"]]
            q("chararray",np.asarray(c))
        else:
            bad("compare_chararrays",lambda:np.char.compare_chararrays(a,b,"==",True))
            q("array",np.asarray(np.char.array(a)))
        for name in ("upper","lower","capitalize","title","swapcase","strip","lstrip","rstrip"):
            q(name,getattr(np.char,name)(a))
        for name in ("isalnum","isalpha","isdigit","islower","isupper","isspace","istitle","isnumeric","isdecimal"):
            f=getattr(np.char,name)
            if kind=="s" and name in ("isnumeric","isdecimal"):bad(name,lambda:f(a))
            else:r(name,f(a))
        r("str_len",np.char.str_len(a))
        q("add",np.char.add(a,b))
        q("multiply",np.char.multiply(a,np.array([2,0,1,-1])))
        q("mod",np.char.mod(np.array(["%03d","%d%%"],dtype=dt),np.array([7,-4])))
        for name in ("find","rfind","count","startswith","endswith"):
            r(name,getattr(np.char,name)(a,b,-3,12))
        for name in ("index","rindex"):r(name,getattr(np.char,name)(a,empty))
        q("replace",np.char.replace(a,b,new,1))
        q("expandtabs",np.char.expandtabs(a,4))
        for name in ("center","ljust","rjust"):q(name,getattr(np.char,name)(a,6,"*"))
        q("zfill",np.char.zfill(a,6))
        for name in ("partition","rpartition"):
            f=getattr(np.char,name)
            if kind=="t":bad(name,lambda:f(a,sep))
            else:q(name,f(a,sep))
        for name in ("split","rsplit"):
            f=getattr(np.char,name)
            lists(base+name,f(a,None,1))
            lists(base+name+"_explicit",f(a,b" " if kind=="s" else " ",1))
        lists(base+"splitlines",np.char.splitlines(a,True))
        if kind=="s":
            bad("join",lambda:np.char.join(b"-",a))
            q("decode",np.char.decode(a,"utf-8"))
            q("translate",np.char.translate(a,bytes.maketrans(b"aA",b"XY"),b" "))
        else:
            q("join",np.char.join(np.array(["-","+"],dtype="U1").reshape(2,1),a))
            q("encode",np.char.encode(a,"utf-8"))
            q("translate",np.char.translate(a,{ord("a"):"xy",ord(" "):None}))

    def nullable(name,a,layout=False):
        a=np.asarray(a)
        na=a.dtype.na_object
        if isinstance(na,str):
            mask=np.array([v==na and not b for v,b in zip(a.flat,a.astype(bool).flat)],dtype=bool)
        elif isinstance(na,float) and np.isnan(na):
            mask=np.isnan(a).ravel()
        else:
            mask=np.array([v is na or type(v)==type(na) and v==na for v in a.flat],dtype=bool)
        words=[b"" if m else v.encode("utf-8") for v,m in zip(a.flat,mask)]
        record(name,np.frombuffer(b"".join(words),dtype=np.uint8))
        record(name+"_offsets",np.cumsum([0]+[len(v) for v in words],dtype=np.int64))
        record(name+"_dtype",np.frombuffer(str(a.dtype).encode("utf-8"),dtype=np.uint8))
        record(name+"_shape",np.array(a.shape,dtype=np.int64))
        record(name+"_mask",mask.reshape(a.shape))
        if layout:record(name+"_strides",np.array(a.strides,dtype=np.int64))

    for name,na in [("none",None),("nan",np.nan),("text","NA"),("int",0)]:
        d=np.dtypes.StringDType(na_object=na)
        a=np.array(["a",na,"","N",na,"A"],dtype=d).reshape(2,3)
        prefix="strna_"+name+"_"
        q=lambda n,x,layout=False:nullable(prefix+n,x,layout)
        r=lambda n,x:record(prefix+n,x)
        bad=lambda n,f:fails(prefix+n,f)
        q("base",a,True);q("reverse",a[:,::-1],True);q("copy",a.copy())
        q("concat",np.concatenate([a,a]))
        q("where",np.where([[True],[False]],a,a[:,::-1]))
        q("zeros",np.zeros_like(a));q("ones",np.ones_like(a))
        q("take",np.take(a,[1,0],axis=1))
        changed=a.copy();changed[0,0]=na;changed[1,1]="real"
        q("assign",changed)
        r("bool",a.astype(bool));r("isnan",np.isnan(a));r("any",np.any(a,axis=1))
        r("eq",np.equal(a,a[:,::-1]));r("ne",np.not_equal(a,a[:,::-1]))
        r("isin",np.isin(a,a))
        put(prefix+"cast_u",a.astype("U8"))
        put(prefix+"cast_default",a.astype("T"))
        bad("cast_num",lambda:np.array(["1",na],dtype=d).astype(float))
        if name in ("none","int"):
            for n,f in [
                ("add",lambda:np.add(a,a)),("multiply",lambda:np.multiply(a,2)),("less",lambda:np.less(a,a[:,::-1])),
                ("sum",lambda:np.sum(a,axis=1)),("min",lambda:np.min(a,axis=1)),("max",lambda:np.max(a,axis=1)),
                ("sort",lambda:np.sort(a,axis=1)),("unique",lambda:np.unique(a)),("argmin",lambda:np.argmin(a,axis=1)),
                ("argmax",lambda:np.argmax(a,axis=1)),("upper",lambda:np.strings.upper(a)),
                ("strip",lambda:np.strings.strip(a)),("alpha",lambda:np.strings.isalpha(a)),
                ("strlen",lambda:np.strings.str_len(a)),("replace",lambda:np.strings.replace(a,"a","X")),
            ]:bad(n,f)
        else:
            q("add",np.add(a,a));q("multiply",np.multiply(a,2));r("less",np.less(a,a[:,::-1]))
            q("sum",np.sum(a,axis=1));q("min",np.min(a,axis=1));q("max",np.max(a,axis=1))
            q("sort",np.sort(a,axis=1));q("unique",np.unique(a))
            r("argmin",np.argmin(a,axis=1));r("argmax",np.argmax(a,axis=1))
            q("strip",np.strings.strip(a));q("replace",np.strings.replace(a,"a","X"))
            r("alpha",np.strings.isalpha(a))
            if name=="nan":
                bad("upper",lambda:np.strings.upper(a));bad("strlen",lambda:np.strings.str_len(a))
            else:
                q("upper",np.strings.upper(a));r("strlen",np.strings.str_len(a))
                q("same_text",np.add(np.array(["N"],dtype=d),np.array(["A"],dtype=d)))
        strict=np.array([na,"x"],dtype=np.dtypes.StringDType(na_object=na,coerce=False))
        q("strict",strict)
        bad("strict_value",lambda:strict.__setitem__(1,123))
        r("repr",np.frombuffer(np.array_repr(a).encode("utf-8"),dtype=np.uint8))
    bad=lambda n,f:fails("strna_"+n,f)
    bad("different_sentinel",lambda:np.add(np.array(["x"],dtype=np.dtypes.StringDType(na_object=None)),np.array(["x"],dtype=np.dtypes.StringDType(na_object=np.nan))))

    # Titles are dtype metadata, not additional fields or storage.
    for kind,dt in [("u","U6"),("s","S6"),("t","T")]:
        a=np.array(["a b","xyab","zx" if kind=="s" else "éx"],dtype=dt).reshape(3,1)
        b=np.array(["a","b","x"],dtype="S1" if kind=="s" else "U1")
        new=np.array(["Q","Z","-"],dtype=b.dtype)
        start=np.array([-2,0,1]);end=np.array([4,3,6]);width=np.array([3,5,6])
        prefix="strarray_"+kind+"_"
        for n in ("find","rfind","count","startswith","endswith"):
            record(prefix+n,getattr(np.strings,n)(a,b,start,end))
        for n in ("index","rindex"):
            record(prefix+n,getattr(np.strings,n)(a,np.array("",dtype=b.dtype),start,end))
        put(prefix+"replace",np.strings.replace(a,b,new,np.array([0,1,-1])))
        for n in ("center","ljust","rjust"):put(prefix+n,getattr(np.strings,n)(a,width,b))
        put(prefix+"zfill",np.strings.zfill(a,width))
        for n in ("strip","lstrip","rstrip"):put(prefix+n,getattr(np.strings,n)(a,b))
        tab=np.array(["a\tb","xy\n\t","z\t" if kind=="s" else "é\t"],dtype=dt).reshape(3,1)
        put(prefix+"expandtabs",np.strings.expandtabs(tab,np.array([0,2,4])))
        put(prefix+"slice",np.strings.slice(a,np.array([0,-1,0]),np.array([3,-3,4]),np.array([1,-1,2])))
        fails(prefix+"slice_step",lambda:np.strings.slice(a,0,3,np.array([1,0,1])))

    dt=np.dtype([(("速度","speed"),">i2"),((7,"v"),"<f8",(2,)),(("other","tag"),"S3")])
    a=np.zeros((2,2),dtype=dt)
    a["速度"][:]=[[1,2],[3,4]]
    a["v"][:]=np.arange(8).reshape(2,2,2)/2
    a["other"][:]=[[b"x",b"ab"],[b"abc",b"z"]]
    a["速度"][:,::-1][0,0]=-9
    record("titles_speed",a["speed"].astype(np.int16))
    record("titles_v",a["v"])
    put("titles_tag",a["tag"])
    record("titles_bytes",np.frombuffer(a.tobytes(),dtype=np.uint8))
    record("titles_itemsize",np.array(dt.itemsize,dtype=np.int64))
    record("titles_names",np.frombuffer("|".join(dt.names).encode("utf-8"),dtype=np.uint8))
    dt2=np.dtype([(("速度","speed"),"<i4"),((7,"v"),"<f8",(2,)),(("other","tag"),"S5")])
    promoted=np.promote_types(dt,dt2)
    record("titles_promote_itemsize",np.array(promoted.itemsize,dtype=np.int64))
    fails("titles_mismatch",lambda:np.promote_types(dt,np.dtype([(("different","speed"),"i2"),((7,"v"),"<f8",(2,)),(("other","tag"),"S3")])))
    fails("titles_duplicate",lambda:np.dtype([(("b","a"),"i2"),("b","i2")]))
    fails("titles_self",lambda:np.dtype([(("a","a"),"i2")]))
    root=pathlib.Path(__file__).resolve().parents[1]/"testdata"/"numpy"
    np.save(root/"structured_titles.npy",a,allow_pickle=False)
    record("titles_npy_file",np.frombuffer((root/"structured_titles.npy").read_bytes(),dtype=np.uint8))
    dt_inner=np.dtype([((True,"ok"),"?"),(("","val"),"<i4")])
    dt_outer=np.dtype([(("meta","data"),dt_inner),(("标签","label"),"U2")])
    b=np.zeros(2,dtype=dt_outer)
    b["meta"]["ok"]=[True,False]
    b["meta"][""]=[6,-5]
    b["标签"]=["中","a"]
    record("titles_nested_ok",b["data"]["ok"])
    record("titles_nested_val",b["meta"]["val"])
    put("titles_nested_label",b["label"])
    record("titles_nested_bytes",np.frombuffer(b.tobytes(),dtype=np.uint8))
    record("titles_nested_mask_itemsize",np.array(np.ma.make_mask_descr(dt_outer).itemsize,dtype=np.int64))
    np.save(root/"structured_titles_nested.npy",b,allow_pickle=False)
    record("titles_nested_file",np.frombuffer((root/"structured_titles_nested.npy").read_bytes(),dtype=np.uint8))
