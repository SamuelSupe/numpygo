"""Character namespace, variable strings, descriptor mmap and external jumps."""
import pathlib
import tempfile
import numpy as np


def populate_strings(record, fails):
    def put(name, a, layout=False):
        a = np.asarray(a)
        if a.dtype.char == "T":
            words = [v.encode("utf-8") for v in a.flat]
            raw = b"".join(words)
            record(name+"_offsets", np.cumsum([0]+[len(v) for v in words], dtype=np.int64))
            dtype = str(a.dtype)
        else:
            raw = a.tobytes(order="C")
            dtype = a.dtype.str
        record(name,np.frombuffer(raw,dtype=np.uint8))
        record(name+"_dtype",np.frombuffer(dtype.encode("ascii"),dtype=np.uint8))
        record(name+"_shape",np.array(a.shape,dtype=np.int64))
        if layout:
            record(name+"_strides",np.array(a.strides,dtype=np.int64))

    for kind, raw, dtype in [
        ("s", [b" Ab ",b"A\xff",b"a\0b",b"",b"ABC",b"xy",b"\xff\t",b"123",b" \t",b"\x80",b"a\tb",b"-7"],"S12"),
        ("u", [" Ab ","éß","a\0b","","ΟΣ","İ","ﬃ","1²½"," \t","中文","a\tb","-7"],"U12"),
        ("t", [" Ab ","éß","a\0b","","ΟΣ","İ","ﬃ","1²½"," \t","中文","a\tb","-7"],"T"),
    ]:
        def q(name, a): put("strings_"+kind+"_"+name,a)
        def r(name, a): record("strings_"+kind+"_"+name,a)
        def bad(name, op): fails("strings_"+kind+"_"+name,op)
        arr=np.array(raw,dtype=dtype).reshape(3,4)[:,::-1]
        sub=np.array(["a","b"," ","-"],dtype="S3" if kind=="s" else "U3")
        new=np.array(["xyz","Q","","+"],dtype=sub.dtype)
        empty=np.array("",dtype=sub.dtype)
        for name in ["equal","not_equal","less","less_equal","greater","greater_equal","find","rfind","count","startswith","endswith"]:
            args=(arr,sub)
            if name in ("find","rfind","count","startswith","endswith"):args+=( -4,-1)
            r(name,getattr(np.strings,name)(*args))
        q("add",np.strings.add(arr,sub))
        q("multiply",np.strings.multiply(arr,np.array([2,0,-2,3])))
        r("str_len",np.strings.str_len(arr))
        for name in ["isalpha","isdigit","isspace","isalnum","islower","isupper","istitle","isdecimal","isnumeric"]:
            f=getattr(np.strings,name)
            if kind=="s" and name in ["isdecimal","isnumeric"]:
                bad(name,lambda:f(arr))
            else:r(name,f(arr))
        for name in ["upper","lower","swapcase","capitalize","title","strip","lstrip","rstrip"]:
            q(name,getattr(np.strings,name)(arr))
        r("index",np.strings.index(arr,empty))
        r("rindex",np.strings.rindex(arr,empty))
        bad("missing_index",lambda:np.strings.index(arr,np.array("not-found",dtype=sub.dtype)))
        q("replace",np.strings.replace(arr,sub,new,1))
        q("replace_empty",np.strings.replace(arr,empty,new,2))
        q("slice",np.strings.slice(arr,None,None,-2))
        q("expandtabs",np.strings.expandtabs(arr,4))
        for name in ["center","ljust","rjust"]:
            q(name,getattr(np.strings,name)(arr,6,"*"))
        q("zfill",np.strings.zfill(arr,6))
        for name in ["partition","rpartition"]:
            f=getattr(np.strings,name)
            if kind=="t":bad(name,lambda:f(arr,sub))
            else:
                for j,a in enumerate(f(arr,sub)):q(name+str(j),a)
        if kind=="s":
            table=bytes.maketrans(b"Ab",b"XY")
            q("translate",np.strings.translate(arr,table,b" "))
            for codec,policy in [("latin-1","strict"),("utf-8","replace"),("ascii","ignore")]:
                q("decode_"+codec,np.strings.decode(arr,codec,policy))
            bad("decode_strict",lambda:np.strings.decode(arr))
        else:
            q("translate",np.strings.translate(arr,{ord("a"):"XY",ord("b"):None}))
            for codec,policy in [("utf-8","strict"),("latin-1","replace"),("ascii","ignore")]:
                q("encode_"+codec,np.strings.encode(arr,codec,policy))
            bad("encode_strict",lambda:np.strings.encode(arr,"ascii"))
        fmt=np.array(["%04d","%+d","value=%d","%d%%"],dtype=dtype)
        q("mod",np.strings.mod(fmt,np.array([[7],[-12]])))
        args=np.empty((),dtype=object)
        args[()]=(b"hi" if kind=="s" else "hi",1.25)
        q("mod_multi",np.strings.mod(np.array("%s %.2f",dtype=dtype),args))
    narrow=np.array(["ß","İ","ﬃ"],dtype="U1")
    put("strings_u_narrow_upper",np.strings.upper(narrow))
    put("strings_u_narrow_lower",np.strings.lower(narrow))
    empty=np.array([],dtype="U4").reshape(2,0)
    put("strings_empty_upper",np.strings.upper(empty))
    fails("strings_empty_multiply",lambda:np.strings.multiply(empty,2))

    a=np.array(["long"*30,"é🙂","a\0","", "12","-7"],dtype="T").reshape(2,3)
    put("varstr_base",a,True)
    put("varstr_reverse",a[::-1,::-1],True)
    put("varstr_transpose",a.T,True)
    put("varstr_reshape",a.T.reshape(6))
    put("varstr_advanced",a[[1,0],[2,0]])
    put("varstr_broadcast",np.broadcast_to(a[:1],(2,3)),True)
    put("varstr_copy",a[:,::-1].copy())
    put("varstr_concat",np.concatenate([a,np.array([["u","V",""]],dtype="U3")]))
    put("varstr_where",np.where(np.array([[True],[False]]),a,a[:,::-1]))
    put("varstr_zero",np.zeros_like(a))
    put("varstr_ones",np.ones_like(a))
    put("varstr_sort",np.sort(a,axis=1))
    record("varstr_argsort",np.argsort(a,axis=1))
    put("varstr_unique",np.unique(a))
    record("varstr_isin",np.isin(a,np.array(["a\0","-7"],dtype="T")))
    trailing=np.array(["a\0","a","\0",""],dtype="T")
    record("varstr_isin_nul",np.isin(trailing,np.array(["a\0","\0"],dtype="T")))
    record("varstr_equal_nul",np.equal(trailing,np.array(["a\0"],dtype="T")))
    for kind,dtype in [("u","U4"),("t","T")]:
        x=np.array(["abc","x","é"],dtype=dtype)
        out=np.array(["old","keep","old"],dtype="U4" if kind=="u" else "T")[::-1]
        np.multiply(x,np.array([3,100,2]),out=out,where=[True,False,True])
        put("strings_"+kind+"_ufunc_multiply",out,True)
        if kind=="u":fails("strings_u_direct_multiply",lambda:np.multiply(x,2))
        else:
            np.add(x,np.array(["!","?",":"],dtype="U1"),out=out,where=[False,True,True])
            put("strings_t_ufunc_add",out,True)
            put("strings_t_direct_multiply",np.multiply(x,2))
    record("varstr_bool",a.astype(bool))
    record("varstr_any",np.any(a,axis=1))
    record("varstr_count",np.count_nonzero(a))
    record("varstr_isnan",np.isnan(a))
    put("varstr_min",np.min(a,axis=1))
    put("varstr_max",np.max(a,axis=1))
    put("varstr_sum",np.sum(a,axis=1))
    put("varstr_cast_u",a.astype("U4"))
    put("varstr_cast_from_u",np.array(["é","abc"],dtype="U4").astype("T"))
    put("varstr_cast_from_num",np.array([1,-7,23],dtype=np.int64).astype("T"))
    record("varstr_cast_num",np.array(["12","-7"," 23 "],dtype="T").astype(np.int64))
    put("varstr_coerce",np.array([b"a",None,1.2,True],dtype="T"))
    strict=np.array(["x","y"],dtype=np.dtypes.StringDType(coerce=False))
    put("varstr_strict",strict.copy())
    put("varstr_strict_ones",np.ones_like(strict))
    put("varstr_strict_add",np.add(strict,np.array(["!","?"],dtype="T")))
    put("varstr_strict_mul",np.multiply(strict,2))
    put("varstr_strict_concat",np.concatenate([strict,np.array(["z"],dtype="T")]))
    put("strings_t_strict_upper",np.strings.upper(strict))
    put("varstr_strict_cast",strict.astype("T"))
    fails("varstr_strict_error",lambda:strict.__setitem__(0,3))
    changed=a.copy()
    changed[:,1:]=changed[:,::-1][:,:2]
    changed[0,0]="replacement"*20
    put("varstr_assign",changed)
    fails("varstr_bad_promote",lambda:np.promote_types(a.dtype,np.int64))
    fails("varstr_bad_view",lambda:a.view(np.uint8))
    fails("varstr_bad_astype",lambda:a.astype("U"))
    record("varstr_repr",np.frombuffer(np.array_repr(np.array(["a\0","é",""],dtype="T")).encode("utf-8"),dtype=np.uint8))

    with tempfile.TemporaryDirectory() as td:
        p=pathlib.Path(td)/"map.bin"
        def file(name):record(name,np.frombuffer(p.read_bytes(),dtype=np.uint8))
        for kind,dtype,values in [
            ("unicode",">U4",["x","é","中","", "a\0b","1234"]),
            ("bytes","S3",[b"x",b"\xff",b"ab",b"",b"xy",b"z"]),
            ("void","V3",[b"abc",b"123",b"\xffxx",b"\0\0\0",b"xyz",b"z\0\0"]),
        ]:
            a=np.memmap(p,dtype=dtype,mode="w+",offset=7,shape=(2,3),order="F")
            a[:]=np.array(values,dtype=dtype).reshape(2,3)
            a[:,::-1][0,0]=values[0]
            a.flush()
            put("descriptor_map_"+kind,a,True);file("descriptor_map_"+kind+"_file")
            del a
        for kind,dtype in [("date",">M8[D]"),("delta",">m8[3h]")]:
            a=np.memmap(p,dtype=dtype,mode="w+",offset=5,shape=(2,3),order="F")
            a.view(">i8")[:]=np.arange(1,7).reshape(2,3)
            a[:,::-1].view(">i8")[0,0]=-9
            a.flush()
            record("descriptor_map_"+kind,a.view(">i8").astype(np.int64))
            file("descriptor_map_"+kind+"_file")
            del a
        dtype=np.dtype([("id",">i2"),("tag","S3"),("text",">U2")])
        a=np.memmap(p,dtype=dtype,mode="w+",offset=3,shape=(2,2),order="F")
        a["id"][:]=[[10,20],[30,40]]
        a["tag"][:]=[[b"a",b"bc"],[b"abc",b"z"]]
        a["text"][:]=[["é","中"],["xy",""]]
        a["id"][:,::-1][0,0]=-7
        a.flush()
        record("descriptor_map_record_id",a["id"].astype(np.int16))
        record("descriptor_map_record_id_strides",np.array(a["id"].strides,dtype=np.int64))
        put("descriptor_map_record_tag",a["tag"],True)
        put("descriptor_map_record_text",a["text"],True)
        file("descriptor_map_record_file")
        del a
        r=np.memmap(p,dtype=dtype,mode="r",offset=3)
        fails("descriptor_map_readonly",lambda:r["id"].__setitem__(0,100))
        c=np.memmap(p,dtype=dtype,mode="c",offset=3)
        c["tag"][0]=b"new"
        put("descriptor_map_private_tag",c["tag"])
        c.flush();file("descriptor_map_private_file")
    seed=7251
    record("external_caps_tape",np.random.PCG64(seed).random_raw(256))
    bit=np.random.PCG64(seed)
    bit.advance(17)
    record("external_caps_advance",bit.random_raw(5))
    bit=np.random.PCG64(seed)
    record("external_caps_jump",bit.jumped(2).random_raw(6))
    record("external_caps_parent",bit.random_raw(4))
    children=bit.spawn(2)
    for i,b in enumerate(children):record("external_caps_spawn"+str(i),b.random_raw(8))
