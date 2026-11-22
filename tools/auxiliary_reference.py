"""Reference fixtures for the public record, stride, and NPY utilities."""
import io
import pathlib
import tempfile
import warnings
import numpy as np
from numpy.lib import recfunctions as rf, stride_tricks as st, format as fmt


def populate_auxiliary(record, fails):
    def blob(n,x):
        record(n,np.frombuffer(x,dtype="u1"))
    def text(n,x):
        blob(n,x.encode("utf-8"))
    def header(n,a):
        f=io.BytesIO()
        fmt.write_array_header_1_0(f,fmt.header_data_from_array_1_0(a))
        blob(n,f.getvalue())
    def rec(n,a,raw_masked=False):
        if isinstance(a,np.ma.MaskedArray):
            rec(n+"_mask",np.ma.getmaskarray(a))
            a=a.data if raw_masked else a.filled()
        header(n+"_header",a)
        if a.dtype.names is not None:
            record(n+"_meta",np.array([a.dtype.itemsize,a.dtype.alignment,int(a.dtype.isalignedstruct)],"i8"))
            def visit(p,a):
                if a.dtype.names is not None:
                    for name in a.dtype.names:
                        visit(p+"_"+name,a[name])
                else:
                    blob(p,a.tobytes(order="C"))
            visit(n,a)
        else:
            blob(n,a.tobytes(order="C"))

    a=np.arange(12,dtype="i4").reshape(3,4)
    v=st.as_strided(a,shape=(3,2),strides=(4,8))
    record("stride_overlap",v)
    v[1,0]=88
    record("stride_writeback",a)
    record("stride_reverse",st.as_strided(a[2,::-1],shape=(3,4),strides=(-16,-4),writeable=False))
    record("stride_zero",st.as_strided(a,shape=(2,3),strides=(0,4)))
    record("stride_unaligned",st.as_strided(np.arange(12,dtype="u1").view("<i2"),shape=(5,),strides=(1,)))
    fails("stride_readonly",lambda:st.as_strided(a,shape=(2,2),strides=(4,4),writeable=False).__setitem__((0,0),1))
    a=np.arange(12,dtype="i4").reshape(3,4)
    for n,v,w,axes in [
        ("all",a,(2,3),None),("negative",a[::-1,::-1],(2,),(-1,)),
        ("repeated",a,(2,2),(1,1)),("transposed",a.T,(2,),(0,)),
        ("zero",a,(0,),(1,)),("scalar",np.array(9,"i4"),(),None),
        ("empty_axis",np.empty((0,3),"i4"),(2,),(1,))]:
        record("window_"+n,st.sliding_window_view(v,w,axis=axes))
    v=st.sliding_window_view(a,2,axis=0,writeable=True)
    v[1,2,1]=99
    record("window_writeback",a)
    record("window_object",st.sliding_window_view(np.array([1,2,3,4],dtype=object),2).astype("i8"))
    fails("window_bad_axis",lambda:st.sliding_window_view(a,2,axis=2))
    fails("window_bad_size",lambda:st.sliding_window_view(a,5,axis=1))
    fails("window_negative_size",lambda:st.sliding_window_view(a,-1,axis=1))
    fails("window_bad_count",lambda:st.sliding_window_view(a,(2,),axis=(0,1)))
    fails("window_readonly",lambda:st.sliding_window_view(a,2,axis=1).__setitem__((0,0,0),1))

    schema=np.dtype([("id","i2"),("nested",[("x",">f4"),("deep",[("y","i4")])]),("vec","i2",(2,))])
    a=np.array([(1,(2.5,(3,)),[4,5]),(6,(7.5,(8,)),[9,10])],dtype=schema)
    text("rf_get_names",repr(rf.get_names(schema)))
    text("rf_get_names_flat","|".join(rf.get_names_flat(schema)))
    text("rf_get_fieldstructure","|".join(k+":"+"/".join(v) for k,v in rf.get_fieldstructure(schema).items()))
    # The flattened description is independent of field offsets and preserves subarrays.
    rec("rf_flatten_descr",np.zeros(1,dtype=list(rf.flatten_descr(schema))))
    b=rf.rename_fields(a,{"id":"ID","x":"X","y":"Y"})
    rec("rf_rename_fields",b)
    record("rf_rename_shares",np.array(np.shares_memory(a,b)))
    b["nested"]["deep"]["Y"][0]=91
    rec("rf_rename_writeback",a)
    rec("rf_drop_fields",rf.drop_fields(a,["x","vec"],usemask=False))
    rec("rf_rec_drop_fields",rf.rec_drop_fields(a,["nested"]))
    rec("rf_drop_all",rf.drop_fields(a,["id","nested","vec"]))
    target=np.dtype([("vec","i4",(2,)),("nested",[("x","f8"),("z","i2")]),("missing","i4")])
    dst=np.ones(2,target)
    rf.assign_fields_by_name(dst,a,zero_unassigned=False)
    rec("rf_assign_keep",dst)
    rf.assign_fields_by_name(dst,a)
    rec("rf_assign_zero",dst)
    rec("rf_require_fields",rf.require_fields(a,target))
    dst=np.ones(3,target)
    rf.recursive_fill_fields(a,dst)
    rec("rf_recursive_fill",dst)
    padded=np.zeros(2,dtype=[("tag","u1"),("point",[("x","i8"),("y","f4")]),("value","f8")])
    padded["tag"]=[2,3];padded["point"]["x"]=[9,13];padded["point"]["y"]=[4.5,-2.5];padded["value"]=[7.25,8.5]
    for align,recurse in [(True,False),(True,True),(False,True)]:
        rep=rf.repack_fields(padded,align=align,recurse=recurse)
        rec(f"rf_repack_{int(align)}_{int(recurse)}",rep)
    rec("rf_repack_already",rf.repack_fields(padded))
    record("rf_repack_shares",np.array(np.shares_memory(padded,rf.repack_fields(padded))))

    mixed=np.array([(1,(2.5,3),[4,5]),(6,(7.5,8),[9,10])],dtype=[("a","i2"),("sub",[("x","f4"),("y","i4")]),("v","i2",(2,))])
    record("rf_struct_to_unstruct",rf.structured_to_unstructured(mixed[::-1]))
    record("rf_apply_along_fields",rf.apply_along_fields(np.mean,mixed))
    fails("rf_struct_bad_cast",lambda:rf.structured_to_unstructured(mixed,dtype="i2",casting="safe"))
    fails("rf_struct_no_fields",lambda:rf.structured_to_unstructured(np.empty(2,dtype=[])))
    # Uniform padded and reversed offsets share storage rather than copying.
    for n,offsets in [("padded",[1,9,17]),("reversed",[17,9,1]),("overlap",[1,1,1]),("irregular",[1,9,22])]:
        d=np.dtype({"names":["a","b","c"],"formats":["i4"]*3,"offsets":offsets,"itemsize":28})
        r=np.zeros(2,d);r["a"]=[1,4];r["b"]=[2,5];r["c"]=[3,6]
        out=rf.structured_to_unstructured(r)
        record("rf_uniform_"+n,out)
        record("rf_uniform_share_"+n,np.array(np.shares_memory(r,out)))
        out[1,0]=77
        rec_name="rf_uniform_write_"+n
        # Out-of-order and overlapping dtypes cannot be written as NPY descriptors.
        record(rec_name,r["a"])
        record("rf_uniform_copy_"+n,rf.structured_to_unstructured(r,copy=True))
    src=np.arange(15,dtype="i4").reshape(3,5)
    s=rf.unstructured_to_structured(src,names=["a","b","c","d","e"])
    rec("rf_unstruct_to_struct",s)
    record("rf_unstruct_shares",np.array(np.shares_memory(src,s)))
    s["d"][-1]=99
    record("rf_unstruct_writeback",src)
    rec("rf_unstruct_nested",rf.unstructured_to_structured(src,dtype=mixed.dtype))
    rec("rf_unstruct_strided",rf.unstructured_to_structured(src[:,::-1]))
    fails("rf_unstruct_bad_size",lambda:rf.unstructured_to_structured(src,dtype=[("a","i4")]))
    fails("rf_unstruct_both",lambda:rf.unstructured_to_structured(src,dtype=mixed.dtype,names=["a"]))

    base=np.array([(2,20),(1,10)],dtype=[("id","i4"),("v","i2")])
    rec("rf_merge_flat",rf.merge_arrays((base,np.ma.array([9,8],dtype="i4",mask=[False,True])),flatten=True,usemask=True),True)
    fails("rf_merge_flat_unequal",lambda:rf.merge_arrays((base,np.array([9],"i4")),flatten=True,usemask=True))
    rec("rf_merge_nested",rf.merge_arrays((base,np.array([9,8,7],"i4")),flatten=False,usemask=False))
    rec("rf_append_fields",rf.append_fields(base,["c","d"],[np.array([3],"i4"),np.array([2.5,4.5,6.5])],dtypes=["i2","f4"]))
    rec("rf_rec_append_fields",rf.rec_append_fields(base,"c",np.array([3],"i4"),dtypes="i4"))
    fails("rf_append_bad_names",lambda:rf.append_fields(base,["c","d"],[[1]]))
    other=np.array([(3,30.5,4),(4,40.5,5)],dtype=[("id","i4"),("v","f4"),("new","i2")])
    rec("rf_stack_arrays",rf.stack_arrays((base,other),defaults={"new":-7},autoconvert=True))
    rec("rf_stack_filled",rf.stack_arrays((base,other),defaults={"new":-7},autoconvert=True,usemask=False))
    fails("rf_stack_bad_dtype",lambda:rf.stack_arrays((base,other)))
    dup=np.ma.array(np.array([(3,30),(1,10),(1,11),(2,20),(3,31),(5,50)],dtype=base.dtype),
                    mask=[(True,False),(False,False),(False,False),(False,False),(True,False),(False,False)])
    for ignore in (True,False):
        d,i=rf.find_duplicates(dup,key="id",ignoremask=ignore,return_index=True)
        rec("rf_dup_"+str(int(ignore)),d)
        record("rf_dup_index_"+str(int(ignore)),i)
    right=np.array([(3,30.5,4),(2,40.5,5)],dtype=other.dtype)
    for kind in ("inner","outer","leftouter"):
        rec("rf_join_"+kind,rf.join_by("id",base,right,jointype=kind,defaults={"v1":-7,"v2":-8,"new":-9}))
    rec("rf_rec_join",rf.rec_join("id",base,right,jointype="outer",defaults={"v1":-7,"v2":-8,"new":-9}))
    fails("rf_join_missing",lambda:rf.join_by("not_here",base,right))
    fails("rf_join_bad_mode",lambda:rf.join_by("id",base,right,jointype="full"))
    fails("rf_join_collision",lambda:rf.join_by("id",base,right,r1postfix="",r2postfix=""))
    l2=np.array([(2,1,8),(1,2,9),(1,1,7)],dtype=[("j","i2"),("i","i4"),("x","i2")])
    r2=np.array([(1,2,15),(1,1,16),(2,2,17)],dtype=[("i","i4"),("j","i2"),("y","i2")])
    rec("rf_join_multiple",rf.join_by(["i","j"],l2,r2,jointype="outer",defaults={"x":-1,"y":-2}))

    for n,v in [("1",(1,0)),("2",(2,0)),("other",(42,255))]:
        b=fmt.magic(*v)
        blob("format_magic_"+n,b)
        record("format_read_magic_"+n,np.array(fmt.read_magic(io.BytesIO(b)),"i8"))
    fails("format_magic_invalid",lambda:fmt.magic(256,0))
    fails("format_read_magic_invalid",lambda:fmt.read_magic(io.BytesIO(b"not_npy!")))
    endian=np.array([[1,-2,3],[7,9,-8]],dtype=">i4",order="F")
    h=fmt.header_data_from_array_1_0(endian)
    for version,write,read in [(1,fmt.write_array_header_1_0,fmt.read_array_header_1_0),(2,fmt.write_array_header_2_0,fmt.read_array_header_2_0)]:
        f=io.BytesIO();write(f,h)
        blob("format_header_"+str(version),f.getvalue())
        f.seek(8);sh,fo,dt=read(f)
        record("format_header_shape_"+str(version),np.array(sh,"i8"))
        record("format_header_order_"+str(version),np.array(fo))
        text("format_header_dtype_"+str(version),dt.str)
        f.seek(8)
        fails("format_header_limit_"+str(version),lambda:read(f,max_header_size=10))
    for n,d in [("plain",np.dtype(">f8")),("record",np.dtype([(("title","a"),"i2"),("b","f8",(2,))],align=True)),("object",np.dtype(object)),("variable",np.dtypes.StringDType())]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            desc=fmt.dtype_to_descr(d)
        if isinstance(desc,str):
            text("format_descr_"+n,desc)
        else:
            x=np.zeros(2,d)
            header("format_descr_"+n,x)
            rec("format_descr_roundtrip",np.zeros(2,fmt.descr_to_dtype(desc)))
    text("format_drop_metadata",fmt.drop_metadata(np.dtype("f4",metadata={"label":"sample"})).str)
    rec("format_header_data",endian)
    outdir=pathlib.Path(__file__).resolve().parents[1]/"testdata/numpy"
    for v in (1,2,3):
        f=io.BytesIO();fmt.write_array(f,endian,version=(v,0),allow_pickle=False)
        blob("format_write_"+str(v),f.getvalue())
        f.seek(0)
        ar=fmt.read_array(f)
        record("format_read_"+str(v),ar.astype("i4",order="F"))
        blob("format_read_raw_"+str(v),ar.tobytes(order="F"))
        if v==2:
            (outdir/"format_endian.npy").write_bytes(f.getvalue())
    uni=np.array([(1,-2),(3,7)],dtype=[("时间","i4"),("café","i2")])
    f=io.BytesIO();fmt.write_array(f,uni,version=(3,0),allow_pickle=False)
    blob("format_unicode",f.getvalue())
    (outdir/"format_unicode.npy").write_bytes(f.getvalue())
    f=io.BytesIO()
    fails("format_unicode_v1",lambda:fmt.write_array(f,uni,version=(1,0),allow_pickle=False))
    d={"descr":[("a"*66000,"|i1")],"fortran_order":False,"shape":(1,)}
    fails("format_long_v1",lambda:fmt.write_array_header_1_0(io.BytesIO(),d))
    f=io.BytesIO();fmt.write_array_header_2_0(f,d)
    record("format_long_v2",np.array(len(f.getvalue()),"i8"))
    f.seek(8)
    fails("format_long_default_limit",lambda:fmt.read_array_header_2_0(f))
    f.seek(8)
    record("format_long_read",np.array(len(fmt.read_array_header_2_0(f,max_header_size=100000)[2].names[0]),"i8"))
    obj=np.array([1],dtype=object)
    fails("format_write_object",lambda:fmt.write_array(io.BytesIO(),obj,allow_pickle=False))
    f=io.BytesIO();fmt.write_array(f,obj)
    f.seek(0)
    fails("format_read_object",lambda:fmt.read_array(f))
    f=io.BytesIO();fmt.write_array_header_1_0(f,fmt.header_data_from_array_1_0(obj))
    blob("format_object_header",f.getvalue())
    with tempfile.TemporaryDirectory() as td:
        path=pathlib.Path(td)/"mapped.npy"
        m=fmt.open_memmap(path,mode="w+",dtype=">i4",shape=(2,3),fortran_order=True,version=(2,0))
        m[:]=endian
        m.flush()
        record("format_mmap_shape",np.array(m.shape,"i8"))
        record("format_mmap_strides",np.array(m.strides,"i8"))
        record("format_mmap_offset",np.array(m.offset,"i8"))
        m=fmt.open_memmap(path,mode="r+");m[1,2]=67;m.flush()
        m=fmt.open_memmap(path,mode="c");m[0,0]=88;m.flush()
        m=fmt.open_memmap(path,mode="r")
        record("format_mmap_value",m.astype("i4",order="F"))
        fails("format_mmap_readonly",lambda:m.__setitem__((0,0),1))
        blob("format_mmap_file",path.read_bytes())
        (outdir/"format_memmap.npy").write_bytes(path.read_bytes())
        with open(path,"rb") as f:
            record("format_isfileobj",np.array([fmt.isfileobj(f),fmt.isfileobj(io.BytesIO())],dtype=bool))
        fails("format_mmap_object",lambda:fmt.open_memmap(pathlib.Path(td)/"obj.npy",mode="w+",dtype=object,shape=(1,)))
