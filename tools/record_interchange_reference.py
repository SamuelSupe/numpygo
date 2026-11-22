"""Structured dtype promotion and genuine NPY descriptor interchange."""
import io
import pathlib
import numpy as np


def populate_records(record,fails):
    def raw(name,a):
        record("recwide_"+name,np.frombuffer(a.tobytes(order="C"),dtype=np.uint8))
    def layout(name,a):
        d=a.dtype
        record("recwide_"+name+"_layout",np.array([d.itemsize]+[d.fields[n][1] for n in d.names],dtype=np.int64))
    def cases(name,a,b):
        d=np.result_type(a,b)
        # Do not materialize malformed layouts from an upstream dtype bug.
        assert all(d.fields[n][1]+d.fields[n][0].itemsize<=d.itemsize for n in d.names)
        for label,v in (
            ("concat",np.concatenate([a,b])),
            ("where",np.where([True,False],a,b)),
            ("cast",a.astype(d)),
        ):
            raw(name+"_"+label,v);layout(name+"_"+label,v)
        record("recwide_"+name+"_equal",a==b)
    a=np.zeros(2,dtype=[("id",">i2"),("tag","S3"),("v","<i2",(2,)),("sub",[("x","<f4")])])
    b=np.zeros(2,dtype=[("id","<i8"),("tag","S5"),("v","<i2",(2,)),("sub",[("x","<f8")])])
    a["id"]=[-4,9];b["id"]=[-4,2**40]
    a["tag"]=[b"abc",b"def"];b["tag"]=[b"abc",b"hello"]
    a["v"]=[[2,3],[4,5]];b["v"]=[[2,3],[-2,-3]]
    a["sub"]["x"]=[1.5,2];b["sub"]["x"]=[1.5,-5.25]
    cases("nested",a,b)
    a=np.array([(1,b"abc"),(2,b"def")],dtype=[("id","i1"),("tag","S3")])
    b=np.array([(1,"abc"),(70000,"中文")],dtype=[("id","i4"),("tag","U2")])
    cases("text",a,b)
    a=np.zeros(2,dtype=[("id","i1"),("x","f4")])
    b=np.zeros(2,dtype=np.dtype([("id","i4"),("x","f8")],align=True))
    a["id"]=[1,2];b["id"]=[1,1000];a["x"]=[3.5,4.5];b["x"]=[3.5,6.5]
    # Aligned dtype padding bytes have unspecified contents after ufuncs;
    # compare field values and offsets instead of those padding bytes.
    for label,v in (("concat",np.concatenate([a,b])),("where",np.where([True,False],a,b)),("cast",a.astype(np.result_type(a,b)))):
        layout("aligned_"+label,v)
        record("recwide_aligned_"+label+"_id",v["id"])
        record("recwide_aligned_"+label+"_x",v["x"])
    for label,dt1,dt2 in (
        ("names",[("a","i4")],[("b","i4")]),
        ("shape",[("a","i2",(2,))],[("a","i2",(3,))]),
        ("void",[("a","V2")],[("a","V3")]),
    ):
        fails("recwide_bad_"+label,lambda:np.concatenate([np.zeros(1,dtype=dt1),np.zeros(1,dtype=dt2)]))
    dt=np.dtype({"names":["a","b"],"formats":["i4","i2"],"offsets":[0,1],"itemsize":4})
    fails("recwide_overlap_npy",lambda:np.save(io.BytesIO(),np.zeros(1,dtype=dt),allow_pickle=False))

    destination=pathlib.Path(__file__).resolve().parents[1]/"testdata/numpy"
    dt=np.dtype([("id",">i4"),("温度'","<f8"),("v","<i2",(2,)),("sub",[("x",">f4"),("name","S4")])])
    packed=np.zeros((2,3),dtype=dt)
    packed["id"]=np.arange(6).reshape(2,3)-2
    packed["温度'"]=np.arange(6).reshape(2,3)+0.5
    packed["v"]=np.arange(12).reshape(2,3,2)
    packed["sub"]["x"]=3.75;packed["sub"]["name"]=b"tag"
    dt=np.dtype([("tag","S3"),("id","<i8"),("vec","<f4",(2,)),("name","U3")],align=True)
    f=np.zeros((2,3),dtype=dt,order="F")
    f["tag"]=b"go";f["id"]=np.arange(6).reshape(2,3)*1000000
    f["vec"]=np.arange(12).reshape(2,3,2)+0.25;f["name"]="中文"
    for name,v in (("structured_packed",packed),("structured_fortran",f)):
        buf=io.BytesIO();np.save(buf,v,allow_pickle=False)
        b=buf.getvalue()
        record("recwide_"+name+"_file",np.frombuffer(b,dtype=np.uint8))
        raw(name,v);layout(name,v)
        record("recwide_"+name+"_stride",np.array(v.strides,dtype=np.int64))
        (destination/(name+".npy")).write_bytes(b)
