"""NumPy library helpers: scimath, axis tools, blocks, versions and lazy I/O."""
import bz2
import gzip
import io
import pathlib
import tempfile
import threading
import warnings
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
from numpy.lib import scimath as sm, array_utils as au, Arrayterator, NumpyVersion
from numpy.lib import introspect, npyio, format as fmt


def populate_library(record, fails):
    def blob(n,b):
        record(n,np.frombuffer(b,"u1"))
    def text(n,s):
        blob(n,s.encode("utf-8"))
    def put(n,a):
        a=np.asarray(a)
        text(n+"_dtype",str(a.dtype))
        if a.dtype in (np.dtype("g"),np.dtype("G")):
            b=np.frombuffer(a.tobytes(order="C"),"u1").reshape(-1,16)
            record(n,b[:,:10].copy())
            record(n+"_shape",np.array(a.shape,"i8"))
        else:
            record(n,a)
    specs=["bool","i1","i2","i4","i8","u1","u2","u4","u8","f2","f4","f8","g","c8","c16","G"]
    ops="sqrt log log2 log10 arcsin arccos arctanh".split()
    text("scimath_types","|".join(str(np.dtype(s)) for s in specs))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i,s in enumerate(specs):
            d=np.dtype(s)
            for mode in ("inside","outside"):
                if mode=="inside":
                    a=np.array([0.0,0.25,0.75,1.0],dtype=d)
                elif d.kind=="b":
                    a=np.array([True,False,True,True],dtype=d)
                elif d.kind=="u":
                    a=np.array([0,1,2,4],dtype=d)
                elif d.kind=="i":
                    a=np.array([-4,-2,-1,0,1,2,4],dtype=d)
                elif d.kind=="f":
                    a=np.array([-4,-2,-1,-0.0,0.0,1,2,4,np.nan,np.inf,-np.inf],dtype=d)
                else:
                    a=np.array([complex(-4,0.0),complex(-4,-0.0),complex(4,0.0),complex(4,-0.0),
                                -2+0.25j,0.5-0.3j,complex(0.0,-0.0),complex(1.0,0.0)],dtype=d)
                key=f"scimath_{i}_{mode}"
                put(key+"_input",a)
                for op in ops:
                    put(key+"_"+op,getattr(sm,op)(a[::-1]))
            # A nonreal complex array does not enter _tocomplex even when
            # components have negative real parts or magnitude > 1.
            if d.kind=="c":
                a=np.array([-2+0.25j,0.5-0.3j,1+2j],dtype=d)
                put(f"scimath_{i}_nonreal_input",a)
                for op in ops:
                    put(f"scimath_{i}_nonreal_"+op,getattr(sm,op)(a))
        for i,s in enumerate(specs):
            d=np.dtype(s)
            base=np.array([2,4] if d.kind in "bu" else [-2,4],dtype=d).reshape(2,1)
            put(f"scimath_binary_a_{i}",base)
            exponent=np.array([2,1] if d.kind in "bu" else [2,-1],dtype=d).reshape(1,2)
            put(f"scimath_binary_b_{i}",exponent)
        for i,s in enumerate(specs):
            d=np.dtype(s)
            x=np.array([2,4] if d.kind in "bu" else [-2,4],dtype=d).reshape(2,1)
            for j,t in enumerate(specs):
                e=np.dtype(t)
                p=np.array([2,1] if e.kind in "bu" else [2,-1],dtype=e).reshape(1,2)
                put(f"scimath_power_{i}_{j}",sm.power(x,p))
                put(f"scimath_logn_{i}_{j}",sm.logn(p,x))
        for n,v in [
            ("half",np.array([-4,4],"f2")),
            ("extended",np.array(["-1e-4000","1.0000000000000000001"],"g")),
            ("extended_positive",np.array(["1e4000","1.0000000000000000001"],"g")),
            ("signed_min",np.array([np.iinfo("i8").min,0,1],"i8")),
            ("branch",np.array([complex(-4,0.0),complex(-4,-0.0),complex(4,0.0),complex(4,-0.0),
                                 complex(0.0,0.0),complex(-0.0,-0.0),complex(-1,0.0),complex(-1,-0.0)],"c16")),
            ("empty",np.empty((2,0),"f4")),
            ("scalar",np.array(-9,"i2"))]:
            put("scimath_edge_"+n+"_input",v)
            for op in ops:
                a=getattr(sm,op)(v)
                put("scimath_edge_"+n+"_"+op,a)
                if n=="branch":
                    record("scimath_branch_sign_"+op,np.stack((np.signbit(a.real),np.signbit(a.imag)),axis=-1))
    fails("scimath_bad_string",lambda:sm.sqrt(np.array(["-1","2"])))
    fails("scimath_bad_broadcast",lambda:sm.power(np.ones(2),np.ones(3)))

    a=np.arange(24,dtype="i4").reshape(2,3,4)
    re=np.empty((3,),dtype=[("tag","u1"),("x",">i4")])
    c=np.array([1+2j,3+4j],"c16")
    arrays={
        "c":a,"f":a.transpose(2,1,0),"reverse":a[::-1,1:,::-2],
        "gapped":a[:,::2,::2],"broadcast":np.broadcast_to(a[0,0,0:1],(5,2)),
        "unaligned":np.lib.stride_tricks.as_strided(np.arange(16,dtype="u1").view("i2"),shape=(4,),strides=(1,)),
        "scalar":np.array(4,"i8"),"empty":np.empty((2,0,3),"f8"),"record":re,"field":re["x"][::-1],
        "real":c.real,"imag":c.imag[::-1],"unicode":np.array(["a","中文"],"U3"),"void0":np.empty(4,"V0"),
        "object":np.array([1,2,3],object)[::-1],
    }
    for n,v in arrays.items():
        lo,hi=au.byte_bounds(v);base=v.__array_interface__["data"][0]
        result=[lo-base,hi-base]
        if n=="object":
            result=[q//v.itemsize for q in result]
        record("lib_bounds_"+n,np.array(result,"i8"))
    axes=np.arange(-9,10,dtype="i8")
    out=np.full((8,len(axes)),-999,dtype="i8")
    for ndim in range(8):
        for j,ax in enumerate(axes):
            try:out[ndim,j]=au.normalize_axis_index(int(ax),ndim)
            except np.exceptions.AxisError:pass
    record("lib_axis_indices",axes);record("lib_axis_matrix",out)
    fails("lib_axis_name",lambda:au.normalize_axis_index(9,3,msg_prefix="window"))
    for n,ax,ndim,dup in [("negative",(-3,-1),3,False),("order",(2,0,-2),3,False),("dup",(0,-3,1),3,True),("empty",(),0,False),("scalar",1,3,False)]:
        record("lib_axis_tuple_"+n,np.array(au.normalize_axis_tuple(ax,ndim,allow_duplicate=dup),"i8"))
    fails("lib_axis_tuple_duplicate",lambda:au.normalize_axis_tuple((0,-3),3))
    fails("lib_axis_tuple_bad",lambda:au.normalize_axis_tuple((0,3),3))

    def blocks(n,a):
        record(n+"_indices",np.array([a.start,a.stop,a.step],"i8"))
        record(n+"_buffer",np.array(a.buf_size,"i8"))
        record(n+"_shape",np.array(a.shape,"i8"))
        record(n+"_array",a.__array__())
        # __getattr__ also exposes the original array-interface; NumPy's
        # asarray takes that path before calling the explicit __array__.
        record(n+"_asarray",np.asarray(a))
        bs=list(a)
        record(n+"_blocks",np.array([b.shape for b in bs],"i8").reshape(len(bs),len(a.shape)))
        record(n+"_flat",np.array(list(a.flat),dtype=a.var.dtype))
        record(n+"_sizes",np.array([b.size for b in bs],"i8"))
        # Keep a boundary record of every block's first/last element.
        record(n+"_edges",np.array([[b.flat[0],b.flat[-1]] for b in bs],dtype=a.var.dtype).reshape(-1,2))
    b=np.arange(60,dtype="i8").reshape(3,4,5)
    for size in (0,1,2,5,7,20,21,59,100):
        blocks("lib_iter_"+str(size),Arrayterator(b,size))
    blocks("lib_iter_slice",Arrayterator(b,4)[1:3:2,...,1:5:2])
    blocks("lib_iter_index",Arrayterator(b,3)[1,0:4:2])
    blocks("lib_iter_nested",Arrayterator(b,3)[1:3,::2,1:5:2][:,1:3,1:3])
    blocks("lib_iter_noncontiguous",Arrayterator(b[::-1].transpose(2,0,1),5))
    blocks("lib_iter_empty",Arrayterator(np.empty((2,0,3),"i8"),3))
    with tempfile.TemporaryDirectory() as td:
        mm=np.memmap(pathlib.Path(td)/"blocks.bin",mode="w+",dtype="i8",shape=b.shape)
        mm[:]=b
        blocks("lib_iter_mmap",Arrayterator(mm,7)[:,1:4:2,1:5])
        mm.flush()
    it=Arrayterator(b,2)
    first=next(iter(it))
    record("lib_iter_shares",np.array(np.shares_memory(b,first)))
    first[0,0,0]=99
    record("lib_iter_writeback",b)
    scalar=Arrayterator(np.array(5,"i8"),2)
    record("lib_iter_scalar",np.asarray(scalar))
    record("lib_iter_scalar_first",next(iter(scalar)))

    vv=["1.9.0","2.0.0","2.0.1","2.5.0","2.5.0.dev-Unknown","2.5.0.dev0+abc","2.5.0a1.dev-x",
        "2.5.0a1","2.5.0a2","2.5.0a10","2.5.0b1.dev-x","2.5.0b1","2.5.0rc1","2.5.0rc2",
        "2.5.0rc10","2.5.0+local","2.5.0.post1","02.05.000","10.0.0"]
    text("lib_version_inputs","|".join(vv))
    for i,s in enumerate(vv):
        v=NumpyVersion(s)
        text("lib_version_meta_"+str(i),"|".join([v.vstring,v.version,str(v.major),str(v.minor),str(v.bugfix),v.pre_release,str(int(v.is_devversion)),repr(v)]))
    record("lib_version_compare",np.array([[(NumpyVersion(a)>b)-(NumpyVersion(a)<b) for b in vv] for a in vv],"i8"))
    fails("lib_version_invalid",lambda:NumpyVersion("2.5"))
    fails("lib_version_bad_compare",lambda:NumpyVersion("2.5.0")==1)
    info=introspect.opt_func_info(func_name="^add$",signature="float64")
    text("lib_opt_add_float64","|".join(sorted(info.get("add",{}))))
    record("lib_opt_empty",np.array(len(introspect.opt_func_info(func_name="^nobody_had_this_name$")),"i8"))
    record("lib_opt_empty_sig",np.array(len(introspect.opt_func_info(signature="^not_a_dtype$")),"i8"))
    fails("lib_opt_bad_regex",lambda:introspect.opt_func_info(func_name="["))
    class Trace(np.lib.mixins.NDArrayOperatorsMixin):
        def __array_ufunc__(self,ufunc,method,*inputs,**kwargs):
            labels=lambda a: ",".join("self" if x is self else str(x) for x in a)
            return ufunc.__name__+"|"+labels(inputs)+"|"+labels(kwargs.get("out",()))
    obj=Trace()
    calls=[]
    for op in "lt le eq ne gt ge add sub mul matmul truediv floordiv mod divmod pow lshift rshift and xor or".split():
        for prefix in ("","r","i"):
            method=getattr(obj,"__"+prefix+op+"__",None)
            if method is not None:
                calls.append(prefix+op+":"+method("other"))
    for op in ("neg","pos","abs","invert"):
        calls.append(op+":"+getattr(obj,"__"+op+"__")())
    text("lib_mixin_dispatch","\n".join(calls))
    class Wrapped(np.lib.mixins.NDArrayOperatorsMixin):
        def __init__(self,x):self.x=np.asarray(x)
        def __array_ufunc__(self,f,method,*inputs,**kwargs):
            a=[x.x if isinstance(x,Wrapped) else x for x in inputs]
            if "out" in kwargs:kwargs["out"]=tuple(x.x if isinstance(x,Wrapped) else x for x in kwargs["out"])
            return getattr(f,method)(*a,**kwargs)
    x=Wrapped(np.array([1,2,3],"i8"))
    record("lib_mixin_binary",x-1)
    record("lib_mixin_reflected",1-x)
    record("lib_mixin_unary",-x)
    x+=2
    record("lib_mixin_inplace",x)
    a,b=divmod(Wrapped(x),2)
    record("lib_mixin_divmod_q",a);record("lib_mixin_divmod_r",b)

    def npy(a):
        f=io.BytesIO();fmt.write_array(f,a,allow_pickle=False);return f.getvalue()
    payload=np.array([[1,-2,3],[7,9,-8]],dtype=">i4",order="F")
    structure=np.array([(1,2.5),(3,4.5)],dtype=[("x","i2"),("y","f8")])
    f=io.BytesIO()
    with zipfile.ZipFile(f,"w") as z:
        for n,raw,comp in [("z.npy",npy(payload),zipfile.ZIP_DEFLATED),("readme.txt",b"hello\n",zipfile.ZIP_STORED),
                           ("a.npy",npy(np.array([3,5,7],"i8")),zipfile.ZIP_STORED),("raw.npy",b"not an array",zipfile.ZIP_DEFLATED),
                           ("nested/record.npy",npy(structure),zipfile.ZIP_DEFLATED)]:
            zi=zipfile.ZipInfo(n);zi.compress_type=comp;z.writestr(zi,raw)
    raw=f.getvalue();blob("lib_npz_archive",raw)
    with npyio.NpzFile(io.BytesIO(raw)) as n:
        text("lib_npz_keys","|".join(n.keys()))
        text("lib_npz_repr",repr(n))
        record("lib_npz_len",np.array(len(n),"i8"))
        record("lib_npz_contains",np.array([k in n for k in ("z","z.npy","readme.txt","raw","missing","nested/record")]))
        record("lib_npz_z",n["z"].astype("i4",order="F"));blob("lib_npz_z_raw",n["z.npy"].tobytes(order="F"))
        record("lib_npz_a",n.get("a"))
        blob("lib_npz_readme",n["readme.txt"]);blob("lib_npz_raw",n["raw.npy"])
        record("lib_npz_attribute",n.f.a)
        blob("lib_npz_struct",npy(n["nested/record"]))
        record("lib_npz_get_default",np.array(n.get("missing",71),"i8"))
        text("lib_npz_items","|".join(k+":"+("array" if isinstance(v,np.ndarray) else "bytes") for k,v in n.items()))
        text("lib_npz_values","|".join("array" if isinstance(v,np.ndarray) else "bytes" for v in n.values()))
        text("lib_npz_zip_names","|".join(q.filename for q in n.zip.infolist()))
        record("lib_npz_zip_meta",np.array([[q.file_size,q.compress_size,q.compress_type] for q in n.zip.infolist()],"i8"))
        fails("lib_npz_missing",lambda:n["missing"])
    record("lib_npz_closed",np.array([n.zip is None,n.fid is None]))
    with tempfile.TemporaryDirectory() as td:
        p=pathlib.Path(td)/"sample.npz";p.write_bytes(raw)
        arch=np.load(p)
        record("lib_npz_owned",np.array(arch.fid is not None))
        fid=arch.fid
        arch.close()
        record("lib_npz_owned_close",np.array(fid.closed))
    f=io.BytesIO()
    with zipfile.ZipFile(f,"w") as z:
        z.writestr("good.npy",npy(np.array([1,2],"i8")))
        obj=io.BytesIO();fmt.write_array(obj,np.array([1],object))
        z.writestr("bad.npy",obj.getvalue())
    blob("lib_npz_lazy_archive",f.getvalue())
    with npyio.NpzFile(io.BytesIO(f.getvalue())) as n:
        record("lib_npz_lazy_good",n["good"])
        fails("lib_npz_lazy_bad",lambda:n["bad"])

    sample=b"1,2\n3,4\n"
    blob("lib_data_gzip",gzip.compress(sample,mtime=0))
    blob("lib_data_bzip2",bz2.compress(sample))
    with tempfile.TemporaryDirectory() as td:
        root=pathlib.Path(td)
        cache=root/"cache";cache.mkdir()
        ds=npyio.DataSource(str(cache))
        path=root/"plain.txt";path.write_bytes(sample)
        g=root/"gzip.txt.gz";g.write_bytes(gzip.compress(sample,mtime=0))
        b=root/"bzip.txt.bz2";b.write_bytes(bz2.compress(sample))
        (cache/"cached.dat").write_bytes(b"cached\n")
        for n,p in [("plain",path),("gzip",root/"gzip.txt"),("bzip",root/"bzip.txt"),("cache",cache/"cached.dat")]:
            with ds.open(str(p),"rb") as r:blob("lib_data_open_"+n,r.read())
        record("lib_data_exists",np.array([ds.exists(str(p)) for p in [path,g,root/"gzip.txt","cached.dat",root/"missing"]]))
        for n,p in [("relative","folder/file.dat"),("absolute","/folder/file.dat"),("up","../../folder/file.dat"),("space","my file.txt")]:
            text("lib_data_abspath_"+n,str(pathlib.Path(ds.abspath(p)).relative_to(cache)))
        with ds.open(str(path),"ab") as o:o.write(b"5,6\n")
        blob("lib_data_append",path.read_bytes())
        with ds.open(str(g),"wb") as o:o.write(b"7,8\n")
        with ds.open(str(g),"rb") as r:blob("lib_data_gzip_write",r.read())
        fails("lib_data_missing",lambda:ds.open(str(root/"missing"),"rb"))
        fails("lib_data_missing_write",lambda:ds.open(str(root/"missing"),"wb"))
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*p):pass
            def do_GET(self):
                if self.path!="/numbers.txt":
                    self.send_error(404);return
                self.send_response(200);self.end_headers();self.wfile.write(sample)
        server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
        t=threading.Thread(target=server.serve_forever,daemon=True);t.start()
        address=f"http://127.0.0.1:{server.server_port}/numbers.txt"
        try:
            record("lib_data_http_exists",np.array(ds.exists(address)))
            with ds.open(address,"rb") as r:blob("lib_data_http",r.read())
            text("lib_data_http_path",str(pathlib.Path(ds.abspath(address)).relative_to(cache).parts[-1]))
            fails("lib_data_http_write",lambda:ds.open(address,"wb"))
        finally:
            server.shutdown();server.server_close();t.join()
        record("lib_data_http_cached_exists",np.array(ds.exists(address)))
        with ds.open(address,"rb") as r:blob("lib_data_http_cached",r.read())
