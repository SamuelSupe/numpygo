import io
import pathlib
import re
import tempfile
import warnings
import numpy as np


def populate_remaining_core(record, fails):
    def r(name,x):record("remaining_"+name,x)
    def bs(name,b):r(name,np.frombuffer(b,dtype=np.uint8))
    def txt(name,s):bs(name,s.encode())
    a=np.arange(24,dtype=np.int64).reshape(2,3,4)
    r("a",a)
    r("apply_along_axis",np.apply_along_axis(lambda v:np.array([[v.min(),v.max()],[v.sum(),v[0]]]),1,a))
    r("apply_along_axis_scalar",np.apply_along_axis(np.sum,-1,a))
    r("apply_over_axes",np.apply_over_axes(np.sum,a,[0,2]))
    r("fromfunction",np.fromfunction(lambda i,j:i*i+j,(3,4),dtype=np.int64))
    r("fromiter",np.fromiter((i*i for i in range(20)),dtype=np.int64,count=7))
    r("block",np.block([[a[0,:2,:2],np.eye(2,dtype=np.int64)],[np.ones((1,2),dtype=np.int64),np.array([[11,12]])]]))
    r("block_vector",np.block([np.arange(3),np.array(9)]))
    x=np.array([-3.,-1.,0.,1.,3.]);r("piecewise_x",x)
    r("piecewise",np.piecewise(x,[x<=0,x>=1],[lambda t:t*t,lambda t:2*t+1,-5]))
    r("piecewise_overlap",np.piecewise(x,[x<2,x>=0],[-1,3]))
    r("vectorize",np.vectorize(lambda v,w:v*v+.5*w,otypes=[float])(np.array([[1],[2]]),np.array([3,4,5])))
    r("ndindex",np.array(list(np.ndindex(2,3,2)),dtype=np.int64))
    v=a.transpose(2,0,1)[::-1]
    r("ndenumerate_indices",np.array([i for i,_ in np.ndenumerate(v)],dtype=np.int64))
    r("ndenumerate_values",np.array([v for _,v in np.ndenumerate(v)]))
    r("flat",np.array(list(v.flat)))
    r("nditer_c",np.array([int(t) for t in np.nditer(v,order="C")],dtype=np.int64))
    r("nditer_f",np.array([int(t) for t in np.nditer(v,order="F")],dtype=np.int64))
    outer,inner=np.nested_iters(a,[[2,0],[1]],order="C")
    r("nested_iters",np.array([[int(v) for v in inner] for _ in outer],dtype=np.int64))
    b=np.broadcast(np.array([[1],[2]]),np.array([4,5,6]))
    r("broadcast",np.array(list(b),dtype=np.int64));r("broadcast_shape",np.array(b.shape,dtype=np.int64))
    r("broadcast_metadata",np.array([b.ndim,b.size],dtype=np.int64))
    for j,arr in enumerate(np.mask_indices(4,np.triu,1)):r("mask_indices_"+str(j),arr)
    e=np.empty((2,3),dtype=np.int32);e.fill(17);r("empty",e)
    e=np.empty_like(a[0].T,dtype=np.float32);e.fill(3.);r("empty_like",e)
    r("asanyarray",np.asanyarray(a,dtype=np.float32));r("asarray_chkfinite",np.asarray_chkfinite(x,dtype=np.float32))
    r("require",np.require(np.broadcast_to(np.array([1,2,3]),(2,3)),dtype=np.int32,requirements=['F','W','O']))
    fails("remaining_chkfinite_invalid",lambda:np.asarray_chkfinite([1,np.nan]))
    fails("remaining_apply_empty",lambda:np.apply_along_axis(np.sum,1,np.zeros((0,2))))
    fails("remaining_block_mismatch",lambda:np.block([[np.ones(2)],np.ones(3)]))
    fails("remaining_fromiter_short",lambda:np.fromiter(iter([1,2]),count=4,dtype=np.int64))
    r("isscalar",np.array([np.isscalar(1),np.isscalar(np.float64(2)),np.isscalar(np.array(1)),np.isscalar([1])]))
    r("iterable",np.array([np.iterable(1),np.iterable(np.array(1)),np.iterable(np.arange(3)),np.iterable([1])]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",PendingDeprecationWarning)
        r("asmatrix",np.asarray(np.asmatrix(np.arange(3,dtype=np.int64))))
        r("matrix",np.asarray(np.matrix([[1.,2.],[3.,4.]],dtype=np.float32)))
        p=np.array([[1,2],[3,4]],dtype=np.int64)
        r("bmat",np.asarray(np.bmat([[p,p],[p,p]])))
    txt("device",a.device);r("to_device",a.to_device("cpu"))
    r("tolist",np.array(v.tolist()))
    own=np.array([1,2,3],dtype=np.int64);own.resize((2,3));r("resize",own)
    own.resize((2,));r("resize_shrink",own)
    own.setflags(write=False);r("setflags_readonly",np.array(own.flags.writeable))
    fails("remaining_setflags_readonly",lambda:own.__setitem__(0,5))
    own.setflags(write=True);own[0]=5;r("setflags",own)
    root=np.arange(8,dtype=np.int64)
    derived=root[1::2][::-1]
    r("base_identity",np.array([root.base is None,derived.base is root]))
    derived.base[1]=99;r("base_mutation",derived)
    fails("remaining_resize_view",lambda:root.resize((12,)))

    # Byte-level behavior uses exact uint8 fixtures to retain bit patterns.
    raw=bytearray(b"X"+np.array([257,-260,4096],dtype=">i2").tobytes())
    bs("buffer_input",raw)
    view=np.frombuffer(raw,dtype=">i2",offset=1)
    r("frombuffer",view.astype(np.int16))
    view[1]=-2
    bs("buffer_writeback",raw)
    ba=np.array([[0x01234567,0x76543210],[-1,7]],dtype=np.int32);r("ba",ba)
    bs("tobytes_c",ba.T.tobytes(order="C"));bs("tobytes_f",ba.T.tobytes(order="F"))
    r("byteswap",ba.byteswap());r("byteswap_complex",np.array([1+2j,3-4j],dtype=np.complex64).byteswap())
    sw=ba.copy();sw[1,::-1].byteswap(inplace=True);r("byteswap_inplace",sw)
    u=ba.view(np.uint8);r("view",u)
    u[0,1]=200;r("view_mutation",ba)
    memory=ba.data.cast("B");memory[0]=23;r("data_mutation",ba)
    fa=np.array([0x12345678,0x43211234],dtype=np.int32)
    r("getfield",fa.getfield(np.uint16,2))
    fa.setfield([0x7777,0x9999],np.uint16,2);r("setfield",fa)
    with tempfile.TemporaryFile() as f:
        ba.T.tofile(f);f.seek(0);bs("tofile",f.read());f.seek(0)
        r("fromfile",np.fromfile(f,dtype=np.int32,count=3))
    fails("remaining_buffer_bad_size",lambda:np.frombuffer(b"xxz",dtype=np.int32))
    fails("remaining_view_axis",lambda:ba.T.view(np.uint8))
    fails("remaining_field_oob",lambda:fa.getfield(np.uint32,1))
    cbuf=np.frombuffer(bytearray(np.array([2+3j,4-5j],dtype=np.complex128).tobytes()),dtype=np.complex128)
    cbuf.imag[0]=17;r("buffer_complex_component",cbuf)

    # Type metadata values are encoded as typed arrays or UTF-8 bytes.
    types=["bool","int8","int16","int32","int64","uint8","uint16","uint32","uint64","float16","float32","float64","complex64","complex128"]
    kinds=["generic","number","inexact","integer","signedinteger","unsignedinteger","floating","complexfloating","bool_"]
    r("issubdtype",np.array([[np.issubdtype(np.dtype(d),getattr(np,k)) for k in kinds]for d in types]))
    for d in types[1:9]:
        info=np.iinfo(d);r("iinfo_"+d,np.array([info.min,info.max],dtype=d));r("iinfo_bits_"+d,np.array(info.bits,dtype=np.int64))
    for d in types[9:]:
        f=np.finfo(d)
        r("finfo_meta_"+d,np.array([f.bits,f.nmant,f.iexp,f.minexp,f.maxexp,f.precision],dtype=np.int64))
        r("finfo_values_"+d,np.array([f.eps,f.epsneg,f.min,f.max,f.tiny,f.smallest_subnormal,f.resolution],dtype=np.float64))
    for i,ds in enumerate([["int8"],["float16"],["float32","int16"],["float32","complex64"],["complex64","int32"]]):
        txt("common_type_"+str(i),np.dtype(np.common_type(*[np.zeros(2,dtype=d) for d in ds])).name)
    scalars=[0,255,256,-128,-129,65535,65536,2**63,0.,64999.,65000.,np.inf,np.nan,1e-20,3.4e38,3.39e38,1+2j,1e40+2j,complex(np.inf,0)]
    for i,s in enumerate(scalars):txt("min_scalar_type_"+str(i),np.min_scalar_type(s).name)
    for i,(types,typeset,default) in enumerate([("fd","GDFgdf","d"),("Fd","GDFgdf","d"),("abceh","GDFgdf","G"),("hif","qih","q")]):
        txt("mintypecode_"+str(i),np.mintypecode(types,typeset,default))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",DeprecationWarning)
        for i,code in enumerate(["?","b","H","i","L","q","f","D","G"]):txt("typename_"+str(i),np.typename(code))
    r("long",np.long(2**40+17));r("ulong",np.ulong(2**63+17))
    r("fromstring",np.fromstring("1.5, -2.25, 3e2, 4",sep=",",count=3))
    r("fromstring_int",np.fromstring("1 -2 3 999999999999999",sep=" ",dtype=np.int64))
    r("fromstring_complex",np.fromstring("1+2j -3.5-4j 5",sep=" ",dtype=np.complex128))
    text="# ignored\n1, 2, 3, 4\n5, 6, 7, 8 # note\n\n9, 10, 11, 12\n"
    txt("loadtxt_input",text)
    r("loadtxt",np.loadtxt(io.StringIO(text),delimiter=",",usecols=(3,1),dtype=np.int64,ndmin=2))
    r("loadtxt_unpack",np.loadtxt(io.StringIO(text),delimiter=",",unpack=True,skiprows=2,max_rows=2))
    r("loadtxt_scalar",np.loadtxt(io.StringIO("5.25")))
    missing="1,2,NA\n4,,6\n7,8,9\n"
    txt("genfromtxt_input",missing)
    r("genfromtxt",np.genfromtxt(io.StringIO(missing),delimiter=",",missing_values="NA",filling_values=-99.))
    r("genfromtxt_default",np.genfromtxt(io.StringIO("1,2\n,4"),delimiter=","))
    buf=io.StringIO()
    np.savetxt(buf,np.array([[1.25,-3.5],[4.5,2.75]]),fmt="%.2f",delimiter=",",header="values\nunits",footer="end")
    txt("savetxt",buf.getvalue())
    buf=io.StringIO();np.savetxt(buf,np.array([1+2j,-3-4j]),fmt="%.1f");txt("savetxt_complex",buf.getvalue())
    fails("remaining_loadtxt_columns",lambda:np.loadtxt(io.StringIO("1 2\n3 4 5")))
    fails("remaining_fromstring_bad",lambda:np.fromstring("1 x",sep=" "))
    for i,(n,base,pad) in enumerate([(123,3,2),(-255,16,3),(0,2,0),(0,10,4),(1234567,36,0)]):
        txt("base_repr_"+str(i),np.base_repr(n,base,pad))
    for i,(n,width) in enumerate([(5,None),(-5,None),(5,8),(-5,8),(-128,8),(0,0)]):
        txt("binary_repr_"+str(i),np.binary_repr(n,width))
    fails("remaining_binary_repr_width",lambda:np.binary_repr(-5,3))
    for index,(value,precision,unique,sign,trim) in enumerate([
        (1.,None,True,False,"k"),(-0.,None,True,False,"k"),(1.23456789,3,True,True,"-"),
        (1.25,4,False,False,"k"),(100.,5,False,True,"0"),(1.23456789e-6,None,True,False,"."),
        (12345.6789,2,False,False,"-"),
    ]):
        kwargs=dict(precision=precision,unique=unique,sign=sign,trim=trim)
        txt("format_float_positional_"+str(index),np.format_float_positional(value,**kwargs))
        txt("format_float_scientific_"+str(index),np.format_float_scientific(value,**kwargs))

    data=np.array([-3.,-2.,-.75,-.3,0,.1,.2,.3,.4,.5,1.,1.1,2.,3.,6.,7.])
    r("hist_input",data)
    for name in ("sqrt","sturges","rice","scott","stone","doane","fd","auto"):
        r("histogram_bin_edges_"+name,np.histogram_bin_edges(data,bins=name,range=(-2,8)))
    r("histogram_bin_edges_count",np.histogram_bin_edges(data,bins=5))
    samples=np.array([[0.,0.,1.],[.1,.8,.5],[.4,.2,.7],[.8,.8,.9],[1.,1.,1.],[1.2,.5,.8],[.9,.1,.3]])
    weights=np.array([1.,2.,3.,.5,1.5,4.,5.])
    r("hist_samples",samples);r("hist_weights",weights)
    for density in (False,True):
        mode="density" if density else "count"
        h,e=np.histogramdd(samples,bins=([0,.3,1.],[0,.5,1.],[0,.2,.8,1.]),weights=weights,density=density)
        r("histogramdd_"+mode,h)
        for j,axis in enumerate(e):r("histogramdd_"+mode+"_edges_"+str(j),axis)
        h,xe,ye=np.histogram2d(samples[:,0],samples[:,1],bins=(3,2),range=((0,1),(0,1)),weights=weights,density=density)
        r("histogram2d_"+mode,h);r("histogram2d_"+mode+"_x",xe);r("histogram2d_"+mode+"_y",ye)
    matrices=[np.arange(12.,dtype=float).reshape(4,3)/10,np.arange(15.,dtype=float).reshape(3,5)/9,np.arange(10.,dtype=float).reshape(5,2)/7,np.arange(12.,dtype=float).reshape(2,6)/8]
    for i,a in enumerate(matrices):r("einsum_path_input_"+str(i),a)
    equation="ab,bc,cd,de->ae"
    for mode,option in (("greedy","greedy"),("optimal","optimal"),("none",False),("limited",("greedy",1))):
        path,report=np.einsum_path(equation,*matrices,optimize=option)
        flattened=[];sizes=[]
        for step in path[1:]:flattened.extend(step);sizes.append(len(step))
        r("einsum_path_"+mode+"_steps",np.array(flattened,dtype=np.int64))
        r("einsum_path_"+mode+"_sizes",np.array(sizes,dtype=np.int64))
        metrics=[int(float(re.search(key+r":\s*([\d.+eE-]+)",report).group(1))) for key in ("Naive FLOP count","Optimized FLOP count","Largest intermediate")]
        r("einsum_path_"+mode+"_metrics",np.array(metrics,dtype=np.int64))
        r("einsum_path_"+mode+"_output",np.einsum(equation,*matrices,optimize=path))
