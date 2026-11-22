"""Reference cases for the audited API-parameter and numerical gaps."""
import io
import json
import pickle
import tempfile
import warnings
import numpy as np
import numpy.ma.mrecords as mr

def populate_audit_completion(record, fails):
    def r(name, value):
        record("audit_"+name,value)
    def f(name,fn):
        fails("audit_"+name,fn)
    def ex(name,value):
        a=np.asarray(value)
        slots=np.frombuffer(a.tobytes(order="C"),dtype="u1").reshape(-1,16)
        r(name,slots[:,:10].copy())
        r(name+"_shape",np.array(a.shape,dtype="i8"))
    def fields(name,a):
        for n in a.dtype.names:
            r(name+"_"+n,a[n].astype(a.dtype[n].newbyteorder("=")))

    dt=np.dtype([("x",">i4"),("y","f8")],align=True)
    a=np.array([(2,1.),(1,3.),(2,1.),(-2,5.),(0,-0.),(0,0.)],dtype=dt)
    for order in (None,["y"]):
        suffix="default" if order is None else "y"
        fields("record_sort_"+suffix,np.sort(a,order=order,kind="stable"))
        r("record_argsort_"+suffix,np.argsort(a,order=order,kind="stable"))
    u,ix,iv,counts=np.unique(a,return_index=True,return_inverse=True,return_counts=True)
    fields("record_unique",u)
    r("record_unique_indices",ix);r("record_unique_inverse",iv);r("record_unique_counts",counts)
    r("record_isin",np.isin(a,a[[0,3]]))
    fields("record_intersect",np.intersect1d(a,a[[0,3]]))
    r("record_search",np.searchsorted(u,a))
    f("record_sort_bad_order",lambda:np.sort(a,order=["missing"]))
    nested=np.dtype([("p",[("x","i2"),("y","i2")],(2,)),("n","i4")])
    v=np.zeros(3,dtype=nested)
    v["p"]["x"]=[[1,0],[1,-1],[1,4]]
    v["p"]["y"]=[[0,0],[1,0],[-1,0]]
    v["n"]=[10,20,30]
    r("record_nested_sort",np.sort(v,kind="stable")["n"])

    z=np.array([[1,2],[1,2],[3,4],[0,1]],dtype="i8")
    for ax in (0,1):
        v,ix,iv,c=np.unique(z[:,::-1],axis=ax,return_index=True,return_inverse=True,return_counts=True)
        r("unique_axis"+str(ax),v);r("unique_axis"+str(ax)+"_idx",ix)
        r("unique_axis"+str(ax)+"_inverse",iv);r("unique_axis"+str(ax)+"_counts",c)
    r("unique_empty_width",np.unique(np.empty((3,0),dtype="i8"),axis=0))
    r("search_sorter",np.searchsorted([30,10,20],[15,25],sorter=[1,2,0]))
    f("search_bad_sorter",lambda:np.searchsorted([1,2],[1],sorter=[0]))

    x=np.fromstring("1.0000000000000000001 1e-4000 -1e4000",dtype=np.longdouble,sep=" ")
    ex("extended_text",x)
    ex("extended_grad_input",np.array(["1","1.0000000000000000001","1.0000000000000000002"],dtype=np.longdouble))
    x=np.array(["1","1.0000000000000000001","1.0000000000000000002"],dtype=np.longdouble)
    ex("extended_grad",np.gradient(x))
    for ord in (1,-1,0,np.inf,-np.inf,3):
        ex("extended_norm_"+str(ord),np.linalg.norm(x,ord=ord,axis=0))
    v=np.array([[-1,2,4],[3,-5,1]],dtype=np.longdouble)
    for ord in (1,-1,np.inf,-np.inf):
        ex("extended_matrix_norm_"+str(ord),np.linalg.norm(v,ord=ord))
    ex("extended_linspace",np.linspace(np.longdouble("1"),np.longdouble("1.0000000000000000004"),3))
    ex("extended_linspace_step",np.linspace(np.longdouble("1"),np.longdouble("1.0000000000000000004"),3,retstep=True)[1])
    ex("extended_arange",np.arange(np.longdouble("1"),np.longdouble("1.0000000000000000004"),np.longdouble("0.0000000000000000001"),dtype=np.longdouble))
    r("arange_uint64",np.arange(2**64-7,2**64-1,2,dtype="u8"))
    r("gradient_nonuniform",np.gradient([0.,1.,9.],[0.,1.,3.],edge_order=2))
    g=np.arange(12,dtype="f4").reshape(3,4)[:,::-1]**2
    for ax,v in enumerate(np.gradient(g,[0.,1.,3.],2.,edge_order=2)):r("gradient_axes"+str(ax),v)
    g=np.array([0,4,10],dtype="datetime64[D]")
    r("gradient_temporal",np.gradient(g).astype("i8"))
    r("reduce_default",np.add.reduce([[1,2],[3,4]]))
    r("reduce_all",np.add.reduce([[1,2],[3,4]],axis=None))
    t=np.array([[1,2],[3,4]],dtype="timedelta64[D]")
    r("reduce_temporal",np.add.reduce(t).astype("i8"))
    r("reduce_temporal_where",np.add.reduce(t,axis=1,where=[[True,False],[True,True]],initial=np.timedelta64(5,"D")).astype("i8"))

    v=np.arange(12,dtype="f8").reshape(3,4)[:,::-1]
    mask=np.array([[True,False,True,True],[False,True,False,True],[True,True,True,True]])
    m=np.mean(v,axis=1,where=mask,keepdims=True)
    r("variance_where",np.var(v,axis=1,where=mask))
    r("variance_mean",np.var(v,axis=1,mean=m,where=mask,correction=1))
    r("std_where",np.std(v,axis=1,mean=m,where=mask,ddof=1))
    r("variance_supplied",np.var([1.,2.,3.],mean=np.array([0.])))
    v[0,0]=np.nan
    r("nanvar_where",np.nanvar(v,axis=1,where=mask))
    r("nanstd_where",np.nanstd(v,axis=1,where=mask))
    r("cov_weighted",np.cov([[1,2,3],[4,7,6]],aweights=[1,2,3]))
    r("cov_both_weights",np.cov([[1,4],[2,7],[3,6]],rowvar=False,fweights=[1,3,2],aweights=[1.,2.,3.],ddof=1))
    r("cov_cross",np.cov([1.,2.,3.],[4.,7.,6.],bias=True))
    r("corr_cross",np.corrcoef([1.,2.,3.],[4.,7.,6.]))
    v=np.arange(24,dtype="f8").reshape(2,3,4)
    w=np.arange(1,9,dtype="f8").reshape(4,2)
    a,s=np.average(v,axis=(2,0),weights=w,returned=True,keepdims=True)
    r("average_multi",a);r("average_multi_weights",s)

    a=np.array([500,500],dtype="i2")
    r("einsum_dtype",np.einsum("i,i->",a,a,dtype="i8"))
    x=np.arange(6,dtype="f8").reshape(2,3)
    y=np.arange(12,dtype="f8").reshape(3,4)
    z=np.einsum("ij,jk->ik",x,y,order="F",optimize=False)
    r("einsum_f",z)
    r("einsum_opt",np.einsum("ij,jk->ik",x,y,optimize="greedy"))
    out=np.zeros((2,8),dtype="f8")[:,::2]
    np.einsum("ij,jk->ik",x,y,out=out)
    r("einsum_out",out)
    f("einsum_cast",lambda:np.einsum("i,i->",a,a,dtype="i1"))
    start=np.array([0,10],dtype="f4");stop=np.array([2,20],dtype="f4")
    v,s=np.linspace(start,stop,3,axis=1,retstep=True)
    r("linspace_array",v);r("linspace_step",s)
    v,s=np.linspace(start,stop,4,endpoint=False,axis=0,retstep=True,dtype="i8")
    r("linspace_int",v);r("linspace_int_step",s)
    r("logspace_array",np.logspace([0.,1.],[2.,3.],3,base=[2.,10.],axis=1))
    r("geomspace_array",np.geomspace([1.,-1.],[100.,-1000.],4,axis=1))
    r("geomspace_complex",np.geomspace([1+1j,-1+1j],[-1-.5j,1-2j],5,axis=0))

    for mode in ("wrap","clip"):
        r("take_"+mode,np.take([10,20,30],[-4,-1,4],mode=mode))
        r("choose_"+mode,np.choose([-2,0,4],[[10,20,30],[40,50,60]],mode=mode))
    r("interp_period",np.interp([-10,370],[0,180],[0,1],period=360))
    a=np.arange(1,7,dtype="i8").reshape(2,3)
    for mode in ("linear_ramp","mean","median","minimum","maximum"):
        kw={"end_values":((0,4),(2,1))} if mode=="linear_ramp" else {"stat_length":((1,2),(2,1))}
        r("pad_"+mode,np.pad(a,((2,1),(1,2)),mode=mode,**kw))
    for mode in ("reflect","symmetric"):
        r("pad_odd_"+mode,np.pad([1,2,4],(5,4),mode=mode,reflect_type="odd"))
    r("pad_constants",np.pad(a,((1,2),(2,1)),constant_values=((3,4),(5,6))))
    def pad_fn(v,pad,axis,kw):
        if pad[0]:v[:pad[0]]=10+axis
        if pad[1]:v[-pad[1]:]=20+axis
    r("pad_callback",np.pad(a,((1,1),(2,1)),pad_fn))

    r("text_quotes",np.loadtxt(io.StringIO('1,"2",3\n4,"5",6 #comment'),delimiter=",",quotechar='"'))
    r("text_converter",np.loadtxt(io.StringIO("1,0x10\n2,0x20"),delimiter=",",converters={1:lambda x:int(x,16)}))
    a=np.loadtxt(io.StringIO("1,2.5\n3,4.5"),delimiter=",",dtype=[("x","i4"),("y","f8")])
    fields("text_structured",a)
    a=np.genfromtxt(io.StringIO("x,y\n1,2.5\n3,4.5"),delimiter=",",names=True,dtype=None)
    fields("text_inferred",a)

    with np.nditer([np.arange(2,dtype="i8"),np.arange(3,dtype="i8")],op_axes=[[0,-1],[-1,0]],flags=["multi_index"],order="C") as it:
        r("nditer_axes_shape",np.array(it.shape,dtype="i8"))
        r("nditer_axes_values",np.array([(int(x),int(y)) for x,y in it],dtype="i8"))
    with np.nditer([np.arange(3,dtype="i2"),np.arange(3,dtype="f4")],flags=["buffered","common_dtype"],order="C") as it:
        r("nditer_common",np.array([(float(x),float(y)) for x,y in it],dtype="f4"))
    with np.errstate(divide="raise"):
        r("fp_where",np.divide([1.,2.],[0.,1.],where=[False,True],out=np.array([9.,9.])))
        f("fp_raise_option",lambda:np.divide([1.,2.],[0.,1.],where=[True,True]))

    s=np.float64(3.25)
    for n in ("all","any","argmax","argmin","argsort","cumprod","cumsum","max","mean","min","prod","std","sum","var"):
        r("scalar_"+n,getattr(s,n)())
    r("scalar_choose",np.int64(2).choose([10,20,30]))
    r("scalar_clip",s.clip(0,3));r("scalar_compress",s.compress([True]))
    r("scalar_repeat",s.repeat(3));r("scalar_round",s.round(1))
    r("scalar_take",s.take([0,0]));r("scalar_to_device",s.to_device("cpu"))
    r("scalar_tolist",s.tolist());r("scalar_flat",np.array(list(s.flat)))
    r("scalar_base",s.base is None);r("scalar_data",np.frombuffer(s.data.tobytes(),dtype="u1"))
    fl=s.flags
    r("scalar_flags",[fl.c_contiguous,fl.f_contiguous,fl.owndata,fl.writeable,fl.aligned,fl.writebackifcopy])
    r("scalar_dumps",pickle.loads(s.dumps()))
    b=io.BytesIO();s.dump(b);b.seek(0);r("scalar_dump",pickle.load(b))
    with tempfile.TemporaryFile() as file:
        s.tofile(file);file.seek(0);r("scalar_tofile",np.frombuffer(file.read(),dtype="u1"))
    for n,fn in (("fill",lambda:s.fill(7)),("put",lambda:s.put([0],[9])),("resize",lambda:s.resize((2,))),("setflags",lambda:s.setflags(write=True))):
        fn();r("scalar_"+n,s)
    for n,fn in (("diagonal",lambda:s.diagonal()),("nonzero",lambda:s.nonzero()),("searchsorted",lambda:s.searchsorted(1)),
                 ("setfield",lambda:s.setfield(1,"f8")),("sort",lambda:s.sort()),("swapaxes",lambda:s.swapaxes(0,0)),("trace",lambda:s.trace())):
        f("scalar_"+n,fn)
    v=np.void(b"\x01\x02\x03\x04")
    r("void_getfield",v.getfield("u1",1))
    v.setfield(5,"u1",1);r("void_setfield",np.frombuffer(v.tobytes(),dtype="u1"))
    r("void_dtype",np.frombuffer(v.dtype.str.encode(),dtype="u1"))
    r("void_base",v.base is None);fl=v.flags
    r("void_flags",[fl.c_contiguous,fl.f_contiguous,fl.owndata,fl.writeable,fl.aligned,fl.writebackifcopy])
    a=np.array([(4,1.5)],dtype=[("x","i4"),("y","f8")]);v=a[0]
    r("void_record_getfield",v.getfield("f8",4));v.setfield(9,"i4",0)
    r("void_record_writeback",a["x"]);r("void_record_base",v.base is a)
    m=mr.fromarrays([np.ma.array([1,2],mask=[False,True]),np.ma.array([3.,4.],mask=[True,True])],names="a,b")
    r("mrecords_data_a",m.data["a"]);r("mrecords_data_b",m.data["b"])
    r("mrecords_mask_a",m.mask["a"]);r("mrecords_mask_b",m.mask["b"])
    r("mrecords_recordmask",m.recordmask)
    r("mrecords_list",np.frombuffer(json.dumps(m.tolist(),separators=(",",":")).encode(),dtype="u1"))
    r("mrecords_view",m.view().mask["a"])
    r("mrecords_copy",m.copy().mask["b"])
    m.harden_mask();r("mrecords_harden",m._hardmask)
    m.soften_mask();r("mrecords_soften",m._hardmask)
    q=mr.addfield(m,np.ma.array([5,6],mask=[True,False]),"c")
    r("mrecords_add_data",q.data["c"]);r("mrecords_add_mask",q.mask["c"])
    q=mr.fromrecords([(1,3.),(2,4.)],names="a,b")
    r("mrecords_fromrecords",q.data["a"]);r("mrecords_fromrecords_mask",q.mask["a"])
    q=mr.fromtextfile(io.StringIO("a,b\n1,2.5\n3,4.5\n"),delimiter=",")
    r("mrecords_fromtextfile_a",q.data["a"]);r("mrecords_fromtextfile_b",q.data["b"])
    r("mrecords_fromtextfile_mask",q.mask["a"])
    classes=(np.polynomial.Polynomial,np.polynomial.Chebyshev,np.polynomial.Legendre,np.polynomial.Hermite,np.polynomial.HermiteE,np.polynomial.Laguerre)
    modules=(np.polynomial.polynomial,np.polynomial.chebyshev,np.polynomial.legendre,np.polynomial.hermite,np.polynomial.hermite_e,np.polynomial.laguerre)
    prefixes=("poly","cheb","leg","herm","herme","lag")
    for i,cls in enumerate(classes):
        a=np.array(["1.0000000000000000001","0.25","0.5"],dtype=np.longdouble)
        b=np.array(["0.5","-0.125"],dtype=np.longdouble)
        p,q=cls(a),cls(b)
        name="poly_ext_"+str(i)+"_"
        for op,value in (("add",(p+q).coef),("sub",(p-q).coef),("mul",(p*q).coef),("pow",(p**2).coef),
                         ("deriv",p.deriv().coef),("integ",p.integ(k=[np.longdouble("0.25")]).coef),
                         ("div",(p/2).coef),("eval",p(np.array(["0.125","0.5"],dtype=np.longdouble))),
                         ("convert",p.convert(kind=np.polynomial.Polynomial).coef),("linear_roots",q.roots())):
            ex(name+op,value)
        mod,prefix=modules[i],prefixes[i]
        ex(name+"vander",getattr(mod,prefix+"vander")(np.array(["0.125","0.5"],dtype=np.longdouble),3))
        ex(name+"companion",getattr(mod,prefix+"companion")(a))
        v=np.arange(1,13,dtype="f8").reshape(3,4)[:,::-1]
        r(name+"deriv_axis",getattr(mod,prefix+"der")(v,m=2,axis=1))
        r(name+"integ_axis",getattr(mod,prefix+"int")(v,m=1,k=[.25],axis=1))
    p=np.polynomial.Polynomial(np.array([1,2,3],dtype="c16"),domain=[1j,3j])
    r("poly_complex_domain",p.domain)
    r("poly_complex_map",np.array(p.mapparms()))
    r("poly_complex_eval",p(np.array([1j,2j,3j])))
    r("poly_complex_deriv",p.deriv().coef)
    r("poly_complex_integ",p.integ().coef)
    r("poly_complex_convert",p.convert().coef)
    r("poly_complex_roots",np.sort_complex(p.roots()))
    x,y=p.linspace(3);r("poly_complex_linspace_x",x);r("poly_complex_linspace_y",y)
    q=np.polynomial.Polynomial([1.,2.,3.],domain=[1j,3j])
    f("poly_complex_real_deriv",q.deriv)
    f("poly_complex_real_integ",q.integ)
    for spacing,label in [(1.,"uniform"),([0.,1.,3.],"nonuniform")]:
        a=np.array([1,2,4],dtype="datetime64[D]")
        r("gradient_temporal_"+label,np.gradient(a,spacing,edge_order=2).astype("i8"))
    a=np.array([10,20,30,40],dtype="i8")
    with np.nditer([a,np.array([True,False,True,False])],flags=["buffered"],op_flags=[["readwrite","writemasked"],["readonly","arraymask"]],
                   op_dtypes=[np.float64,None],casting="unsafe",buffersize=2) as it:
        for x,mask in it:x[...]=99
    r("nditer_writemasked",a)
    f("extended_complex_fromstring",lambda:np.fromstring("1+2j",sep=" ",dtype=np.clongdouble))
    ex("extended_complex_loadtxt",np.loadtxt(io.StringIO("1.0000000000000000001+0.0000000000000000001j"),dtype=np.clongdouble))
    c=np.empty((),dtype=np.clongdouble);c.real=np.longdouble("1.0000000000000000001");c.imag=np.longdouble("0.0000000000000000001")
    ex("extended_complex_explicit",c)
    rs=np.random.RandomState(17)
    r("legacy_choice_array",rs.choice([10,20,30],size=4))
    f("legacy_choice_bad_dim",lambda:rs.choice([[10,20],[30,40]],size=2))
    r("legacy_choice_next",rs.randint(0,100,size=4))
