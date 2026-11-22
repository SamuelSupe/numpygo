import numpy as np


def populate_masked(record):
    ma = np.ma

    def r(name, value):
        record("ma_" + name, value)

    def m(name, value):
        value = ma.asarray(value)
        r(name+"_mask", ma.getmaskarray(value))
        r(name+"_values", ma.filled(value, 0))

    x = ma.array([-2., -.5, 0., .5, 2., np.nan, np.inf], mask=[0,1,0,0,0,0,0])
    a = ma.array([[1.25,-2.,0.,4.],[5.,6.,7.,8.]], mask=[[0,1,0,0],[1,1,1,1]])
    b = ma.array([2.,.5,0.,-1.], mask=[0,0,1,0])
    g = ma.array([[1,2,3],[4,5,6]],mask=[[0,1,0],[0,0,0]])
    for name, value in (("x",x),("a",a),("b",b),("g",g)):
        r(name+"_data",value.data); r(name+"_input_mask",ma.getmaskarray(value))
    with np.errstate(all="ignore"):
        for name in ("sqrt","log","log2","log10","arcsin","arccos","arccosh","arctanh","tan","exp","conjugate","sin","cos","arctan","arcsinh","sinh","cosh","tanh","absolute","fabs","negative","floor","ceil","logical_not"):
            m(name,getattr(ma,name)(x))
        for name in ("add","subtract","multiply","divide","floor_divide","remainder","fmod","power","arctan2","equal","not_equal","less","less_equal","greater","greater_equal","logical_and","logical_or","logical_xor","hypot","minimum","maximum"):
            m(name,getattr(ma,name)(a,b))
        for name in ("bitwise_and","bitwise_or","bitwise_xor","left_shift","right_shift"):
            m(name,getattr(ma,name)(g,np.array([1,2,0])))
        m("angle",ma.angle(x,deg=True)); m("round",ma.round(a,1)); m("clip",ma.clip(a,-1,2))
        for name in ("sum","prod","min","max","all","any","ptp","mean","median"):
            m(name,getattr(ma,name)(a,axis=1,keepdims=True))
        for name in ("var","std"):
            m(name,getattr(ma,name)(a,axis=1,ddof=1,keepdims=True))
        for d in (np.float16,np.float32,np.complex64):
            z=ma.array([[1,2,5],[4,5,6]],dtype=d,mask=[[False,True,False],[True,True,False]])
            if d==np.complex64:z[0,0]=1+2j
            r("typed_"+d.__name__+"_data",z.data);r("typed_"+d.__name__+"_mask",z.mask)
            for name in ("mean","median","var","std"):
                m(name+"_"+d.__name__,getattr(ma,name)(z,axis=1))
        r("count",ma.count(a,axis=1,keepdims=True))
        r("count_masked",ma.count_masked(a,axis=1))
        r("argmin",ma.argmin(a,axis=1));r("argmax",ma.argmax(a,axis=1))
        m("cumsum",ma.cumsum(a,axis=1));m("cumprod",ma.cumprod(a,axis=1))
        m("anomalies",ma.anomalies(a,axis=1))
        weights=ma.array([1.,2.,3.,4.],mask=[0,0,1,0])
        r("weights_data",weights.data);r("weights_mask",weights.mask)
        avg,total=ma.average(a,axis=1,weights=weights,returned=True)
        m("average",avg);m("average_weight",total)
        r("allclose",np.array(ma.allclose(a,a)));r("allequal",np.array(ma.allequal(a,a)))
    r("filled",g.filled(-9));r("compressed",g.compressed())
    r("getdata",ma.getdata(g));r("getmask",ma.getmask(g));r("getmaskarray",ma.getmaskarray(g))
    r("is_masked",np.array(ma.is_masked(g)));r("is_mask",np.array(ma.is_mask(g.mask)))
    r("make_mask",ma.make_mask([0,1,2]));r("make_mask_none",ma.make_mask_none((2,3)))
    r("mask_or",ma.mask_or(np.array([0,1,0],dtype=bool),np.array([1,0,0],dtype=bool)))
    m("masked_all",ma.masked_all((2,3),np.int32));m("masked_all_like",ma.masked_all_like(g))
    m("masked_where",ma.masked_where(g.data>4,g))
    for name in ("masked_equal","masked_not_equal","masked_greater","masked_greater_equal","masked_less","masked_less_equal"):
        m(name,getattr(ma,name)(g,4))
    m("masked_inside",ma.masked_inside(g,2,5));m("masked_outside",ma.masked_outside(g,2,5))
    m("masked_values",ma.masked_values(a,4.))
    m("masked_invalid",ma.masked_invalid(x));m("fix_invalid",ma.fix_invalid(x,fill_value=99))
    h=g.copy();h.harden_mask();h[0,:]=9;m("hardmask",h);h.soften_mask();h[0,:]=8;m("softmask",h)
    m("reshape",ma.reshape(g,(3,2)));m("transpose",ma.transpose(g));m("ravel",ma.ravel(g))
    for n in (1,2,3):m("atleast"+str(n),getattr(ma,f"atleast_{n}d")(g[0]))
    m("expand_dims",ma.expand_dims(g,0));m("squeeze",ma.squeeze(g[None,:,:]));m("swapaxes",ma.swapaxes(g,0,1))
    m("resize",ma.resize(g,(3,4)));m("repeat",ma.repeat(g,2,axis=1));m("take",ma.take(g,[2,0],axis=1))
    m("compress",ma.compress([True,False,True],g,axis=1))
    m("diag",ma.diag(g,1));m("diagflat",ma.diagflat(g,1));m("diagonal",ma.diagonal(g,offset=1));m("trace",ma.trace(g,offset=1))
    m("concatenate",ma.concatenate([g,g],axis=0));m("stack",ma.stack([g,g],axis=1));m("append",ma.append(g,g,axis=1))
    for name in ("hstack","vstack","dstack","column_stack"):m(name,getattr(ma,name)([g[0],g[1]]))
    for name, fn in (("split",lambda:np.split(g,2,axis=0)), ("hsplit",lambda:ma.hsplit(g,3)), ("vsplit",lambda:np.vsplit(g,2))):
        for k, value in enumerate(fn()):m(name+"_"+str(k),value)
    m("diff",ma.diff(g,axis=1));m("ediff1d",ma.ediff1d(g,to_begin=-1,to_end=9))
    m("where",ma.where(g>3,g,0));m("choose",ma.choose([1,0,1],[g[0],g[1]]))
    m("sort",ma.sort(g,axis=1,endwith=True,kind="stable"));r("argsort",ma.argsort(g,axis=1,endwith=True,kind="stable"))
    for k,v in enumerate(ma.nonzero(g)):r("nonzero_"+str(k),v)
    m("dot",ma.dot(g,g.T));m("inner",ma.inner(g,g));m("outer",ma.outer(g[0],g[1]))
    # isin masks are tested through numpy.ma's membership wrapper if present.
    for name in ("clump_masked","clump_unmasked"):
        r(name,np.array([(s.start,s.stop) for s in getattr(ma,name)(g.ravel())],dtype=np.int64))
    r("compress_rows",ma.compress_rows(g));r("compress_cols",ma.compress_cols(g))
    m("mask_rows",ma.mask_rows(g.copy()));m("mask_cols",ma.mask_cols(g.copy()))
    m("zeros",ma.zeros((2,3),np.int32));m("ones",ma.ones((2,3),np.int32))
    m("identity",ma.identity(3,dtype=np.int32));m("arange",ma.arange(1,5,2,dtype=np.int32))
    m("zeros_like",ma.zeros_like(g));m("ones_like",ma.ones_like(g))
