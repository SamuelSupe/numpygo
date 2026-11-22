"""Parameterized random calls and matrix/tensor behavior beyond scalar APIs."""
import numpy as np


def populate_parameter(record, fails):
    factories = {
        "pcg": lambda: np.random.Generator(np.random.PCG64(12345)),
        "dxsm": lambda: np.random.Generator(np.random.PCG64DXSM(12345)),
        "mt": lambda: np.random.Generator(np.random.MT19937(12345)),
        "philox": lambda: np.random.Generator(np.random.Philox(12345)),
        "sfc": lambda: np.random.Generator(np.random.SFC64(12345)),
        "legacy": lambda: np.random.RandomState(12345),
    }
    def column(a,b): return np.array([[a],[b]])
    def row(a,b,c): return np.array([[a,b,c]])
    def matrix(a,b,c,d,e,f): return np.array([[a,b,c],[d,e,f]])
    one=matrix(.2,.5,1.,2.,3.,6.)
    loc=column(-2.,3.)
    scale=row(.5,0.,2.)
    distributions = {
        "normal": (loc,scale), "uniform": (column(-3.,1.),row(2.,8.,10.)),
        "exponential": (matrix(0.,.5,1.,2.,3.,4.),),
        "gamma": (column(.3,4.),row(0.,1.,3.)), "standard_gamma": (matrix(0.,.3,1.,2.,3.,7.),),
        "beta": (column(.2,3.),row(.7,1.2,5.)), "chisquare": (one,),
        "f": (column(1.,5.),row(.7,3.,8.)), "standard_t":(one,),
        "pareto":(one,), "weibull":(matrix(0.,.5,1.,2.,3.,4.),), "power":(one,),
        "laplace":(loc,scale), "gumbel":(loc,scale), "logistic":(loc,scale),
        "lognormal":(loc,scale), "rayleigh":(matrix(0.,.5,1.,2.,3.,4.),),
        "triangular":(column(-3.,0.),row(1.,2.,3.),np.array(5.)),
        "wald":(column(.7,3.),row(.5,2.,6.)),
        "noncentral_chisquare":(column(.7,3.),row(0.,2.,5.)),
        "noncentral_f":(column(.7,3.),row(1.,2.,5.),np.array(.4)),
        "vonmises":(loc,row(0.,.5,4.)),
        "poisson":(matrix(0.,1.,9.,10.,45.,100.),),
        "negative_binomial":(column(.8,5.),row(.3,.7,1.)),
        "binomial":(column(4,80),row(0.,.27,1.)),
        "geometric":(matrix(.1,.2,.34,.5,.9,1.),),
        "logseries":(matrix(.1,.2,.34,.5,.8,.9),),
        "zipf":(matrix(1.2,1.5,2.,3.,8.,20.),),
        "hypergeometric":(column(40,80),row(20,30,100),np.array(11,dtype=np.int64)),
    }
    for name,args in distributions.items():
        for j,arg in enumerate(args):
            record(f"param_dist_input_{name}_{j}",arg)
        views=[a[::-1,::-1] if a.ndim==2 else a for a in args]
        for engine,factory in factories.items():
            for mode,size in (("infer",None),("explicit",(2,2,3))):
                rng=factory()
                prefix=f"param_dist_{name}_{engine}_{mode}"
                record(prefix,getattr(rng,name)(*views,size=size))
                record(prefix+"_next",rng.random(4))

    for engine,factory in factories.items():
        for dtype in ("bool","int8","int16","int32","int64","uint8","uint16","uint32","uint64"):
            for endpoint in (False,True):
                if engine=="legacy" and endpoint: continue
                label="closed" if endpoint else "open"
                if dtype=="bool":
                    low=column(0,0)
                    high=row(0,1,1) if endpoint else row(1,2,2)
                elif dtype.startswith("uint"):
                    low=column(0,2); high=row(4,10,20)
                else:
                    low=column(-3,0); high=row(2,10,20)
                if dtype=="uint64":
                    low=column(np.uint64(2**63+1),np.uint64(2**63+9))
                    high=row(np.uint64(2**63+21),np.uint64(2**63+99),np.uint64(2**63+111))
                if dtype=="int64":
                    low=column(-2**63,-7)
                    high=row(2**63-1,100,200)
                record(f"param_int_low_{dtype}_{label}",low)
                record(f"param_int_high_{dtype}_{label}",high)
                rng=factory();prefix=f"param_int_{engine}_{dtype}_{label}"
                if engine=="legacy":
                    value=rng.randint(low[::-1],high[:,::-1],size=(2,2,3),dtype=dtype)
                else:
                    value=rng.integers(low[::-1],high[:,::-1],size=(2,2,3),dtype=dtype,endpoint=endpoint)
                record(prefix,value);record(prefix+"_next",rng.random(4))
        rng=factory()
        high=np.array([2,10,50],dtype=np.int32)
        record("param_int_one_"+engine,(rng.randint if engine=="legacy" else rng.integers)(high))
        record("param_int_one_"+engine+"_next",rng.random(4))
        rng=factory()
        record("param_empty_"+engine,rng.normal(column(1.,2.),row(0.,1.,2.),size=(0,2,3)))
        record("param_empty_"+engine+"_next",rng.random(4))
        rng=factory()
        fails("param_bad_scale_"+engine,lambda:rng.normal(column(1.,2.),row(1.,2.,-1.),size=(2,2,3)))
        record("param_bad_scale_"+engine+"_next",rng.random(4))
        rng=factory()
        fails("param_empty_bad_scale_"+engine,lambda:rng.normal(np.array([]),np.array([-1.])))
        fails("param_negative_zero_scale_"+engine,lambda:rng.normal(0.,-0.))
        record("param_empty_bad_scale_"+engine+"_next",rng.random(4))
        rng=factory()
        fails("param_bad_size_"+engine,lambda:rng.normal(column(1.,2.),row(1.,2.,3.),size=(3,)))
        record("param_bad_size_"+engine+"_next",rng.random(4))
        rng=factory()
        f=rng.randint if engine=="legacy" else rng.integers
        fails("param_bad_bound_"+engine,lambda:f(np.array([0,2]),np.array([3,1]),size=(3,2),dtype=np.int8))
        record("param_bad_bound_"+engine+"_next",rng.random(4))
        rng=factory()
        fails("param_bad_trials_"+engine,lambda:rng.binomial(np.array([4.9,80.2]),.5))
        record("param_bad_trials_"+engine+"_next",rng.random(4))

    pvals=np.array([[[.2,.3,.5],[.1,.1,.8],[0.,.2,.8]]])
    n=np.array([[10],[80]],dtype=np.int64)
    record("param_multinomial_pvals",pvals); record("param_multinomial_n",n)
    for engine,factory in factories.items():
        if engine=="legacy":continue
        rng=factory()
        record("param_multinomial_"+engine,rng.multinomial(n,pvals[:,::-1],size=(2,2,3)))
        record("param_multinomial_"+engine+"_next",rng.random(4))
        rng=factory()
        fails("param_multinomial_bad_"+engine,lambda:rng.multinomial(n,np.array([.7,.6,.1])))
        record("param_multinomial_bad_"+engine+"_next",rng.random(4))

        for dtype in ("float32","float64"):
            for name,method in (("random",""),("standard_normal",""),("standard_exponential","zig"),
                                ("standard_exponential","inv"),("standard_gamma","")):
                for order in ("C","F"):
                    if name=="standard_gamma" and order=="F":continue
                    rng=factory();out=np.full((2,3),-99.,dtype=dtype,order=order)
                    kw={"out":out,"dtype":np.dtype(dtype)}
                    if method:kw["method"]=method
                    a=[np.array([[.3],[4.]],dtype=dtype)] if name=="standard_gamma" else []
                    prefix=f"param_fill_{engine}_{dtype}_{name}_{method or 'default'}_{order}"
                    record(prefix,getattr(rng,name)(*a,**kw))
                    record(prefix+"_next",rng.random(4))
    rng=factories["pcg"]()
    fails("param_fill_bad_shape",lambda:rng.random(size=(3,2),out=np.empty((2,3))))
    fails("param_fill_bad_dtype",lambda:rng.random(out=np.empty((2,3),dtype=np.float32)))
    fails("param_fill_bad_layout",lambda:rng.random(out=np.empty((2,6))[:,::2]))
    fails("param_fill_readonly",lambda:rng.random(out=np.broadcast_to(1.,(2,3))))
    fails("param_fill_bad_byteorder",lambda:rng.random(out=np.empty((2,3),dtype=">f8")))
    fails("param_fill_gamma_f",lambda:rng.standard_gamma(2.,out=np.empty((2,3),order="F")))
    record("param_fill_bad_next",rng.random(4))
