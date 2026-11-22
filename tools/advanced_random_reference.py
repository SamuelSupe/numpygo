"""Random vector calls and configurable SeedSequence replay."""
import numpy as np
import warnings

ENGINES = {"pcg": np.random.PCG64, "dxsm": np.random.PCG64DXSM,
           "mt": np.random.MT19937, "philox": np.random.Philox, "sfc": np.random.SFC64}

def populate_random_advanced(record, fails):
    for pool in (4,5,8,16):
        for key,entropy in (("empty",[]),("short",[123]),("long",list(range(1,12)))):
            label=f"adv_seed_{pool}_{key}"
            seq=np.random.SeedSequence(entropy,spawn_key=(5,17),pool_size=pool,n_children_spawned=3)
            record(label+"_pool",seq.pool)
            record(label+"_32",seq.generate_state(19))
            record(label+"_64",seq.generate_state(11,dtype=np.uint64))
            for i,child in enumerate(seq.spawn(2)):
                cl=f"{label}_child{i}"
                record(cl+"_key",np.array(child.spawn_key,dtype=np.uint32))
                record(cl+"_pool",child.pool)
                record(cl+"_state",child.generate_state(17))
                record(cl+"_grand",child.spawn(1)[0].generate_state(17))
            rebuilt=np.random.SeedSequence(**seq.state)
            record(label+"_resume",rebuilt.spawn(1)[0].generate_state(17))
            for en,engine in ENGINES.items():
                g=engine(np.random.SeedSequence(entropy,spawn_key=(5,17),pool_size=pool,n_children_spawned=3))
                record(label+"_"+en,g.random_raw(9))
                record(label+"_"+en+"_spawn",g.spawn(1)[0].random_raw(9))
    fails("adv_seed_bad_pool",lambda:np.random.SeedSequence(42,pool_size=3))

    base=np.arange(24,dtype=np.int64).reshape(2,3,4)
    record("adv_permuted_input",base)
    for en,engine in ENGINES.items():
        for kind,a in (("c",base),("f",np.asfortranarray(base)),("transpose",base.transpose(1,2,0)),("reverse",base[:,::-1,:])):
            for ax in (None,0,1,2):
                for mode in ("auto","c","f","stride","alias"):
                    g=np.random.Generator(engine(12345))
                    label=f"adv_permuted_{en}_{kind}_{ax}_{mode}"
                    if mode=="auto":
                        result=g.permuted(a,axis=ax)
                    elif mode=="alias":
                        out=a.copy(order="K")
                        result=g.permuted(out,axis=ax,out=out)
                        assert result is out
                    elif mode=="stride":
                        sh=list(a.shape);sh[-1]*=2
                        backing=np.full(sh,-99,dtype=np.int64)
                        out=backing[...,::-2]
                        result=g.permuted(a,axis=ax,out=out)
                        assert result is out
                        record(label+"_backing",backing)
                    else:
                        out=np.empty_like(a,order=mode.upper())
                        result=g.permuted(a,axis=ax,out=out)
                        assert result is out
                    record(label,result)
                    record(label+"_next",g.random(3))
        g=np.random.Generator(engine(12345))
        record("adv_permuted_cast_"+en,g.permuted(base,axis=1,out=np.empty(base.shape,np.float64)))
        record("adv_permuted_cast_"+en+"_next",g.random(3))
        g=np.random.Generator(engine(12345))
        fails("adv_permuted_bad_cast_"+en,lambda:g.permuted(base,out=np.empty(base.shape,np.int32)))
        record("adv_permuted_bad_cast_"+en+"_next",g.random(3))

    for en in (*ENGINES,"legacy"):
        def factory():
            return np.random.RandomState(12345) if en=="legacy" else np.random.Generator(ENGINES[en](12345))
        for key,alpha in (("allzero",[0.,0.,0.]),("first",[.01,0.,0.]),("middle",[0.,.01,0.]),
                          ("last",[0.,0.,.01]),("tailzero",[.02,.03,0.]),("mixed",[0.,.1,0.]),
                          ("singleton",[2.]),("empty",[]),("tiny",[1e-120,2e-120,3e-120]),("regular",[1.,2.,3.])):
            if en=="legacy" and key=="tiny":
                continue # historical beta underflow flow is not used by legacy dirichlet
            g=factory();label=f"adv_dirichlet_{en}_{key}"
            if en=="legacy" and 0 in alpha:
                fails(label,lambda:g.dirichlet(alpha,size=(2,3)))
            else:
                record(label,g.dirichlet(alpha,size=(2,3)))
            record(label+"_next",g.random(3))
        if en!="legacy":
            for method in ("count","marginals"):
                for key,col,n in (("zero",[0,4,0,8],0),("full",[0,4,0,8],12),("more",[0,4,0,8],9),
                                  ("less",[5,0,8,2],4),("empty",[],0),("allzero",[0,0],0)):
                    g=factory();label=f"adv_mvhg_{en}_{method}_{key}"
                    record(label,g.multivariate_hypergeometric(col,n,size=(2,3),method=method))
                    record(label+"_next",g.random(3))
        methods=("svd",) if en=="legacy" else ("svd","eigh","cholesky")
        for method in methods:
            for key,diag in (("pd",[4.,1.]),("singular",[1.,0.]),("negative",[1.,-.5]),("tolerance",[1.,-1e-10])):
                cov=np.diag(diag)
                for check in ("raise","warn","ignore","bad"):
                    g=factory();label=f"adv_mvnormal_{en}_{method}_{key}_{check}"
                    kw={"method":method} if en!="legacy" else {}
                    def call():
                        return g.multivariate_normal([1.,2.],cov,size=(2,2),check_valid=check,tol=1e-8,**kw)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        if method=="cholesky" and min(diag)<=0 or method!="cholesky" and (check=="bad" or check=="raise" and key=="negative"):
                            fails(label,call)
                        else:
                            record(label,call())
                    record(label+"_warnings",np.int64(len(caught)))
                    record(label+"_next",g.random(3))
        if en!="legacy":
            g=factory();label="adv_mvnormal_bad_method_"+en
            fails(label,lambda:g.multivariate_normal([1.,2.],np.eye(2),size=3,method="bad"))
            record(label+"_next",g.random(3))
