"""Seed, raw bit streams, state transitions and legacy NumPy 2.5.0 draws."""
import warnings
import numpy as np


def populate_random_legacy(record, fails):
    seed = 12345
    for algorithm in ("MT19937", "Philox", "SFC64", "PCG64", "PCG64DXSM"):
        cls = getattr(np.random, algorithm)
        tag = "bit_" + algorithm.lower() + "_"
        b = cls(seed)
        record(tag+"raw", b.random_raw(700 if algorithm == "MT19937" else 35))
        g = np.random.Generator(cls(seed))
        record(tag+"double", g.random(9))
        record(tag+"single", g.random(9, dtype=np.float32))
        record(tag+"i64", g.integers(-2**48, 2**48, size=13, dtype=np.int64))
        record(tag+"u16", g.integers(0,65535,size=17,dtype=np.uint16,endpoint=True))
        record(tag+"normal",g.standard_normal(9))
        state = g.bit_generator.state
        if algorithm=="MT19937":
            record(tag+"state_key",state["state"]["key"])
            record(tag+"state_pos",np.array(state["state"]["pos"],dtype=np.int64))
        else:
            record(tag+"state_has32",np.array(state["has_uint32"],dtype=np.bool_))
            record(tag+"state_cached",np.array(state["uinteger"],dtype=np.uint32))
            if algorithm=="Philox":
                record(tag+"state_counter",state["state"]["counter"])
                record(tag+"state_philoxkey",state["state"]["key"])
                record(tag+"state_buffer",state["buffer"])
                record(tag+"state_buffer_pos",np.array(state["buffer_pos"],dtype=np.int64))
            elif algorithm=="SFC64":
                record(tag+"state_sfc",state["state"]["state"])
            else:
                def words(x):return np.array([x%2**64,x//2**64],dtype=np.uint64)
                record(tag+"state_pcg",words(state["state"]["state"]))
                record(tag+"state_inc",words(state["state"]["inc"]))
        after = g.random(5, dtype=np.float32)
        record(tag+"restored",after)
        g.bit_generator.state=state
        assert np.array_equal(g.random(5,dtype=np.float32),after)
        b = cls(seed)
        for j,child in enumerate(b.spawn(2)):
            record(tag+"spawn_"+str(j),child.random_raw(8))
        record(tag+"seed_entropy",np.atleast_1d(np.array(b.seed_seq.entropy,dtype=np.uint32)))
        locked=cls(seed)
        interface=locked.ctypes
        with locked.lock:
            record(tag+"lock_u32",np.array([interface.next_uint32(interface.state) for _ in range(3)],dtype=np.uint32))
            record(tag+"lock_u64",np.array([interface.next_uint64(interface.state) for _ in range(5)],dtype=np.uint64))
            record(tag+"lock_double",np.array([interface.next_double(interface.state) for _ in range(4)],dtype=np.float64))
        record(tag+"lock_followup",locked.random_raw(5))
        if hasattr(b,"jumped"):
            b.random_raw(3)
            record(tag+"jumped",b.jumped(2).random_raw(10))
            record(tag+"jumped_original",b.random_raw(6))
        if hasattr(b,"advance"):
            b=cls(seed)
            b.random_raw(3)
            b.advance(2**70+12345)
            record(tag+"advance",b.random_raw(11))
    explicit=np.random.Philox(counter=np.array([2**64-1,7,0,0],dtype=np.uint64),key=[123,456])
    record("bit_philox_explicit",explicit.random_raw(12))
    explicit.advance(2**192+2**128+17)
    record("bit_philox_advance256",explicit.random_raw(7))
    s=np.random.SeedSequence([1,2,3,4]);child=s.spawn(3)[1]
    for label,obj in (("parent",s),("child",child)):
        tag="bit_seed_"+label+"_"
        record(tag+"entropy",np.array(obj.entropy,dtype=np.uint32))
        record(tag+"pool",obj.pool)
        record(tag+"pool_size",np.array(obj.pool_size,dtype=np.int64))
        record(tag+"spawn_key",np.array(obj.spawn_key,dtype=np.uint32))
        record(tag+"children",np.array(obj.n_children_spawned,dtype=np.uint64))
        assert obj.state["spawn_key"]==obj.spawn_key

    # Independently seeded tests validate the exact legacy algorithms, not the
    # modern Generator's statistically equivalent distributions.
    tests = {
        "random":("random",()), "random_sample":("random_sample",()),
        "standard_normal":("standard_normal",()),"normal":("normal",(1.25,2.5)),
        "uniform":("uniform",(-3.,4.)), "standard_exponential":("standard_exponential",()),
        "exponential":("exponential",(2.3,)), "gamma":("gamma",(2.5,1.3)),
        "gamma_small":("gamma",(.24,2.)), "standard_gamma":("standard_gamma",(.7,)),
        "beta":("beta",(2.5,3.2)), "beta_small":("beta",(.3,.4)),
        "chisquare":("chisquare",(3.3,)), "f":("f",(5.,2.5)),
        "standard_cauchy":("standard_cauchy",()), "standard_t":("standard_t",(4.2,)),
        "pareto":("pareto",(2.2,)), "weibull":("weibull",(1.7,)), "power":("power",(2.3,)),
        "laplace":("laplace",(1.,2.)), "gumbel":("gumbel",(-1.,.8)), "logistic":("logistic",(1.2,.7)),
        "lognormal":("lognormal",(-.2,.7)), "rayleigh":("rayleigh",(2.,)),
        "poisson":("poisson",(3.5,)), "poisson_large":("poisson",(35.,)),
        "negative_binomial":("negative_binomial",(4.1,.3)), "binomial":("binomial",(20,.3)),
        "binomial_large":("binomial",(400,.43)), "geometric":("geometric",(.16,)),
        "geometric_search":("geometric",(.65,)), "triangular":("triangular",(-2.,1.,5.)),
        "wald":("wald",(1.7,2.4)), "logseries":("logseries",(.78,)), "zipf":("zipf",(2.1,)),
        "noncentral_chisquare":("noncentral_chisquare",(3.5,2.1)),
        "noncentral_chisquare_lowdf":("noncentral_chisquare",(.5,2.)),
        "noncentral_f":("noncentral_f",(4.,3.,1.8)), "vonmises":("vonmises",(.3,3.)),
        "hypergeometric":("hypergeometric",(19,30,7)),
        "hypergeometric_large":("hypergeometric",(40,67,62)),
        "multinomial":("multinomial",(20,[.2,.35,.45])),
        "dirichlet":("dirichlet",([.5,2.,3.],)),
        "dirichlet_small":("dirichlet",([.05,.08,.09],)),
        "multivariate_normal":("multivariate_normal",([1.,2.],[[4.,0.],[0.,1.]])),
        "randint":("randint",(-7,29)), "randint_big":("randint",(-2**48,2**48)),
        "random_integers":("random_integers",(-2,6)),
        "tomaxint":("tomaxint",()),
        "choice":("choice",(7,)),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",DeprecationWarning)
        for name,(fn,args) in tests.items():
            r=np.random.RandomState(seed)
            record("legacy_"+name,getattr(r,fn)(*args,size=(2,3)))
            if hasattr(np.random,fn):
                np.random.seed(seed)
                record("legacy_global_"+name,getattr(np.random,fn)(*args,size=(2,3)))
        for name in ("rand","randn","ranf","sample"):
            np.random.seed(seed)
            data=getattr(np.random,name)(2,3) if name in ("rand","randn") else getattr(np.random,name)((2,3))
            record("legacy_global_"+name,data)
        for name in ("rand","randn"):
            record("legacy_"+name,getattr(np.random.RandomState(seed),name)(2,3))
        for label,kwargs in (("without",{"replace":False}),("weighted",{"p":[.1,.2,.3,.4]}),("weighted_without",{"p":[.1,.2,.3,.4],"replace":False})):
            population=4 if "weighted" in label else 10
            record("legacy_choice_"+label,np.random.RandomState(seed).choice(population,3,**kwargs))
        for d in ("int8","uint16","int32","int64"):
            record("legacy_randint_"+d,np.random.RandomState(seed).randint(0,93,size=19,dtype=d))
        r=np.random.RandomState(seed)
        record("legacy_bytes",np.frombuffer(r.bytes(23),dtype=np.uint8))
        np.random.seed(seed);record("legacy_global_bytes",np.frombuffer(np.random.bytes(23),dtype=np.uint8))
        for label,rs in (("",np.random.RandomState(seed)),("global_",np.random)):
            if label:np.random.seed(seed)
            record("legacy_"+label+"permutation",rs.permutation(12))
            values=np.arange(15).reshape(5,3);rs.shuffle(values)
            record("legacy_"+label+"shuffle",values)
        for label,rs in (("",np.random.RandomState(seed)),("global_",np.random)):
            if label:np.random.seed(seed)
            rs.standard_normal(3)
            state=rs.get_state()
            record("legacy_"+label+"state_key",state[1])
            record("legacy_"+label+"state_pos",np.array(state[2],dtype=np.int64))
            record("legacy_"+label+"state_has_gauss",np.array(state[3],dtype=bool))
            record("legacy_"+label+"state_gauss",np.array(state[4]))
            record("legacy_"+label+"state_future",rs.standard_normal(7))
            rs.set_state(state);record("legacy_"+label+"state_restored",rs.standard_normal(7))
            rs.seed(54321);record("legacy_"+label+"reseed",rs.random_sample(8))
        r=np.random.RandomState([1,2,3,4])
        record("legacy_array_seed",r.random_sample(11))
        r=np.random.RandomState(seed)
        r.binomial(0,.5,8)
        record("legacy_binomial_zero_followup",r.random_sample(4))
        r=np.random.RandomState(seed)
        r.binomial(20,0.,8)
        record("legacy_binomial_pzero_followup",r.random_sample(4))
    fails("legacy_invalid_seed",lambda:np.random.RandomState(-1))
    fails("legacy_invalid_seed_array",lambda:np.random.RandomState([]))
    fails("legacy_invalid_randint",lambda:np.random.RandomState(seed).randint(5,2))
    fails("legacy_invalid_choice",lambda:np.random.RandomState(seed).choice(2,3,replace=False))
