"""Pinned NumPy results for decomposition and reduction fast paths."""
import numpy as np


def populate_decomposition(record, fails):
    def r(name,value):
        record("decomp_"+name,value)
    rng=np.random.default_rng(426)

    # A non-multiple-of-block-sized tall matrix, its non-contiguous wide
    # view, and all standard NumPy linalg precisions.
    re=rng.uniform(-1,1,(67,35))
    im=rng.uniform(-1,1,re.shape)
    for dtype in ("float32","float64","complex64","complex128"):
        a=(re+1j*im if "complex" in dtype else re).astype(dtype)
        r("qr_input_"+dtype,a)
        for layout,x in (("tall",a[::-1]),("wide",a.T[:,::-1])):
            for mode in ("reduced","complete"):
                q,rr=np.linalg.qr(x,mode=mode)
                r(f"qr_{layout}_{mode}_{dtype}_q",q)
                r(f"qr_{layout}_{mode}_{dtype}_r",rr)
            r(f"qr_{layout}_r_{dtype}",np.linalg.qr(x,mode="r"))
        h,tau=np.linalg.qr(a,mode="raw")
        r("qr_raw_"+dtype+"_h",h);r("qr_raw_"+dtype+"_tau",tau)
    for name,sh in (("empty_rows",(0,3)),("empty_cols",(5,0))):
        for dtype in ("float64","complex128"):
            a=np.empty(sh,dtype=dtype)
            q,rr=np.linalg.qr(a,mode="complete")
            r(f"qr_{name}_{dtype}_q",q);r(f"qr_{name}_{dtype}_r",rr)
    a=re.astype("complex128")+1j*im
    a[:,3]=a[:,1]
    r("qr_rankdef_input",a)
    q,rr=np.linalg.qr(a,mode="reduced")
    r("qr_rankdef_q",q);r("qr_rankdef_r",rr)

    # Full rectangular U/V completion, reduced vectors and values only.
    re=rng.uniform(-1,1,(45,35))
    a=re+1j*rng.uniform(-1,1,re.shape)
    for dtype in ("complex64","complex128"):
        x=a.astype(dtype)
        r("svd_input_"+dtype,x)
        for layout,v in (("tall",x[::-1]),("wide",x.T[:,::-1])):
            for full in (False,True):
                u,s,h=np.linalg.svd(v,full_matrices=full)
                tag=f"svd_{layout}_{'full' if full else 'reduced'}_{dtype}"
                r(tag+"_u",u);r(tag+"_s",s);r(tag+"_vh",h)
            r(f"svd_{layout}_values_{dtype}",np.linalg.svd(v,compute_uv=False))
    z=np.zeros((40,33),dtype="complex128")
    z[:,:11]=a[:40,:11]
    r("svd_zero_input",z)
    u,s,h=np.linalg.svd(z,full_matrices=True)
    r("svd_zero_u",u);r("svd_zero_s",s);r("svd_zero_vh",h)
    r("svd_zero_values",np.linalg.svd(z,compute_uv=False))
    graded=np.diag(np.geomspace(1.,1e-200,33)).astype("complex128")
    r("svd_graded_input",graded)
    r("svd_graded_values",np.linalg.svd(graded,compute_uv=False))

    re=rng.uniform(-.5,.5,(64,64))
    z=re+1j*rng.uniform(-.5,.5,re.shape)
    for dtype in ("complex64","complex128"):
        a=z.astype(dtype)
        r("eig_input_"+dtype,a)
        e,v=np.linalg.eig(a.T)
        r("eig_values_"+dtype,e);r("eig_vectors_"+dtype,v)
        r("eig_only_"+dtype,np.linalg.eigvals(a.T))
    r("real_eig_input",re)
    r("real_eig_only",np.linalg.eigvals(re))
    sym=(re+re.T)/2
    r("real_sym_input",sym)
    r("real_sym_eig_only",np.linalg.eigvals(sym))
    r("real_sym_eigvalsh",np.linalg.eigvalsh(sym))
    for dtype in ("float32","float64","int64"):
        a=np.array([[2,1,0],[1,3,1],[0,1,4]],dtype=dtype)
        r("real_eig_dtype_input_"+dtype,a)
        e,v=np.linalg.eig(a)
        r("real_eig_dtype_values_"+dtype,e);r("real_eig_dtype_vectors_"+dtype,v)
        r("real_eig_dtype_only_"+dtype,np.linalg.eigvals(a))
        r("real_eig_empty_"+dtype,np.linalg.eigvals(np.empty((0,0),dtype=dtype)))

    # The polynomial APIs retain real output for all-real roots even
    # though NumPy 2.5 linalg.eigvals itself always returns complex.
    for dtype in ("float32","float64","int64","complex64","complex128"):
        for name,coeffs in (("real",[1,0,-1]),("complex",[1,0,1])):
            if "complex" in dtype and name=="complex":
                # Distinct real parts avoid sort ordering depending on
                # eigensolver rounding of mathematically tied roots.
                coeffs=[1,1-3j,-4]
            a=np.array(coeffs,dtype=dtype)
            tag=f"poly_roots_{name}_{dtype}"
            r(tag+"_input",a)
            r(tag+"_legacy",np.sort(np.roots(a)))
            r(tag+"_power",np.polynomial.polynomial.polyroots(a[::-1]))

    for dtype in ("complex64","complex128"):
        a=((z+z.conj().T)/2).astype(dtype)
        r("hermitian_input_"+dtype,a)
        s,v=np.linalg.eigh(a)
        r("eigh_values_"+dtype,s);r("eigh_vectors_"+dtype,v)
        r("eigvalsh_"+dtype,np.linalg.eigvalsh(a))
        r("svd_hermitian_values_"+dtype,np.linalg.svd(a,compute_uv=False,hermitian=True))
        # The invalid opposite triangle must not leak into the only-values
        # decomposition, even when values are repeated.
        block=np.kron(np.eye(20),np.array([[2.,1.+2j],[1.-2j,5.]]))
        bad=block.astype(dtype)
        bad[np.tril_indices(40,-1)]=13+17j
        r("hermitian_upper_input_"+dtype,bad)
        r("hermitian_upper_values_"+dtype,np.linalg.eigvalsh(bad,UPLO="U"))
    bad=np.eye(33,dtype="complex128");bad[0,0]=np.nan
    r("nonfinite_input",bad)
    fails("decomp_eig_nonfinite",lambda:np.linalg.eigvals(bad))
    # NumPy eigvalsh may return NaN-containing arrays rather than throw on
    # some nonfinite input; do not treat a platform-dependent result as an
    # exact finite contract.

    # Preserve existing NumPy fast-axis pairwise rounding and slow-axis
    # sequential rounding, with cancellation, reversed/broadcast strides.
    for dtype,huge in (("float32",1e8),("float64",1e16)):
        raw=np.arange(37*41).reshape(37,41)
        a=((raw%17-8)/4).astype(dtype)
        a[:,0::7]=huge
        a[:,2::7]=-huge
        r("reduce_input_"+dtype,a)
        with np.errstate(all="ignore"):
            for name,view,axis in (("all",a,None),("axis0",a,0),("axis1",a,1),
                       ("transpose",a.T,1),("reverse",a[::-1,::-1],1),
                       ("broadcast",np.broadcast_to(a[:1,:],(37,41)),0),
                       ("nonlinear",a[::2,::2],None)):
                r(f"sum_{name}_{dtype}",np.sum(view,axis=axis))
                r(f"mean_{name}_{dtype}",np.mean(view,axis=axis))
            r("sum_noaxes_"+dtype,np.sum(a,axis=()))
            r("mean_empty_"+dtype,np.mean(np.empty((3,0),dtype=dtype),axis=1))
            r("sum_empty_"+dtype,np.sum(np.empty((3,0),dtype=dtype),axis=1))
            x=np.array([[np.inf,-np.inf,2.],[-0.,0.,-0.],[np.nan,1.,2.]],dtype=dtype)
            r("reduce_special_input_"+dtype,x)
            r("sum_special_"+dtype,np.sum(x,axis=1))
            r("mean_special_"+dtype,np.mean(x,axis=1))
    a=(np.arange(2*4093).reshape(2,4093)%137-68)/17.
    r("fft_prime_input",a)
    r("fft_prime",np.fft.fft(a))
    z=np.fft.rfft(a)
    r("rfft_prime",z)
    r("irfft_prime",np.fft.irfft(z,n=4093))
    c=a+1j*a[:,::-1]/7
    r("ifft_prime_input",c)
    r("ifft_prime",np.fft.ifft(c))
