"""Reference checks for native complex reductions and packed SIMD kernels."""
import numpy as np


def populate_spectral(record, fails):
    def r(name, value):
        record("spectral_"+name, value)
    rng=np.random.default_rng(872)
    for dtype in ("complex64","complex128"):
        a=rng.uniform(-1,1,(2,33,33))+1j*rng.uniform(-1,1,(2,33,33))
        a=((a+a.conj().swapaxes(-1,-2))/2).astype(dtype)
        r("herm_"+dtype,a)
        b=a[:,::-1,::-1]
        w,v=np.linalg.eigh(b)
        r("herm_w_"+dtype,w)
        r("herm_v_"+dtype,v)
        r("herm_vals_"+dtype,np.linalg.eigvalsh(b))
        u,s,vh=np.linalg.svd(b,hermitian=True)
        r("herm_svd_u_"+dtype,u)
        r("herm_svd_s_"+dtype,s)
        r("herm_svd_vh_"+dtype,vh)
        # Repeated blocks, ignored lower triangle and imaginary diagonal.
        a=np.kron(np.eye(12),np.array([[2,1j],[-1j,4]])).astype(dtype)
        bad=a.copy()
        bad[np.tril_indices(24,-1)]=np.nan+1j*np.nan
        bad.flat[::25]+=4j
        r("upper_"+dtype,bad)
        r("upper_clean_"+dtype,a)
        w,v=np.linalg.eigh(bad,UPLO="U")
        r("upper_w_"+dtype,w)
        r("upper_v_"+dtype,v)
        r("upper_vals_"+dtype,np.linalg.eigvalsh(bad,UPLO="U"))
    a=rng.uniform(-1,1,(25,25))+1j*rng.uniform(-1,1,(25,25))
    a=a+a.conj().T
    for tag,scale in (("small",1e-200),("large",1e200)):
        x=a*scale
        r("herm_"+tag,x)
        r("herm_"+tag+"_vals",np.linalg.eigvalsh(x))
    for tag,x in (("zero",np.zeros((16,16),dtype=np.complex128)),
                  ("diag",np.diag(np.arange(16,dtype=float)+3j))):
        r("herm_"+tag,x)
        w,v=np.linalg.eigh(x)
        r("herm_"+tag+"_vals",w)
        r("herm_"+tag+"_vecs",v)
    x=np.diag(np.geomspace(1e-200,1e200,16)).astype("complex128")
    r("herm_dynamic",x)
    r("herm_dynamic_vals",np.linalg.eigvalsh(x))
    a=rng.uniform(-1,1,(97,47))+1j*rng.uniform(-1,1,(97,47))
    for dtype in ("complex64","complex128"):
        x=a.astype(dtype)
        r("svd_"+dtype,x)
        for name,b in (("tall",x[::-1]),("wide",x.T[:,::-1])):
            u,s,vh=np.linalg.svd(b,full_matrices=False)
            key="svd_"+name+"_"+dtype
            r(key+"_u",u);r(key+"_s",s);r(key+"_vh",vh)
            r(key+"_vals",np.linalg.svd(b,compute_uv=False))
    # A diagonal graded matrix makes relative accuracy observable.
    for tag,scale in (("small",1e-200),("large",1e200)):
        x=a[:41,:31]*scale
        r("svd_"+tag,x)
        r("svd_"+tag+"_vals",np.linalg.svd(x,compute_uv=False))
    x=np.zeros((21,21),dtype=np.complex128)
    np.fill_diagonal(x,np.geomspace(1,1e-220,21)*(1+1j))
    r("svd_graded",x)
    r("svd_graded_vals",np.linalg.svd(x,compute_uv=False))
    for dtype in ("float32","float64"):
        # Non-aligned starting offset, a vector tail and signed zero,
        # infinities, NaN, subnormal and maximum finite values.
        f=np.finfo(dtype)
        a=np.resize(np.array([.1,-.2,3.7,-0.,0.,np.inf,-np.inf,np.nan,
                              f.tiny,f.smallest_subnormal,-f.max],dtype=dtype),72)
        b=np.resize(np.array([2.3,.7,-1.1,0.,-0.,0.,np.inf,1.,.25,2.,2.],dtype=dtype),72)
        r("simd_a_"+dtype,a)
        r("simd_b_"+dtype,b)
        with np.errstate(all="ignore"):
            for name,fn in (("add",np.add),("sub",np.subtract),("mul",np.multiply),("div",np.divide)):
                r("simd_"+name+"_"+dtype,fn(a[1:70],b[2:71]))
    # Nontrivial complex phases and negative strides exercising the radix
    # path and the Bluestein convolution's inverse butterflies.
    a=(rng.uniform(-1,1,(3,1024))+1j*rng.uniform(-1,1,(3,1024))).astype("complex128")
    r("fft_input",a)
    r("fft",np.fft.fft(a[:,::-1],norm="ortho"))
    r("ifft",np.fft.ifft(a[:,::-1],norm="forward"))
    r("fft_axis0",np.fft.fft(a.T,axis=0))
    r("prime_fft",np.fft.fft(a[:,:509]))
    r("prime_ifft",np.fft.ifft(a[:,:509]))
    x=a.real.astype("float64")
    r("rfft_input",x)
    r("rfft",np.fft.rfft(x,n=509,norm="ortho"))
    z=np.fft.rfft(x,n=509,norm="ortho")
    r("irfft",np.fft.irfft(z,n=509,norm="ortho"))
