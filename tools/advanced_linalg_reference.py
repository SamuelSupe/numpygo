"""Graded complex SVD, pseudoinverses, raw QR and dtype norm limits."""
import numpy as np
import warnings

def populate_linalg_advanced(record, fails):
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        matrices = {
            "graded": np.diag([1+1j,1e-200j,3e-100+4e-100j]),
            "singular": np.diag([1+1j,0j]),
            "triangular": np.array([[1+2j,1e-180*(2+3j)], [0,2e-180j]],complex),
            "small": 1e-200*np.array([[1+2j,3-1j],[2+1j,4+3j],[0-1j,-1+2j]],complex),
            "large": 1e200*np.array([[1+2j,3-1j],[2+1j,4+3j],[0-1j,-1+2j]],complex),
            "wide": 1e-180*np.array([[1+2j,3-1j,2-2j],[2+1j,4+3j,1+3j]],complex),
        }
        for key,a in matrices.items():
            label="adv_linalg_"+key
            record(label+"_input",a)
            record(label+"_s",np.linalg.svdvals(a))
            record(label+"_pinv",np.linalg.pinv(a,rcond=0))
            if key in ("graded","triangular"):
                record(label+"_inv",np.linalg.inv(a))
            h,tau=np.linalg.qr(a,mode="raw")
            record(label+"_raw_h",h);record(label+"_raw_tau",tau)
        for dt,large in (("float16",256.),("float32",1e25),("float64",1e200),
                         ("complex64",1e25),("complex128",1e200)):
            for key,val in (("tiny",1/large),("large",large)):
                a=np.array([[val,val],[val*0.5,val]],dtype=dt)
                label=f"adv_norm_{dt}_{key}"
                record(label+"_input",a)
                record(label+"_flat",np.linalg.norm(a))
                record(label+"_rows",np.linalg.norm(a,axis=1))
                record(label+"_p3",np.linalg.norm(a,axis=1,ord=3))
                record(label+"_fro",np.linalg.norm(a,ord="fro",axis=(0,1)))
                record(label+"_vector",np.linalg.vector_norm(a,axis=(0,1),keepdims=True))
                record(label+"_matrix",np.linalg.matrix_norm(a,keepdims=True))
