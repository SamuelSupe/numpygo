"""Advanced linear algebra and typed contraction oracles."""
import numpy as np


def populate_matrix(record, fails):
    for dtype in ("float32", "float64", "complex64", "complex128"):
        a = np.array([[[3, 1, 0], [1, -2, 0], [0, 0, 0]],
                      [[2, 0, 1], [0, 4, 0], [1, 0, -1]]], dtype=dtype)
        if a.dtype.kind == "c":
            a[0, 0, 1] += 2j
            a[0, 1, 0] -= 2j
            a[1, 0, 2] -= 1j
            a[1, 2, 0] += 1j
        prefix = "param_linalg_hermitian_" + dtype
        record(prefix+"_input", a)
        u, s, vh = np.linalg.svd(a, hermitian=True)
        for name, v in (("u",u), ("s",s), ("vh",vh)):
            record(prefix+"_"+name, v)
        record(prefix+"_values", np.linalg.svd(a, compute_uv=False, hermitian=True))
        cut = np.array([.05, .5], dtype=np.float32)
        record(prefix+"_pinv", np.linalg.pinv(a, rcond=cut, hermitian=True))
        record(prefix+"_pinv_default", np.linalg.pinv(a, rtol=None, hermitian=True))
        record(prefix+"_pinv_relative", np.linalg.pinv(a, rtol=cut, hermitian=True))
        record(prefix+"_rank", np.linalg.matrix_rank(a, hermitian=True))
        record(prefix+"_rank_relative", np.linalg.matrix_rank(a, rtol=cut, hermitian=True))
        record(prefix+"_rank_expand", np.linalg.matrix_rank(a, tol=np.array([[.01],[3.]]), hermitian=True))
        q = np.array([[3,-1,2],[1,4,0],[-2,2,5],[6,1,-1]], dtype=dtype)
        if q.dtype.kind == "c":
            q += 1j*np.array([[1,0,-2],[0,1,3],[2,1,0],[1,-2,3]])
        for kind, x in (("tall",q),("wide",q.T),("batch",np.stack((q,q[::-1]))),("empty",q[:0])):
            p = f"param_linalg_raw_{dtype}_{kind}"
            record(p+"_input",x)
            h,tau = np.linalg.qr(x,mode="raw")
            record(p+"_h",h)
            record(p+"_tau",tau)
    a = np.eye(2)
    fails("param_linalg_pinv_conflict",lambda: np.linalg.pinv(a,rcond=.1,rtol=.1))
    fails("param_linalg_rank_conflict",lambda: np.linalg.matrix_rank(a,tol=.1,rtol=.1))
    fails("param_linalg_pinv_shape",lambda: np.linalg.pinv(np.stack((a,a)),rcond=np.ones(3)))
    fails("param_linalg_hermitian_shape",lambda: np.linalg.svd(np.ones((2,3)),hermitian=True))
    fails("param_linalg_raw_half",lambda: np.linalg.qr(np.eye(2,dtype=np.float16),mode="raw"))
    record("param_linalg_pinv_empty",np.linalg.pinv(np.empty((0,3),dtype=np.int16)))

    for dtype in ("float16","float32","float64","complex64","complex128","int64","bool"):
        prefix = "param_product_"+dtype
        x = (np.arange(24).reshape(2,1,3,4)%7-3).astype(dtype)
        y = (np.arange(16).reshape(1,2,4,2)%5-2).astype(dtype)
        if x.dtype.kind == "c":
            x += 1j*x[::-1]
            y -= 1j*y[:,::-1]
        record(prefix+"_a",x)
        record(prefix+"_b",y)
        record(prefix+"_matmul",np.matmul(x[...,::-1],y[...,::-1,:]))
        a = (np.arange(24).reshape(2,3,4)%7-3).astype(dtype)
        b = (np.arange(60).reshape(4,3,5)%11-5).astype(dtype)
        if a.dtype.kind == "c":
            a += 1j*a[::-1]
            b += 1j*b[::-1]
        record(prefix+"_ta",a)
        record(prefix+"_tb",b)
        a = a[:,::-1,::-1]
        b = b[::-1,::-1,:]
        record(prefix+"_tensordot",np.tensordot(a,b,axes=([2,1],[0,1])))
        record(prefix+"_einsum",np.einsum("ijk,kjl->li",a,b,optimize=True))
    with np.errstate(all="ignore"):
        for dtype in ("float32","float64","complex64","complex128"):
            p = "param_product_nonfinite_"+dtype
            a = np.array([[0,1],[2,0]],dtype=dtype)
            b = np.array([[np.inf,1],[-np.inf,np.nan]],dtype=dtype)
            record(p+"_a",a)
            record(p+"_b",b)
            record(p+"_matmul",a@b)
            record(p+"_tensordot",np.tensordot(a,b,axes=1))
    a=np.arange(12,dtype=np.float32).reshape(3,4)
    b=np.arange(20,dtype=np.complex64).reshape(4,5)
    b+=1j*b[::-1]
    record("param_product_mixed_a",a)
    record("param_product_mixed_b",b)
    record("param_product_mixed_matmul",a@b)
    record("param_product_mixed_dot",np.dot(a,b))
    record("param_product_mixed_inner",np.inner(a,b.T))
    record("param_product_mixed_einsum",np.einsum("ik,kj->ji",a,b,optimize=True))
    record("param_product_zero_tensordot",np.tensordot(np.zeros((3,0)),np.zeros((0,5)),axes=1))
    record("param_product_empty_tensordot",np.tensordot(np.zeros((0,3)),np.zeros((3,5)),axes=1))
    record("param_product_broadcast_einsum",np.einsum("bij,bjk->bik",a[None,:,:],
           np.stack((b,b*2)),optimize=True))
