"""Targeted NumPy 2.5.0 references for SIMD ufuncs and mixed radix FFT."""
import numpy as np


def populate_vector(record, fails):
    def r(name, x):
        record("vector_" + name, x)

    rng = np.random.default_rng(9791)
    with np.errstate(all="ignore"):
        for dtype in ("float32", "float64"):
            span = 80 if dtype == "float32" else 700
            x = rng.uniform(-span, span, 1027).astype(dtype)
            x[:7] = [0., -0., np.log(2)/2, -np.log(2)/2, .1, -.1, 1.]
            r("exp_input_" + dtype, x)
            r("exp_result_" + dtype, np.exp(x))
            edge = np.resize(np.array([-np.inf, np.inf, np.nan, -0., 0., -745., -708., -700.,
                                       -104., -103., -90., 88., 89., 700., 709., 710.], dtype=dtype), 137)
            r("exp_edge_input_" + dtype, edge)
            r("exp_edge_result_" + dtype, np.exp(edge))
            a = np.resize(np.array([0., -0., np.nan, 3., np.inf, -np.inf, .25, -3.,
                                    -0., 0., np.nan, 9., -4.], dtype=dtype), 259)
            b = np.resize(np.array([-0., 0., 3., np.nan, -np.inf, np.inf, .25, 4.,
                                    -0., 0., np.nan, 9., -4.], dtype=dtype), 259)
            mask = np.arange(a.size) % 97 < 64
            r("extrema_a_" + dtype, a)
            r("extrema_b_" + dtype, b)
            r("mask_" + dtype, mask)
            for name in ("minimum", "maximum"):
                f = getattr(np,name)
                r(name+"_"+dtype, f(a,b))
                out = np.full(261, -7, dtype=dtype)
                f(a,b,out=out[1:-1],where=mask)
                r(name+"_masked_"+dtype,out)
        for dtype in ("complex64", "complex128"):
            a = (rng.uniform(-10, 10, 259)+1j*rng.uniform(-10, 10, 259)).astype(dtype)
            b = (rng.uniform(-5, 5, 259)+1j*rng.uniform(-5, 5, 259)).astype(dtype)
            b[:8] = [0, 1j, 3, -2j, 2+2j, -3+3j, 1-4j, 3-1j]
            r("complex_a_" + dtype, a)
            r("complex_b_" + dtype, b)
            r("mask_" + dtype, mask)
            for name in ("add","subtract","multiply","divide"):
                f = getattr(np,name)
                r(name+"_"+dtype,f(a,b))
                out = np.full(261,-7,dtype=dtype)
                f(a,b,out=out[1:-1],where=mask)
                r(name+"_masked_"+dtype,out)
        for dtype in ("float32","float64"):
            cd = "complex64" if dtype=="float32" else "complex128"
            a = rng.uniform(-1,1,(3,1025)).astype(dtype)
            c = (a+1j*rng.uniform(-1,1,a.shape)).astype(cd)
            r("fft_real_input_"+dtype,a);r("fft_complex_input_"+dtype,c)
            for n,norm in ((90,"ortho"),(150,"backward"),(729,"forward"),(1000,"backward"),(1500,"ortho"),(1536,"forward")):
                tag=f"{dtype}_{n}_{norm}"
                z=np.fft.rfft(a,n=n,norm=norm)
                r("rfft_"+tag,z)
                r("irfft_"+tag,np.fft.irfft(z,n=n,norm=norm))
                r("fft_"+tag,np.fft.fft(c,n=n,norm=norm))
                r("ifft_"+tag,np.fft.ifft(c,n=n,norm=norm))
            r("fft_real_"+dtype,np.fft.fft(a,n=150))
            r("fft_axis0_"+dtype,np.fft.fft(c[:,:7],n=150,axis=0,norm="ortho"))
            r("rfft_reverse_"+dtype,np.fft.rfft(a[:,::-1],n=150,norm="forward"))
            r("fft_alias_"+dtype,np.fft.fft(c[:,:150],norm="ortho"))


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    assert np.__version__ == "2.5.0"
    arrays = {}

    def record(name, value):
        value = np.asarray(value)
        arrays[name] = {"dtype":str(value.dtype),"shape":list(value.shape),"strides":list(value.strides),
                        "values":[[str(int(v)),"0"] if value.dtype.kind in "biu" else
                                  [repr(float(v.real)),repr(float(v.imag))] for v in value.flat]}

    populate_vector(record, None)
    Path(sys.argv[1]).write_text(json.dumps({"version":np.__version__,"arrays":arrays,"errors":{}},
                                           indent=2,allow_nan=False)+"\n")
    print("generated",len(arrays),"vector references")
