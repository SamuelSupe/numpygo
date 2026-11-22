"""NumPy 2.5.0 references for register conversion and packed real FFT."""
import numpy as np


def populate_register(record, fails):
    def r(name,x):
        record("register_"+name,x)
    rng=np.random.default_rng(32050)
    with np.errstate(all="ignore"):
        a=(rng.uniform(-4,4,257)+1j*rng.uniform(-4,4,257)).astype("complex64")
        b=(rng.uniform(-2,2,257)+1j*rng.uniform(-2,2,257)).astype("complex64")
        b[:8]=[0,1j,3,-2j,2+2j,-3+3j,1-4j,3-1j]
        a[8:14]=[complex(0.,-0.),complex(-0.,0.),complex(-0.,-0.),1+1j,-1-1j,1-1j]
        mask=np.arange(257)%43<31
        r("complex_a",a); r("complex_b",b); r("complex_mask",mask)
        for name in ("multiply","divide"):
            fn=getattr(np,name)
            r(name,fn(a,b))
            # An independent binary64 intermediate oracle protects
            # cancellation precision across the register conversion.
            r(name+"_wide",fn(a.astype("complex128"),b.astype("complex128")).astype("complex64"))
            out=np.full(259,-7,dtype="complex64")
            fn(a,b,out=out[1:-1],where=mask)
            r(name+"_masked",out)
        for dtype in ("float32","float64"):
            cd="complex64" if dtype=="float32" else "complex128"
            a=rng.normal(0,1,(3,1130)).astype(dtype)
            # Independent, non-Hermitian DC and both pad and truncate.
            c=(rng.normal(0,1,(3,406))+1j*rng.normal(0,1,(3,406))).astype(cd)
            c[:,0]=[2+3j,-1-7j,.25+9j]
            r("real_input_"+dtype,a); r("spectral_input_"+dtype,c)
            for n,norm in ((75,"forward"),(81,"backward"),(125,"ortho"),(243,"backward"),
                           (625,"forward"),(729,"ortho"),(1125,"backward"),(2025,"ortho"),(3125,"forward")):
                tag=f"{dtype}_{n}_{norm}"
                z=np.fft.rfft(a,n=n,norm=norm)
                r("rfft_"+tag,z)
                r("irfft_"+tag,np.fft.irfft(c,n=n,norm=norm))
                r("roundtrip_"+tag,np.fft.irfft(z,n=n,norm=norm))
            r("rfft_reverse_"+dtype,np.fft.rfft(a[:,::-1],n=729,norm="ortho"))
            r("irfft_reverse_"+dtype,np.fft.irfft(c[:,::-1],n=729,norm="forward"))
            r("rfft_axis0_"+dtype,np.fft.rfft(a[:,:7],n=125,axis=0,norm="ortho"))
            r("irfft_axis0_"+dtype,np.fft.irfft(c[:,:7],n=125,axis=0,norm="ortho"))
            # All-constant and impulse rows exercise exact DC and very
            # small bins in the fused real butterflies.
            special=np.zeros((3,729),dtype=dtype)
            special[0]=1
            special[1,101]=1
            special[2]=np.cos(2*np.pi*17*np.arange(729)/729).astype(dtype)
            r("signal_"+dtype,special)
            z=np.fft.rfft(special)
            r("signal_fft_"+dtype,z)
            r("signal_roundtrip_"+dtype,np.fft.irfft(z,n=729))
            wide=np.zeros((3,365),dtype=cd)
            wide[:,0]=[.5+3j,-.5+2j,0]
            wide[:,1]=[1e12+.01j,-1e12+.001j,1e12-.003j]
            r("wide_spectrum_"+dtype,wide)
            r("wide_native_irfft_"+dtype,np.fft.irfft(wide,n=729))
            r("wide_irfft_"+dtype,np.fft.irfft(wide.astype("complex128"),n=729).astype(dtype))
            # Complex and even-real paths cover the new arena, digit
            # gather map, odd half-length and twiddle powers.
            cx=(a+1j*rng.normal(0,1,a.shape)).astype(cd)
            r("mixed_complex_input_"+dtype,cx)
            for n in (150,1000,1500,1536,3125):
                r(f"mixed_fft_{dtype}_{n}",np.fft.fft(cx,n=n,norm="ortho"))
                if n%2==0:
                    r(f"mixed_rfft_{dtype}_{n}",np.fft.rfft(a,n=n,norm="ortho"))
                    r(f"mixed_irfft_{dtype}_{n}",np.fft.irfft(c,n=n,norm="forward"))


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path
    assert np.__version__=="2.5.0"
    arrays={}
    def record(name,value):
        value=np.asarray(value)
        arrays[name]={"dtype":str(value.dtype),"shape":list(value.shape),"strides":list(value.strides),
                      "values":[[str(int(v)),"0"] if value.dtype.kind in "biu" else
                                [repr(float(v.real)),repr(float(v.imag))] for v in value.flat]}
    populate_register(record,None)
    Path(sys.argv[1]).write_text(json.dumps({"version":np.__version__,"arrays":arrays,"errors":{}},
                                           indent=2,allow_nan=False)+"\n")
    print("generated",len(arrays),"register/FFT references")
