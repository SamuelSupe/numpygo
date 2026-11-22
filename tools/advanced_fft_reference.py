"""Fourier shape, normalization, output-buffer and dtype contracts."""
import numpy as np
import warnings

TYPES = ("bool", "int8", "int16", "int32", "int64", "uint8", "uint16",
         "uint32", "uint64", "float16", "float32", "float64", "complex64", "complex128")
ONE = ("fft", "ifft", "rfft", "irfft", "hfft", "ihfft")
MULTI = ("fftn", "ifftn", "rfftn", "irfftn", "fft2", "ifft2", "rfft2", "irfft2")

def populate_fft(record, fails):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proto=np.arange(24,dtype=np.int64).reshape(2,3,4)
        for key,a in (("c",proto),("f",np.asfortranarray(proto)),
                      ("transpose",proto.transpose(1,2,0)),("reverse",proto[:,::-1,:]),
                      ("scalar",np.array(2)),("empty",np.empty((2,0,3),dtype=np.int64))):
            for order in ("C","F","A","K"):
                for sk,shape in (("same",a.shape),("resize",(3,5,2)),("rank",(5,2)),("scalar",())):
                    label=f"adv_like_{key}_{order}_{sk}"
                    empty=np.empty_like(a,dtype=np.int16,order=order,shape=shape)
                    empty.fill(4)
                    record(label+"_empty",empty)
                    record(label+"_zeros",np.zeros_like(a,dtype=np.int16,order=order,shape=shape))
                    record(label+"_ones",np.ones_like(a,dtype=np.int16,order=order,shape=shape))
                    record(label+"_full",np.full_like(a,7,dtype=np.int16,order=order,shape=shape))
            record(f"adv_like_{key}_default_zeros",np.zeros_like(a))
            record(f"adv_like_{key}_default_ones",np.ones_like(a))
            record(f"adv_like_{key}_default_full",np.full_like(a,7))
        for dt in TYPES:
            base = (np.arange(24).reshape(2, 3, 4) % 7 - 3).astype(np.int64)
            if dt.startswith("uint"):
                base += 3
            if dt.startswith("complex"):
                base = base + 1j * (base[::-1] % 3 - 1)
            base = base.astype(dt)
            record("adv_fft_input_"+dt, base)
            a = base.transpose(1, 2, 0)[::-1]
            for name in ONE:
                for norm in ("backward", "forward", "ortho"):
                    for ax in (1, -1):
                        for n in (None, 1, 5):
                            label = f"adv_fft_{name}_{dt}_{norm}_{ax}_{n}"
                            call = lambda: getattr(np.fft, name)(a, n=n, axis=ax, norm=norm)
                            if np.iscomplexobj(a) and name in ("rfft", "ihfft"):
                                fails(label, call)
                            else:
                                record(label, call())
        for dt in ("float32", "float64", "complex64"):
            a = np.asarray((np.arange(24).reshape(2,3,4)%7-3).astype(dt))
            if np.iscomplexobj(a):
                a += 1j*(np.arange(24).reshape(2,3,4)%3)
            record("adv_fft_multi_input_"+dt,a)
            for name in MULTI:
                if np.iscomplexobj(a) and name in ("rfft2","rfftn"):
                    continue
                for key,axes in (("repeat",(1,1)),("outer",(0,-1)),("reverse",(2,0))):
                    for sk,s in (("default",None),("full",(-1,-1)),("resize",(5,7))):
                        for norm in ("backward","forward","ortho"):
                            record(f"adv_fft_{name}_{dt}_{key}_{sk}_{norm}",getattr(np.fft,name)(a,s=s,axes=axes,norm=norm))
        a = np.arange(12., dtype=np.float64).reshape(3,4)-3
        record("adv_fft_out_input",a)
        for name in ONE:
            fn=getattr(np.fft,name)
            for norm in ("backward","forward","ortho"):
                for n in (4,7):
                    orig=fn(a,n=n,norm=norm)
                    types=("float16","float32","float64","complex128") if orig.dtype.kind=="f" else ("complex64","complex128")
                    for dt in types:
                        shape=list(orig.shape);shape[-1]*=2
                        backing=np.full(shape,-99,dtype=dt)
                        out=backing[...,::-2]
                        label=f"adv_fft_out_{name}_{norm}_{n}_{dt}"
                        assert fn(a,n=n,norm=norm,out=out) is out
                        record(label,out)
                        record(label+"_backing",backing)
        for dt in ("float16","float32","complex64"):
            a=np.array([1.25,-2.,3.5,4.,-1.],dtype=dt)
            if dt=="complex64":
                a+=1j*np.array([.5,-1.,2.,.25,1.5],dtype=np.float32)
            record("adv_fft_wide_input_"+dt,a)
            for name in ONE:
                if dt=="complex64" and name in ("rfft","ihfft"):
                    continue
                fn=getattr(np.fft,name)
                orig=fn(a,n=7,norm="ortho")
                out=np.empty(orig.shape,dtype=np.float64 if orig.dtype.kind=="f" else np.complex128)
                record(f"adv_fft_wide_{name}_{dt}", fn(a,n=7,norm="ortho",out=out))
        a=np.arange(12.,dtype=np.float64).reshape(3,4)
        for name in MULTI:
            fn=getattr(np.fft,name)
            s=(5,7) if name.startswith("irfft") else (3,7)
            out=np.empty(fn(a,s=s,axes=(0,1)).shape,dtype=np.float64 if name.startswith("irfft") else np.complex128,order="F")
            assert fn(a,s=s,axes=(0,1),out=out) is out
            record("adv_fft_ndout_"+name,out)
        a=np.arange(4.,dtype=np.float64).reshape(1,4)
        record("adv_fft_broadcast",np.fft.fft(a,out=np.empty((3,4),complex)))
        v=np.array([[1+2j,3-4j,5+1j,2-1j]])
        record("adv_fft_alias_input",v.copy())
        record("adv_fft_alias",np.fft.fft(v,out=v))
        fails("adv_fft_out_shape",lambda:np.fft.fft(a,out=np.empty((1,3),complex)))
        fails("adv_fft_out_rank",lambda:np.fft.fft(a,out=np.empty((1,1,4),complex)))
        fails("adv_fft_out_bad_dtype",lambda:np.fft.fft(a,out=np.empty(a.shape,float)))
        fails("adv_fft_out_readonly",lambda:np.fft.fft(a,out=np.broadcast_to(1+1j,a.shape)))
        fails("adv_fft_ndout_resize",lambda:np.fft.fftn(np.ones((3,4)),s=(5,7),axes=(0,1),out=np.empty((5,7),complex)))
        fails("adv_fft_empty_realaxes",lambda:np.fft.rfftn(a,axes=()))
        fails("adv_fft_inverse_empty_realaxes",lambda:np.fft.irfftn(a,axes=()))
        for n in (1,2,5,6):
            for k,d in (("normal",.25),("negative",-.5),("nan",float("nan")),("inf",float("inf"))):
                record(f"adv_fft_freq_{n}_{k}",np.fft.fftfreq(n,d))
                record(f"adv_fft_rfreq_{n}_{k}",np.fft.rfftfreq(n,d))
