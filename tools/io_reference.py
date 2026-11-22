"""Generate native NumPy numeric file fixtures."""
from pathlib import Path

import numpy as np


def populate_io(record):
    dest = Path(__file__).resolve().parents[1] / "testdata/numpy"
    dest.mkdir(parents=True, exist_ok=True)
    arrays = {
        "bool":np.array([[True, False],[False,True]]),
        "int8":np.array([-128,-1,0,127],dtype=np.int8),
        "int16":np.array([-32768,12345,32767],dtype=np.int16),
        "int32":np.array([-2000000000,1,2000000000],dtype=np.int32),
        "int64":np.array([-2**63,2**53+1,2**63-1],dtype=np.int64),
        "uint8":np.array([0,1,255],dtype=np.uint8),
        "uint16":np.array([0,1,65535],dtype=np.uint16),
        "uint32":np.array([0,1,2**32-1],dtype=np.uint32),
        "uint64":np.array([0,2**53+1,2**64-1],dtype=np.uint64),
        "float16":np.array([0.,-0.,.1,-.25,np.inf,np.nan],dtype=np.float16),
        "float32":np.array([0.25,1.5,-2.75],dtype=np.float32),
        "float64":np.array([[1.,2.],[3.,4.]]),
        "complex64":np.array([1+2j,3-4j],dtype=np.complex64),
        "complex128":np.array([1+2j,3-4j],dtype=np.complex128),
        "fortran":np.asfortranarray(np.arange(6.).reshape(2,3)),
        "bigendian":np.array([[1,-2],[300,400]],dtype=">i4"),
        "scalar":np.array(3.25),
        "empty":np.zeros((2,0,3),dtype=np.float32),
        "version2":np.array([11,22,33],dtype=np.int16),
        "version3":np.array([1+2j,-3+4j],dtype=np.complex128),
    }
    for name,a in arrays.items():
        record("io_npy_"+name,a.astype(a.dtype.newbyteorder("="),copy=False))
        path=dest/(name+".npy")
        if name in ("version2","version3"):
            with path.open("wb") as f:
                np.lib.format.write_array(f,a,version=(int(name[-1]),0),allow_pickle=False)
        else:
            np.save(path,a,allow_pickle=False)
    np.savez(dest/"stored.npz",a=arrays["uint64"],b=arrays["fortran"])
    np.savez_compressed(dest/"compressed.npz",a=arrays["uint64"],b=arrays["fortran"])
    np.save(dest/"structured.npy",np.array([(1,2.)],dtype=[("x","i4"),("y","f8")]),allow_pickle=False)
