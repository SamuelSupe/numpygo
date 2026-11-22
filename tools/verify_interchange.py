#!/usr/bin/env python3
"""Validate files emitted by the Go interchange tests with NumPy itself."""
import pathlib
import sys
import numpy as np
assert np.__version__ == "2.5.0"
generated=pathlib.Path(sys.argv[1])
reference=pathlib.Path(__file__).resolve().parents[1]/"testdata/numpy"

def check_values(actual,expected,label):
    # Record equality treats even matching NaT fields as unequal.
    if expected.dtype.names:
        for name in expected.dtype.names:
            check_values(actual[name],expected[name],f"{label}:{name}")
        return
    if expected.dtype.kind in "Mm":
        actual=actual.view(actual.dtype.str[0]+"i8")
        expected=expected.view(expected.dtype.str[0]+"i8")
    np.testing.assert_array_equal(actual,expected,err_msg=label)

count=0
for path in sorted(generated.glob("*.npy")):
    expected=np.load(reference/path.name,allow_pickle=False)
    actual=np.load(path,allow_pickle=False)
    assert actual.dtype==(expected.dtype if expected.dtype.names or expected.dtype.kind in "SUV" or path.stem.startswith("format_") else expected.dtype.newbyteorder("=")),path
    assert actual.shape==expected.shape,path
    check_values(actual,expected,str(path))
    if path.stem in ("fortran","structured_fortran","flex_unicode","format_endian","format_memmap"):
        assert actual.flags.f_contiguous and not actual.flags.c_contiguous
    count+=1
for path in sorted(generated.glob("*.npz")):
    with np.load(path,allow_pickle=False) as actual, np.load(reference/path.name,allow_pickle=False) as expected:
        assert set(actual.files)==set(expected.files)
        for key in actual.files:
            assert actual[key].dtype==expected[key].dtype
            check_values(actual[key],expected[key],f"{path}:{key}")
            count+=1
assert count==46,f"expected 32 NPY arrays and 14 NPZ members, got {count}"
print(f"NumPy verified {count} Go-written arrays")
