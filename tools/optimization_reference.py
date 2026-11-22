"""NumPy oracles for optimized kernels, including the non-contiguous paths."""
import numpy as np


def populate_optimization(record, fails):
    def r(name, value):
        record("optimization_" + name, value)

    for dtype in ("bool", "int8", "int16", "int32", "int64",
                  "uint8", "uint16", "uint32", "uint64"):
        lo, hi = (False, True) if dtype == "bool" else (
            np.iinfo(dtype).min, np.iinfo(dtype).max)
        a = np.array([lo, hi, 0, 1, hi, lo, 1, 0, hi, lo, hi, 1],
                     dtype=dtype).reshape(3, 4)
        r("unique_input_" + dtype, a)
        v, ix, inv, cnt = np.unique(a[::-1, ::2], return_index=True,
                                    return_inverse=True, return_counts=True)
        for label, val in zip(("values", "indices", "inverse", "counts"),
                              (v, ix, inv, cnt)):
            r("unique_" + dtype + "_" + label, val)
    for name, a in (("empty", np.empty((2, 0), dtype="i8")),
                    ("scalar", np.array(42, dtype="i8")),
                    ("endian", np.array([3, -5, -5, 7, 1, 3], dtype=">i4")[::-1])):
        for label, val in zip(("values", "indices", "inverse", "counts"),
                              np.unique(a, return_index=True, return_inverse=True, return_counts=True)):
            r("unique_" + name + "_" + label, val.astype(val.dtype.newbyteorder("=")))

    grid = np.array([8, 99, 4, 98, 4, 97, 1, 96, -2, 95], dtype=">i4")
    q = np.array([[-4, 4, 8], [1, 9, -2]], dtype="i4")
    for side in ("left", "right"):
        r("search_strided_" + side, np.searchsorted(grid[::2][::-1], q, side=side))
        for kind in ("datetime64", "timedelta64"):
            a = np.array([0, 1, 2, "NaT"], dtype=kind + "[D]")
            b = np.array([12, 24, 25, 48, 72, "NaT"], dtype=kind + "[h]")
            r("search_" + kind + "_" + side, np.searchsorted(a, b, side=side))
            r("isin_" + kind, np.isin(a, b))
    td = np.array([0, 1, 2, "NaT"], dtype="timedelta64[D]")
    q = np.array([0, 2, 3], dtype="i8")
    r("search_delta_integer", np.searchsorted(td, q))
    r("isin_delta_integer", np.isin(td, q))
    r("isin_incompatible", np.isin(td.astype("datetime64[D]"), td))
    fails("optimization_search_incompatible", lambda: np.searchsorted(td, td.astype("datetime64[D]")))

    a = np.array(["-0", "1.0000000000000000001", "1", "inf", "nan"], dtype=np.longdouble)
    b = np.array(["0", "1.0000000000000000001", "inf", "nan"], dtype=np.longdouble)
    r("isin_extended", np.isin(a, b))
    r("isin_extended_complex", np.isin(a.astype(np.clongdouble)+1j, b.astype(np.clongdouble)+1j))
    a, b = np.empty(4, dtype=object), np.empty(3, dtype=object)
    a[:] = [[1, 2], [2, 3], "x", 2]
    b[:] = [[1, 2], "x", 2.0]
    r("isin_object", np.isin(a, b))
    r("isin_object_invert", np.isin(a, b, invert=True))

    dt = np.dtype([("x", ">i4"), ("y", "f8")], align=True)
    a = np.array([(i % 5, float(i % 7)) for i in range(24)], dtype=dt)
    a["y"][0] = -0.
    a["y"][7] = np.nan
    b = a[::2].copy()
    b["y"][0] = 0.
    r("record_isin", np.isin(a[::-1], b))
    r("record_isin_invert", np.isin(a[::-1], b, invert=True))
    dt = np.dtype([("x", "f8", (2,))])
    a, b = np.zeros(3, dtype=dt), np.zeros(10, dtype=dt)
    a["x"][:] = [[-0., 1.], [1., 2.], [np.nan, 3.]]
    b["x"][:, 1] = 1.
    r("record_isin_subarray", np.isin(a, b))

    with np.errstate(all="ignore"):
        a = np.arange(24., dtype="f8").reshape(4, 6)**2
        r("gradient_input", a)
        a = a[::-1, ::2].T
        for axis in (0, 1):
            for edge in (1, 2):
                r(f"gradient_{axis}_{edge}", np.gradient(a, -0.5, axis=axis, edge_order=edge))
        a = np.array([0., 1., 4., 9.])
        for name, h in (("tiny", 1e-200), ("huge", 1e200), ("zero", 0.)):
            r("gradient_" + name, np.gradient(a, np.array(h), edge_order=2))
        a = np.array([np.nan, np.inf, -np.inf, 5.])
        r("gradient_nonfinite", np.gradient(a, edge_order=2))
        for dtype in ("float16", "float32", "float64", "complex64", "complex128"):
            start = np.array([-2, 1, 2] if "complex" not in dtype else
                             [-2+0.5j, 1j, 2+1j], dtype=dtype)
            stop = np.array([-64, 16, -8] if "complex" not in dtype else
                            [-8-0.5j, 16j, 8-1j], dtype=dtype)
            r("geom_start_" + dtype, start)
            r("geom_stop_" + dtype, stop)
            r("geom_" + dtype, np.geomspace(start, stop, num=7, axis=1, endpoint=False))
        a = (np.arange(48, dtype="f8").reshape(8, 6) % 13)**2 + 3
        r("cov_input", a)
        a = a[::-1, ::2].T
        r("cov_strided", np.cov(a))
        r("cov_float32", np.cov(a, dtype="float32"))
        r("cov_complex", np.cov(a+1j*a[:, ::-1], aweights=np.arange(1., 9.)))
        r("cov_nonfinite", np.cov([[1., 2., np.inf], [4., 5., 6.]]))
