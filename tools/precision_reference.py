"""Differential references for native-precision FFT and ordered reductions."""
import numpy as np


def populate_precision(record, fails):
    def r(name, x):
        record("precision_" + name, x)

    rng = np.random.default_rng(872)
    with np.errstate(all="ignore"):
        for dtype in ("float32", "float64"):
            # Cross SIMD tails and the pairwise recursion boundary, as well
            # as the sequential slow-axis path with contiguous output lanes.
            x = rng.uniform(-1, 1, (35, 137)).astype(dtype)
            x[:, ::17] *= 2**20 if dtype == "float32" else 2**45
            r("reduce_" + dtype, x)
            for tag, data in (("packed", x), ("reverse", x[:, ::-1])):
                for axis in (None, 0, 1):
                    for name, f in (("sum", np.sum), ("mean", np.mean), ("var", np.var), ("std", np.std)):
                        if name in ("var", "std") and tag == "reverse":
                            continue
                        kw = {"ddof": .5} if name in ("var", "std") else {}
                        r(f"{name}_{tag}_{axis}_{dtype}", f(data, axis=axis, **kw))
            edge = np.zeros((4, 137), dtype=dtype)
            edge[0, :] = -0.
            edge[1, 5] = np.nan
            edge[2, 4] = np.inf
            edge[3, :] = np.linspace(-2, 2, 137, dtype=dtype)
            r("reduce_edge_" + dtype, edge)
            for name, f in (("sum", np.sum), ("mean", np.mean), ("var", np.var), ("std", np.std)):
                r(f"{name}_edge_{dtype}", f(edge, axis=1))
            for name, f in (("var", np.var), ("std", np.std)):
                r(f"{name}_zero_den_{dtype}", f(edge, axis=1, ddof=137))
                r(f"{name}_empty_{dtype}", f(np.empty((2, 0), dtype=dtype), axis=1))
            z = np.resize(np.array([.1, -.2, 3.7, -0., 0., np.inf, -np.inf, np.nan, .25, 2.], dtype=dtype), 137)
            y = np.resize(np.array([2., .5, -1., 1., -2., np.nan, .7, np.inf, .5, 2.], dtype=dtype), 137)
            mask = np.arange(137) % 11 < 7
            r("unary_" + dtype, z)
            r("binary_" + dtype, y)
            r("mask_" + dtype, mask)
            for name in ("positive", "negative", "absolute", "fabs", "square", "reciprocal", "sqrt",
                         "exp", "log", "sin", "cos", "floor", "ceil", "trunc", "rint",
                         "conjugate", "minimum", "maximum", "fmin", "fmax"):
                fn = getattr(np, name)
                args = (z, y) if name in ("minimum", "maximum", "fmin", "fmax") else (z,)
                out = np.full(139, -7, dtype=dtype)
                fn(*args, out=out[1:-1], where=mask)
                r(f"{name}_masked_{dtype}", out)
                alias = z.copy()
                fn(alias, *args[1:], out=alias)
                r(f"{name}_alias_{dtype}", alias)
            x = np.arange(138, dtype=dtype) / 7
            r("shift_input_" + dtype, x)
            np.square(x[:-1], out=x[1:])
            r("shift_output_" + dtype, x)
        for dtype in ("complex64", "complex128"):
            x = (rng.uniform(-2, 2, 137) + 1j*rng.uniform(-2, 2, 137)).astype(dtype)
            y = (rng.uniform(.5, 2, 137) + 1j*rng.uniform(.5, 2, 137)).astype(dtype)
            r("unary_" + dtype, x)
            r("binary_" + dtype, y)
            r("mask_" + dtype, mask)
            for name in ("positive", "negative", "conjugate", "square", "reciprocal", "add", "subtract", "multiply", "divide"):
                fn = getattr(np, name)
                args = (x, y) if name in ("add", "subtract", "multiply", "divide") else (x,)
                out = np.full(139, -7, dtype=dtype)
                fn(*args, out=out[1:-1], where=mask)
                r(f"{name}_masked_{dtype}", out)
                alias = x.copy()
                fn(alias, *args[1:], out=alias)
                r(f"{name}_alias_{dtype}", alias)
        a = rng.uniform(-1, 1, (3, 1025)).astype("float32")
        c = (a + 1j*rng.uniform(-1, 1, a.shape)).astype("complex64")
        r("fft_real", a)
        r("fft_complex", c)
        for n in (1024, 1000, 509, 2048):
            for norm in ("backward", "ortho", "forward"):
                tag = f"{n}_{norm}"
                z = np.fft.rfft(a, n=n, norm=norm)
                r("rfft_" + tag, z)
                r("irfft_" + tag, np.fft.irfft(z, n=n, norm=norm))
                r("fft_" + tag, np.fft.fft(c, n=n, norm=norm))
                r("ifft_" + tag, np.fft.ifft(c, n=n, norm=norm))
        r("fft_realcomplex", np.fft.fft(a, n=1024))
        r("fft_axis0", np.fft.fft(c, n=64, axis=0, norm="ortho"))
        r("rfft_reverse", np.fft.rfft(a[:, ::-1], n=1024))
        u = rng.uniform(-1, 1, (137, 2)).astype("float32")
        v = (u + 1j*rng.uniform(-1, 1, u.shape)).astype("complex64")
        r("fft2_real_input", u)
        r("fft2_complex_input", v)
        r("fft2_real", np.fft.fft(u))
        r("fft2_complex", np.fft.fft(v))
        r("rfft2", np.fft.rfft(u))
        r("ifft2", np.fft.ifft(v, norm="forward"))
        r("irfft2", np.fft.irfft(v, norm="forward"))
        r("fft_alias", np.fft.fft(c[:, :1024], norm="ortho"))


if __name__ == "__main__":
    # A standalone fixture permits the same focused Go differential suite
    # to run against NumPy built for a different CPU architecture.
    import json
    import sys
    from pathlib import Path

    assert np.__version__ == "2.5.0"
    arrays = {}

    def record(name, value):
        value = np.asarray(value)
        arrays[name] = {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "strides": list(value.strides),
            "values": [[str(int(v)), "0"] if value.dtype.kind in "biu" else
                       [repr(float(v.real)), repr(float(v.imag))] for v in value.flat],
        }

    populate_precision(record, None)
    Path(sys.argv[1]).write_text(json.dumps({
        "version": np.__version__, "arrays": arrays, "errors": {},
    }, indent=2, allow_nan=False) + "\n")
    print("generated", len(arrays), "precision references")
