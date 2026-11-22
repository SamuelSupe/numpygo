"""NumPy 2.5.0 counterpart of BenchmarkPrecisionKernels."""
import time
import numpy as np

assert np.__version__ == "2.5.0"
print("NumPy", np.__version__)
np.show_runtime()
v = (np.arange(1024*1024) % 137-68)/17.
a64 = v.reshape(1024, 1024)
a32 = a64.astype("float32")
cases = []
for a in (a64, a32):
    for tag, axis in (("all", None), ("axis0", 0), ("axis1", 1)):
        for name in ("sum", "mean", "var", "std"):
            f = getattr(np, name)
            cases.append((f"{name}_{a.dtype}_{tag}", lambda a=a, axis=axis, f=f: f(a, axis=axis)))
a = v[:128*1024].reshape(128, 1024)
x = a.astype("float32")
c = (a + 1j*((np.arange(a.size).reshape(a.shape) % 97-48)/31.)).astype("complex64")
half = c[:, :513].copy()
co = np.zeros_like(c)
ho = np.zeros_like(half)
ro = np.zeros_like(x)
mixed = np.zeros((128, 501), dtype="complex64")
prime = v[:4*4093].reshape(4, 4093).astype("float32")
prime_out = np.zeros((4, 2047), dtype="complex64")
cases += [
    ("fft32_alloc", lambda: np.fft.fft(c)),
    ("fft32_out", lambda: np.fft.fft(c, out=co)),
    ("ifft32_out", lambda: np.fft.ifft(c, out=co)),
    ("rfft32_alloc", lambda: np.fft.rfft(x)),
    ("rfft32_out", lambda: np.fft.rfft(x, out=ho)),
    ("irfft32_out", lambda: np.fft.irfft(half, n=1024, out=ro)),
    ("fftpack_rfft32_out", lambda: np.fft.rfft(x, n=1000, out=mixed)),
    ("prime_rfft32_out", lambda: np.fft.rfft(prime, out=prime_out)),
]
mask = (np.arange(a.size) % 256 < 128).reshape(a.shape)
for q in (a, x):
    out = np.zeros_like(q)
    for name in ("negative", "absolute", "square", "reciprocal", "sqrt", "exp", "minimum"):
        f = getattr(np, name)
        args = (q, q) if name == "minimum" else (q,)
        cases.append((f"out_{name}_{q.dtype}", lambda f=f, args=args, out=out: f(*args, out=out)))
    cases.append((f"out_mask_square_{q.dtype}", lambda q=q,out=out: np.square(q, out=out, where=mask)))
for name in ("conjugate", "square", "multiply", "divide"):
    f = getattr(np, name)
    args = (c,c) if name in ("multiply", "divide") else (c,)
    cases.append((f"out_{name}_complex64", lambda f=f,args=args: f(*args, out=co)))
aa = x.reshape(-1, 2)
cc = c.reshape(-1, 2)
doo = np.zeros_like(cc)
dr = np.zeros_like(aa)
cases += [("rfft2_out", lambda: np.fft.rfft(aa, out=doo)),
          ("fft2_out", lambda: np.fft.fft(cc, out=doo)),
          ("irfft2_out", lambda: np.fft.irfft(cc, out=dr, norm="forward"))]
with np.errstate(all="ignore"):
    for name, fn in cases:
        start = time.perf_counter_ns()
        result = fn()
        loops = max(1, min(10000, round(100_000_000/max(1,time.perf_counter_ns()-start))))
        for _ in range(5):
            start = time.perf_counter_ns()
            for _ in range(loops):
                result = fn()
            print(f"NumPyPrecisionKernels/{name} {loops} {(time.perf_counter_ns()-start)/loops:.0f} ns/op", flush=True)
