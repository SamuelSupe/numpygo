"""Measure cumulative float32 radix-2 rounding against a wide FFT oracle."""
import numpy as np

assert np.__version__ == "2.5.0"
n = 4096
rng = np.random.default_rng(872)
x = (rng.uniform(-1, 1, (3, n)) + 1j*rng.uniform(-1, 1, (3, n))).astype("complex64")
rev = np.array([int(f"{i:012b}"[::-1], 2) for i in range(n)])
y = x[:, rev].copy()
h = 1
while h < n:
    block = y.reshape(3, -1, 2*h)
    left = block[:, :, :h].copy()
    right = block[:, :, h:].copy() * np.exp(-1j*np.pi*np.arange(h)/h).astype("complex64")
    block[:, :, :h] = left + right
    block[:, :, h:] = left - right
    h *= 2
ref = np.fft.fft(x.astype("complex128")).astype("complex64").view("float32")
err = np.abs(y.view("float32") - ref)
print("NumPy", np.__version__)
print("components", err.size, "outside 1e-6 + 2e-6*abs(ref)",
      np.count_nonzero(err > 1e-6 + 2e-6*np.abs(ref)), "max abs error", err.max())
