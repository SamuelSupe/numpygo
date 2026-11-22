# Numerical computing

All numerical packages consume the shared `*np.Array` type. Import the module root and the relevant subpackage, for example:

```go
import (
    np "github.com/SamuelSupe/numpygo"
    "github.com/SamuelSupe/numpygo/fft"
    "github.com/SamuelSupe/numpygo/linalg"
    "github.com/SamuelSupe/numpygo/random"
)
```

## Linear algebra

The [linalg package](https://pkg.go.dev/github.com/SamuelSupe/numpygo/linalg) supports batched matrices, real and complex arithmetic, and NumPy-style broadcasting of leading dimensions. Its API includes `MatMul`, `MatVec`, `VecMat`, `Solve`, `Inv`, `Det`, `Slogdet`, `SVD`, `SVDVals`, `QR`, `QRRaw`, `Cholesky`, `Eig`, `Eigh`, `Eigvals`, `Eigvalsh`, `Pinv`, `MatrixRank`, `Lstsq`, `MatrixPower`, `MultiDot`, `Norm` and `Cond`.

```go
a, err := np.FromFloat64([]float64{3, 1, 2, 4}, 2, 2)
if err != nil {
    return err
}
b, err := np.FromFloat64([]float64{7, 11}, 2)
if err != nil {
    return err
}
x, err := linalg.Solve(a, b)
if err != nil {
    return err
}
u, s, vh, err := linalg.SVD(a, linalg.SVDOptions{Reduced: true})
if err != nil {
    return err
}
_, _, _, _ = x, u, s, vh
```

The solve interface distinguishes a one-dimensional right-hand vector from matrix right-hand sides. Singular matrices return errors. SVD and eigendecomposition support real and complex single/double precision; integer inputs use the corresponding promoted types. Hermitian eigenvalue routines use the specified triangular part. Equivalent eigenvector phases, signs, degenerate eigenspaces and unspecified eigenvalue ordering are valid.

`PinvWithOptions` and `MatrixRankWithOptions` accept batched tolerance arrays and Hermitian options. `SVDOptions.Hermitian` selects the Hermitian decomposition path. General complex eigendecomposition uses a scaled Hessenberg/shifted QR/Schur approach. The reference suite includes defective matrices, repeated eigenvalues, non-normal matrices and extreme magnitude tests; it does not establish stability for every matrix size and condition number.

The core namespace also provides `Dot`, `VDot`, `Inner`, `Outer`, `Kron`, `Cross`, `Tensordot`, `Einsum` and `EinsumPath`. `EinsumWithOptions` accepts an output array, layout, casting, calculation dtype and optimization/path options.

## Fourier transforms

The [fft package](https://pkg.go.dev/github.com/SamuelSupe/numpygo/fft) includes FFT/IFFT, RFFT/IRFFT, HFFT/IHFFT, their multidimensional variants, frequency grids and frequency shifting.

```go
signal, err := np.FromSlice([]float32{1, -2, 3, 0.5, 4}, 5)
if err != nil {
    return err
}
spectrum, err := fft.RFFT(signal, fft.Options{Norm: fft.Ortho})
if err != nil {
    return err
}
restored, err := fft.IRFFT(spectrum, fft.Options{
    N: np.Int(5), Norm: fft.Ortho,
})
if err != nil {
    return err
}
values, err := np.Values[float32](restored)
if err != nil {
    return err
}
fmt.Println(values)
```

Always pass the original length to IRFFT for odd-length signals. Without it the inverse infers an even length of twice the spectral length minus two. The default axis is the last axis. Explicit transform lengths truncate or zero-pad the input. Normalization:

| Norm | Forward | Inverse |
|:--|:--|:--|
| `Backward` (default) | 1 | 1/n |
| `Forward` | 1/n | 1 |
| `Ortho` | 1/sqrt(n) | 1/sqrt(n) |

Use `fft.NDOptions` for explicit axes and shape. `Out` accepts supported output dtypes, writable strided arrays and applicable aliases. Single-precision real/complex transforms use native-precision input/output and associated rounding, with wider working precision where necessary. Optimized paths cover radix-2 and mixed-radix lengths, including odd real transforms; general lengths use fallback algorithms.

## Statistics and signal operations

The root package includes sums, products, means, variances, standard deviations, extrema, quantiles, percentiles, covariance, correlation, histograms, weighted averages and NaN-aware variants. Most reductions accept axes and keepdims. `VarOptions` additionally handles explicit mean, correction, masks and output. `QuantileArray` supports array-valued quantiles and 13 quantile methods; weights are supported for inverted-CDF quantiles.

Sorting, partitioning, unique/set operations, searchsorted, digitization, interpolation, gradients, differences, convolution, correlation and common signal windows are also available. The order within an unspecified partition and the ordering of equal keys in an unstable sort are not compatibility guarantees.

## Random sampling

Use a seeded generator for reproducible results:

```go
g := random.DefaultRNG(12345)
sample, err := g.StandardNormal(2, 3)
if err != nil {
    return err
}
fmt.Println(sample.Shape()) // [2 3]
```

The [random package](https://pkg.go.dev/github.com/SamuelSupe/numpygo/random) implements SeedSequence and five built-in bit generators: MT19937, PCG64, PCG64DXSM, Philox and SFC64.

Modern Generator and legacy RandomState can share one bit generator. Restoring the legacy state restores the shared engine and that instance's Gaussian cache:

```go
bit := random.NewPhilox(12345)
legacy, err := random.NewRandomStateFromBitGenerator(bit)
if err != nil {
    return err
}
modern, err := random.NewGeneratorFromBitGenerator(bit)
if err != nil {
    return err
}
saved := legacy.GetState()
if _, err := modern.Random(3); err != nil {
    return err
}
if err := legacy.SetState(saved); err != nil {
    return err
}
```

Raw bit streams, supported jumps/advances, spawn and state transitions are tested against NumPy. Legacy MT19937 seed and Gaussian caching behavior differ from modern Generator, as in NumPy. Some decomposition-based distributions can differ by a valid eigenvector phase or basis. Array-valued distribution parameters use broadcast semantics through `Sample`; use `FillRandom` for supported preallocated float32/64 outputs.

External Go algorithms can implement `ExternalBitSource` and use `NewExternalBitGenerator` with explicit state snapshots. Optional advance/jump/spawn capabilities follow the external algorithm's own contract. No C pointers or Python capsules are involved.

## Other scientific packages

| Package | Use |
|:--|:--|
| [ma](https://pkg.go.dev/github.com/SamuelSupe/numpygo/ma) | Numeric and structured masked arrays, masks, filled results and missing-data operations |
| [poly](https://pkg.go.dev/github.com/SamuelSupe/numpygo/poly) | Power, Chebyshev, Legendre, Hermite, HermiteE and Laguerre bases; evaluation, fits, roots, calculus and conversion |
| [scimath](https://pkg.go.dev/github.com/SamuelSupe/numpygo/scimath) | Complex promotion for domain-sensitive square roots, logarithms, powers and inverse trig functions |
| [datetime](https://pkg.go.dev/github.com/SamuelSupe/numpygo/datetime) | Explicit date/time units, NaT, business-day counts and offsets |
| [matrix](https://pkg.go.dev/github.com/SamuelSupe/numpygo/matrix) | Legacy two-dimensional matrix semantics |

Masked arrays have a data array and a Boolean mask (true means missing), with hard-mask and fill-value options. Polynomial series use ascending-degree coefficients; the legacy poly1d interface uses descending-degree coefficients.
