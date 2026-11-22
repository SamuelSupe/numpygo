<div align="center">

# numpygo

**NumPy-style scientific computing, built for Go.**

Multidimensional arrays · Broadcasting · Linear algebra · FFT · Random sampling

[![CI](https://github.com/SamuelSupe/numpygo/actions/workflows/ci.yml/badge.svg)](https://github.com/SamuelSupe/numpygo/actions/workflows/ci.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/SamuelSupe/numpygo.svg)](https://pkg.go.dev/github.com/SamuelSupe/numpygo)
[![Release](https://img.shields.io/github/v/release/SamuelSupe/numpygo)](https://github.com/SamuelSupe/numpygo/releases)
[![Go 1.24+](https://img.shields.io/badge/Go-1.24%2B-00ADD8?logo=go&logoColor=white)](https://go.dev)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

</div>

numpygo brings NumPy's multidimensional array model and numerical APIs to native Go applications. A shared strided array representation handles dtype promotion, broadcasting, advanced indexing and non-contiguous views. Numerical kernels use Go, Gonum and architecture-specific SIMD where appropriate.

There is no Python runtime or C toolchain requirement: production applications and the differential test suite run with `CGO_ENABLED=0`.

## Installation

Requires Go 1.24 or later.

```sh
go get github.com/SamuelSupe/numpygo@v1.0.0
```

The root package is named `np`.

## Quick start

```go
package main

import (
    "fmt"

    np "github.com/SamuelSupe/numpygo"
    "github.com/SamuelSupe/numpygo/fft"
    "github.com/SamuelSupe/numpygo/linalg"
)

func must(a *np.Array, err error) *np.Array {
    if err != nil {
        panic(err)
    }
    return a
}

func main() {
    a := must(np.FromSlice([]float64{3, 1, 2, 4}, 2, 2))
    b := must(np.FromSlice([]float64{7, 11}, 2))
    x := must(linalg.Solve(a, b))
    values, err := np.Values[float64](x)
    if err != nil {
        panic(err)
    }
    fmt.Println(x.Shape(), values) // [2] [1.7 1.9]

    signal := must(np.FromSlice([]float64{1, -2, 3, 0.5, 4}, 5))
    spectrum := must(fft.RFFT(signal))
    restored := must(fft.IRFFT(spectrum, fft.Options{N: np.Int(5)}))
    values, err = np.Values[float64](restored)
    if err != nil {
        panic(err)
    }
    fmt.Println(values) // approximately [1 -2 3 0.5 4]
}
```

See the runnable [scientific computing example](examples/scientific/main.go) for batched solves, SVD and non-contiguous Fourier transforms.

## Highlights

| Capability | What's included |
|:--|:--|
| Arrays and dtypes | Booleans, signed/unsigned integers, float16/32/64, complex64/128, x87-style extended precision, temporal, string, record and Go object arrays |
| Array operations | Shared views, negative strides, reshaping, C/F layouts, integer and boolean indexing, broadcasting, ufuncs, masked and preallocated outputs |
| Statistics | Reductions, weighted statistics, quantiles, NaN-aware functions, histograms, sorting and set operations |
| Tensor algebra | Dot, tensor contraction, einsum and contraction path planning |
| Linear algebra | Batched solve, matmul, LU-based routines, Cholesky, QR, real/complex SVD, eigenvalues and least squares |
| Fourier transforms | Real, complex and Hermitian FFTs, multi-axis transforms, arbitrary lengths, normalization and output buffers |
| Random sampling | SeedSequence, PCG64, PCG64DXSM, MT19937, Philox, SFC64, Generator, RandomState and shared engine state |
| Scientific utilities | Masked arrays, six polynomial bases, calendar/business-day arithmetic, strings and record utilities |
| Interchange | NPY/NPZ, binary/text I/O, dtype descriptors and OS-backed memory mapping |

## NumPy compatibility

The numerical reference is **NumPy 2.5.0**. Compatibility is tracked at the symbol and tested-behavior level:

| Metric | Count |
|:--|--:|
| Tracked public symbols and members | 2,260 |
| Verified behavior subsets | 2,075 |
| Verified Go-equivalent contracts | 185 |
| Pending entries in the inventory | 0 |
| Reference arrays | 31,662 |
| Reference error records | 2,732 |

Every inventory entry links to its Go counterpart and reference fixtures. This is complete mapping coverage of the **defined inventory**, not a claim that every possible NumPy argument, dtype, platform, Python object protocol or C-extension interface is identical. See the [compatibility contract](docs/compatibility.md), [namespace summary](API_COVERAGE.md) and [machine-readable inventory](API_COVERAGE.csv).

## Performance

Algorithms use dense fast paths, strided fallbacks, direct output buffers, AVX/NEON kernels and specialized radix-based Fourier plans. The latest native ARM64 benchmark compares 65 complex-arithmetic and FFT workloads against the preceding Go implementation and NumPy 2.5.0:

| Workload | Go speedup vs previous build | Go after |
|:--|--:|--:|
| Complex64 multiply, 131,072 elements, preallocated | 1.60× | 0.084 ms |
| Batched float64 RFFT, 128 × 81 | 1.68× | 0.026 ms |
| Batched float64 RFFT, 32 × 3,125 | 1.76× | 0.481 ms |

These are workload-specific medians, not universal speedups over NumPy. The [benchmark report](docs/benchmarks.md) includes NumPy timings, allocations, regressions, environment details and the limits of the amd64 measurements.

## Documentation

| Guide | Contents |
|:--|:--|
| [Arrays and ufuncs](docs/arrays.md) | Array ownership, exact typed values, indexing, broadcasting and output options |
| [Numerical computing](docs/numerics.md) | Linear algebra, FFT, statistics, random generators, masked arrays and polynomials |
| [Dtypes and data interchange](docs/io.md) | Extended, structured and temporal arrays, NPY/NPZ and memory mapping |
| [Compatibility](docs/compatibility.md) | Scope, differences from Python NumPy and validation criteria |
| [Release validation](docs/validation.md) | Test results and architecture checks |
| [Performance](docs/benchmarks.md) | Benchmark results and reproduction commands |
| [Contributing](CONTRIBUTING.md) | Development, tests and fixture generation |
| [Go package reference](https://pkg.go.dev/github.com/SamuelSupe/numpygo) | Exported packages, types and functions |

## Development

```sh
git clone https://github.com/SamuelSupe/numpygo.git
cd numpygo
CGO_ENABLED=0 go test -p 2 ./...
CGO_ENABLED=0 go vet ./...
CGO_ENABLED=0 go run ./examples/scientific
```

The compressed NumPy reference dataset is included in the repository and embedded by the test helper, so Python is needed only to regenerate references or run NumPy-side comparisons.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [THIRD_PARTY_LICENSES.txt](THIRD_PARTY_LICENSES.txt) for component notices. numpygo is an independent Go implementation; it is not affiliated with the NumPy project.
