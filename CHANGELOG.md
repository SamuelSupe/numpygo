# Changelog

## v1.0.0

First stable public release of numpygo, a native Go scientific computing library targeting NumPy 2.5.0 numerical behavior.

### Features

- Unified strided ndarray model with typed storage, broadcasting, advanced indexing, writable views, dtype promotion and ufunc options.
- Integer, half/single/double precision, complex, x87-style extended, date/time, flexible string, structured and object dtypes.
- Statistics, sorting, signal utilities, tensor contraction and einsum path planning.
- Batched real and complex linear algebra, including solve, SVD, QR, Hermitian and general eigenvalue decomposition.
- Multidimensional real/complex/Hermitian FFTs, explicit normalization, odd-length reconstruction and preallocated output.
- Five NumPy-compatible bit generators, SeedSequence, Generator, RandomState, state sharing and external Go bit-source adapters.
- Masked arrays, polynomial series, calendar utilities, string/record operations, NPY/NPZ interoperability and memory mapping.
- SIMD acceleration on amd64 and ARM64, direct-output kernels, mixed-radix FFT plans and native single-precision data paths.

### Validation and distribution

- 2,260 mapped symbols and members: 2,075 verified behavior subsets, 185 verified Go equivalents, and 0 pending inventory entries.
- Embedded reference dataset: 31,662 arrays and 2,732 error records generated from NumPy 2.5.0.
- Go module path: `github.com/SamuelSupe/numpygo`.
- Go 1.24+, with no Python or cgo required for normal operation.
- Apache-2.0 license, English documentation, reproducible counting and benchmark scripts, Linux amd64/arm64 CI.

Compatibility is limited to the documented behaviors; Python object dispatch, pickle and C-extension binary compatibility are not provided. The published [compatibility guide](docs/compatibility.md) and [benchmarks](docs/benchmarks.md) describe the numerical and platform boundaries.
