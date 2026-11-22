# Performance and benchmark results

numpygo uses native Go, Gonum and architecture-specific SIMD. Fast paths include contiguous and strided array operations, selected direct-output ufuncs, vector reductions, native-width random generation, matrix kernels and specialized complex/real FFT plans. Scalar paths preserve functionality where SIMD is unavailable or inappropriate.

## Latest native ARM64 comparison

The final optimization series focused on register-local Complex64 widening, odd-length real FFTs and workspaces for mixed-radix plans. The following is a measured comparison of the immediately preceding implementation and the release implementation, alongside NumPy 2.5.0. Lower times are better.

**Environment:** Apple M5 Max, OrbStack Linux/arm64 (native ARM64 NEON), Go 1.24.4, NumPy 2.5.0. A single guest vCPU was used for all three implementations; Go used `GOMAXPROCS=1` and `CGO_ENABLED=0`; Python numerical thread limits were set to one. Reported values are the medians of five approximately 100 ms runs, grouped by workload with adjacent before/after runs on the same CPU. Allocating and preallocated cases were compared separately.

| Workload | Go before (ms) | Go after (ms) | NumPy (ms) | Go speedup |
|:--|--:|--:|--:|--:|
| Complex64 multiply, 131,072, out | 0.13406 | 0.08359 | 0.03776 | 1.60× |
| Complex64 divide, 131,072, out | 0.22171 | 0.18150 | 0.20824 | 1.22× |
| float32 RFFT, 128 × 81 | 0.04525 | 0.02730 | 0.02636 | 1.66× |
| float64 RFFT, 128 × 81 | 0.04292 | 0.02553 | 0.01906 | 1.68× |
| float32 IRFFT, 128 × 729 | 0.48855 | 0.32904 | 0.11762 | 1.48× |
| float64 RFFT, 64 × 2025 | 0.90852 | 0.55104 | 0.26235 | 1.65× |
| float64 RFFT, 32 × 3125 | 0.84751 | 0.48061 | 0.22467 | 1.76× |
| float32 FFT, 128 × 1536 | 1.27174 | 1.21956 | 1.29010 | 1.04× |
| float32 IRFFT, 128 × 1500 | 0.86546 | 0.88598 | 0.23492 | 0.98× |

Across all 65 workloads, 57 Go median times improved and 8 regressed. The largest regression was roughly 2.4%; the maximum improvement was 1.76×. As with any shared-system microbenchmark, small percentage differences are subject to scheduling and measurement variation. NumPy remains faster for many cases, notably several inverse-real and single-signal transforms.

The [complete 65-workload CSV](benchmarks-arm64.csv) includes timing, relative performance, bytes allocated and allocations per operation. Mixed-radix workspace changes reduced allocation counts substantially; for example, float64 RFFT at 32 × 3125 changed from 17 to 9 allocations and from 136,736 to 106,864 bytes per operation.

## Architecture status

| Platform | Numerical validation | Physical performance comparison |
|:--|:--|:--|
| Linux/arm64 | Full Go test suite on native ARM64 OrbStack Linux | Measured, CSV above |
| Linux/amd64 | Cross-compilation and focused AVX tests in emulation; native Linux CI is configured | Pending; the designated remote Linux host was unavailable via SSH |

No emulator timings are included as physical amd64 measurements. GitHub CI validates behavior on the configured architecture but is not a controlled physical-hardware benchmark. Results from ARM64 cannot be extrapolated to x86 workloads.

## Reproduce

Run the Go and Python sides on the same machine. Pin CPU affinity where possible, restrict NumPy/BLAS thread counts, match the data shape/dtype and output allocation policy, and use repeated medians.

```sh
GOMAXPROCS=1 CGO_ENABLED=0 go test -run '^$' \
  -bench '^BenchmarkRegisterKernels$' -benchtime=100ms -benchmem -count=5 .

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
  python3 tools/benchmark_register.py
```

Install NumPy 2.5.0 in a separate Python development environment. The Go benchmark requires no Python. The published CSV uses nanoseconds per operation; the table above converts those values to milliseconds. To reproduce just the current implementation, use the v1.0.0 source; the before measurements refer to the preceding development snapshot with the same benchmark workloads.

Other benchmark families are available for general workloads, array and FFT paths, decomposition kernels, SIMD, direct-output kernels and precision-specific paths. List Go benchmark functions with `go test -list Benchmark ./...`; matching Python scripts are in [tools](../tools).
