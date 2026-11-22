# Compatibility contract

numpygo targets NumPy 2.5.0 numerical behavior in an independent Go runtime. Its compatibility inventory records public callable exports, selected public class members and explicitly tracked numeric protocols. It counts aliases separately where they are distinct public entries; inherited class members are grouped as described in the [inventory](../API_COVERAGE.md).

## Evidence

The [CSV inventory](../API_COVERAGE.csv) contains 2,260 rows, each classified as:

- **verified-subset (2,075):** the documented Go numerical behavior has corresponding reference fixtures.
- **verified-go-equivalent (185):** an explicit Go contract is verified where a Python runtime or platform interface cannot be reproduced directly.
- **pending (0):** no verified mapping in the defined inventory.

Zero pending entries does not establish full semantic parity across arbitrary input combinations. Consult each row's scope, mapped API and fixture identifiers. Private NumPy APIs, the C ABI and unlisted namespaces are not part of this inventory.

The compressed reference contains 31,662 array results and 2,732 error records from NumPy 2.5.0. Its decoded SHA-256 is `74508e03febbaf639df8cc6cfea76c7ab69f2ef9a518156c70020277daf571f2`. Go tests embed the compressed reference, decompress it and validate the pinned version before using it. Python is not needed to run the tests.

## Comparison rules

| Behavior | Criterion |
|:--|:--|
| Shape, dtype and relevant strides | Exact comparison |
| Integer values and random raw streams | Exact typed comparison |
| Floating-point outputs | Dtype- and algorithm-appropriate tolerances, including NaN/Inf and specified signed-zero cases |
| Errors | Go error category matched to the corresponding reference exception |
| Shared views and assignment | Observable storage sharing and writeback |
| Decompositions | Singular/eigenvalues, residuals, orthogonality, rank and reconstruction; valid sign/phase and basis choices may differ |
| File interchange | Numeric values, descriptors, layout and supported on-disk fields |

The test suite also contains regression tests that exercise overlap, overflow, non-contiguous arrays, random-state boundaries, direct output, SIMD, mixed-radix FFT and file I/O. See [Contributing](../CONTRIBUTING.md) for test and fixture regeneration commands.

## Explicit Go equivalents

| Python/NumPy interface | Go contract |
|:--|:--|
| Python operators, subclass dispatch, frompyfunc | Explicit Go methods, ufunc descriptors and callbacks; no Python type dispatch |
| ctypes, cffi, capsules, DLPack | Typed handles or guarded bit-source access; no ABI-compatible Python capsules or arbitrary foreign pointers |
| ndarray.dump/dumps | Numeric NPY serialization, not Python pickle |
| NumPy object scalars | Go reference values and callbacks |
| Python thread-local floating-point/print options | Go configuration and explicit contexts/formatters |
| Python documentation mutation and introspection | Explicit Go registries, metadata and diagnostics |
| NumPy extension C API | Not implemented |

## Numerical and platform boundaries

- The library does not guarantee NumPy's unspecified uninitialized memory, unstable-sort tie order, partition-internal ordering or unordered repeated assignment; Go uses safe zero initialization and deterministic repeated-index assignment.
- Bit generators and many fixed-seed distributions match their reference; not all possible seeds, parameter combinations or backend decomposition choices have been tested. Non-finite random parameters may be rejected conservatively.
- x87 extended types target the Linux/x86 representation even on ARM; other platform-specific long-double representations are not asserted equivalent. Numeric casts outside the representable range may vary across NumPy SIMD/platform implementations.
- Object dispatch, arbitrary extension-defined dtypes, pickle, NumPy C-extension binary compatibility and unsupported descriptor/format combinations are excluded.
- SIMD selection depends on build target, CPU support and array layout, with scalar fallback kernels. Numerical correctness and physical performance are distinct measurements.
- Full Linux ARM64 tests and focused emulated amd64 AVX tests were run during development. Emulation and cross-compilation do not prove physical x86 throughput. See the [benchmark report](benchmarks.md) and [CI runs](https://github.com/SamuelSupe/numpygo/actions/workflows/ci.yml) for current evidence.

Compatibility is a testable maintenance goal. Please report a mismatched NumPy behavior with a minimal example, dtype, shape, arguments, expected result and NumPy version.
