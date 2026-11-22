# Release validation

The v1.0.0 release uses the same pinned NumPy 2.5.0 numerical references as the final development snapshot. The fixture was losslessly compressed for distribution; its decoded SHA-256 is recorded in the [compatibility contract](compatibility.md).

## Release-candidate test results

The release source, including its final module import path and compressed test fixture, was validated with Go 1.24.4 on a native ARM64 Linux environment (OrbStack Ubuntu, Apple M5 Max).

| Check | Result |
|:--|:--|
| `CGO_ENABLED=0 go test -p 2 -count=1 ./...` | 21,005 passing tests and subtests; 0 failing or skipped tests |
| Packages | 21 test-bearing packages passed; 3 packages have no test files |
| `go vet -p 2 ./...` | Passed |
| `go mod tidy -diff` | Passed; no dependency changes |
| `go run ./examples/scientific` | Passed: batched solve, SVD and odd-length FFT reconstruction |
| Linux/amd64 cross-compile of all Go test packages | Passed; binaries were not executed as part of this check |

The repository configures separate [GitHub Actions jobs](https://github.com/SamuelSupe/numpygo/actions/workflows/ci.yml) for Linux amd64 and arm64. Consult their live results for the tagged source.

Physical amd64 benchmarks remain pending because the designated remote Linux host did not complete the SSH handshake. The [ARM64 benchmark report](benchmarks.md) describes the measured environment and avoids treating cross-compilation or emulated timings as physical performance evidence.
