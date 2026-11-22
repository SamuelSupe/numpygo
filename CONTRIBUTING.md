# Contributing

Contributions to numerical correctness, clear APIs, test coverage and performance are welcome.

## Development

Use Go 1.24 or later. The runtime and regular tests must work without cgo or Python.

```sh
git clone https://github.com/SamuelSupe/numpygo.git
cd numpygo
CGO_ENABLED=0 go test -p 2 -count=1 ./...
CGO_ENABLED=0 go vet ./...
go mod tidy -diff
```

Use `gofmt` for Go sources. Keep changes focused, preserve dtype/layout semantics and add regression tests for meaningful observable behavior. Test inputs should cover strided/empty/aliased arrays when relevant. CI runs native Linux amd64 and arm64 jobs with cgo disabled.

## Reference generation

Only maintainers regenerating the differential reference need Python. Use a separate development environment with the exact pinned NumPy version:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install numpy==2.5.0 cffi pytest hypothesis
python tools/generate_ufunc_metadata.py
python tools/generate_reference.py
python tools/generate_inventory.py
CGO_ENABLED=0 go test -p 2 -count=1 ./...
```

The reference generator writes `internal/refcheck/numpy-2.5.0.json.gz`, and the inventory generator writes `API_COVERAGE.csv` and `API_COVERAGE.md`. The published reference is the pinned Linux/x86 oracle; platform-specific fixtures may differ on other NumPy builds. Inspect any changed reference behavior and document the reason rather than blindly accepting new baselines.

For cross-language file verification:

```sh
interchange_dir=$(mktemp -d)
NUMPYGO_INTERCHANGE_DIR="$interchange_dir" CGO_ENABLED=0 go test -p 2 -count=1 -run '^(TestNumPyInterchange|TestStructuredNPYNumPy|TestFlexibleFilesNumPy|TestTitledStructuredNPYNumPy|TestStructuredTemporalNumPy|TestNPYFormatNumPy|TestNPYOpenMemMapNumPy|TestNestedSubarrayAndLayoutNumPy)$' ./...
python tools/verify_interchange.py "$interchange_dir"
```

## Performance changes

Use representative benchmark families and equivalent dtype, shape, layout and output allocation across revisions and implementations. Record CPU architecture/model, Go and NumPy versions, threads, median latency, allocations and regressions. Do not report emulator timings as physical hardware results. See [benchmark methodology](docs/benchmarks.md).

## Pull requests

Include the behavioral motivation, tested scope, and any known limitations; update documentation and the inventory when the public compatibility contract changes. New third-party code and derived algorithms require appropriate attribution.
