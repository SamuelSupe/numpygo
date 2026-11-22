package np

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
)

// TensorHandle retains the Go allocation and exposes safe typed operations
// and strided metadata instead of a Python capsule or an unchecked C pointer.
type TensorHandle struct{ array *Array }

func (a *Array) CTypes() (*TensorHandle, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	// A handle must continue to refer to this allocation; resizing would
	// otherwise replace the array's backing memory under the handle.
	a.viewBase()
	return &TensorHandle{a}, nil
}
func (a *Array) DLPack() (*TensorHandle, error)        { return a.CTypes() }
func (h *TensorHandle) Shape() []int                   { return h.array.Shape() }
func (h *TensorHandle) Strides() []int                 { return h.array.Strides() }
func (h *TensorHandle) DType() DType                   { return h.array.DType() }
func (h *TensorHandle) Writable() bool                 { return h.array.Writable() }
func (h *TensorHandle) DataIdentity() uintptr          { return h.array.DataIdentity() }
func (h *TensorHandle) At(indices ...int) (any, error) { return h.array.AtValue(indices...) }
func (h *TensorHandle) Set(value any, indices ...int) error {
	return h.array.SetValue(value, indices...)
}

// FromDLPack imports a retained native Go tensor handle as a shared view,
// preserving its writeability as in NumPy 2.5. Foreign capsules and non-CPU
// devices are not supported.
func FromDLPack(h *TensorHandle) (*Array, error) {
	if h == nil || h.array.DType() == 0 {
		return nil, ErrDType
	}
	v, e := h.array.Slice()
	if e != nil {
		return nil, e
	}
	return v, nil
}

// Dump and Dumps use the portable numerical NPY representation rather than
// executing or producing Python pickle payloads. LoadNPY is their inverse.
func (a *Array) Dump(w io.Writer) error { return SaveNPY(w, a) }
func (a *Array) Dumps() ([]byte, error) { var b bytes.Buffer; e := a.Dump(&b); return b.Bytes(), e }

type NamespaceCapabilities struct {
	BooleanIndexing     bool
	DataDependentShapes bool
	MaxDimensions       int
}
type NamespaceInfo struct{}

func ArrayNamespaceInfo() NamespaceInfo { return NamespaceInfo{} }
func (NamespaceInfo) Capabilities() NamespaceCapabilities {
	return NamespaceCapabilities{true, true, maxDims}
}
func (NamespaceInfo) DefaultDevice() string { return "cpu" }
func (NamespaceInfo) Devices() []string     { return []string{"cpu"} }
func (NamespaceInfo) DefaultDTypes() map[string]DType {
	return map[string]DType{"real floating": Float64, "complex floating": Complex128, "integral": Int64, "indexing": Int64}
}
func (NamespaceInfo) DTypes(kinds ...string) (map[string]DType, error) {
	out := map[string]DType{}
	for _, d := range []DType{Bool, Int8, Int16, Int32, Int64, Uint8, Uint16, Uint32, Uint64, Float32, Float64, Complex64, Complex128} {
		if len(kinds) == 0 {
			out[d.String()] = d
			continue
		}
		ok, e := IsDType(d, kinds...)
		if e != nil {
			return nil, e
		}
		if ok {
			out[d.String()] = d
		}
	}
	return out, nil
}

const ModuleImportPath = "github.com/SamuelSupe/numpygo"

// GetInclude returns the installed Go source directory if available; clients
// import ModuleImportPath instead of compiling against NumPy C headers.
func GetInclude() (string, error) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		return "", os.ErrNotExist
	}
	dir := filepath.Dir(file)
	if _, e := os.Stat(filepath.Join(dir, "array.go")); e != nil {
		return "", e
	}
	return dir, nil
}
func Info(w io.Writer, a *Array) error {
	if w == nil || a.DType() == 0 {
		return ErrValue
	}
	_, e := fmt.Fprintf(w, "shape: %v\nstrides: %v\nitemsize: %d\ndtype: %s\nwritable: %t\n", a.Shape(), a.Strides(), a.DType().ItemSize(), a.DType(), a.Writable())
	return e
}

type RuntimeInfo struct {
	OS, Architecture, Version string
	LogicalCPUs               int
}

func ShowRuntime() RuntimeInfo {
	return RuntimeInfo{runtime.GOOS, runtime.GOARCH, runtime.Version(), runtime.NumCPU()}
}

type BuildConfig struct {
	Module, GoVersion string
	Dependencies      map[string]string
}

func ShowConfig() BuildConfig {
	c := BuildConfig{Module: ModuleImportPath, GoVersion: runtime.Version(), Dependencies: map[string]string{}}
	if b, ok := debug.ReadBuildInfo(); ok {
		for _, d := range b.Deps {
			c.Dependencies[d.Path] = d.Version
		}
	}
	return c
}

// SelfTest checks a small numerical deployment smoke test without importing
// Python or invoking a compiler/test runner at runtime.
func SelfTest() error {
	a, e := FromFloat64([]float64{1, 2, 3}, 3)
	if e != nil {
		return e
	}
	v, e := Add(a, Scalar(2))
	if e != nil {
		return e
	}
	s, e := Sum(v)
	if e != nil {
		return e
	}
	x, e := s.At()
	if e != nil {
		return e
	}
	if x != 12 {
		return fmt.Errorf("numerical self-test failed: %v", x)
	}
	return nil
}
