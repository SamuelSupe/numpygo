// Package refcheck loads the pinned NumPy fixtures for Go-only differential tests.
package refcheck

import (
	"bytes"
	"compress/gzip"
	_ "embed"
	"encoding/json"
	"errors"
	"io"
	"math"
	"slices"
	"strconv"
	"testing"

	"github.com/SamuelSupe/numpygo"
)

//go:embed numpy-2.5.0.json.gz
var referenceGzip []byte

type referenceArray struct {
	DType   string
	Shape   []int
	Strides []int
	Values  [][2]string
}

var reference struct {
	Version string
	Arrays  map[string]referenceArray
	Errors  map[string]string
}

func init() {
	r, err := gzip.NewReader(bytes.NewReader(referenceGzip))
	if err != nil {
		panic(err)
	}
	defer r.Close()
	referenceJSON, err := io.ReadAll(r)
	if err != nil {
		panic(err)
	}
	if err := json.Unmarshal(referenceJSON, &reference); err != nil {
		panic(err)
	}
	if reference.Version != "2.5.0" {
		panic("incorrect NumPy reference version")
	}
}

func Require(t testing.TB) func(*np.Array, error) *np.Array {
	t.Helper()
	return func(a *np.Array, err error) *np.Array {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
		return a
	}
}

func RequireValue[T any](t testing.TB) func(T, error) T {
	t.Helper()
	return func(v T, e error) T {
		t.Helper()
		if e != nil {
			t.Fatal(e)
		}
		return v
	}
}
func Bytes(t testing.TB, name string, data []byte) {
	t.Helper()
	Check(t, name, Require(t)(np.WrapSlice(data, len(data))))
}
func Text(t testing.TB, name, data string) {
	t.Helper()
	want := Input(t, name)
	v, e := np.Values[uint8](want)
	if e != nil {
		t.Fatal(e)
	}
	if string(v) != data {
		t.Errorf("%s: got %q, NumPy %q", name, data, string(v))
	}
}

func lookup(t testing.TB, name string) referenceArray {
	t.Helper()
	value, ok := reference.Arrays[name]
	if !ok {
		t.Fatalf("missing NumPy array %q", name)
	}
	return value
}

func number(t testing.TB, value string) float64 {
	t.Helper()
	x, err := strconv.ParseFloat(value, 64)
	if err != nil {
		t.Fatalf("invalid reference number %q: %v", value, err)
	}
	return x
}

func Input(t testing.TB, name string) *np.Array {
	t.Helper()
	r := lookup(t, name)
	d, err := np.ParseDType(r.DType)
	if err != nil {
		t.Fatal(err)
	}
	must := Require(t)
	if d.IsSigned() {
		values := make([]int64, len(r.Values))
		for i, x := range r.Values {
			values[i], err = strconv.ParseInt(x[0], 10, 64)
			if err != nil {
				t.Fatal(err)
			}
		}
		return must(must(np.WrapSlice(values, r.Shape...)).Astype(d))
	}
	if d.IsUnsigned() || d == np.Bool {
		values := make([]uint64, len(r.Values))
		for i, x := range r.Values {
			values[i], err = strconv.ParseUint(x[0], 10, 64)
			if err != nil {
				t.Fatal(err)
			}
		}
		return must(must(np.WrapSlice(values, r.Shape...)).Astype(d))
	}
	if d.IsFloat() {
		values := make([]float64, len(r.Values))
		for i, x := range r.Values {
			values[i] = number(t, x[0])
		}
		return must(must(np.WrapFloat64(values, r.Shape...)).Astype(d))
	}
	if d.IsComplex() {
		values := make([]complex128, len(r.Values))
		for i, x := range r.Values {
			values[i] = complex(number(t, x[0]), number(t, x[1]))
		}
		return must(must(np.WrapComplex128(values, r.Shape...)).Astype(d))
	}
	t.Fatalf("unsupported fixture dtype %q", r.DType)
	return nil
}

func Metadata(t testing.TB, name string, a *np.Array) {
	t.Helper()
	want := lookup(t, name)
	if a == nil || a.DType().String() != want.DType || !slices.Equal(a.Shape(), want.Shape) {
		t.Fatalf("%s: expected %s %v, got %v", name, want.DType, want.Shape, a)
	}
}

func Check(t testing.TB, name string, a *np.Array) {
	t.Helper()
	Metadata(t, name, a)
	r := lookup(t, name)
	if a.DType().IsSigned() {
		b, err := a.Astype(np.Int64)
		if err != nil {
			t.Fatal(err)
		}
		v, err := np.Values[int64](b)
		if err != nil {
			t.Fatal(err)
		}
		for i, x := range r.Values {
			want, err := strconv.ParseInt(x[0], 10, 64)
			if err != nil {
				t.Fatal(err)
			}
			if v[i] != want {
				t.Errorf("%s[%d]: got %d; want %d", name, i, v[i], want)
			}
		}
		return
	}
	if a.DType().IsUnsigned() || a.DType() == np.Bool {
		b, err := a.Astype(np.Uint64)
		if err != nil {
			t.Fatal(err)
		}
		v, err := np.Values[uint64](b)
		if err != nil {
			t.Fatal(err)
		}
		for i, x := range r.Values {
			want, err := strconv.ParseUint(x[0], 10, 64)
			if err != nil {
				t.Fatal(err)
			}
			if v[i] != want {
				t.Errorf("%s[%d]: got %d; want %d", name, i, v[i], want)
			}
		}
		return
	}
	values, err := a.Complex128s()
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range r.Values {
		wantReal, wantImag := number(t, v[0]), number(t, v[1])
		if !CloseDType(real(values[i]), wantReal, a.DType()) || !CloseDType(imag(values[i]), wantImag, a.DType()) {
			t.Errorf("%s[%d]: got %v; want (%g, %g)", name, i, values[i], wantReal, wantImag)
		}
	}
}

func CheckLayout(t testing.TB, name string, a *np.Array) {
	t.Helper()
	Check(t, name, a)
	want := lookup(t, name).Strides
	if !slices.Equal(a.Strides(), want) {
		t.Errorf("%s strides: got %v; want %v", name, a.Strides(), want)
	}
}

// CheckExact compares represented numerical values, including signed zero.
// NaN payloads are not part of the portable numerical casting contract.
func CheckExact(t testing.TB, name string, a *np.Array) {
	t.Helper()
	if a.DType().IsInteger() || a.DType() == np.Bool {
		Check(t, name, a)
		return
	}
	Metadata(t, name, a)
	r := lookup(t, name)
	values, e := a.Complex128s()
	if e != nil {
		t.Fatal(e)
	}
	same := func(a, b float64) bool {
		return math.Float64bits(a) == math.Float64bits(b) || math.IsNaN(a) && math.IsNaN(b)
	}
	for i, v := range r.Values {
		x, y := number(t, v[0]), number(t, v[1])
		if !same(real(values[i]), x) || !same(imag(values[i]), y) {
			t.Errorf("%s[%d]: exact got %v; want (%g, %g)", name, i, values[i], x, y)
		}
	}
}

func Close(got, want float64) bool {
	return CloseDType(got, want, np.Float64)
}

func CloseDType(got, want float64, d np.DType) bool {
	if math.IsNaN(want) {
		return math.IsNaN(got)
	}
	if math.IsInf(want, 0) || math.IsInf(got, 0) {
		return got == want
	}
	if d == np.Float16 {
		return math.Abs(got-want) <= 1e-4+2e-3*math.Abs(want)
	}
	if d == np.Float32 || d == np.Complex64 {
		return math.Abs(got-want) <= 1e-6+2e-6*math.Abs(want)
	}
	return math.Abs(got-want) <= 1e-12+1e-11*math.Abs(want)
}

func Error(t testing.TB, name, class string, got, target error) {
	t.Helper()
	if reference.Errors[name] != class {
		t.Fatalf("%s NumPy exception: got %q; expected %q", name, reference.Errors[name], class)
	}
	if !errors.Is(got, target) {
		t.Errorf("%s: got %v; expected %v (NumPy %s)", name, got, target, class)
	}
}

// Case compares a generated NumPy success or error. The systematic grid uses
// the same fixtures and type-specific comparison as the original tests.
func Case(t testing.TB, name string, a *np.Array, err error) {
	t.Helper()
	if class, ok := reference.Errors[name]; ok {
		switch class {
		case "TypeError", "UFuncTypeError", "UFuncNoLoopError", "_UFuncInputCastingError", "_UFuncOutputCastingError", "_UFuncNoLoopError":
			Error(t, name, class, err, np.ErrDType)
		case "ValueError":
			Error(t, name, class, err, np.ErrValue)
		case "OverflowError":
			Error(t, name, class, err, np.ErrOverflow)
		default:
			t.Fatalf("%s: unhandled NumPy exception %s", name, class)
		}
		return
	}
	if err != nil {
		t.Errorf("%s: Go error %v; NumPy succeeded (%s)", name, err, lookup(t, name).DType)
		return
	}
	if a == nil {
		t.Fatalf("%s: nil result", name)
	}
	Check(t, name, a)
}
