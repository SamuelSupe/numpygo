package refcheck

import (
	"math/big"
	"slices"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/x87"
)

// Extended references contain the ten significant bytes of each x87 slot.
// NumPy's six padding bytes in a 16-byte slot are intentionally unspecified.
func ExtendedInput(t testing.TB, name string, d np.DType) *np.Array {
	t.Helper()
	m := Require(t)
	raw, e := np.Values[uint8](Input(t, name))
	if e != nil {
		t.Fatal(e)
	}
	shape64, e := np.Values[int64](Input(t, name+"_shape"))
	if e != nil {
		t.Fatal(e)
	}
	shape := make([]int, len(shape64))
	for i, n := range shape64 {
		shape[i] = int(n)
	}
	if len(raw)%10 != 0 {
		t.Fatal("invalid x87 reference")
	}
	parts := make([]x87.Float, len(raw)/10)
	for i := range parts {
		parts[i], e = x87.FromRaw(raw[10*i : 10*(i+1)])
		if e != nil {
			t.Fatal(e)
		}
	}
	if d == np.Float128 {
		return m(np.WrapSlice(parts, shape...))
	}
	if d != np.Complex256 || len(parts)%2 != 0 {
		t.Fatal("invalid extended reference precision")
	}
	v := make([]x87.Complex, len(parts)/2)
	for i := range v {
		v[i] = x87.Complex{Real: parts[2*i], Imag: parts[2*i+1]}
	}
	return m(np.WrapSlice(v, shape...))
}
func extendedParts(t testing.TB, a *np.Array) []x87.Float {
	t.Helper()
	if a.DType() == np.Float128 {
		x, e := np.Values[x87.Float](a)
		if e != nil {
			t.Fatal(e)
		}
		return x
	}
	if a.DType() != np.Complex256 {
		t.Fatalf("expected extended output, got %s", a.DType())
	}
	z, e := np.Values[x87.Complex](a)
	if e != nil {
		t.Fatal(e)
	}
	x := make([]x87.Float, 0, 2*len(z))
	for _, v := range z {
		x = append(x, v.Real, v.Imag)
	}
	return x
}
func x87ULP(x x87.Float) *big.Rat {
	forward, back := x.Nextafter(x87.Inf(false)), x.Nextafter(x87.Inf(true))
	var width big.Rat
	for _, v := range []x87.Float{forward, back} {
		if v.IsInf() {
			continue
		}
		d := new(big.Rat).Sub(x.Rat(), v.Rat())
		d.Abs(d)
		if d.Cmp(&width) > 0 {
			width.Set(d)
		}
	}
	return &width
}

// CheckExtended compares the extended exponent range and actual x87 values
// with a bounded ULP error. No conversion through binary64 is involved.
func CheckExtended(t testing.TB, name string, a *np.Array, d np.DType, ulps int64) {
	t.Helper()
	w := ExtendedInput(t, name, d)
	if a.DType() != d || !slices.Equal(a.Shape(), w.Shape()) {
		t.Fatalf("%s metadata: %s %v; want %s %v", name, a.DType(), a.Shape(), d, w.Shape())
	}
	got, want := extendedParts(t, a), extendedParts(t, w)
	for i, x := range want {
		v := got[i]
		if x.IsNaN() && v.IsNaN() {
			continue
		}
		if x.Equal(v) && (!x.IsZero() || x.Signbit() == v.Signbit()) {
			continue
		}
		if x.IsZero() || !x.IsFinite() || !v.IsFinite() {
			t.Errorf("%s[%d]: got %s want %s", name, i, v, x)
			continue
		}
		delta := new(big.Rat).Sub(x.Rat(), v.Rat())
		delta.Abs(delta)
		unit := x87ULP(x)
		if delta.Cmp(new(big.Rat).Mul(unit, new(big.Rat).SetInt64(ulps))) > 0 {
			err, _ := new(big.Rat).Quo(delta, unit).Float64()
			t.Errorf("%s[%d]: got %s want %s (%.2f ULP)", name, i, v, x, err)
		}
	}
}

// CheckExtendedFFT permits cancellation-sensitive near-zero outputs using a
// scale-relative absolute bound. The scale is itself an exact x87 value.
// Exact cancellation/precision cases are tested separately with ULP checks.
func CheckExtendedFFT(t testing.TB, name string, a *np.Array, d np.DType, ulps int64) {
	t.Helper()
	w := ExtendedInput(t, name, d)
	if a.DType() != d || !slices.Equal(a.Shape(), w.Shape()) {
		t.Fatalf("%s metadata: %s %v; want %s %v", name, a.DType(), a.Shape(), d, w.Shape())
	}
	got, want := extendedParts(t, a), extendedParts(t, w)
	scale := x87.Zero(false)
	for _, x := range want {
		if !x.IsFinite() {
			continue
		}
		if c, _ := x.Abs().Compare(scale); c > 0 {
			scale = x.Abs()
		}
	}
	tol := new(big.Rat).Mul(x87ULP(scale), new(big.Rat).SetInt64(ulps))
	for i, x := range want {
		v := got[i]
		if x.IsNaN() && v.IsNaN() || x.Equal(v) {
			continue
		}
		if !x.IsFinite() || !v.IsFinite() {
			t.Errorf("%s[%d]: got %s want %s", name, i, v, x)
			continue
		}
		delta := new(big.Rat).Sub(x.Rat(), v.Rat())
		delta.Abs(delta)
		if delta.Cmp(tol) > 0 {
			t.Errorf("%s[%d]: got %s want %s, scale %s (tolerance %d ULP of scale)", name, i, v, x, scale, ulps)
		}
	}
}
