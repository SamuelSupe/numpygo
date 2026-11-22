package refcheck

import (
	"bytes"
	"github.com/SamuelSupe/numpygo"
	"slices"
	"testing"
)

// CheckCharacter compares logical slots rather than NumPy's process-local
// variable-length string metadata. Fixed-width storage is still byte exact.
func CheckCharacter(t testing.TB, name string, a *np.Array, layout bool) {
	t.Helper()
	dtype := string(RequireValue[[]uint8](t)(np.Values[uint8](Input(t, name+"_dtype"))))
	if dtype != "StringDType()" && dtype != "StringDType(coerce=False)" {
		CheckFlexible(t, name, a, layout)
		return
	}
	if a.DType() != np.StringDType {
		t.Fatalf("%s: dtype %s; NumPy %s", name, a.DType(), dtype)
	}
	strict := RequireValue[bool](t)(a.StringStrict())
	if strict != (dtype == "StringDType(coerce=False)") {
		t.Errorf("%s: incorrect coerce descriptor", name)
	}
	u := RequireValue[[]int64](t)(np.Values[int64](Input(t, name+"_shape")))
	shape := make([]int, len(u))
	for i, v := range u {
		shape[i] = int(v)
	}
	if !slices.Equal(shape, a.Shape()) {
		t.Fatalf("%s: shape %v; NumPy %v", name, a.Shape(), shape)
	}
	s := RequireValue[[]string](t)(a.CharacterValues())
	off := make([]int64, len(s)+1)
	var raw bytes.Buffer
	for i, v := range s {
		raw.WriteString(v)
		off[i+1] = int64(raw.Len())
	}
	Bytes(t, name, raw.Bytes())
	Check(t, name+"_offsets", Require(t)(np.WrapSlice(off, len(off))))
	if layout {
		strides := make([]int64, a.NDim())
		for i, v := range a.Strides() {
			strides[i] = int64(v)
		}
		Check(t, name+"_strides", Require(t)(np.WrapSlice(strides, len(strides))))
	}
}
