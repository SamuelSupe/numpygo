package refcheck

import (
	"slices"
	"testing"

	"github.com/SamuelSupe/numpygo"
)

func flexibleMetadata(t testing.TB, name string) (np.FlexibleDType, []int, []byte) {
	t.Helper()
	typ := RequireValue[[]uint8](t)(np.Values[uint8](Input(t, name+"_dtype")))
	d := RequireValue[np.FlexibleDType](t)(np.ParseFlexibleDType(string(typ)))
	shape64 := RequireValue[[]int64](t)(np.Values[int64](Input(t, name+"_shape")))
	shape := make([]int, len(shape64))
	for i, v := range shape64 {
		shape[i] = int(v)
	}
	raw := RequireValue[[]uint8](t)(np.Values[uint8](Input(t, name)))
	return d, shape, raw
}
func FlexibleInput(t testing.TB, name string) *np.Array {
	t.Helper()
	d, shape, raw := flexibleMetadata(t, name)
	return Require(t)(np.FromFlexibleBuffer(raw, d, shape, false))
}
func CheckFlexible(t testing.TB, name string, a *np.Array, layout bool) {
	t.Helper()
	d, shape, raw := flexibleMetadata(t, name)
	actual := RequireValue[np.FlexibleDType](t)(a.FlexibleType())
	if !d.Equal(actual) || !slices.Equal(a.Shape(), shape) {
		t.Fatalf("%s: dtype %s shape %v; NumPy %s %v", name, actual, a.Shape(), d, shape)
	}
	b := RequireValue[[]byte](t)(a.ToBytes("C"))
	if !slices.Equal(b, raw) {
		t.Errorf("%s: bytes %x; NumPy %x", name, b, raw)
	}
	if layout {
		v := make([]int64, a.NDim())
		for i, s := range a.Strides() {
			v[i] = int64(s)
		}
		Check(t, name+"_strides", Require(t)(np.WrapSlice(v, len(v))))
	}
}
