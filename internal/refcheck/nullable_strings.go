package refcheck

import (
	"github.com/SamuelSupe/numpygo"
	"math"
	"reflect"
	"slices"
	"strings"
	"testing"
)

func CheckNullableString(t testing.TB, name string, a *np.Array, layout bool) {
	t.Helper()
	m := Require(t)
	d := RequireValue[np.StringDescriptor](t)(a.StringDescriptor())
	Text(t, name+"_dtype", d.String())
	shape := RequireValue[[]int64](t)(np.Values[int64](Input(t, name+"_shape")))
	actual := a.Shape()
	if len(shape) != len(actual) {
		t.Fatalf("%s shape %v vs %v", name, actual, shape)
	}
	for i, v := range shape {
		if int(v) != actual[i] {
			t.Fatalf("%s shape %v vs %v", name, actual, shape)
		}
	}
	values, mask, e := a.StringValues()
	if e != nil {
		t.Fatal(e)
	}
	Check(t, name+"_mask", m(np.WrapSlice(mask, actual...)))
	offsets := make([]int64, len(values)+1)
	var b strings.Builder
	for i, v := range values {
		if mask[i] {
			equal := reflect.DeepEqual(v, d.NA)
			if x, ok := v.(float64); ok {
				if y, ok := d.NA.(float64); ok && math.IsNaN(x) && math.IsNaN(y) {
					equal = true
				}
			}
			if !equal {
				t.Fatalf("%s missing slot %d has %v; sentinel %v", name, i, v, d.NA)
			}
		} else {
			s, ok := v.(string)
			if !ok {
				t.Fatalf("%s nonmissing scalar: %T", name, v)
			}
			b.WriteString(s)
		}
		offsets[i+1] = int64(b.Len())
	}
	Bytes(t, name, []byte(b.String()))
	Check(t, name+"_offsets", m(np.WrapSlice(offsets, len(offsets))))
	if layout {
		strides := RequireValue[[]int64](t)(np.Values[int64](Input(t, name+"_strides")))
		v := make([]int64, a.NDim())
		for i, x := range a.Strides() {
			v[i] = int64(x)
		}
		if !slices.Equal(v, strides) {
			t.Fatalf("%s strides %v vs %v", name, v, strides)
		}
	}
}
