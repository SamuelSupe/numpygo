package np_test

import (
	"crypto/sha256"
	"encoding/binary"
	"math"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
)

func TestBinary16ExhaustiveNumPy(t *testing.T) {
	m := refcheck.Require(t)
	values := make([]np.Half, 65536)
	for i := range values {
		values[i] = np.Half(i)
	}
	a := m(np.FromSlice(values, len(values)))
	check16 := func(name string, v []np.Half) {
		t.Helper()
		b := make([]byte, 0, 2*len(v))
		for _, x := range v {
			raw := uint16(x)
			if raw&0x7c00 == 0x7c00 && raw&0x03ff != 0 {
				raw = 0x7e00
			}
			b = binary.LittleEndian.AppendUint16(b, raw)
		}
		h := sha256.Sum256(b)
		refcheck.Bytes(t, "closure_"+name, h[:])
	}
	decoded := m(a.Astype(np.Float64))
	f64 := refcheck.RequireValue[[]float64](t)(np.Values[float64](decoded))
	b := make([]byte, 0, len(f64)*8)
	for _, x := range f64 {
		raw := math.Float64bits(x)
		if math.IsNaN(x) {
			raw = 0x7ff8000000000000
		}
		b = binary.LittleEndian.AppendUint64(b, raw)
	}
	h := sha256.Sum256(b)
	refcheck.Bytes(t, "closure_half_decode", h[:])
	halves := func(x *np.Array) []np.Half { return refcheck.RequireValue[[]np.Half](t)(np.Values[np.Half](x)) }
	check16("half_roundtrip", halves(m(decoded.Astype(np.Float16))))
	for _, tc := range []struct {
		name   string
		toward np.Half
	}{
		{"half_next_pos", np.Half(0x7c00)}, {"half_next_neg", np.Half(0xfc00)}, {"half_next_zero", np.Half(0)},
	} {
		t.Run(tc.name, func(t *testing.T) { check16(tc.name, halves(m(np.Nextafter(a, np.ScalarOf(tc.toward))))) })
	}
	check16("half_spacing", halves(m(np.Spacing(a))))
	around := make([]float64, 0, 3*0x7c00)
	for i := 0; i < 0x7c00; i++ {
		mid := 65520.0
		if i < 0x7bff {
			mid = (np.Half(i).Float64() + np.Half(i+1).Float64()) / 2
		}
		around = append(around, math.Nextafter(mid, math.Inf(-1)), mid, math.Nextafter(mid, math.Inf(1)))
	}
	rounded := make([]np.Half, 2*len(around))
	for i, x := range around {
		rounded[i] = np.HalfFromFloat64(x)
		rounded[len(around)+i] = np.HalfFromFloat64(-x)
	}
	check16("half_round_ties", rounded)
}
