package random

import (
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"testing"
)

func TestBitGeneratorHandleParity(t *testing.T) {
	m := refcheck.Require(t)
	newGenerator := refcheck.RequireValue[*Generator](t)
	for _, name := range []string{"MT19937", "PCG64", "PCG64DXSM", "Philox", "SFC64"} {
		t.Run(name, func(t *testing.T) {
			prefix := "final_rng_" + name
			var b BitGenerator = newGenerator(NewBitGenerator(name, 123))
			refcheck.Check(t, prefix+"_base", np.ScalarOf(b.Algorithm() == name))
			c := b.(*Generator).Capsule()
			refcheck.Check(t, prefix+"_capsule_valid", np.ScalarOf(c.Algorithm() == name))
			refcheck.Check(t, prefix+"_capsule_raw", m(c.RandomRaw(4)))
			for _, variant := range []string{"ctypes", "cffi"} {
				b := newGenerator(NewBitGenerator(name, 123))
				var h *RawHandle
				if variant == "ctypes" {
					h = b.CTypes()
				} else {
					h = b.CFFI()
				}
				var u64 []uint64
				var u32 []uint32
				var d []float64
				if e := h.WithLock(func(s *LockedStream) error {
					for range 2 {
						u64 = append(u64, s.Uint64())
					}
					for range 3 {
						u32 = append(u32, s.Uint32())
					}
					for range 2 {
						d = append(d, s.Double())
					}
					return nil
				}); e != nil {
					t.Fatal(e)
				}
				p := prefix + "_" + variant
				refcheck.Check(t, p+"_u64", m(np.WrapSlice(u64, 2)))
				refcheck.Check(t, p+"_u32", m(np.WrapSlice(u32, 3)))
				refcheck.Check(t, p+"_double", m(np.WrapSlice(d, 2)))
				refcheck.Check(t, p+"_following", m(b.RandomRaw(4)))
			}
		})
	}
}
