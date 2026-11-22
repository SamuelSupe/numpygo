package random

import (
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
)

// A wrapped engine must share the exact stream via the BitGenerator contract,
// not be rejected merely because its Go type is not *Generator.
type wrappedBitGenerator struct{ BitGenerator }

func TestSharedBitGeneratorNumPy(t *testing.T) {
	for _, name := range []string{"MT19937", "PCG64", "PCG64DXSM", "Philox", "SFC64"} {
		t.Run(name, func(t *testing.T) {
			m := refcheck.Require(t)
			mg := refcheck.RequireValue[*Generator](t)
			mr := refcheck.RequireValue[*RandomState](t)
			bit := mg(NewBitGenerator(name, 12345))
			wrapped := wrappedBitGenerator{bit}
			g := mg(NewGeneratorFromBitGenerator(wrapped))
			r := mr(NewRandomStateFromBitGenerator(wrapped))
			r2 := mr(NewRandomStateFromBitGenerator(bit))
			check := func(s string, a *np.Array) { t.Helper(); refcheck.Check(t, "unified_random_"+name+"_"+s, a) }
			check("uint32", m(g.IntegersDType(0, 123456, np.Uint32, false, 3)))
			check("legacy_normal", m(r.StandardNormal(1)))
			s := r.GetState()
			check("cache_flag", np.ScalarOf(s.HasGauss))
			check("cache_value", np.ScalarOf(s.Gauss))
			check("modern", m(g.Random(3)))
			check("raw", m(bit.RandomRaw(5)))
			check("second_legacy", m(r2.StandardNormal(1)))
			check("cached", m(r.StandardNormal(1)))
			check("randint", m(r.Randint(-15, 21, np.Int8, 7)))
			check("modern_normal", m(g.StandardNormal(4)))
			check("second_cached", m(r2.StandardNormal(1)))
			if e := r.SetState(s); e != nil {
				t.Fatal(e)
			}
			check("replay_modern", m(g.Random(3)))
			check("replay_raw", m(bit.RandomRaw(5)))
			check("replay_cached", m(r.StandardNormal(1)))
			if name == "MT19937" {
				if e := r.Seed(42); e != nil {
					t.Fatal(e)
				}
				check("reseed_raw", m(bit.RandomRaw(4)))
				check("reseed_legacy", m(r.RandomSample(3)))
			} else {
				refcheck.Error(t, "unified_random_"+name+"_seed", "TypeError", r.Seed(42), np.ErrDType)
				check("after_rejected_seed", m(bit.RandomRaw(4)))
			}
		})
	}
}
