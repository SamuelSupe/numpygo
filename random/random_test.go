package random_test

import (
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"github.com/SamuelSupe/numpygo/random"
	"math"
	"testing"
)

const seed = 12345

func TestSeedSequenceAndBitGenerators(t *testing.T) {
	must := refcheck.Require(t)
	s := random.Seed(seed)
	u32, err := s.GenerateState32(12)
	if err != nil {
		t.Fatal(err)
	}
	refcheck.Check(t, "rng_seed32", must(np.WrapSlice(u32, 12)))
	u64, err := s.GenerateState64(8)
	if err != nil {
		t.Fatal(err)
	}
	refcheck.Check(t, "rng_seed64", must(np.WrapSlice(u64, 8)))
	check := func(name string, ss *random.SeedSequence) {
		v, err := ss.GenerateState64(8)
		if err != nil {
			t.Fatal(err)
		}
		refcheck.Check(t, name, must(np.WrapSlice(v, 8)))
	}
	check("rng_seed_large", random.Seed(1<<60+123))
	check("rng_seed_words", random.NewSeedSequence(5, 7, 9, 11, 13, 15))
	children, err := s.Spawn(2)
	if err != nil {
		t.Fatal(err)
	}
	check("rng_spawn_0", children[0])
	check("rng_spawn_1", children[1])
	children, err = s.Spawn(1)
	if err != nil {
		t.Fatal(err)
	}
	check("rng_spawn_2", children[0])
	refcheck.Check(t, "rng_raw", must(random.NewPCG64(seed).RandomRaw(32)))
	refcheck.Check(t, "rng_dxsm_raw", must(random.NewPCG64DXSM(seed).RandomRaw(32)))
	pcg, dxsm := random.NewPCG64(seed), random.NewPCG64DXSM(seed)
	pcg.Advance128(1, 123)
	dxsm.Advance128(1, 123)
	refcheck.Check(t, "rng_advance_pcg", must(pcg.RandomRaw(16)))
	refcheck.Check(t, "rng_advance_dxsm", must(dxsm.RandomRaw(16)))
	g := random.DefaultRNG(seed)
	refcheck.Check(t, "rng_cache_first", must(g.Integers(0, 100, 3)))
	state := g.State()
	refcheck.Check(t, "rng_cache_double", must(g.Random(2)))
	refcheck.Check(t, "rng_cache_last", must(g.Integers(0, 100, 5)))
	if err := g.SetState(state); err != nil {
		t.Fatal(err)
	}
	refcheck.Check(t, "rng_cache_double", must(g.Random(2)))
	refcheck.Check(t, "rng_cache_last", must(g.Integers(0, 100, 5)))
}
func TestBoundedIntegers(t *testing.T) {
	must := refcheck.Require(t)
	for _, d := range []np.DType{np.Bool, np.Int8, np.Int16, np.Int32, np.Int64, np.Uint8, np.Uint16, np.Uint32, np.Uint64} {
		t.Run(d.String(), func(t *testing.T) {
			low, high := int64(-55), int64(100)
			if d.IsUnsigned() {
				low, high = 5, 230
			}
			if d == np.Bool {
				low, high = 0, 2
			}
			refcheck.Check(t, "rng_integer_"+d.String(), must(random.DefaultRNG(seed).IntegersDType(low, high, d, false, 64)))
		})
	}
	refcheck.Check(t, "rng_uint64_full", must(random.DefaultRNG(seed).UnsignedIntegers(0, math.MaxUint64, np.Uint64, true, 32)))
	refcheck.Check(t, "rng_int64_full", must(random.DefaultRNG(seed).IntegersDType(math.MinInt64, math.MaxInt64, np.Int64, true, 32)))
}
func TestDistributions(t *testing.T) {
	tests := []struct {
		name   string
		sample func(*random.Generator) (*np.Array, error)
	}{
		{"random", func(g *random.Generator) (*np.Array, error) { return g.Random(32) }},
		{"random32", func(g *random.Generator) (*np.Array, error) { return g.RandomDType(np.Float32, 32) }},
		{"normal", func(g *random.Generator) (*np.Array, error) { return g.Normal(2, .5, 64) }},
		{"standard_normal", func(g *random.Generator) (*np.Array, error) { return g.StandardNormal(1024) }},
		{"standard_normal32", func(g *random.Generator) (*np.Array, error) { return g.StandardNormalDType(np.Float32, 512) }},
		{"standard_exponential", func(g *random.Generator) (*np.Array, error) { return g.StandardExponential(1024) }},
		{"standard_exponential32", func(g *random.Generator) (*np.Array, error) { return g.StandardExponentialDType(np.Float32, 512) }},
		{"standard_exponential_inv", func(g *random.Generator) (*np.Array, error) { return g.StandardExponentialInverse(64) }},
		{"uniform", func(g *random.Generator) (*np.Array, error) { return g.Uniform(-4, 8, 32) }},
		{"exponential", func(g *random.Generator) (*np.Array, error) { return g.Exponential(2.5, 64) }},
		{"gamma_small", func(g *random.Generator) (*np.Array, error) { return g.Gamma(.4, 2, 64) }},
		{"gamma_big", func(g *random.Generator) (*np.Array, error) { return g.Gamma(5, .25, 64) }},
		{"standard_gamma", func(g *random.Generator) (*np.Array, error) { return g.StandardGamma(2.5, 64) }},
		{"beta_small", func(g *random.Generator) (*np.Array, error) { return g.Beta(.4, .7, 64) }},
		{"beta_big", func(g *random.Generator) (*np.Array, error) { return g.Beta(2, 4, 64) }},
		{"chisquare", func(g *random.Generator) (*np.Array, error) { return g.Chisquare(3, 64) }},
		{"f", func(g *random.Generator) (*np.Array, error) { return g.F(3, 5, 64) }},
		{"standard_cauchy", func(g *random.Generator) (*np.Array, error) { return g.StandardCauchy(64) }},
		{"standard_t", func(g *random.Generator) (*np.Array, error) { return g.StandardT(4, 64) }},
		{"pareto", func(g *random.Generator) (*np.Array, error) { return g.Pareto(3, 64) }},
		{"weibull", func(g *random.Generator) (*np.Array, error) { return g.Weibull(2.5, 64) }},
		{"power", func(g *random.Generator) (*np.Array, error) { return g.Power(3, 64) }},
		{"laplace", func(g *random.Generator) (*np.Array, error) { return g.Laplace(2, 3, 64) }},
		{"gumbel", func(g *random.Generator) (*np.Array, error) { return g.Gumbel(-1, 2, 64) }},
		{"logistic", func(g *random.Generator) (*np.Array, error) { return g.Logistic(2, 1.5, 64) }},
		{"lognormal", func(g *random.Generator) (*np.Array, error) { return g.Lognormal(.5, .25, 64) }},
		{"rayleigh", func(g *random.Generator) (*np.Array, error) { return g.Rayleigh(2, 64) }},
		{"poisson_small", func(g *random.Generator) (*np.Array, error) { return g.Poisson(3, 64) }},
		{"poisson_big", func(g *random.Generator) (*np.Array, error) { return g.Poisson(20, 64) }},
		{"negative_binomial", func(g *random.Generator) (*np.Array, error) { return g.NegativeBinomial(3, .4, 64) }},
		{"binomial_small", func(g *random.Generator) (*np.Array, error) { return g.Binomial(10, .2, 64) }},
		{"binomial_big", func(g *random.Generator) (*np.Array, error) { return g.Binomial(200, .7, 64) }},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			refcheck.Check(t, "rng_"+tc.name, refcheck.Require(t)(tc.sample(random.DefaultRNG(seed))))
		})
	}
}
func TestRandomSelection(t *testing.T) {
	must := refcheck.Require(t)
	cases := []struct {
		name          string
		population, n int
		opt           random.ChoiceOptions
	}{
		{"choice_replace", 10, 20, random.ChoiceOptions{}},
		{"choice_no_replace", 9, 5, random.ChoiceOptions{WithoutReplacement: true}},
		{"choice_no_shuffle", 12, 8, random.ChoiceOptions{WithoutReplacement: true, NoShuffle: true}},
		{"choice_weighted", 4, 20, random.ChoiceOptions{Probabilities: []float64{.1, .2, 0, .7}}},
		{"choice_weighted_no_replace", 4, 3, random.ChoiceOptions{WithoutReplacement: true, Probabilities: []float64{.1, .2, 0, .7}}},
		{"choice_large", 12000, 300, random.ChoiceOptions{WithoutReplacement: true}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			refcheck.Check(t, "rng_"+tc.name, must(random.DefaultRNG(seed).Choice(tc.population, tc.opt, tc.n)))
		})
	}
	refcheck.Check(t, "rng_permutation", must(random.DefaultRNG(seed).Permutation(20)))
	a := must(must(np.ArangeInt64(0, 12, 1)).Astype(np.Int32))
	a = must(a.Reshape(3, 4))
	refcheck.Check(t, "rng_permutation_array", must(random.DefaultRNG(seed).PermutationArray(a, 1)))
	if err := random.DefaultRNG(seed).Shuffle(a, 1); err != nil {
		t.Fatal(err)
	}
	refcheck.Check(t, "rng_shuffle", a)
}
func TestRandomErrors(t *testing.T) {
	g := random.DefaultRNG(seed)
	_, err := g.Normal(0, -1)
	refcheck.Error(t, "rng_invalid_normal", "ValueError", err, np.ErrValue)
	_, err = g.Integers(3, 2)
	refcheck.Error(t, "rng_invalid_integer", "ValueError", err, np.ErrValue)
	_, err = g.Choice(2, random.ChoiceOptions{Probabilities: []float64{.2, .2}}, 3)
	refcheck.Error(t, "rng_invalid_probability", "ValueError", err, np.ErrValue)
	_, err = g.Choice(2, random.ChoiceOptions{WithoutReplacement: true}, 3)
	refcheck.Error(t, "rng_invalid_choice", "ValueError", err, np.ErrValue)
	_, err = g.Poisson(-1)
	refcheck.Error(t, "rng_invalid_poisson", "ValueError", err, np.ErrValue)
	_, err = g.Binomial(5, 1.2)
	refcheck.Error(t, "rng_invalid_binomial", "ValueError", err, np.ErrValue)
}
