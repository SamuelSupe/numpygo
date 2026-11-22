package random_test

import (
	"slices"
	"strings"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"github.com/SamuelSupe/numpygo/random"
)

func TestBitGeneratorFamilies(t *testing.T) {
	constructors := map[string]func(uint64) *random.Generator{
		"MT19937": random.NewMT19937, "Philox": random.NewPhilox, "SFC64": random.NewSFC64,
		"PCG64": random.NewPCG64, "PCG64DXSM": random.NewPCG64DXSM,
	}
	for name, makeGenerator := range constructors {
		t.Run(name, func(t *testing.T) {
			tag := "bit_" + strings.ToLower(name) + "_"
			must := refcheck.Require(t)
			check := func(n string, a *np.Array) { t.Helper(); refcheck.Check(t, tag+n, a) }
			check32 := func(n string, v []uint32) { check(n, must(np.FromSlice(v, len(v)))) }
			check64 := func(n string, v []uint64) { check(n, must(np.FromSlice(v, len(v)))) }
			g := makeGenerator(12345)
			length := 35
			if name == "MT19937" {
				length = 700
			}
			check("raw", must(g.RandomRaw(length)))
			g = makeGenerator(12345)
			if g.Algorithm() != name {
				t.Fatalf("algorithm %s", g.Algorithm())
			}
			check("double", must(g.Random(9)))
			check("single", must(g.RandomDType(np.Float32, 9)))
			check("i64", must(g.Integers(-1<<48, 1<<48, 13)))
			check("u16", must(g.UnsignedIntegers(0, 65535, np.Uint16, true, 17)))
			check("normal", must(g.StandardNormal(9)))
			state := g.State()
			if name == "MT19937" {
				check32("state_key", state.MT.Key[:])
				check("state_pos", np.ScalarOf(int64(state.MT.Pos)))
			} else {
				check("state_has32", np.ScalarOf(state.HasUint32))
				check("state_cached", np.ScalarOf(state.Uint32))
				switch name {
				case "Philox":
					check64("state_counter", state.Philox.Counter[:])
					check64("state_philoxkey", state.Philox.Key[:])
					check64("state_buffer", state.Philox.Buffer[:])
					check("state_buffer_pos", np.ScalarOf(int64(state.Philox.BufferPos)))
				case "SFC64":
					check64("state_sfc", state.SFC[:])
				default:
					check64("state_pcg", []uint64{state.StateLow, state.StateHigh})
					check64("state_inc", []uint64{state.IncrementLow, state.IncrementHigh})
				}
			}
			want := must(g.RandomDType(np.Float32, 5))
			check("restored", want)
			if err := g.SetState(state); err != nil {
				t.Fatal(err)
			}
			check("restored", must(g.RandomDType(np.Float32, 5)))
			g = makeGenerator(12345)
			children, e := g.Spawn(2)
			if e != nil {
				t.Fatal(e)
			}
			for i, child := range children {
				check("spawn_"+string(rune('0'+i)), must(child.RandomRaw(8)))
			}
			check32("seed_entropy", g.SeedSequence().Entropy())
			locked := makeGenerator(12345)
			var u32 []uint32
			var u64 []uint64
			var doubles []float64
			e = locked.WithLock(func(s *random.LockedStream) error {
				for i := 0; i < 3; i++ {
					u32 = append(u32, s.Uint32())
				}
				for i := 0; i < 5; i++ {
					u64 = append(u64, s.Uint64())
				}
				for i := 0; i < 4; i++ {
					doubles = append(doubles, s.Double())
				}
				return nil
			})
			if e != nil {
				t.Fatal(e)
			}
			check32("lock_u32", u32)
			check64("lock_u64", u64)
			check("lock_double", must(np.FromFloat64(doubles, len(doubles))))
			check("lock_followup", must(locked.RandomRaw(5)))
			if name != "SFC64" {
				must(g.RandomRaw(3))
				j, e := g.Jumped(2)
				if e != nil {
					t.Fatal(e)
				}
				check("jumped", must(j.RandomRaw(10)))
				check("jumped_original", must(g.RandomRaw(6)))
			}
			if name != "SFC64" && name != "MT19937" {
				g = makeGenerator(12345)
				must(g.RandomRaw(3))
				if e = g.Advance128(64, 12345); e != nil {
					t.Fatal(e)
				}
				check("advance", must(g.RandomRaw(11)))
			}
		})
	}
	must := refcheck.Require(t)
	ph := random.PhiloxFromKey([2]uint64{123, 456}, [4]uint64{^uint64(0), 7})
	refcheck.Check(t, "bit_philox_explicit", must(ph.RandomRaw(12)))
	if err := ph.Advance256([4]uint64{17, 0, 1, 1}); err != nil {
		t.Fatal(err)
	}
	refcheck.Check(t, "bit_philox_advance256", must(ph.RandomRaw(7)))
}

func TestSeedSequenceMetadata(t *testing.T) {
	must := refcheck.Require(t)
	s := random.NewSeedSequence(1, 2, 3, 4)
	children, err := s.Spawn(3)
	if err != nil {
		t.Fatal(err)
	}
	for _, item := range []struct {
		name string
		s    *random.SeedSequence
	}{{"parent", s}, {"child", children[1]}} {
		state := item.s.State()
		pool := item.s.Pool()
		tag := "bit_seed_" + item.name + "_"
		refcheck.Check(t, tag+"entropy", must(np.FromSlice(state.Entropy, len(state.Entropy))))
		refcheck.Check(t, tag+"pool", must(np.FromSlice(pool[:], 4)))
		refcheck.Check(t, tag+"pool_size", np.ScalarOf(int64(state.PoolSize)))
		refcheck.Check(t, tag+"spawn_key", must(np.FromSlice(state.SpawnKey, len(state.SpawnKey))))
		refcheck.Check(t, tag+"children", np.ScalarOf(state.NChildrenSpawned))
		if !slices.Equal(state.Entropy, item.s.Entropy()) || !slices.Equal(state.SpawnKey, item.s.SpawnKey()) ||
			state.PoolSize != item.s.PoolSize() || state.NChildrenSpawned != item.s.NChildrenSpawned() {
			t.Fatal("inconsistent SeedSequence snapshot")
		}
	}
}

type legacyOperation func(*random.RandomState) (*np.Array, error)

func TestLegacyRandomStateDistributions(t *testing.T) {
	shape := []int{2, 3}
	ops := map[string]legacyOperation{
		"random":                     func(r *random.RandomState) (*np.Array, error) { return r.Random(shape...) },
		"random_sample":              func(r *random.RandomState) (*np.Array, error) { return r.RandomSample(shape...) },
		"rand":                       func(r *random.RandomState) (*np.Array, error) { return r.Rand(shape...) },
		"randn":                      func(r *random.RandomState) (*np.Array, error) { return r.Randn(shape...) },
		"standard_normal":            func(r *random.RandomState) (*np.Array, error) { return r.StandardNormal(shape...) },
		"normal":                     func(r *random.RandomState) (*np.Array, error) { return r.Normal(1.25, 2.5, shape...) },
		"uniform":                    func(r *random.RandomState) (*np.Array, error) { return r.Uniform(-3., 4., shape...) },
		"standard_exponential":       func(r *random.RandomState) (*np.Array, error) { return r.StandardExponential(shape...) },
		"exponential":                func(r *random.RandomState) (*np.Array, error) { return r.Exponential(2.3, shape...) },
		"gamma":                      func(r *random.RandomState) (*np.Array, error) { return r.Gamma(2.5, 1.3, shape...) },
		"gamma_small":                func(r *random.RandomState) (*np.Array, error) { return r.Gamma(.24, 2., shape...) },
		"standard_gamma":             func(r *random.RandomState) (*np.Array, error) { return r.StandardGamma(.7, shape...) },
		"beta":                       func(r *random.RandomState) (*np.Array, error) { return r.Beta(2.5, 3.2, shape...) },
		"beta_small":                 func(r *random.RandomState) (*np.Array, error) { return r.Beta(.3, .4, shape...) },
		"chisquare":                  func(r *random.RandomState) (*np.Array, error) { return r.Chisquare(3.3, shape...) },
		"f":                          func(r *random.RandomState) (*np.Array, error) { return r.F(5., 2.5, shape...) },
		"standard_cauchy":            func(r *random.RandomState) (*np.Array, error) { return r.StandardCauchy(shape...) },
		"standard_t":                 func(r *random.RandomState) (*np.Array, error) { return r.StandardT(4.2, shape...) },
		"pareto":                     func(r *random.RandomState) (*np.Array, error) { return r.Pareto(2.2, shape...) },
		"weibull":                    func(r *random.RandomState) (*np.Array, error) { return r.Weibull(1.7, shape...) },
		"power":                      func(r *random.RandomState) (*np.Array, error) { return r.Power(2.3, shape...) },
		"laplace":                    func(r *random.RandomState) (*np.Array, error) { return r.Laplace(1., 2., shape...) },
		"gumbel":                     func(r *random.RandomState) (*np.Array, error) { return r.Gumbel(-1., .8, shape...) },
		"logistic":                   func(r *random.RandomState) (*np.Array, error) { return r.Logistic(1.2, .7, shape...) },
		"lognormal":                  func(r *random.RandomState) (*np.Array, error) { return r.Lognormal(-.2, .7, shape...) },
		"rayleigh":                   func(r *random.RandomState) (*np.Array, error) { return r.Rayleigh(2., shape...) },
		"poisson":                    func(r *random.RandomState) (*np.Array, error) { return r.Poisson(3.5, shape...) },
		"poisson_large":              func(r *random.RandomState) (*np.Array, error) { return r.Poisson(35., shape...) },
		"negative_binomial":          func(r *random.RandomState) (*np.Array, error) { return r.NegativeBinomial(4.1, .3, shape...) },
		"binomial":                   func(r *random.RandomState) (*np.Array, error) { return r.Binomial(20, .3, shape...) },
		"binomial_large":             func(r *random.RandomState) (*np.Array, error) { return r.Binomial(400, .43, shape...) },
		"geometric":                  func(r *random.RandomState) (*np.Array, error) { return r.Geometric(.16, shape...) },
		"geometric_search":           func(r *random.RandomState) (*np.Array, error) { return r.Geometric(.65, shape...) },
		"triangular":                 func(r *random.RandomState) (*np.Array, error) { return r.Triangular(-2., 1., 5., shape...) },
		"wald":                       func(r *random.RandomState) (*np.Array, error) { return r.Wald(1.7, 2.4, shape...) },
		"logseries":                  func(r *random.RandomState) (*np.Array, error) { return r.Logseries(.78, shape...) },
		"zipf":                       func(r *random.RandomState) (*np.Array, error) { return r.Zipf(2.1, shape...) },
		"noncentral_chisquare":       func(r *random.RandomState) (*np.Array, error) { return r.NoncentralChisquare(3.5, 2.1, shape...) },
		"noncentral_chisquare_lowdf": func(r *random.RandomState) (*np.Array, error) { return r.NoncentralChisquare(.5, 2., shape...) },
		"noncentral_f":               func(r *random.RandomState) (*np.Array, error) { return r.NoncentralF(4., 3., 1.8, shape...) },
		"vonmises":                   func(r *random.RandomState) (*np.Array, error) { return r.Vonmises(.3, 3., shape...) },
		"hypergeometric":             func(r *random.RandomState) (*np.Array, error) { return r.Hypergeometric(19, 30, 7, shape...) },
		"hypergeometric_large":       func(r *random.RandomState) (*np.Array, error) { return r.Hypergeometric(40, 67, 62, shape...) },
		"multinomial": func(r *random.RandomState) (*np.Array, error) {
			return r.Multinomial(20, []float64{.2, .35, .45}, shape...)
		},
		"dirichlet":       func(r *random.RandomState) (*np.Array, error) { return r.Dirichlet([]float64{.5, 2., 3.}, shape...) },
		"dirichlet_small": func(r *random.RandomState) (*np.Array, error) { return r.Dirichlet([]float64{.05, .08, .09}, shape...) },
		"multivariate_normal": func(r *random.RandomState) (*np.Array, error) {
			c, e := np.FromFloat64([]float64{4, 0, 0, 1}, 2, 2)
			if e != nil {
				return nil, e
			}
			return r.MultivariateNormal([]float64{1., 2.}, c, shape...)
		},
		"randint":         func(r *random.RandomState) (*np.Array, error) { return r.Randint(-7, 29, np.Int64, shape...) },
		"randint_big":     func(r *random.RandomState) (*np.Array, error) { return r.Randint(-1<<48, 1<<48, np.Int64, shape...) },
		"random_integers": func(r *random.RandomState) (*np.Array, error) { return r.RandomIntegers(-2, 6, shape...) },
		"tomaxint":        func(r *random.RandomState) (*np.Array, error) { return r.ToMaxInt(shape...) },
		"choice":          func(r *random.RandomState) (*np.Array, error) { return r.Choice(7, random.ChoiceOptions{}, shape...) },
	}
	for name, op := range ops {
		t.Run(name, func(t *testing.T) {
			must := refcheck.Require(t)
			refcheck.Check(t, "legacy_"+name, must(op(random.NewRandomState(12345))))
			if name != "tomaxint" {
				r := random.GlobalRandomState()
				r.Seed(12345)
				refcheck.Check(t, "legacy_global_"+name, must(op(r)))
			}
		})
	}
	for _, name := range []string{"ranf", "sample"} {
		r := random.GlobalRandomState()
		r.Seed(12345)
		refcheck.Check(t, "legacy_global_"+name, refcheck.Require(t)(r.RandomSample(shape...)))
	}
}

func TestLegacySelectionAndState(t *testing.T) {
	must := refcheck.Require(t)
	for _, item := range []struct {
		name string
		n    int
		o    random.ChoiceOptions
	}{
		{"without", 10, random.ChoiceOptions{WithoutReplacement: true}},
		{"weighted", 4, random.ChoiceOptions{Probabilities: []float64{.1, .2, .3, .4}}},
		{"weighted_without", 4, random.ChoiceOptions{WithoutReplacement: true, Probabilities: []float64{.1, .2, .3, .4}}},
	} {
		r := random.NewRandomState(12345)
		refcheck.Check(t, "legacy_choice_"+item.name, must(r.Choice(item.n, item.o, 3)))
	}
	for _, d := range []np.DType{np.Int8, np.Uint16, np.Int32, np.Int64} {
		r := random.NewRandomState(12345)
		refcheck.Check(t, "legacy_randint_"+d.String(), must(r.Randint(0, 93, d, 19)))
	}
	for _, item := range []struct {
		tag string
		r   *random.RandomState
	}{{"legacy_", random.NewRandomState(12345)}, {"legacy_global_", random.GlobalRandomState()}} {
		r, tag := item.r, item.tag
		r.Seed(12345)
		bytes, e := r.Bytes(23)
		if e != nil {
			t.Fatal(e)
		}
		refcheck.Check(t, tag+"bytes", must(np.FromSlice(bytes, len(bytes))))
		r.Seed(12345)
		refcheck.Check(t, tag+"permutation", must(r.Permutation(12)))
		a := must(np.ArangeInt64(0, 15, 1))
		a = must(a.Reshape(5, 3))
		if e = r.Shuffle(a, 0); e != nil {
			t.Fatal(e)
		}
		refcheck.Check(t, tag+"shuffle", a)
		r.Seed(12345)
		must(r.StandardNormal(3))
		state := r.GetState()
		refcheck.Check(t, tag+"state_key", must(np.FromSlice(state.MT.Key[:], 624)))
		refcheck.Check(t, tag+"state_pos", np.ScalarOf(int64(state.MT.Pos)))
		refcheck.Check(t, tag+"state_has_gauss", np.ScalarOf(state.HasGauss))
		refcheck.Check(t, tag+"state_gauss", np.Scalar(state.Gauss))
		refcheck.Check(t, tag+"state_future", must(r.StandardNormal(7)))
		if e = r.SetState(state); e != nil {
			t.Fatal(e)
		}
		refcheck.Check(t, tag+"state_restored", must(r.StandardNormal(7)))
		r.Seed(54321)
		refcheck.Check(t, tag+"reseed", must(r.RandomSample(8)))
	}
	r, e := random.NewRandomStateArray(1, 2, 3, 4)
	if e != nil {
		t.Fatal(e)
	}
	refcheck.Check(t, "legacy_array_seed", must(r.RandomSample(11)))
	r.Seed(12345)
	must(r.Binomial(0, .5, 8))
	refcheck.Check(t, "legacy_binomial_zero_followup", must(r.RandomSample(4)))
	r.Seed(12345)
	must(r.Binomial(20, 0., 8))
	refcheck.Check(t, "legacy_binomial_pzero_followup", must(r.RandomSample(4)))
	refcheck.Error(t, "legacy_invalid_seed", "ValueError", r.SeedInt(-1), np.ErrValue)
	refcheck.Error(t, "legacy_invalid_seed_array", "ValueError", r.SeedArray(), np.ErrValue)
	_, e = r.Randint(5, 2, np.Int64, 3)
	refcheck.Error(t, "legacy_invalid_randint", "ValueError", e, np.ErrValue)
	_, e = r.Choice(2, random.ChoiceOptions{WithoutReplacement: true}, 3)
	refcheck.Error(t, "legacy_invalid_choice", "ValueError", e, np.ErrValue)
}
