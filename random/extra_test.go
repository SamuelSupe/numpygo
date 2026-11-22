package random_test

import (
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"github.com/SamuelSupe/numpygo/random"
	"testing"
)

func TestAdditionalDistributions(t *testing.T) {
	tests := []struct {
		name   string
		sample func(*random.Generator) (*np.Array, error)
	}{
		{"geometric_search", func(g *random.Generator) (*np.Array, error) { return g.Geometric(.6, 40) }},
		{"geometric_inv", func(g *random.Generator) (*np.Array, error) { return g.Geometric(.1, 40) }},
		{"triangular", func(g *random.Generator) (*np.Array, error) { return g.Triangular(-2, 1, 5, 40) }},
		{"wald", func(g *random.Generator) (*np.Array, error) { return g.Wald(2, 3, 40) }},
		{"logseries", func(g *random.Generator) (*np.Array, error) { return g.Logseries(.8, 40) }},
		{"zipf", func(g *random.Generator) (*np.Array, error) { return g.Zipf(2, 40) }},
		{"noncentral_chisquare", func(g *random.Generator) (*np.Array, error) { return g.NoncentralChisquare(3, 2, 40) }},
		{"noncentral_chisquare_small", func(g *random.Generator) (*np.Array, error) { return g.NoncentralChisquare(.5, 2, 40) }},
		{"noncentral_f", func(g *random.Generator) (*np.Array, error) { return g.NoncentralF(3, 4, 2, 40) }},
		{"vonmises", func(g *random.Generator) (*np.Array, error) { return g.Vonmises(.3, 2, 40) }},
		{"multinomial", func(g *random.Generator) (*np.Array, error) { return g.Multinomial(10, []float64{.1, .2, .7}, 20) }},
		{"dirichlet", func(g *random.Generator) (*np.Array, error) { return g.Dirichlet([]float64{1, 2, 3}, 20) }},
		{"dirichlet_small", func(g *random.Generator) (*np.Array, error) { return g.Dirichlet([]float64{.03, .04, .05}, 20) }},
		{"multivariate_normal", func(g *random.Generator) (*np.Array, error) {
			return g.MultivariateNormal([]float64{1, 2}, refcheck.Require(t)(np.FromFloat64([]float64{4, 1, 1, 3}, 2, 2)), "cholesky", 20)
		}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			refcheck.Check(t, "rng_"+tc.name, refcheck.Require(t)(tc.sample(random.DefaultRNG(seed))))
		})
	}
}
func TestAdditionalRandomOperations(t *testing.T) {
	must := refcheck.Require(t)
	g := random.DefaultRNG(seed)
	by, err := g.Bytes(17)
	if err != nil {
		t.Fatal(err)
	}
	refcheck.Check(t, "rng_bytes", must(np.WrapSlice(by, len(by))))
	children, err := random.DefaultRNG(seed).Spawn(2)
	if err != nil {
		t.Fatal(err)
	}
	refcheck.Check(t, "rng_generator_spawn_0", must(children[0].Random(16)))
	refcheck.Check(t, "rng_generator_spawn_1", must(children[1].Random(16)))
	x := must(must(np.ArangeInt64(0, 12, 1)).Astype(np.Int32))
	x = must(x.Reshape(3, 4))
	refcheck.Check(t, "rng_permuted", must(random.DefaultRNG(seed).Permuted(x, np.Int(1))))
	refcheck.Check(t, "rng_permuted_flat", must(random.DefaultRNG(seed).Permuted(x, nil)))
	refcheck.Check(t, "rng_hypergeometric_small", must(random.DefaultRNG(seed).Hypergeometric(7, 13, 5, 64)))
	refcheck.Check(t, "rng_hypergeometric_big", must(random.DefaultRNG(seed).Hypergeometric(40, 60, 30, 64)))
	refcheck.Check(t, "rng_hypergeometric_complement", must(random.DefaultRNG(seed).Hypergeometric(60, 40, 95, 64)))
	for _, method := range []string{"marginals", "count"} {
		refcheck.Check(t, "rng_multivariate_hypergeometric_"+method, must(random.DefaultRNG(seed).MultivariateHypergeometric([]int64{12, 7, 5}, 9, method, 24)))
	}
	refcheck.Check(t, "rng_multivariate_hypergeometric_complement", must(random.DefaultRNG(seed).MultivariateHypergeometric([]int64{12, 7, 5}, 19, "marginals", 24)))
}
