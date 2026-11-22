package random_test

import (
	"fmt"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"github.com/SamuelSupe/numpygo/linalg"
	"github.com/SamuelSupe/numpygo/random"
)

func TestSeedSequenceConfigAndReplay(t *testing.T) {
	seeds := map[string][]uint32{"empty": {}, "short": {123}, "long": {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}}
	for _, pool := range []int{4, 5, 8, 16} {
		for key, entropy := range seeds {
			name := fmt.Sprintf("adv_seed_%d_%s", pool, key)
			t.Run(name, func(t *testing.T) {
				m := refcheck.Require(t)
				requireSeed := refcheck.RequireValue[*random.SeedSequence](t)
				check32 := func(key string, v []uint32) { refcheck.Check(t, key, m(np.WrapSlice(v, len(v)))) }
				check64 := func(key string, v []uint64) { refcheck.Check(t, key, m(np.WrapSlice(v, len(v)))) }
				state := random.SeedSequenceState{Entropy: entropy, SpawnKey: []uint32{5, 17}, PoolSize: pool, NChildrenSpawned: 3}
				s := requireSeed(random.SeedSequenceFromState(state))
				check32(name+"_pool", s.PoolWords())
				check32(name+"_32", refcheck.RequireValue[[]uint32](t)(s.GenerateState32(19)))
				check64(name+"_64", refcheck.RequireValue[[]uint64](t)(s.GenerateState64(11)))
				children := refcheck.RequireValue[[]*random.SeedSequence](t)(s.Spawn(2))
				for i, child := range children {
					k := fmt.Sprintf("%s_child%d", name, i)
					check32(k+"_key", child.SpawnKey())
					check32(k+"_pool", child.PoolWords())
					check32(k+"_state", refcheck.RequireValue[[]uint32](t)(child.GenerateState32(17)))
					gc := refcheck.RequireValue[[]*random.SeedSequence](t)(child.Spawn(1))[0]
					check32(k+"_grand", refcheck.RequireValue[[]uint32](t)(gc.GenerateState32(17)))
				}
				rebuilt := requireSeed(random.SeedSequenceFromState(s.State()))
				rc := refcheck.RequireValue[[]*random.SeedSequence](t)(rebuilt.Spawn(1))[0]
				check32(name+"_resume", refcheck.RequireValue[[]uint32](t)(rc.GenerateState32(17)))
				for en, makeG := range map[string]func(*random.SeedSequence) *random.Generator{
					"pcg":  func(s *random.SeedSequence) *random.Generator { return random.FromSeedSequence(s, false) },
					"dxsm": func(s *random.SeedSequence) *random.Generator { return random.FromSeedSequence(s, true) },
					"mt":   random.MT19937FromSeedSequence, "philox": random.PhiloxFromSeedSequence, "sfc": random.SFC64FromSeedSequence,
				} {
					g := makeG(requireSeed(random.SeedSequenceFromState(state)))
					refcheck.Check(t, name+"_"+en, m(g.RandomRaw(9)))
					child := refcheck.RequireValue[[]*random.Generator](t)(g.Spawn(1))[0]
					refcheck.Check(t, name+"_"+en+"_spawn", m(child.RandomRaw(9)))
				}
			})
		}
	}
	_, err := random.SeedSequenceFromState(random.SeedSequenceState{Entropy: []uint32{42}, PoolSize: 3})
	refcheck.Error(t, "adv_seed_bad_pool", "ValueError", err, np.ErrValue)
}

func TestPermutedOutputAndStream(t *testing.T) {
	m := refcheck.Require(t)
	base := refcheck.Input(t, "adv_permuted_input")
	inputs := map[string]*np.Array{"c": base, "f": m(np.AsFortranArray(base)), "transpose": m(base.Transpose(1, 2, 0)), "reverse": m(base.Slice(np.Slice{}, np.Reverse()))}
	for en, factory := range factories {
		if en == "legacy" {
			continue
		}
		for kind, a := range inputs {
			for _, ax := range []int{-1, 0, 1, 2} {
				ak := "None"
				var axis *int
				if ax >= 0 {
					ak = fmt.Sprint(ax)
					axis = np.Int(ax)
				}
				for _, mode := range []string{"auto", "c", "f", "stride", "alias"} {
					key := fmt.Sprintf("adv_permuted_%s_%s_%s_%s", en, kind, ak, mode)
					t.Run(key, func(t *testing.T) {
						m := refcheck.Require(t)
						g := factory()
						var out, backing *np.Array
						input := a
						switch mode {
						case "auto":
						case "c":
							out = m(np.Zeros(np.Int64, a.Shape()...))
						case "f":
							out = m(np.EmptyLike(a, np.Int64, "F"))
						case "stride":
							sh := a.Shape()
							sh[len(sh)-1] *= 2
							backing = m(np.Full(np.Int64, -99, sh...))
							out = m(backing.Slice(np.Ellipsis, np.Slice{Step: np.Int(-2)}))
						case "alias":
							out = m(a.AstypeWithOptions(a.DType(), np.CastOptions{}))
							input = out
						}
						got := m(g.PermutedWithOut(input, axis, out))
						if out != nil && got != out {
							t.Fatal("out identity")
						}
						refcheck.CheckLayout(t, key, got)
						if backing != nil {
							refcheck.Check(t, key+"_backing", backing)
						}
						refcheck.CheckExact(t, key+"_next", m(g.Random(3)))
					})
				}
			}
		}
		g := factory()
		refcheck.Check(t, "adv_permuted_cast_"+en, m(g.PermutedWithOut(base, np.Int(1), m(np.Zeros(np.Float64, base.Shape()...)))))
		refcheck.CheckExact(t, "adv_permuted_cast_"+en+"_next", m(g.Random(3)))
		g = factory()
		_, err := g.PermutedWithOut(base, nil, m(np.Zeros(np.Int32, base.Shape()...)))
		refcheck.Error(t, "adv_permuted_bad_cast_"+en, "TypeError", err, np.ErrDType)
		refcheck.CheckExact(t, "adv_permuted_bad_cast_"+en+"_next", m(g.Random(3)))
	}
}

func TestVectorDistributionEdges(t *testing.T) {
	for en, factory := range factories {
		for key, alpha := range map[string][]float64{"allzero": {0, 0, 0}, "first": {.01, 0, 0}, "middle": {0, .01, 0}, "last": {0, 0, .01}, "tailzero": {.02, .03, 0}, "mixed": {0, .1, 0}, "singleton": {2}, "empty": {}, "tiny": {1e-120, 2e-120, 3e-120}, "regular": {1, 2, 3}} {
			if en == "legacy" && key == "tiny" {
				continue
			}
			name := fmt.Sprintf("adv_dirichlet_%s_%s", en, key)
			t.Run(name, func(t *testing.T) {
				g := factory()
				got, err := g.Dirichlet(alpha, 2, 3)
				refcheck.Case(t, name, got, err)
				refcheck.CheckExact(t, name+"_next", refcheck.Require(t)(g.Random(3)))
			})
		}
		if en == "legacy" {
			continue
		}
		for _, method := range []string{"count", "marginals"} {
			for _, c := range []struct {
				key    string
				colors []int64
				n      int64
			}{
				{"zero", []int64{0, 4, 0, 8}, 0}, {"full", []int64{0, 4, 0, 8}, 12}, {"more", []int64{0, 4, 0, 8}, 9},
				{"less", []int64{5, 0, 8, 2}, 4}, {"empty", []int64{}, 0}, {"allzero", []int64{0, 0}, 0},
			} {
				name := fmt.Sprintf("adv_mvhg_%s_%s_%s", en, method, c.key)
				t.Run(name, func(t *testing.T) {
					g := factory()
					m := refcheck.Require(t)
					refcheck.Check(t, name, m(g.MultivariateHypergeometric(c.colors, c.n, method, 2, 3)))
					refcheck.CheckExact(t, name+"_next", m(g.Random(3)))
				})
			}
		}
	}
}

func TestMultivariateNormalValidationAndReplay(t *testing.T) {
	for en, factory := range factories {
		methods := []string{"svd", "eigh", "cholesky"}
		if en == "legacy" {
			methods = []string{"svd"}
		}
		for _, method := range methods {
			for ck, diag := range map[string][]float64{"pd": {4, 1}, "singular": {1, 0}, "negative": {1, -.5}, "tolerance": {1, -1e-10}} {
				for _, check := range []string{"raise", "warn", "ignore", "bad"} {
					name := fmt.Sprintf("adv_mvnormal_%s_%s_%s_%s", en, method, ck, check)
					t.Run(name, func(t *testing.T) {
						m := refcheck.Require(t)
						g := factory()
						warnings := int64(0)
						cov := m(np.FromFloat64([]float64{diag[0], 0, 0, diag[1]}, 2, 2))
						tol := 1e-8
						out, err := g.MultivariateNormalWithOptions([]float64{1, 2}, cov, random.MultivariateNormalOptions{
							Method: method, CheckValid: check, Tol: &tol, Warn: func(err error) { warnings++ },
						}, 2, 2)
						if method == "cholesky" && diag[1] <= 0 {
							refcheck.Error(t, name, "LinAlgError", err, linalg.ErrNotPositiveDefinite)
						} else if method != "cholesky" && (check == "bad" || check == "raise" && ck == "negative") {
							refcheck.Error(t, name, "ValueError", err, np.ErrValue)
						} else {
							if err != nil {
								t.Fatal(err)
							}
							if method == "cholesky" {
								refcheck.Check(t, name, out)
							} else {
								refcheck.Metadata(t, name, out)
								// SVD/eigenvector signs are backend-dependent. For
								// these diagonal cases allow one consistent sign
								// per feature, never different signs per sample.
								want, _ := refcheck.Input(t, name).Float64s()
								got, _ := out.Float64s()
								for j := 0; j < 2; j++ {
									mean := float64(j + 1)
									direct, flipped := true, true
									for i := j; i < len(want); i += 2 {
										direct = direct && refcheck.Close(got[i], want[i])
										flipped = flipped && refcheck.Close(got[i], 2*mean-want[i])
									}
									if !direct && !flipped {
										t.Errorf("feature %d: inconsistent normal transform: got %v want %v", j, got, want)
									}
								}
							}
						}
						refcheck.Check(t, name+"_warnings", np.ScalarOf(warnings))
						refcheck.CheckExact(t, name+"_next", m(g.Random(3)))
					})
				}
			}
		}
		if en != "legacy" {
			name := "adv_mvnormal_bad_method_" + en
			g := factory()
			m := refcheck.Require(t)
			_, e := g.MultivariateNormalWithOptions([]float64{1, 2}, m(np.FromFloat64([]float64{1, 0, 0, 1}, 2, 2)), random.MultivariateNormalOptions{Method: "bad"}, 3)
			refcheck.Error(t, name, "ValueError", e, np.ErrValue)
			refcheck.CheckExact(t, name+"_next", m(g.Random(3)))
		}
	}
}
