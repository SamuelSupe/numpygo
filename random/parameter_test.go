package random_test

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"github.com/SamuelSupe/numpygo/random"
)

var factories = map[string]func() *random.Generator{
	"pcg":    func() *random.Generator { return random.NewPCG64(12345) },
	"dxsm":   func() *random.Generator { return random.NewPCG64DXSM(12345) },
	"mt":     func() *random.Generator { return random.NewMT19937(12345) },
	"philox": func() *random.Generator { return random.NewPhilox(12345) },
	"sfc":    func() *random.Generator { return random.NewSFC64(12345) },
	"legacy": func() *random.Generator { return random.NewRandomState(12345).Generator },
}
var parameterizations = map[string]int{"normal": 2, "uniform": 2, "exponential": 1, "gamma": 2, "standard_gamma": 1, "beta": 2, "chisquare": 1, "f": 2, "standard_t": 1, "pareto": 1, "weibull": 1, "power": 1, "laplace": 2, "gumbel": 2, "logistic": 2, "lognormal": 2, "rayleigh": 1, "triangular": 3, "wald": 2, "noncentral_chisquare": 2, "noncentral_f": 3, "vonmises": 2, "poisson": 1, "negative_binomial": 2, "binomial": 2, "geometric": 1, "logseries": 1, "zipf": 1, "hypergeometric": 3}

func TestArrayDistributionStreams(t *testing.T) {
	for name, n := range parameterizations {
		t.Run(name, func(t *testing.T) {
			m := refcheck.Require(t)
			var inputs []*np.Array
			for j := 0; j < n; j++ {
				a := refcheck.Input(t, "param_dist_input_"+name+"_"+string(rune('0'+j)))
				if a.NDim() == 2 {
					a = m(a.Slice(np.Reverse(), np.Reverse()))
				}
				inputs = append(inputs, a)
			}
			for engine, newGen := range factories {
				for _, tc := range []struct {
					label string
					size  []int
				}{{"infer", nil}, {"explicit", []int{2, 2, 3}}} {
					t.Run(engine+"/"+tc.label, func(t *testing.T) {
						g := newGen()
						prefix := "param_dist_" + name + "_" + engine + "_" + tc.label
						v, e := g.Sample(name, tc.size, inputs...)
						refcheck.Case(t, prefix, v, e)
						refcheck.Check(t, prefix+"_next", refcheck.Require(t)(g.Random(4)))
					})
				}
			}
		})
	}
}
func TestIntegerArraysAndState(t *testing.T) {
	for engine, newGen := range factories {
		t.Run(engine, func(t *testing.T) {
			m := refcheck.Require(t)
			for _, dtype := range []np.DType{np.Bool, np.Int8, np.Int16, np.Int32, np.Int64, np.Uint8, np.Uint16, np.Uint32, np.Uint64} {
				for _, closed := range []bool{false, true} {
					if closed && engine == "legacy" {
						continue
					}
					label := "open"
					if closed {
						label = "closed"
					}
					t.Run(dtype.String()+"/"+label, func(t *testing.T) {
						m := refcheck.Require(t)
						l := m(refcheck.Input(t, "param_int_low_"+dtype.String()+"_"+label).Slice(np.Reverse()))
						h := m(refcheck.Input(t, "param_int_high_"+dtype.String()+"_"+label).Slice(np.Slice{}, np.Reverse()))
						g := newGen()
						prefix := "param_int_" + engine + "_" + dtype.String() + "_" + label
						v, e := g.IntegersArray(l, h, random.IntegerOptions{Size: []int{2, 2, 3}, DType: dtype, Endpoint: closed})
						refcheck.Case(t, prefix, v, e)
						refcheck.Check(t, prefix+"_next", m(g.Random(4)))
					})
				}
			}
			g := newGen()
			refcheck.Check(t, "param_int_one_"+engine, m(g.IntegersArray(m(np.WrapSlice([]int32{2, 10, 50}, 3)), nil, random.IntegerOptions{})))
			refcheck.Check(t, "param_int_one_"+engine+"_next", m(g.Random(4)))
			g = newGen()
			loc := m(np.WrapSlice([]float64{1, 2}, 2, 1))
			scale := m(np.WrapSlice([]float64{0, 1, 2}, 1, 3))
			refcheck.Check(t, "param_empty_"+engine, m(g.Sample("normal", []int{0, 2, 3}, loc, scale)))
			refcheck.Check(t, "param_empty_"+engine+"_next", m(g.Random(4)))
			g = newGen()
			_, e := g.Sample("normal", []int{2, 2, 3}, loc, m(np.WrapSlice([]float64{1, 2, -1}, 1, 3)))
			refcheck.Error(t, "param_bad_scale_"+engine, "ValueError", e, np.ErrValue)
			refcheck.Check(t, "param_bad_scale_"+engine+"_next", m(g.Random(4)))
			g = newGen()
			_, e = g.Sample("normal", nil, m(np.Zeros(np.Float64, 0)), m(np.WrapSlice([]float64{-1}, 1)))
			refcheck.Error(t, "param_empty_bad_scale_"+engine, "ValueError", e, np.ErrValue)
			_, e = g.Normal(0, math.Copysign(0, -1), 1)
			refcheck.Error(t, "param_negative_zero_scale_"+engine, "ValueError", e, np.ErrValue)
			refcheck.Check(t, "param_empty_bad_scale_"+engine+"_next", m(g.Random(4)))
			g = newGen()
			_, e = g.Sample("normal", []int{3}, loc, scale)
			refcheck.Error(t, "param_bad_size_"+engine, "ValueError", e, np.ErrShape)
			refcheck.Check(t, "param_bad_size_"+engine+"_next", m(g.Random(4)))
			g = newGen()
			_, e = g.IntegersArray(m(np.WrapSlice([]int64{0, 2}, 2)), m(np.WrapSlice([]int64{3, 1}, 2)), random.IntegerOptions{Size: []int{3, 2}, DType: np.Int8})
			refcheck.Error(t, "param_bad_bound_"+engine, "ValueError", e, np.ErrValue)
			refcheck.Check(t, "param_bad_bound_"+engine+"_next", m(g.Random(4)))
			g = newGen()
			v, e := g.Sample("binomial", nil, m(np.WrapSlice([]float64{4.9, 80.2}, 2)), np.Scalar(.5))
			refcheck.Case(t, "param_bad_trials_"+engine, v, e)
			refcheck.Check(t, "param_bad_trials_"+engine+"_next", m(g.Random(4)))
		})
	}
}
func TestMultinomialAndOutputArrays(t *testing.T) {
	for engine, newGen := range factories {
		if engine == "legacy" {
			continue
		}
		t.Run(engine, func(t *testing.T) {
			m := refcheck.Require(t)
			g := newGen()
			n := refcheck.Input(t, "param_multinomial_n")
			p := m(refcheck.Input(t, "param_multinomial_pvals").Slice(np.Slice{}, np.Reverse()))
			refcheck.Check(t, "param_multinomial_"+engine, m(g.MultinomialArray(n, p, []int{2, 2, 3})))
			refcheck.Check(t, "param_multinomial_"+engine+"_next", m(g.Random(4)))
			g = newGen()
			_, e := g.MultinomialArray(n, m(np.WrapSlice([]float64{.7, .6, .1}, 3)), nil)
			refcheck.Error(t, "param_multinomial_bad_"+engine, "ValueError", e, np.ErrValue)
			refcheck.Check(t, "param_multinomial_bad_"+engine+"_next", m(g.Random(4)))
			for _, d := range []np.DType{np.Float32, np.Float64} {
				for _, tc := range []struct{ name, method string }{{"random", ""}, {"standard_normal", ""}, {"standard_exponential", "zig"}, {"standard_exponential", "inv"}, {"standard_gamma", ""}} {
					for _, order := range []string{"C", "F"} {
						if tc.name == "standard_gamma" && order == "F" {
							continue
						}
						t.Run(d.String()+"/"+tc.name+"/"+tc.method+"/"+order, func(t *testing.T) {
							m := refcheck.Require(t)
							g := newGen()
							out := m(np.Full(d, -99., 2, 3))
							if order == "F" {
								out = m(np.AsFortranArray(out))
							}
							var args []*np.Array
							if tc.name == "standard_gamma" {
								args = []*np.Array{m(m(np.WrapSlice([]float64{.3, 4}, 2, 1)).Astype(d))}
							}
							method := tc.method
							if method == "" {
								method = "default"
							}
							prefix := "param_fill_" + engine + "_" + d.String() + "_" + tc.name + "_" + method + "_" + order
							got, e := g.FillRandom(tc.name, random.FillOptions{DType: d, Out: out, Method: tc.method}, args...)
							if e == nil && got != out {
								t.Fatal("out not returned")
							}
							refcheck.Case(t, prefix, got, e)
							refcheck.Check(t, prefix+"_next", m(g.Random(4)))
						})
					}
				}
			}
		})
	}
	m := refcheck.Require(t)
	g := factories["pcg"]()
	_, e := g.FillRandom("random", random.FillOptions{Size: []int{3, 2}, Out: m(np.Zeros(np.Float64, 2, 3))})
	refcheck.Error(t, "param_fill_bad_shape", "ValueError", e, np.ErrShape)
	_, e = g.FillRandom("random", random.FillOptions{Out: m(np.Zeros(np.Float32, 2, 3))})
	refcheck.Error(t, "param_fill_bad_dtype", "TypeError", e, np.ErrDType)
	_, e = g.FillRandom("random", random.FillOptions{Out: m(m(np.Zeros(np.Float64, 2, 6)).Slice(np.Slice{}, np.Strided(0, 6, 2)))})
	refcheck.Error(t, "param_fill_bad_layout", "ValueError", e, np.ErrValue)
	_, e = g.FillRandom("random", random.FillOptions{Out: m(np.Scalar(1).BroadcastTo(2, 3))})
	refcheck.Error(t, "param_fill_readonly", "ValueError", e, np.ErrReadOnly)
	foreign := m(m(np.FromBuffer(make([]byte, 48), np.Float64, 6, 0, binary.BigEndian)).Reshape(2, 3))
	_, e = g.FillRandom("random", random.FillOptions{Out: foreign})
	refcheck.Error(t, "param_fill_bad_byteorder", "ValueError", e, np.ErrValue)
	_, e = g.FillRandom("standard_gamma", random.FillOptions{Out: m(np.AsFortranArray(m(np.Zeros(np.Float64, 2, 3))))}, np.Scalar(2))
	refcheck.Error(t, "param_fill_gamma_f", "ValueError", e, np.ErrValue)
	refcheck.Check(t, "param_fill_bad_next", m(g.Random(4)))
}
