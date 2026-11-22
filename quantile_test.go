package np_test

import (
	"math"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
)

var quantileMethods = []string{"inverted_cdf", "averaged_inverted_cdf", "closest_observation", "interpolated_inverted_cdf", "hazen", "weibull", "linear", "median_unbiased", "normal_unbiased", "lower", "higher", "midpoint", "nearest"}

func TestQuantileMethodGrid(t *testing.T) {
	m := refcheck.Require(t)
	for _, d := range []np.DType{np.Bool, np.Int16, np.Int64, np.Float16, np.Float32, np.Float64} {
		a := m(refcheck.Input(t, "behavior_quantile_input_"+d.String()).Slice(np.Reverse(), np.Slice{}, np.Reverse()))
		for _, qd := range []np.DType{np.Float16, np.Float32, np.Float64} {
			q := refcheck.Input(t, "behavior_quantile_q_"+qd.String())
			for _, method := range quantileMethods {
				t.Run(d.String()+"/"+qd.String()+"/"+method, func(t *testing.T) {
					v, e := np.QuantileArray(a, q, np.QuantileOptions{Axes: []int{0, 2}, KeepDims: true, Method: method})
					refcheck.Case(t, "behavior_quantile_"+method+"_"+d.String()+"_"+qd.String(), v, e)
				})
			}
		}
	}
	a := m(refcheck.Input(t, "behavior_quantile_nan_input").Slice(np.Reverse(), np.Slice{}, np.Reverse()))
	q := refcheck.Input(t, "behavior_quantile_nan_q")
	for _, method := range quantileMethods {
		t.Run("nan/"+method, func(t *testing.T) {
			v, e := np.NanQuantileArray(a, q, np.QuantileOptions{Axes: []int{0, 2}, Method: method})
			refcheck.Case(t, "behavior_quantile_nan_"+method, v, e)
		})
	}
}
func TestQuantileWeightsAndParameters(t *testing.T) {
	m := refcheck.Require(t)
	a := refcheck.Input(t, "behavior_quantile_input_int64")
	q := refcheck.Input(t, "behavior_quantile_nan_q")
	weights := refcheck.Input(t, "behavior_quantile_weights")
	opt := np.QuantileOptions{Axes: []int{0, 2}, Method: "inverted_cdf", Weights: weights}
	v := m(a.Slice(np.Reverse(), np.Slice{}, np.Reverse()))
	refcheck.Check(t, "behavior_quantile_weighted", m(np.QuantileArray(v, q, opt)))
	opt.Axes = []int{2, 0}
	opt.Weights = m(weights.Transpose())
	refcheck.Check(t, "behavior_quantile_weighted_reversed_axes", m(np.QuantileArray(a, q, opt)))
	opt.Axes = []int{0, 2}
	opt.Weights = refcheck.Input(t, "behavior_quantile_fullweights")
	refcheck.Check(t, "behavior_quantile_nan_weighted", m(np.NanQuantileArray(refcheck.Input(t, "behavior_quantile_nan_input"), q, opt)))
	exact := refcheck.Input(t, "behavior_quantile_exact_input")
	refcheck.Check(t, "behavior_quantile_exact", m(np.QuantileArray(exact, q, np.QuantileOptions{Method: "nearest"})))
	refcheck.Check(t, "behavior_quantile_integer_q", m(np.QuantileArray(exact, m(np.WrapSlice([]int32{0, 1}, 2)), np.QuantileOptions{})))
	refcheck.Check(t, "behavior_quantile_percentile", m(np.PercentileArray(a, m(np.WrapSlice([]float32{20, 70}, 2)), np.QuantileOptions{Axes: []int{1}, Method: "hazen"})))
	base := m(np.Full(np.Float64, -99., 5, 3))
	out := m(base.Slice(np.Slice{}, np.Reverse()))
	got := m(np.QuantileArray(a, q, np.QuantileOptions{Axes: []int{0, 2}, Out: out}))
	if got != out {
		t.Fatal("out replaced")
	}
	refcheck.Check(t, "behavior_quantile_out_base", base)
	refcheck.Check(t, "behavior_quantile_infinity", m(np.QuantileArray(m(np.WrapSlice([]float64{math.Inf(-1), 0, math.Inf(1)}, 3)), m(np.WrapSlice([]float64{0, .5, 1}, 3)), np.QuantileOptions{})))
	empty := m(np.Zeros(np.Float64, 0))
	refcheck.Check(t, "behavior_quantile_nan_empty", m(np.NanQuantileArray(empty, q, np.QuantileOptions{})))
	q64 := m(np.WrapSlice([]float64{.25, .75}, 2))
	refcheck.Check(t, "behavior_quantile_nan_all", m(np.NanQuantileArray(m(np.WrapSlice([]float32{float32(math.NaN()), float32(math.NaN())}, 2)), q64, np.QuantileOptions{})))
	refcheck.Check(t, "behavior_quantile_nan_first_lane", m(np.NanQuantileArray(m(np.WrapSlice([]float32{float32(math.NaN()), float32(math.NaN()), 1, 2}, 2, 2)), q64, np.QuantileOptions{Axes: []int{1}})))
	refcheck.Check(t, "behavior_quantile_nan_scalar", m(np.QuantileArray(m(np.WrapSlice([]float32{float32(math.NaN()), 2}, 2)), np.Scalar(.25), np.QuantileOptions{})))
	refcheck.Check(t, "behavior_quantile_nanpercentile", m(np.NanPercentileArray(refcheck.Input(t, "behavior_quantile_nan_input"), m(np.WrapSlice([]float32{20, 70}, 2)), np.QuantileOptions{Axes: []int{1}, Method: "weibull"})))
	for _, d := range []np.DType{np.Float16, np.Float32} {
		val := refcheck.Input(t, "behavior_quantile_input_"+d.String())
		for _, method := range quantileMethods {
			t.Run(d.String()+"/scalar/"+method, func(t *testing.T) {
				v, e := np.QuantileWithOptions(val, .33, np.QuantileOptions{Axes: []int{1}, Method: method})
				refcheck.Case(t, "behavior_quantile_scalar_"+d.String()+"_"+method, v, e)
			})
		}
	}
	refcheck.Check(t, "behavior_quantile_bool_median", m(np.Median(m(np.WrapSlice([]bool{true, false, true, false}, 4)))))
	_, e := np.QuantileArray(empty, q, np.QuantileOptions{})
	refcheck.Error(t, "behavior_quantile_empty", "IndexError", e, np.ErrIndex)
	_, e = np.QuantileArray(a, m(np.WrapSlice([]float64{-.1, .5}, 2)), np.QuantileOptions{})
	refcheck.Error(t, "behavior_quantile_bad_q", "ValueError", e, np.ErrValue)
	_, e = np.QuantileArray(a, q, np.QuantileOptions{Method: "foo"})
	refcheck.Error(t, "behavior_quantile_bad_method", "ValueError", e, np.ErrValue)
	_, e = np.QuantileArray(a, q, np.QuantileOptions{Axes: []int{0, 2}, Weights: weights})
	refcheck.Error(t, "behavior_quantile_bad_weights_method", "ValueError", e, np.ErrValue)
	_, e = np.QuantileArray(a, q, np.QuantileOptions{Axes: []int{0, 2}, Method: "inverted_cdf", Weights: m(weights.Transpose())})
	refcheck.Error(t, "behavior_quantile_bad_weights_shape", "ValueError", e, np.ErrShape)
	for _, tc := range []struct {
		name    string
		weights *np.Array
	}{{"negative_weights", m(np.Subtract(weights, np.Scalar(2)))}, {"zero_weights", m(np.Zeros(np.Float64, 2, 4))}} {
		_, e = np.QuantileArray(a, q, np.QuantileOptions{Axes: []int{0, 2}, Method: "inverted_cdf", Weights: tc.weights})
		refcheck.Error(t, "behavior_quantile_"+tc.name, "ValueError", e, np.ErrValue)
	}
}
