package np_test

import (
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"testing"
)

func TestWeightedStatisticsAndSamples(t *testing.T) {
	must := refcheck.Require(t)
	s := refcheck.Input(t, "sample_input")
	w := must(np.FromFloat64([]float64{1, 3, 2}, 3))
	avg, sum, err := np.AverageWithWeights(s, w, np.SumOptions{Axes: []int{1}})
	if err != nil {
		t.Fatal(err)
	}
	refcheck.Check(t, "sample_average", avg)
	refcheck.Check(t, "sample_average_weights", sum)
	refcheck.Check(t, "sample_average_unweighted", must(np.Average(s, nil)))
	h, e, err := np.Histogram(s, np.HistogramOptions{Count: 4, Range: &[2]float64{1, 9}})
	if err != nil {
		t.Fatal(err)
	}
	refcheck.Check(t, "sample_histogram", h)
	refcheck.Check(t, "sample_histogram_edges", e)
	h, e, err = np.Histogram(s, np.HistogramOptions{Edges: []float64{0, 2, 5, 10}, Weights: must(np.FromFloat64([]float64{1, 2, 3, .5, 1.5, 2.5}, 2, 3)), Density: true})
	if err != nil {
		t.Fatal(err)
	}
	refcheck.Check(t, "sample_hist_density", h)
	refcheck.Check(t, "sample_hist_density_edges", e)
	x := must(np.FromSlice([]int64{0, 1, 1, 3, 3, 3}, 6))
	refcheck.Check(t, "sample_bincount", must(np.Bincount(x, nil, 5)))
	refcheck.Check(t, "sample_bincount_weights", must(np.Bincount(x, must(np.FromFloat64([]float64{1, 2, .5, 3, 1, .5}, 6)), 5)))
	refcheck.Check(t, "sample_gradient", must(np.Gradient(s, 1, .25, 2)))
	refcheck.Check(t, "sample_trapezoid", must(np.Trapezoid(s, nil, .5, 1)))
	refcheck.Check(t, "sample_trapezoid_x", must(np.Trapezoid(s, must(np.FromFloat64([]float64{0, .5, 2}, 3)), 1, 1)))
	left, right := complex128(-3+1i), complex128(9+2i)
	refcheck.Check(t, "sample_interp", must(np.Interp(must(np.FromFloat64([]float64{-1, .5, 1.5, 3}, 4)), must(np.FromFloat64([]float64{0, 1, 2}, 3)), must(np.FromComplex128([]complex128{1 + 2i, 2 - 1i, 4}, 3)), &left, &right)))
}
