package np_test

import (
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"testing"
)

func TestSetSearchAndAlongAxis(t *testing.T) {
	must := refcheck.Require(t)
	a := refcheck.Input(t, "unique_input")
	u, err := np.Unique(a)
	if err != nil {
		t.Fatal(err)
	}
	refcheck.Check(t, "unique_values", u.Values)
	refcheck.Check(t, "unique_indices", u.Indices)
	refcheck.Check(t, "unique_inverse", u.Inverse)
	refcheck.Check(t, "unique_counts", u.Counts)
	refcheck.Check(t, "unique_values_separate", must(np.UniqueValues(a)))
	x, y := refcheck.Input(t, "set_input_a"), refcheck.Input(t, "set_input_b")
	for name, fn := range map[string]binaryFn{"union1d": np.Union1D, "intersect1d": np.Intersect1D, "setdiff1d": np.Setdiff1D, "setxor1d": np.Setxor1D} {
		refcheck.Check(t, "set_"+name, must(fn(x, y)))
	}
	refcheck.Check(t, "set_isin", must(np.Isin(x, y, false)))
	refcheck.Check(t, "set_isin_invert", must(np.Isin(x, y, true)))
	grid, q := refcheck.Input(t, "search_grid"), refcheck.Input(t, "search_query")
	refcheck.Check(t, "search_left", must(np.SearchSorted(grid, q, false)))
	refcheck.Check(t, "search_right", must(np.SearchSorted(grid, q, true)))
	bins := must(np.FromFloat64([]float64{1, 2, 4}, 3))
	rev := must(bins.Slice(np.Reverse()))
	for _, tc := range []struct {
		name  string
		bins  *np.Array
		right bool
	}{{"00", bins, false}, {"01", bins, true}, {"10", rev, false}, {"11", rev, true}} {
		refcheck.Check(t, "digitize_"+tc.name, must(np.Digitize(q, tc.bins, tc.right)))
	}
	refcheck.Check(t, "complex_sort", must(np.Sort(refcheck.Input(t, "complex_sort_input"), np.Int(0))))
	a, ix := refcheck.Input(t, "along_input"), refcheck.Input(t, "along_index")
	refcheck.Check(t, "along_take", must(np.TakeAlongAxis(a, ix, np.Int(1))))
	refcheck.Check(t, "along_take_broadcast", must(np.TakeAlongAxis(a, must(ix.Slice(np.Range(0, 1))), np.Int(1))))
	if err = np.PutAlongAxis(a, ix, must(np.FromSlice([]int64{99, 88, 77}, 3, 1)), np.Int(1)); err != nil {
		t.Fatal(err)
	}
	refcheck.Check(t, "along_put", a)
}
