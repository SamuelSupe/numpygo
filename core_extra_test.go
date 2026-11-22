package np_test

import (
	"math"
	"strconv"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"github.com/SamuelSupe/numpygo/linalg"
)

func TestExtraMath(t *testing.T) {
	must := refcheck.Require(t)
	in := func(s string) *np.Array { return refcheck.Input(t, "extra_"+s) }
	check := func(s string, a *np.Array) { refcheck.Check(t, "extra_"+s, a) }
	x, y, c, i, j := in("x"), in("y"), in("c"), in("i"), in("j")
	for name, fn := range map[string]unaryFn{"fabs": np.Fabs, "fix": np.Fix, "i0": np.I0, "sinc": np.Sinc, "spacing": np.Spacing} {
		t.Run(name, func(t *testing.T) { check(name, must(fn(x))) })
	}
	for name, fn := range map[string]binaryFn{"float_power": np.FloatPower, "fmod": np.Fmod, "heaviside": np.Heaviside, "logaddexp2": np.LogAddExp2, "nextafter": np.Nextafter} {
		t.Run(name, func(t *testing.T) { check(name, must(fn(x, y))) })
	}
	check("angle", must(np.Angle(c, true)))
	check("round", must(np.Round(x, 1)))
	check("round_int", must(np.Round(i, -1)))
	q, r, e := np.Divmod(i, j)
	if e != nil {
		t.Fatal(e)
	}
	check("divmod_0", q)
	check("divmod_1", r)
	f32 := must(x.Astype(np.Float32))
	q, r, e = np.Frexp(f32)
	if e != nil {
		t.Fatal(e)
	}
	check("frexp_0", q)
	check("frexp_1", r)
	check("ldexp", must(np.Ldexp(f32, j)))
	check("gcd", must(np.GCD(i, j)))
	check("lcm", must(np.LCM(i, j)))
	q, r, e = np.Modf(x)
	if e != nil {
		t.Fatal(e)
	}
	check("modf_0", q)
	check("modf_1", r)
	check("bitwise_count", must(np.BitwiseCount(i)))
	check("left_shift", must(np.LeftShift(i, j)))
	check("right_shift", must(np.RightShift(i, j)))
	for _, d := range []np.DType{np.Float16, np.Float32, np.Float64} {
		a, b := must(x.Astype(d)), must(y.Astype(d))
		for name, result := range map[string]*np.Array{"nextafter": must(np.Nextafter(a, b)), "spacing": must(np.Spacing(a))} {
			name += "_" + d.String()
			check(name, result)
			if eq, err := np.ArrayEqual(result, in(name), false); err != nil || !eq {
				t.Errorf("%s not exact: %v", name, err)
			}
		}
	}
	s := in("special")
	check("nan_to_num", must(np.NanToNum(s)))
	check("isposinf", must(np.IsPosInf(s)))
	check("isneginf", must(np.IsNegInf(s)))
	check("iscomplex", must(np.IsComplex(c)))
	check("isreal", must(np.IsReal(c)))
	check("iscomplexobj", np.ScalarOf(np.IsComplexObj(c)))
	check("isrealobj", np.ScalarOf(np.IsRealObj(x)))
}

func TestExtraViews(t *testing.T) {
	must := refcheck.Require(t)
	in := func(s string) *np.Array { return refcheck.Input(t, "extra_"+s) }
	check := func(s string, a *np.Array) { refcheck.Check(t, "extra_"+s, a) }
	a := in("complex_matrix")
	at := must(a.Transpose())
	r, i := must(np.Real(at)), must(np.Imag(at))
	refcheck.CheckLayout(t, "extra_real", r)
	refcheck.CheckLayout(t, "extra_imag", i)
	if err := r.SetValue(11., 0, 1); err != nil {
		t.Fatal(err)
	}
	if err := i.SetValue(-12., 1, 0); err != nil {
		t.Fatal(err)
	}
	check("component_mutation", a)
	check("real_if_close", must(np.RealIfClose(must(np.FromSlice([]complex128{1 + 1e-15i, 2 - 1e-15i}, 2)), 100)))
	g, x := in("g"), in("x")
	sizes := make([]int64, len(g.Shape()))
	for i, n := range g.Shape() {
		sizes[i] = int64(n)
	}
	check("shape", must(np.FromSlice(sizes, len(sizes))))
	check("ndim", np.ScalarOf(int64(g.NDim())))
	check("size", np.ScalarOf(int64(g.Size())))
	check("size_axis", np.ScalarOf(int64(g.Shape()[g.NDim()-1])))
	check("nbytes", np.ScalarOf(int64(g.NBytes())))
	check("fliplr", must(np.Flip(g, 1)))
	check("flipud", must(np.Flip(g, 0)))
	refcheck.CheckLayout(t, "extra_ascontiguousarray", must(np.AsContiguousArray(must(g.Transpose()))))
	refcheck.CheckLayout(t, "extra_asfortranarray", must(np.AsFortranArray(g)))
	check("isfortran", np.ScalarOf(np.IsFortran(must(g.Transpose()))))
	for idx, fn := range []unaryFn{np.Atleast1D, np.Atleast2D, np.Atleast3D} {
		check("atleast_"+strconv.Itoa(idx+1), must(fn(x)))
	}
	eq, err := np.ArrayEqual(in("special"), in("special"), true)
	if err != nil {
		t.Fatal(err)
	}
	check("array_equal", np.ScalarOf(eq))
	eq, err = np.ArrayEquiv(g, must(g.Reshape(1, 2, 3)))
	if err != nil {
		t.Fatal(err)
	}
	check("array_equiv", np.ScalarOf(eq))
	back := must(np.FromSlice([]int64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 10))
	left := must(back.Slice(np.Slice{Step: np.Int(2)}))
	right := must(back.Slice(np.Slice{Start: np.Int(1), Step: np.Int(2)}))
	check("shares_memory", np.ScalarOf(np.SharesMemory(left, right)))
	check("may_share_memory", np.ScalarOf(np.MayShareMemory(left, right)))
	dest := must(g.Copy())
	v := must(np.FromSlice([]int64{7, 8, 9}, 3))
	mask := must(np.Equal(must(np.Remainder(g, np.ScalarOf(int64(2)))), np.ScalarOf(int64(0))))
	if err := np.CopyTo(dest, v, mask); err != nil {
		t.Fatal(err)
	}
	check("copyto", dest)
	if err := must(must(dest.Transpose()).Slice(np.AtIndex(1))).Fill(int64(-1)); err != nil {
		t.Fatal(err)
	}
	check("fill", dest)
	arr, err := np.BroadcastArrays(g, v)
	if err != nil {
		t.Fatal(err)
	}
	for j, b := range arr {
		check("broadcast_arrays_"+strconv.Itoa(j), b)
	}
}

func TestExtraManipulation(t *testing.T) {
	must := refcheck.Require(t)
	in := func(s string) *np.Array { return refcheck.Input(t, "extra_"+s) }
	check := func(s string, a *np.Array) { refcheck.Check(t, "extra_"+s, a) }
	pair := func(name string, a []*np.Array, e error) {
		t.Helper()
		if e != nil {
			t.Fatal(name, e)
		}
		for j, b := range a {
			check(name+"_"+strconv.Itoa(j), b)
		}
	}
	g, cube := in("g"), in("cube")
	ax0, ax1 := 0, 1
	ints := func(v ...int64) *np.Array { return must(np.FromSlice(v, len(v))) }
	bools := func(v ...bool) *np.Array { return must(np.FromSlice(v, len(v))) }
	r0, r1 := must(g.Slice(np.AtIndex(0))), must(g.Slice(np.AtIndex(1)))
	check("append", must(np.Append(g, g, &ax0)))
	for name, fn := range map[string]func(...*np.Array) (*np.Array, error){"hstack": np.HStack, "vstack": np.VStack, "dstack": np.DStack, "column_stack": np.ColumnStack} {
		check(name, must(fn(r0, r1)))
	}
	a, e := np.Split(g, 2, 0)
	pair("split", a, e)
	a, e = np.ArraySplit(g, 2, 1)
	pair("array_split", a, e)
	a, e = np.SplitAt(g, []int{1, 2}, 1)
	pair("split_at", a, e)
	a, e = np.HSplit(g, 3)
	pair("hsplit", a, e)
	a, e = np.VSplit(g, 2)
	pair("vsplit", a, e)
	a, e = np.DSplit(cube, 2)
	pair("dsplit", a, e)
	a, e = np.Unstack(g, 1)
	pair("unstack", a, e)
	check("rot90", must(np.Rot90(g, 2, 0, 1)))
	check("rollaxis", must(np.RollAxis(cube, 2, 0)))
	check("resize", must(np.Resize(g, 3, 4)))
	check("compress", must(np.Compress(bools(true, false, true), g, &ax1)))
	mask := must(np.Equal(must(np.Remainder(g, np.ScalarOf(int64(2)))), np.ScalarOf(int64(0))))
	check("extract", must(np.Extract(mask, g)))
	check("argwhere", must(np.Argwhere(g)))
	check("delete", must(np.Delete(g, ints(0, 2), &ax1)))
	check("choose", must(np.Choose(ints(1, 0, 1), []*np.Array{r0, r1})))
	conds := []*np.Array{must(np.Less(g, np.ScalarOf(int64(2)))), must(np.Greater(g, np.ScalarOf(int64(3))))}
	choices := []*np.Array{must(np.Add(g, np.ScalarOf(int64(10)))), must(np.Add(g, np.ScalarOf(int64(20))))}
	check("select", must(np.Select(conds, choices, np.ScalarOf(int64(-1)))))
	dest := must(g.Copy())
	if err := np.Put(dest, ints(-1, 1, 7), ints(20, 30), "wrap"); err != nil {
		t.Fatal(err)
	}
	check("put", dest)
	dest = must(g.Copy())
	if err := np.Putmask(dest, mask, ints(20, 30)); err != nil {
		t.Fatal(err)
	}
	check("putmask", dest)
	dest = must(g.Copy())
	if err := np.Place(dest, mask, ints(20, 30)); err != nil {
		t.Fatal(err)
	}
	check("place", dest)
	p := in("partition_input")
	pr := must(np.Partition(p, []int{3}, nil))
	check("partition_sorted", must(np.Sort(pr, nil)))
	check("partition_kth", must(pr.Slice(np.AtIndex(3))))
	ix := must(np.ArgPartition(p, []int{3}, nil))
	check("argpartition_sorted", must(np.Sort(must(np.Take(p, ix, nil)), nil)))
	v, _ := np.Values[int64](pr)
	for k, n := range v {
		if k < 3 && n > v[3] || k > 3 && n < v[3] {
			t.Fatalf("invalid partition %v", v)
		}
	}
	check("lexsort", must(np.Lexsort([]*np.Array{ints(3, 1, 2, 0), ints(1, 1, 0, 1)}, -1)))
}

func TestExtraMatrixIndices(t *testing.T) {
	must := refcheck.Require(t)
	in := func(s string) *np.Array { return refcheck.Input(t, "extra_"+s) }
	check := func(s string, a *np.Array) { refcheck.Check(t, "extra_"+s, a) }
	pair := func(name string, a []*np.Array, e error) {
		t.Helper()
		if e != nil {
			t.Fatal(name, e)
		}
		for j, b := range a {
			check(name+"_"+strconv.Itoa(j), b)
		}
	}
	ints := func(v ...int64) *np.Array { return must(np.FromSlice(v, len(v))) }
	g, x, i := in("g"), in("x"), in("i")
	check("tri", must(np.Tri(3, 4, -1, np.Int32)))
	check("tril", must(np.Tril(g, 1)))
	check("triu", must(np.Triu(g, -1)))
	check("diag", must(np.Diag(g, 1)))
	check("diag_vector", must(np.Diag(x, -1)))
	check("diagflat", must(np.Diagflat(g, 1)))
	a, e := np.DiagIndices(3, 3)
	pair("diag_indices", a, e)
	a, e = np.DiagIndicesFrom(must(np.Zeros(np.Float64, 2, 2)))
	pair("diag_indices_from", a, e)
	a, e = np.TrilIndices(3, -1, 4)
	pair("tril_indices", a, e)
	a, e = np.TriuIndices(3, 1, 4)
	pair("triu_indices", a, e)
	a, e = np.TrilIndicesFrom(g, 1)
	pair("tril_indices_from", a, e)
	a, e = np.TriuIndicesFrom(g, -1)
	pair("triu_indices_from", a, e)
	dest := must(np.Zeros(np.Int64, 5, 2))
	if err := np.FillDiagonal(dest, ints(11, 12), true); err != nil {
		t.Fatal(err)
	}
	check("fill_diagonal", dest)
	check("vander", must(np.Vander(i, 3, true)))
	check("indices", must(np.DenseIndices([]int{2, 3})))
	a, e = np.Indices([]int{2, 3}, true)
	pair("indices_sparse", a, e)
	a, e = np.Meshgrid("xy", false, x, i)
	pair("meshgrid", a, e)
	a, e = np.Meshgrid("ij", true, x, i)
	pair("meshgrid_sparse", a, e)
	a, e = np.Ix(must(np.FromSlice([]bool{true, false, true}, 3)), ints(2, 0))
	pair("ix", a, e)
	check("ravel_multi_index", must(np.RavelMultiIndex([]*np.Array{ints(0, 1, -1), ints(1, 2, 3)}, []int{2, 3}, "wrap", "F")))
	a, e = np.UnravelIndex(ints(0, 4, 5), []int{2, 3}, "F")
	pair("unravel_index", a, e)
	p := in("partition_input")
	u, e := np.UniqueAll(p)
	if e != nil {
		t.Fatal(e)
	}
	pair("unique_all", []*np.Array{u.Values, u.Indices, u.Inverse, u.Counts}, nil)
	v, k, e := np.UniqueCounts(p)
	pair("unique_counts", []*np.Array{v, k}, e)
	v, k, e = np.UniqueInverse(p)
	pair("unique_inverse", []*np.Array{v, k}, e)
	check("trim_zeros", must(np.TrimZeros(ints(0, 0, 2, 1, 0, 0), true, true)))
	check("ediff1d", must(np.Ediff1D(g, np.ScalarOf(int64(-1)), np.ScalarOf(int64(9)))))
}

func TestExtraSignal(t *testing.T) {
	must := refcheck.Require(t)
	in := func(s string) *np.Array { return refcheck.Input(t, "extra_"+s) }
	check := func(s string, a *np.Array) { refcheck.Check(t, "extra_"+s, a) }
	for name, fn := range map[string]func(int) (*np.Array, error){"bartlett": np.Bartlett, "blackman": np.Blackman, "hamming": np.Hamming, "hanning": np.Hanning} {
		check(name, must(fn(8)))
	}
	check("kaiser", must(np.Kaiser(7, 5)))
	check("geomspace", must(np.Geomspace(1, 16i, 5, true, np.Complex128)))
	for _, mode := range []np.SignalMode{np.FullMode, np.SameMode, np.ValidMode} {
		check("convolve_"+string(mode), must(np.Convolve(in("c"), in("x"), mode)))
		check("correlate_"+string(mode), must(np.Correlate(in("c"), in("x"), mode)))
	}
	axis, n := 1, -7
	packed := must(np.Packbits(in("bits"), &axis, np.LittleBits))
	check("packbits", packed)
	check("unpackbits", must(np.Unpackbits(packed, &axis, &n, np.LittleBits)))
	check("sort_complex", must(np.SortComplex(must(np.FromSlice([]float32{4, 2, 1}, 3)))))
	check("unwrap", must(np.Unwrap(in("phase"), 2*math.Pi, math.Pi, -1)))
	g := in("g")
	for _, mode := range []string{"constant", "edge", "wrap", "reflect", "symmetric"} {
		check("pad_"+mode, must(np.Pad(g, [][2]int{{3, 1}, {1, 4}}, mode, int64(-5))))
	}
	check("insert", must(np.Insert(g, np.ScalarOf(int64(1)), must(np.FromSlice([]int64{8, 9}, 2)), &axis)))
	check("insert_multiple", must(np.Insert(g, must(np.FromSlice([]int64{0, 2}, 2)), must(np.FromSlice([]int64{8, 9, 10, 11}, 2, 2)), &axis)))
	_, err := np.Pad(g, [][2]int{{1, 2}, {1, 2}, {1, 2}}, "constant", 0)
	refcheck.Error(t, "extra_pad_shape", "ValueError", err, np.ErrShape)
	_, err = np.Split(g, 2, 1)
	refcheck.Error(t, "extra_split_uneven", "ValueError", err, np.ErrShape)
}

func TestExtraMatrixVector(t *testing.T) {
	must := refcheck.Require(t)
	in := func(s string) *np.Array { return refcheck.Input(t, "extra_"+s) }
	m, v := in("matvec_matrix"), in("matvec_vector")
	mt := must(np.SwapAxes(m, -1, -2))
	refcheck.Check(t, "extra_matvec", must(linalg.MatVec(m, v)))
	refcheck.Check(t, "extra_vecmat", must(linalg.VecMat(v, mt)))
	refcheck.Check(t, "extra_vecmat_broadcast", must(linalg.VecMat(must(v.Slice(np.AtIndex(0))), mt)))
}

func TestExtraDTypeInfo(t *testing.T) {
	must := refcheck.Require(t)
	types := []np.DType{np.Bool, np.Int8, np.Int16, np.Int32, np.Int64, np.Uint8, np.Uint16, np.Uint32, np.Uint64, np.Float16, np.Float32, np.Float64, np.Complex64, np.Complex128}
	for _, rule := range []string{"no", "equiv", "safe", "same_kind", "unsafe"} {
		var values []bool
		for _, f := range types {
			for _, d := range types {
				v, err := np.CanCast(f, d, rule)
				if err != nil {
					t.Fatal(err)
				}
				values = append(values, v)
			}
		}
		refcheck.Check(t, "extra_can_cast_"+rule, must(np.FromSlice(values, len(types), len(types))))
	}
	kinds := []string{"bool", "signed integer", "unsigned integer", "integral", "real floating", "complex floating", "numeric"}
	var values []bool
	for _, d := range types {
		for _, k := range kinds {
			v, err := np.IsDType(d, k)
			if err != nil {
				t.Fatal(err)
			}
			values = append(values, v)
		}
	}
	refcheck.Check(t, "extra_isdtype", must(np.FromSlice(values, len(types), len(kinds))))
}
