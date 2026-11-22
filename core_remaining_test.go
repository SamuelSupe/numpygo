package np_test

import (
	"bytes"
	"encoding/binary"
	"math"
	"strconv"
	"strings"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
)

func remainingHelpers(t *testing.T) (func(string) *np.Array, func(string, *np.Array), func(*np.Array, error) *np.Array) {
	return func(n string) *np.Array { return refcheck.Input(t, "remaining_"+n) }, func(n string, a *np.Array) { refcheck.Check(t, "remaining_"+n, a) }, refcheck.Require(t)
}
func TestRemainingFunctional(t *testing.T) {
	in, check, must := remainingHelpers(t)
	a := in("a")
	ints := func(v ...int64) *np.Array { return must(np.FromSlice(v, len(v))) }
	check("apply_along_axis", must(np.ApplyAlongAxis(a, 1, func(v *np.Array) (*np.Array, error) {
		values := []*np.Array{must(np.Min(v)), must(np.Max(v)), must(np.Sum(v)), must(v.Slice(np.AtIndex(0)))}
		return must(np.Stack(values, 0)).Reshape(2, 2)
	})))
	check("apply_along_axis_scalar", must(np.ApplyAlongAxis(a, -1, func(v *np.Array) (*np.Array, error) { return np.Sum(v) })))
	check("apply_over_axes", must(np.ApplyOverAxes(a, []int{0, 2}, func(v *np.Array, axis int) (*np.Array, error) { return np.Sum(v, np.SumOptions{Axes: []int{axis}}) })))
	check("fromfunction", must(np.FromFunction([]int{3, 4}, np.Int64, func(v ...*np.Array) (*np.Array, error) { return np.Add(must(np.Multiply(v[0], v[0])), v[1]) })))
	seq := func(yield func(int64) bool) {
		for i := int64(0); i < 20; i++ {
			if !yield(i * i) {
				return
			}
		}
	}
	check("fromiter", must(np.FromIter(seq, 7)))
	block := []any{
		[]*np.Array{must(a.Slice(np.AtIndex(0), np.Range(0, 2), np.Range(0, 2))), must(np.Identity(2, np.Int64))},
		[]*np.Array{must(np.Ones(np.Int64, 1, 2)), must(ints(11, 12).Reshape(1, 2))},
	}
	check("block", must(np.Block(block)))
	check("block_vector", must(np.Block([]*np.Array{ints(0, 1, 2), np.ScalarOf(int64(9))})))
	x := in("piecewise_x")
	conds := []*np.Array{must(np.LessEqual(x, np.Scalar(0))), must(np.GreaterEqual(x, np.Scalar(1)))}
	check("piecewise", must(np.Piecewise(x, conds, []func(*np.Array) (*np.Array, error){
		func(v *np.Array) (*np.Array, error) { return np.Multiply(v, v) },
		func(v *np.Array) (*np.Array, error) { return np.Add(must(np.Multiply(v, np.Scalar(2))), np.Scalar(1)) },
	}, func(v *np.Array) (*np.Array, error) { return np.Scalar(-5), nil })))
	check("piecewise_overlap", must(np.Piecewise(x, []*np.Array{must(np.Less(x, np.Scalar(2))), must(np.GreaterEqual(x, np.Scalar(0)))}, []func(*np.Array) (*np.Array, error){
		func(v *np.Array) (*np.Array, error) { return np.Scalar(-1), nil }, func(v *np.Array) (*np.Array, error) { return np.Scalar(3), nil },
	}, nil)))
	check("vectorize", must(np.Vectorize(np.Float64, func(v ...any) (any, error) {
		a, b := v[0].(int64), v[1].(int64)
		return float64(a*a) + .5*float64(b), nil
	}, must(ints(1, 2).Reshape(2, 1)), ints(3, 4, 5))))
	index, e := np.NDIndex(2, 3, 2)
	if e != nil {
		t.Fatal(e)
	}
	var ix []int64
	for row := range index {
		for _, v := range row {
			ix = append(ix, int64(v))
		}
	}
	check("ndindex", must(np.FromSlice(ix, 12, 3)))
	v := must(must(a.Transpose(2, 0, 1)).Slice(np.Reverse()))
	enumerate, e := np.NDEnumerate(v)
	if e != nil {
		t.Fatal(e)
	}
	ix = nil
	var values []int64
	for i, val := range enumerate {
		for _, j := range i {
			ix = append(ix, int64(j))
		}
		values = append(values, val.(int64))
	}
	check("ndenumerate_indices", must(np.FromSlice(ix, v.Size(), v.NDim())))
	check("ndenumerate_values", ints(values...))
	for _, order := range []string{"C", "F"} {
		it, e := np.NDIter(v, order)
		if e != nil {
			t.Fatal(e)
		}
		values = nil
		for value := range it {
			values = append(values, value.(int64))
		}
		check("nditer_"+map[string]string{"C": "c", "F": "f"}[order], ints(values...))
	}
	f, e := v.Flat()
	if e != nil {
		t.Fatal(e)
	}
	values = nil
	for value := range f {
		values = append(values, value.(int64))
	}
	check("flat", ints(values...))
	nested, e := np.NestedIters(a, [2][]int{{2, 0}, {1}})
	if e != nil {
		t.Fatal(e)
	}
	values = nil
	rows := 0
	for _, lane := range nested {
		v, e := np.Values[int64](lane)
		if e != nil {
			t.Fatal(e)
		}
		values = append(values, v...)
		rows++
	}
	check("nested_iters", must(np.FromSlice(values, rows, 3)))
	b, e := np.NewBroadcast(must(ints(1, 2).Reshape(2, 1)), ints(4, 5, 6))
	if e != nil {
		t.Fatal(e)
	}
	values = nil
	for pair := range b.Iter() {
		for _, v := range pair {
			values = append(values, v.(int64))
		}
	}
	check("broadcast", must(np.FromSlice(values, b.Size(), 2)))
	sh := []int64{}
	for _, d := range b.Shape() {
		sh = append(sh, int64(d))
	}
	check("broadcast_shape", ints(sh...))
	check("broadcast_metadata", ints(int64(b.NDim()), int64(b.Size())))
	indices, e := np.MaskIndices(4, 1, np.Triu)
	if e != nil {
		t.Fatal(e)
	}
	for j, index := range indices {
		check("mask_indices_"+strconv.Itoa(j), index)
	}
	empty := must(np.Empty(np.Int32, 2, 3))
	if e = empty.Fill(17); e != nil {
		t.Fatal(e)
	}
	check("empty", empty)
	template := must(must(a.Slice(np.AtIndex(0))).Transpose())
	empty = must(np.EmptyLike(template, np.Float32, "K"))
	if e = empty.Fill(3.); e != nil {
		t.Fatal(e)
	}
	refcheck.CheckLayout(t, "remaining_empty_like", empty)
	check("asanyarray", must(np.AsAnyArray(a, np.Float32)))
	check("asarray_chkfinite", must(np.AsArrayChkFinite(x, np.Float32)))
	readOnly := must(ints(1, 2, 3).BroadcastTo(2, 3))
	req := must(np.Require(readOnly, np.Int32, "F", true, true))
	refcheck.CheckLayout(t, "remaining_require", req)
	if !req.Writable() || np.SharesMemory(req, readOnly) {
		t.Fatal("require must return writable independent array")
	}
	check("isscalar", must(np.FromSlice([]bool{np.IsScalar(1), np.IsScalar(2.), np.IsScalar(np.Scalar(1)), np.IsScalar([]int{1})}, 4)))
	check("iterable", must(np.FromSlice([]bool{np.Iterable(1), np.Iterable(np.Scalar(1)), np.Iterable(ints(1, 2, 3)), np.Iterable([]int{1})}, 4)))
	check("asmatrix", must(np.AsMatrix(ints(0, 1, 2), 0, false)))
	check("matrix", must(np.AsMatrix(must(np.FromFloat64([]float64{1, 2, 3, 4}, 2, 2)), np.Float32, true)))
	p := must(ints(1, 2, 3, 4).Reshape(2, 2))
	check("bmat", must(np.BMat([]any{[]*np.Array{p, p}, []*np.Array{p, p}})))
	check("device", must(np.FromSlice([]byte(a.Device()), len(a.Device()))))
	check("to_device", must(a.ToDevice("cpu")))
	list, e := v.ToList()
	if e != nil {
		t.Fatal(e)
	}
	values = nil
	for _, row := range list.([]any) {
		for _, col := range row.([]any) {
			for _, scalar := range col.([]any) {
				values = append(values, scalar.(int64))
			}
		}
	}
	check("tolist", must(np.FromSlice(values, v.Shape()...)))
	own := ints(1, 2, 3)
	if e = own.ResizeInPlace(2, 3); e != nil {
		t.Fatal(e)
	}
	check("resize", own)
	if e = own.ResizeInPlace(2); e != nil {
		t.Fatal(e)
	}
	check("resize_shrink", own)
	if e = own.SetWritable(false); e != nil {
		t.Fatal(e)
	}
	check("setflags_readonly", np.ScalarOf(own.Writable()))
	e = own.SetValue(int64(5), 0)
	refcheck.Error(t, "remaining_setflags_readonly", "ValueError", e, np.ErrReadOnly)
	if e = own.SetWritable(true); e != nil {
		t.Fatal(e)
	}
	if e = own.SetValue(int64(5), 0); e != nil {
		t.Fatal(e)
	}
	check("setflags", own)
	root := must(np.ArangeInt64(0, 8, 1))
	derived := must(must(root.Slice(np.Slice{Start: np.Int(1), Step: np.Int(2)})).Slice(np.Reverse()))
	check("base_identity", must(np.FromSlice([]bool{root.Base() == nil, derived.Base() == root}, 2)))
	if e = derived.Base().SetValue(int64(99), 1); e != nil {
		t.Fatal(e)
	}
	check("base_mutation", derived)
	e = root.ResizeInPlace(12)
	refcheck.Error(t, "remaining_resize_view", "ValueError", e, np.ErrShape)
	_, e = np.AsArrayChkFinite(np.Scalar(math.NaN()), 0)
	refcheck.Error(t, "remaining_chkfinite_invalid", "ValueError", e, np.ErrValue)
	_, e = np.ApplyAlongAxis(must(np.Zeros(np.Float64, 0, 2)), 1, func(a *np.Array) (*np.Array, error) { return np.Sum(a) })
	refcheck.Error(t, "remaining_apply_empty", "ValueError", e, np.ErrValue)
	_, e = np.Block([]any{[]*np.Array{ints(1, 2)}, ints(1, 2, 3)})
	refcheck.Error(t, "remaining_block_mismatch", "ValueError", e, np.ErrShape)
	_, e = np.FromIter(func(yield func(int64) bool) { yield(1); yield(2) }, 4)
	refcheck.Error(t, "remaining_fromiter_short", "ValueError", e, np.ErrValue)
}

func TestRemainingByteViews(t *testing.T) {
	in, check, must := remainingHelpers(t)
	checkBytes := func(name string, b []byte) { check(name, must(np.FromSlice(b, len(b)))) }
	raw, e := np.Values[uint8](in("buffer_input"))
	if e != nil {
		t.Fatal(e)
	}
	view := must(np.FromBuffer(raw, np.Int16, -1, 1, binary.BigEndian))
	check("frombuffer", view)
	if e = view.SetValue(int16(-2), 1); e != nil {
		t.Fatal(e)
	}
	checkBytes("buffer_writeback", raw)
	a := in("ba")
	at := must(a.Transpose())
	for _, order := range []string{"C", "F"} {
		b, e := at.ToBytes(order)
		if e != nil {
			t.Fatal(e)
		}
		checkBytes("tobytes_"+map[string]string{"C": "c", "F": "f"}[order], b)
	}
	check("byteswap", must(a.ByteSwap(false)))
	c := must(np.FromSlice([]complex64{1 + 2i, 3 - 4i}, 2))
	check("byteswap_complex", must(c.ByteSwap(false)))
	sw := must(a.Copy())
	row := must(sw.Slice(np.AtIndex(1), np.Reverse()))
	must(row.ByteSwap(true))
	check("byteswap_inplace", sw)
	u := must(a.View(np.Uint8))
	refcheck.CheckLayout(t, "remaining_view", u)
	if !np.SharesMemory(u, a) {
		t.Fatal("view should share data")
	}
	if e = u.SetValue(uint8(200), 0, 1); e != nil {
		t.Fatal(e)
	}
	check("view_mutation", a)
	memory, e := a.DataBytes()
	if e != nil {
		t.Fatal(e)
	}
	memory[0] = 23
	check("data_mutation", a)
	fa := must(np.FromSlice([]int32{0x12345678, 0x43211234}, 2))
	field := must(fa.GetField(np.Uint16, 2))
	refcheck.CheckLayout(t, "remaining_getfield", field)
	if e = fa.SetField(must(np.FromSlice([]uint16{0x7777, 0x9999}, 2)), np.Uint16, 2); e != nil {
		t.Fatal(e)
	}
	check("setfield", fa)
	var file bytes.Buffer
	if e = at.ToFile(&file); e != nil {
		t.Fatal(e)
	}
	checkBytes("tofile", file.Bytes())
	check("fromfile", must(np.FromFile(&file, np.Int32, 3, 0, binary.NativeEndian)))
	_, e = np.FromBuffer([]byte("xxz"), np.Int32, -1, 0, nil)
	refcheck.Error(t, "remaining_buffer_bad_size", "ValueError", e, np.ErrValue)
	_, e = at.View(np.Uint8)
	refcheck.Error(t, "remaining_view_axis", "ValueError", e, np.ErrShape)
	_, e = fa.GetField(np.Uint32, 1)
	refcheck.Error(t, "remaining_field_oob", "ValueError", e, np.ErrValue)
	cbytes, e := must(np.FromComplex128([]complex128{2 + 3i, 4 - 5i}, 2)).ToBytes("C")
	if e != nil {
		t.Fatal(e)
	}
	cbuf := must(np.FromBuffer(cbytes, np.Complex128, -1, 0, nil))
	imag := must(np.Imag(cbuf))
	if e = imag.SetValue(17., 0); e != nil {
		t.Fatal(e)
	}
	check("buffer_complex_component", cbuf)
	var npy bytes.Buffer
	if e = np.SaveNPY(&npy, cbuf); e != nil {
		t.Fatal(e)
	}
	check("buffer_complex_component", must(np.LoadNPY(&npy)))
	// Reinterpreting to a new dtype is valid even after the original typed
	// array goes out of scope; the bytes retain its backing allocation.
}

func TestRemainingDtypeMetadata(t *testing.T) {
	_, check, must := remainingHelpers(t)
	types := []np.DType{np.Bool, np.Int8, np.Int16, np.Int32, np.Int64, np.Uint8, np.Uint16, np.Uint32, np.Uint64, np.Float16, np.Float32, np.Float64, np.Complex64, np.Complex128}
	kinds := []string{"generic", "number", "inexact", "integer", "signedinteger", "unsignedinteger", "floating", "complexfloating", "bool_"}
	var flags []bool
	for _, d := range types {
		for _, k := range kinds {
			v, e := np.IsSubdtype(d, k)
			if e != nil {
				t.Fatal(e)
			}
			flags = append(flags, v)
		}
	}
	check("issubdtype", must(np.FromSlice(flags, len(types), len(kinds))))
	for _, d := range types[1:9] {
		info, e := np.IInfo(d)
		if e != nil {
			t.Fatal(e)
		}
		out := must(np.Zeros(d, 2))
		if e = out.SetValue(info.Min, 0); e != nil {
			t.Fatal(e)
		}
		if e = out.SetValue(info.Max, 1); e != nil {
			t.Fatal(e)
		}
		check("iinfo_"+d.String(), out)
		check("iinfo_bits_"+d.String(), np.ScalarOf(int64(info.Bits)))
	}
	for _, d := range types[9:] {
		info, e := np.FInfo(d)
		if e != nil {
			t.Fatal(e)
		}
		check("finfo_meta_"+d.String(), must(np.FromSlice([]int64{int64(info.Bits), int64(info.Mantissa), int64(info.Exponent), int64(info.MinExp), int64(info.MaxExp), int64(info.Precision)}, 6)))
		v := must(np.FromFloat64([]float64{info.Eps, info.EpsNeg, info.Min, info.Max, info.Tiny, info.SmallestSubnormal, info.Resolution}, 7))
		check("finfo_values_"+d.String(), v)
		want := refcheck.Input(t, "remaining_finfo_values_"+d.String())
		ok, e := np.ArrayEqual(v, want, false)
		if e != nil || !ok {
			t.Errorf("finfo %s must match IEEE values exactly", d)
		}
	}
	checkString := func(n, s string) { check(n, must(np.FromSlice([]byte(s), len(s)))) }
	for i, ds := range [][]np.DType{{np.Int8}, {np.Float16}, {np.Float32, np.Int16}, {np.Float32, np.Complex64}, {np.Complex64, np.Int32}} {
		inputs := make([]*np.Array, len(ds))
		for j, d := range ds {
			inputs[j] = must(np.Zeros(d, 2))
		}
		d, e := np.CommonType(inputs...)
		if e != nil {
			t.Fatal(e)
		}
		checkString("common_type_"+strconv.Itoa(i), d.String())
	}
	for i, v := range []any{0, 255, 256, -128, -129, 65535, 65536, uint64(1) << 63, 0., 64999., 65000., math.Inf(1), math.NaN(), 1e-20, 3.4e38, 3.39e38, 1 + 2i, 1e40 + 2i, complex(math.Inf(1), 0)} {
		d, e := np.MinScalarType(v)
		if e != nil {
			t.Fatal(e)
		}
		checkString("min_scalar_type_"+strconv.Itoa(i), d.String())
	}
	for i, v := range [][3]string{{"fd", "GDFgdf", "d"}, {"Fd", "GDFgdf", "d"}, {"abceh", "GDFgdf", "G"}, {"hif", "qih", "q"}} {
		d, e := np.MinTypeCode(v[0], v[1], v[2])
		if e != nil {
			t.Fatal(e)
		}
		checkString("mintypecode_"+strconv.Itoa(i), d)
	}
	for i, v := range []string{"?", "b", "H", "i", "L", "q", "f", "D", "G"} {
		s, e := np.TypeName(v)
		if e != nil {
			t.Fatal(e)
		}
		checkString("typename_"+strconv.Itoa(i), s)
	}
	check("long", np.ScalarOf(int64(1<<40+17)))
	check("ulong", np.ScalarOf(uint64(1<<63+17)))
}

func TestRemainingHistograms(t *testing.T) {
	in, check, must := remainingHelpers(t)
	a := in("hist_input")
	for _, name := range []string{"sqrt", "sturges", "rice", "scott", "stone", "doane", "fd", "auto"} {
		check("histogram_bin_edges_"+name, must(np.HistogramBinEdges(a, np.HistogramOptions{Range: &[2]float64{-2, 8}}, name)))
	}
	check("histogram_bin_edges_count", must(np.HistogramBinEdges(a, np.HistogramOptions{Count: 5}, "")))
	sample, weights := in("hist_samples"), in("hist_weights")
	for _, density := range []bool{false, true} {
		mode := "count"
		if density {
			mode = "density"
		}
		bins := []np.HistogramOptions{{Edges: []float64{0, .3, 1}}, {Edges: []float64{0, .5, 1}}, {Edges: []float64{0, .2, .8, 1}}}
		h, e, err := np.HistogramDD(sample, bins, weights, density)
		if err != nil {
			t.Fatal(err)
		}
		check("histogramdd_"+mode, h)
		for j, edge := range e {
			check("histogramdd_"+mode+"_edges_"+strconv.Itoa(j), edge)
		}
		x, y := must(sample.Slice(np.Slice{}, np.AtIndex(0))), must(sample.Slice(np.Slice{}, np.AtIndex(1)))
		h, xe, ye, err := np.Histogram2D(x, y, []np.HistogramOptions{{Count: 3, Range: &[2]float64{0, 1}}, {Count: 2, Range: &[2]float64{0, 1}}}, weights, density)
		if err != nil {
			t.Fatal(err)
		}
		check("histogram2d_"+mode, h)
		check("histogram2d_"+mode+"_x", xe)
		check("histogram2d_"+mode+"_y", ye)
	}
}

func TestRemainingTextIO(t *testing.T) {
	in, check, must := remainingHelpers(t)
	text := func(name string) string {
		b, e := np.Values[uint8](in(name))
		if e != nil {
			t.Fatal(e)
		}
		return string(b)
	}
	checkString := func(name, value string) { check(name, must(np.FromSlice([]byte(value), len(value)))) }
	check("fromstring", must(np.FromString("1.5, -2.25, 3e2, 4", np.Float64, ",", 3)))
	check("fromstring_int", must(np.FromString("1 -2 3 999999999999999", np.Int64, " ", -1)))
	check("fromstring_complex", must(np.FromString("1+2j -3.5-4j 5", np.Complex128, " ", -1)))
	check("loadtxt", must(np.LoadText(strings.NewReader(text("loadtxt_input")), np.TextOptions{DType: np.Int64, Delimiter: ",", UseCols: []int{3, 1}, NDMin: 2})))
	check("loadtxt_unpack", must(np.LoadText(strings.NewReader(text("loadtxt_input")), np.TextOptions{Delimiter: ",", Unpack: true, SkipRows: 2, MaxRows: 2})))
	check("loadtxt_scalar", must(np.LoadText(strings.NewReader("5.25"), np.TextOptions{})))
	check("genfromtxt", must(np.GenFromText(strings.NewReader(text("genfromtxt_input")), np.TextOptions{Delimiter: ",", MissingValues: []string{"NA"}, FillingValue: -99.})))
	check("genfromtxt_default", must(np.GenFromText(strings.NewReader("1,2\n,4"), np.TextOptions{Delimiter: ","})))
	var b bytes.Buffer
	a := must(np.FromFloat64([]float64{1.25, -3.5, 4.5, 2.75}, 2, 2))
	if e := np.SaveText(&b, a, np.SaveTextOptions{Format: "%.2f", Delimiter: ",", Header: "values\nunits", Footer: "end"}); e != nil {
		t.Fatal(e)
	}
	checkString("savetxt", b.String())
	b.Reset()
	c := must(np.FromComplex128([]complex128{1 + 2i, -3 - 4i}, 2))
	if e := np.SaveText(&b, c, np.SaveTextOptions{Format: "%.1f"}); e != nil {
		t.Fatal(e)
	}
	checkString("savetxt_complex", b.String())
	_, e := np.LoadText(strings.NewReader("1 2\n3 4 5"), np.TextOptions{})
	refcheck.Error(t, "remaining_loadtxt_columns", "ValueError", e, np.ErrShape)
	_, e = np.FromString("1 x", np.Float64, " ", -1)
	refcheck.Error(t, "remaining_fromstring_bad", "ValueError", e, np.ErrValue)
	for i, v := range []struct {
		n         int64
		base, pad int
	}{{123, 3, 2}, {-255, 16, 3}, {0, 2, 0}, {0, 10, 4}, {1234567, 36, 0}} {
		s, e := np.BaseRepr(v.n, v.base, v.pad)
		if e != nil {
			t.Fatal(e)
		}
		checkString("base_repr_"+strconv.Itoa(i), s)
	}
	for i, v := range []struct {
		n     int64
		width *int
	}{{5, nil}, {-5, nil}, {5, np.Int(8)}, {-5, np.Int(8)}, {-128, np.Int(8)}, {0, np.Int(0)}} {
		s, e := np.BinaryRepr(v.n, v.width)
		if e != nil {
			t.Fatal(e)
		}
		checkString("binary_repr_"+strconv.Itoa(i), s)
	}
	_, e = np.BinaryRepr(-5, np.Int(3))
	refcheck.Error(t, "remaining_binary_repr_width", "ValueError", e, np.ErrValue)
	for i, v := range []struct {
		value float64
		opt   np.FloatFormatOptions
	}{
		{1., np.FloatFormatOptions{Trim: "k"}}, {math.Copysign(0, -1), np.FloatFormatOptions{Trim: "k"}},
		{1.23456789, np.FloatFormatOptions{Precision: np.Int(3), Sign: true, Trim: "-"}},
		{1.25, np.FloatFormatOptions{Precision: np.Int(4), Fixed: true, Trim: "k"}},
		{100, np.FloatFormatOptions{Precision: np.Int(5), Fixed: true, Sign: true, Trim: "0"}},
		{1.23456789e-6, np.FloatFormatOptions{Trim: "."}},
		{12345.6789, np.FloatFormatOptions{Precision: np.Int(2), Fixed: true, Trim: "-"}},
	} {
		s, e := np.FormatFloatPositional(v.value, v.opt)
		if e != nil {
			t.Fatal(e)
		}
		checkString("format_float_positional_"+strconv.Itoa(i), s)
		s, e = np.FormatFloatScientific(v.value, v.opt)
		if e != nil {
			t.Fatal(e)
		}
		checkString("format_float_scientific_"+strconv.Itoa(i), s)
	}
}

func TestRemainingEinsumPath(t *testing.T) {
	in, check, must := remainingHelpers(t)
	inputs := []*np.Array{in("einsum_path_input_0"), in("einsum_path_input_1"), in("einsum_path_input_2"), in("einsum_path_input_3")}
	const equation = "ab,bc,cd,de->ae"
	for label, opt := range map[string]np.EinsumPathOptions{
		"greedy": {Optimize: "greedy"}, "optimal": {Optimize: "optimal"}, "none": {Optimize: "none"}, "limited": {Optimize: "greedy", MemoryLimit: 1},
	} {
		plan, e := np.EinsumPath(equation, opt, inputs...)
		if e != nil {
			t.Fatal(e)
		}
		var steps, sizes []int64
		for _, step := range plan.Steps {
			sizes = append(sizes, int64(len(step)))
			for _, n := range step {
				steps = append(steps, int64(n))
			}
		}
		prefix := "einsum_path_" + label + "_"
		check(prefix+"steps", must(np.FromSlice(steps, len(steps))))
		check(prefix+"sizes", must(np.FromSlice(sizes, len(sizes))))
		check(prefix+"metrics", must(np.FromSlice([]int64{plan.NaiveCost, plan.OptimizedCost, plan.LargestIntermediate}, 3)))
		check(prefix+"output", must(np.EinsumWithPath(equation, plan, inputs...)))
	}
}
