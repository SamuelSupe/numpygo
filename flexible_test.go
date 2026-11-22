package np_test

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"github.com/SamuelSupe/numpygo/records"
)

func TestFlexibleNDArrayNumPy(t *testing.T) {
	for _, label := range []string{"s", "u"} {
		t.Run(label, func(t *testing.T) {
			m := refcheck.Require(t)
			a := refcheck.FlexibleInput(t, "flex_"+label+"_input")
			check := func(name string, v *np.Array, layout bool) {
				t.Helper()
				refcheck.CheckFlexible(t, "flex_"+label+"_"+name, v, layout)
			}
			value := func(name string, v *np.Array) { t.Helper(); refcheck.Check(t, "flex_"+label+"_"+name, v) }
			rev := m(a.Slice(np.Reverse(), np.Reverse()))
			check("reverse", rev, true)
			check("transpose", m(a.Transpose()), true)
			if !np.SharesMemory(a, rev) {
				t.Fatal("negative view lost storage")
			}
			check("reshape", m(m(a.Transpose()).Reshape(6)), false)
			i := m(np.FromSlice([]int64{1, 0}, 2))
			j := m(np.FromSlice([]int64{2, 1}, 2))
			gather := m(a.Index(i, j))
			check("index", gather, false)
			if np.SharesMemory(a, gather) {
				t.Fatal("advanced indexing must copy")
			}
			check("take", m(np.Take(a, m(np.FromSlice([]int64{2, 0}, 2)), np.Int(1))), false)
			broad := m(m(a.Slice(np.Range(0, 1))).BroadcastTo(2, 3))
			check("broadcast", broad, true)
			check("copy", m(rev.Copy()), false)
			check("concat", m(np.Concatenate([]*np.Array{a, m(a.Slice(np.Reverse()))}, 0)), false)
			mask := m(np.FromSlice([]bool{true, false}, 2, 1))
			check("where", m(np.Where(mask, a, m(a.Slice(np.Slice{}, np.Reverse())))), false)
			check("zeros", m(np.ZerosLike(a)), false)
			check("ones", m(np.OnesLike(a)), false)
			check("sort", m(np.Sort(a, np.Int(1))), false)
			value("argsort", m(np.Argsort(a, np.Int(1))))
			value("argmin", m(np.ArgMin(a, np.ArgOptions{Axis: np.Int(1)})))
			value("argmax", m(np.ArgMax(a, np.ArgOptions{Axis: np.Int(1)})))
			value("bool", m(a.Astype(np.Bool)))
			o := np.SumOptions{Axes: []int{0}}
			value("any", m(np.Any(a, o)))
			value("all", m(np.All(a, o)))
			value("count", m(np.CountNonzero(a)))
			coords := refcheck.RequireValue[[]*np.Array](t)(np.Nonzero(a))
			value("nonzero0", coords[0])
			value("nonzero1", coords[1])
			u := refcheck.RequireValue[np.UniqueResult](t)(np.Unique(a))
			check("unique", u.Values, false)
			value("unique_indices", u.Indices)
			value("unique_inverse", u.Inverse)
			value("unique_counts", u.Counts)
			changed := m(a.Copy())
			rhs := m(m(changed.Slice(np.Slice{}, np.Reverse())).Slice(np.Slice{}, np.Range(0, 2)))
			if e := changed.Assign([]np.Index{np.Slice{}, np.Range(1, 3)}, rhs); e != nil {
				t.Fatal(e)
			}
			check("assign", changed, false)
			check("byteswap", m(a.ByteSwap(false)), false)
			v := refcheck.RequireValue[any](t)(a.AtValue(0, 1))
			refcheck.Error(t, "flex_"+label+"_readonly", "ValueError", broad.SetValue(v, 1, 0), np.ErrReadOnly)
			_, e := np.Min(a)
			refcheck.Case(t, "flex_"+label+"_min_error", nil, e)
			_, e = np.Sum(a)
			refcheck.Case(t, "flex_"+label+"_sum_error", nil, e)
			_, e = np.LogicalNot(a)
			refcheck.Case(t, "flex_"+label+"_logical_not_error", nil, e)
		})
	}
}

func TestFlexibleSetAndCastNumPy(t *testing.T) {
	m := refcheck.Require(t)
	for _, label := range []string{"s", "u"} {
		t.Run(label, func(t *testing.T) {
			a := refcheck.FlexibleInput(t, "flex_"+label+"_set_a")
			b := refcheck.FlexibleInput(t, "flex_"+label+"_set_b")
			check := func(name string, v *np.Array) {
				t.Helper()
				refcheck.CheckFlexible(t, "flex_"+label+"_"+name, v, false)
			}
			value := func(name string, v *np.Array) { t.Helper(); refcheck.Check(t, "flex_"+label+"_"+name, v) }
			column := m(a.Reshape(a.Size(), 1))
			value("equal", m(np.Equal(column, b)))
			value("less", m(np.Less(column, b)))
			check("add", m(np.Add(column, b)))
			value("isin", m(np.Isin(a, b, false)))
			check("union", m(np.Union1D(a, b)))
			check("intersect", m(np.Intersect1D(a, b)))
			check("setdiff", m(np.Setdiff1D(a, b)))
			check("setxor", m(np.Setxor1D(a, b)))
			s := m(np.Sort(a, nil))
			value("search", m(np.SearchSorted(s, b, false)))
			value("searchright", m(np.SearchSorted(s, b, true)))
		})
	}
	s := m(np.FromByteStrings([]string{"abc"}, 3, 1))
	u := m(np.FromUnicodeStrings([]string{"中文"}, 2, 1))
	refcheck.CheckFlexible(t, "flex_mixed_char", m(np.Concatenate([]*np.Array{s, u}, 0)), false)
	refcheck.CheckFlexible(t, "flex_mixed_num", m(np.Concatenate([]*np.Array{s, m(np.FromSlice([]int64{math.MaxInt64}, 1))}, 0)), false)
	refcheck.CheckFlexible(t, "flex_mixed_where", m(np.Where(m(np.FromSlice([]bool{false, true}, 2)), m(np.FromUnicodeStrings([]string{"a", "b"}, 1, 2)), m(np.FromSlice([]int32{-1, 22}, 2)))), false)
	nums := m(np.FromSlice([]int64{math.MinInt64, math.MaxInt64, 0}, 3))
	refcheck.CheckFlexible(t, "flex_cast_int", m(nums.Astype(np.BytesDType)), false)
	f := m(np.FromSlice([]float64{0, math.Copysign(0, -1), 1, 1e15, 1e16, 1e-4, 1e-5, 1.23456789}, 8))
	refcheck.CheckFlexible(t, "flex_cast_float", m(f.Astype(np.UnicodeDType)), false)
	z := m(np.FromSlice([]complex128{1 + 2i, 0, complex(math.Copysign(0, -1), math.Copysign(0, -1)), complex(math.NaN(), math.Inf(-1))}, 4))
	refcheck.CheckFlexible(t, "flex_cast_complex", m(z.Astype(np.UnicodeDType)), false)
	refcheck.CheckFlexible(t, "flex_cast_bool", m(m(np.FromSlice([]bool{true, false}, 2)).Astype(np.BytesDType)), false)
	refcheck.Check(t, "flex_cast_back", m(m(np.FromByteStrings([]string{"12", "-3", " 7"}, 4, 3)).Astype(np.Int64)))
	refcheck.CheckFlexible(t, "flex_cast_text", m(m(np.FromByteStrings([]string{"ab", "xyz"}, 3, 2)).CastFlexible(np.FixedUnicode(4, nil))), false)
	_, e := u.CastFlexible(np.FixedBytes(3))
	refcheck.Error(t, "flex_bad_encode", "UnicodeEncodeError", e, np.ErrValue)
	_, e = m(np.FromByteStrings([]string{"\xff"}, 1, 1)).CastFlexible(np.FixedUnicode(1, nil))
	refcheck.Error(t, "flex_bad_decode", "UnicodeDecodeError", e, np.ErrValue)
	_, e = np.Equal(s, u)
	refcheck.Case(t, "flex_bad_mixed_equal", nil, e)
}

func TestFlexibleCastingAndUFuncNumPy(t *testing.T) {
	m := refcheck.Require(t)
	parse := func(s string) np.FlexibleDType {
		return refcheck.RequireValue[np.FlexibleDType](t)(np.ParseFlexibleDType(s))
	}
	source := func(s string) *np.Array {
		switch s {
		case "i8":
			return m(np.Zeros(np.Int64, 1))
		case "f4":
			return m(np.Zeros(np.Float32, 1))
		case "bool":
			return m(np.Zeros(np.Bool, 1))
		}
		return m(np.ZerosFlexible(parse(s), 1))
	}
	pairs := [][2]string{{"S3", "S3"}, {"S3", "S2"}, {"S3", "S5"}, {"S3", "U3"}, {"U3", "S3"}, {">U3", "<U3"}, {"V3", "V3"}, {"V3", "V2"}, {"i8", "S20"}, {"i8", "S21"}, {"f4", "U32"}, {"bool", "S4"}}
	flags := make([]bool, 0, len(pairs)*5)
	for _, p := range pairs {
		a := source(p[0])
		for _, r := range []string{"no", "equiv", "safe", "same_kind", "unsafe"} {
			flags = append(flags, refcheck.RequireValue[bool](t)(a.CanCastFlexible(parse(p[1]), r)))
		}
	}
	refcheck.Check(t, "flex_can_cast", m(np.FromSlice(flags, len(pairs), 5)))
	u := refcheck.FlexibleInput(t, "flex_u_input")
	refcheck.CheckFlexible(t, "flex_cast_fortran", m(u.CastFlexibleWithOptions(parse("<U4"), np.CastOptions{Order: "F"})), true)
	if m(u.CastFlexibleWithOptions(parse(">U4"), np.CastOptions{NoCopy: true})) != u {
		t.Fatal("cast no-copy did not retain header")
	}
	lookup := func(name string) *np.UFunc { return refcheck.RequireValue[*np.UFunc](t)(np.LookupUFunc(name)) }
	a := m(np.FromByteStrings([]string{"abc", "a", "z"}, 3, 3))
	b := m(np.FromByteStrings([]string{"xy", "b", "q"}, 2, 3))
	mask := m(np.FromSlice([]bool{true, false, true}, 3))
	out := m(m(np.FromByteStrings([]string{"old", "old", "old"}, 4, 3)).Slice(np.Reverse()))
	v := m(lookup("add").CallWithOptions(np.UFuncOptions{Where: mask, Out: out}, a, b))
	if v != out {
		t.Fatal("ufunc out identity")
	}
	refcheck.CheckFlexible(t, "flex_ufunc_add_out", v, true)
	bo := m(m(np.FromSlice([]bool{false, true, false}, 3)).Slice(np.Reverse()))
	refcheck.Check(t, "flex_ufunc_equal_out", m(lookup("equal").CallWithOptions(np.UFuncOptions{Where: mask, Out: bo}, a, b)))
	_, e := lookup("add").CallWithOptions(np.UFuncOptions{Out: out, Casting: "safe"}, a, b)
	refcheck.Case(t, "flex_ufunc_bad_safe", nil, e)
}

func TestFlexibleFormattingNumPy(t *testing.T) {
	m := refcheck.Require(t)
	s := refcheck.RequireValue[string](t)
	for _, label := range []string{"s", "u", "v"} {
		a := refcheck.FlexibleInput(t, "flex_fmt_"+label+"_input")
		refcheck.Text(t, "flex_fmt_"+label+"_array2string", s(np.Array2String(a)))
		refcheck.Text(t, "flex_fmt_"+label+"_repr", s(np.ArrayRepr(a)))
		refcheck.Text(t, "flex_fmt_"+label+"_str", s(np.ArrayStr(a)))
		refcheck.Text(t, "flex_fmt_"+label+"_scalar", s(np.ArrayStr(m(m(a.Slice(np.Range(0, 1))).Reshape()))))
	}
}
func TestOpaqueNDArrayNumPy(t *testing.T) {
	m := refcheck.Require(t)
	v := refcheck.FlexibleInput(t, "flex_void_input")
	refcheck.CheckFlexible(t, "flex_void_input", v, true)
	refcheck.CheckFlexible(t, "flex_void_sort", m(np.Sort(v, nil)), false)
	un := refcheck.RequireValue[np.UniqueResult](t)(np.Unique(v))
	refcheck.CheckFlexible(t, "flex_void_unique", un.Values, false)
	refcheck.Check(t, "flex_void_bool", m(v.Astype(np.Bool)))
	reverse := m(v.Slice(np.Reverse()))
	refcheck.Check(t, "flex_void_equal", m(np.Equal(v, reverse)))
	refcheck.Check(t, "flex_void_isin", m(np.Isin(v, m(v.Slice(np.Range(0, 2))), false)))
	refcheck.CheckFlexible(t, "flex_void_concat", m(np.Concatenate([]*np.Array{v, reverse}, 0)), false)
	_, e := np.Concatenate([]*np.Array{v, m(np.ZerosFlexible(np.FixedVoid(2), 1))}, 0)
	refcheck.Error(t, "flex_void_bad_width", "DTypePromotionError", e, np.ErrDType)
	_, e = np.Less(v, v)
	refcheck.Case(t, "flex_void_bad_order", nil, e)
	zero := m(np.ZerosFlexible(np.FixedVoid(0), 2, 3))
	refcheck.CheckFlexible(t, "flex_void_zero", zero, true)
	refcheck.CheckFlexible(t, "flex_void_zero_unique", refcheck.RequireValue[np.UniqueResult](t)(np.Unique(zero)).Values, false)
	refcheck.Check(t, "flex_void_zero_bool", m(zero.Astype(np.Bool)))
	raw := m(np.FromSlice([]uint8{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, 2, 6))
	view := m(raw.ViewFlexible(np.FixedVoid(3)))
	refcheck.CheckFlexible(t, "flex_void_view", view, true)
	if !np.SharesMemory(raw, view) {
		t.Fatal("opaque view lost backing")
	}
	if e := view.SetValue(np.VoidScalar{255, 170, 0}, 1, 0); e != nil {
		t.Fatal(e)
	}
	refcheck.Check(t, "flex_void_view_backing", raw)
}
func TestFlexibleRecordFieldsNumPy(t *testing.T) {
	m := refcheck.Require(t)
	ms := refcheck.RequireValue[*records.Schema](t)
	mr := refcheck.RequireValue[*records.Array](t)
	d := ms(records.NewSchema([]records.FieldSpec{
		{Name: "id", Type: records.Numeric(np.Int8, nil)}, {Name: "s", Type: records.FixedBytes(3)},
		{Name: "u", Type: records.Subarray(records.FixedUnicode(2, binary.BigEndian), 2)},
		{Name: "v", Type: records.Opaque(2)}, {Name: "zero", Type: records.Opaque(0)},
	}, false, 0))
	a := mr(records.FromRows(d, [][]any{
		{0, "abc", []string{"中", "a"}, []byte("ab"), []byte{}},
		{1, "de", []string{"b", "🙂"}, []byte("cd"), []byte{}},
		{2, "xy", []string{"cd", "é"}, []byte("ef"), []byte{}},
		{3, "", []string{"", "z"}, []byte("gh"), []byte{}},
	}, 2, 2))
	nd := a.NDArray()
	for _, name := range []string{"s", "u", "v", "zero"} {
		f := m(nd.Field(name))
		refcheck.CheckFlexible(t, "flex_field_"+name, f, true)
		if name != "zero" && !np.SharesMemory(f, nd) {
			t.Fatalf("%s does not share record data", name)
		}
	}
	c := refcheck.RequireValue[*records.Column](t)(a.Field("u"))
	shared := m(c.NDArray())
	if e := shared.SetValue("中文x", 0, 1, 1); e != nil {
		t.Fatal(e)
	}
	f := m(m(m(nd.Transpose()).Slice(np.Reverse())).Field("s"))
	if e := f.SetValue("12345", 0, 1); e != nil {
		t.Fatal(e)
	}
	if e := m(nd.Field("v")).SetValue(np.VoidScalar{255, 0}, 1, 0); e != nil {
		t.Fatal(e)
	}
	refcheck.Bytes(t, "flex_field_backing", a.Bytes())
	broad := m(m(nd.BroadcastTo(2, 2, 2)).Field("s"))
	refcheck.Error(t, "flex_field_readonly", "ValueError", broad.SetValue("x", 1, 0, 0), np.ErrReadOnly)
	s1 := ms(records.NewSchema([]records.FieldSpec{{Name: "key", Type: records.Numeric(np.Int64, nil)}, {Name: "value", Type: records.FixedBytes(2)}}, false, 0))
	s2 := ms(records.NewSchema([]records.FieldSpec{{Name: "key", Type: records.FixedUnicode(3, nil)}, {Name: "value", Type: records.Numeric(np.Int16, nil)}}, false, 0))
	x := mr(records.FromRows(s1, [][]any{{1, "x"}}, 1)).NDArray()
	y := mr(records.FromRows(s2, [][]any{{"abc", 7}}, 1)).NDArray()
	p := m(np.Concatenate([]*np.Array{x, y}, 0))
	refcheck.Bytes(t, "flex_field_mixed_promotion", refcheck.RequireValue[[]byte](t)(p.ToBytes("C")))
	sch := p.StructuredSchema().(*records.Schema)
	layout := []int64{int64(sch.ItemSize())}
	for _, f := range sch.Fields() {
		layout = append(layout, int64(f.Offset))
	}
	refcheck.Check(t, "flex_field_mixed_offsets", m(np.FromSlice(layout, len(layout))))
}
func TestFlexibleFilesNumPy(t *testing.T) {
	m := refcheck.Require(t)
	for _, name := range []string{"flex_bytes", "flex_unicode", "flex_void"} {
		t.Run(name, func(t *testing.T) {
			raw := refcheck.RequireValue[[]uint8](t)(np.Values[uint8](refcheck.Input(t, "flex_"+name+"_file")))
			a := m(np.LoadNPY(bytes.NewReader(raw)))
			refcheck.CheckFlexible(t, "flex_"+name+"_payload", a, true)
			var buf bytes.Buffer
			if e := np.SaveNPY(&buf, a); e != nil {
				t.Fatal(e)
			}
			refcheck.CheckFlexible(t, "flex_"+name+"_payload", m(np.LoadNPY(bytes.NewReader(buf.Bytes()))), true)
			if path := os.Getenv("NUMPYGO_INTERCHANGE_DIR"); path != "" {
				if e := os.WriteFile(filepath.Join(path, name+".npy"), buf.Bytes(), 0600); e != nil {
					t.Fatal(e)
				}
			}
		})
	}
	for _, name := range []string{"flex_mixed", "flex_mixed_compressed"} {
		t.Run(name, func(t *testing.T) {
			raw := refcheck.RequireValue[[]uint8](t)(np.Values[uint8](refcheck.Input(t, "flex_"+name+"_file")))
			a := refcheck.RequireValue[map[string]*np.Array](t)(records.LoadNPZ(bytes.NewReader(raw), int64(len(raw))))
			if len(a) != 5 || a["record"].DType() != np.RecordDType || a["words"].DType() != np.UnicodeDType || a["raw"].DType() != np.VoidDType || a["numeric"].DType() != np.Int64 || a["dates"].DType() != np.DateTime64 {
				t.Fatal("mixed archive lost descriptors")
			}
			refcheck.Bytes(t, "flex_field_backing", refcheck.RequireValue[[]byte](t)(a["record"].ToBytes("C")))
			refcheck.CheckFlexible(t, "flex_flex_unicode_payload", a["words"], true)
			refcheck.CheckFlexible(t, "flex_void_input", a["raw"], true)
			var out bytes.Buffer
			if e := np.SaveNPZ(&out, a, name == "flex_mixed_compressed"); e != nil {
				t.Fatal(e)
			}
			b := refcheck.RequireValue[map[string]*np.Array](t)(records.LoadNPZ(bytes.NewReader(out.Bytes()), int64(out.Len())))
			for k, v := range a {
				w := b[k]
				if w == nil || w.DType() != v.DType() {
					t.Fatal("archive roundtrip dtype")
				}
				x := refcheck.RequireValue[[]byte](t)(v.ToBytes("C"))
				y := refcheck.RequireValue[[]byte](t)(w.ToBytes("C"))
				if !bytes.Equal(x, y) {
					t.Fatalf("archive member %s payload changed", k)
				}
			}
			if path := os.Getenv("NUMPYGO_INTERCHANGE_DIR"); path != "" {
				if e := os.WriteFile(filepath.Join(path, name+".npz"), out.Bytes(), 0600); e != nil {
					t.Fatal(e)
				}
			}
		})
	}
}
