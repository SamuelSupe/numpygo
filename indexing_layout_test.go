package np_test

import (
	"encoding/binary"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/dtypes"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"github.com/SamuelSupe/numpygo/records"
)

func TestAdvancedIndexLayoutNumPy(t *testing.T) {
	m := refcheck.Require(t)
	ar := func(n int, sh ...int) *np.Array {
		return m(m(m(np.ArangeInt64(0, int64(n), 1)).Astype(np.Int32)).Reshape(sh...))
	}
	index := func(name string, a *np.Array, key ...np.Index) {
		t.Run(name, func(t *testing.T) {
			v := refcheck.Require(t)(a.Index(key...))
			refcheck.CheckLayout(t, "ixdeep_"+name, v)
			base := []int64{0}
			if b := v.Base(); b != nil {
				base[0] = 1
				for _, d := range b.Shape() {
					base = append(base, int64(d))
				}
			}
			refcheck.Check(t, "ixdeep_"+name+"_base", m(np.FromSlice(base, len(base))))
			if !v.Writable() {
				t.Fatal("advanced result must be writable")
			}
			if v.Size() > 0 {
				before := m(a.Copy())
				coord := make([]int, v.NDim())
				if e := v.SetValue(int32(-999), coord...); e != nil {
					t.Fatal(e)
				}
				equal := m(np.Equal(before, a))
				all := m(np.All(equal))
				if got, e := all.AtValue(); e != nil || got != true {
					t.Fatal("advanced result shares input")
				}
			}
		})
	}
	a := ar(120, 2, 3, 4, 5)
	i := m(np.FromSlice([]int64{0, 1}, 2, 1))
	j := m(np.FromSlice([]int64{0, 1, 0}, 1, 3))
	f := m(m(np.FromSlice([]int64{0, 1, 0, 1, 0, 1}, 2, 3)).AstypeWithOptions(np.Int64, np.CastOptions{Order: "F"}))
	all := np.Slice{}
	for _, p := range []struct {
		name string
		a    *np.Array
	}{
		{"c", a},
		{"f", m(a.AstypeWithOptions(np.Int32, np.CastOptions{Order: "F"}))},
		{"perm", m(a.Transpose(1, 0, 3, 2))},
		{"reverse", m(a.Slice(np.Reverse(), np.Reverse(), np.Slice{Step: np.Int(2)}, np.Reverse()))},
	} {
		sh := p.a.Shape()
		mask := make([]bool, sh[0])
		for k := range mask {
			mask[k] = k%2 == 0
		}
		mask2 := make([]bool, sh[0]*sh[1])
		for k := range mask2 {
			mask2[k] = k%3 == 0
		}
		for _, tc := range []struct {
			name string
			ix   []np.Index
		}{
			{"first", []np.Index{i}}, {"middle", []np.Index{all, i}}, {"last", []np.Index{np.Ellipsis, i}},
			{"adjacent", []np.Index{all, i, j}}, {"separated", []np.Index{all, i, all, j}},
			{"newaxis", []np.Index{all, i, np.NewAxis, j}}, {"prefix_new", []np.Index{np.NewAxis, i}},
			{"middle_f", []np.Index{all, f}}, {"single_f", []np.Index{f}}, {"zero_idx", []np.Index{np.ScalarOf(int64(1))}},
			{"middle_zero", []np.Index{all, np.ScalarOf(int64(1))}}, {"bool0", []np.Index{np.ScalarOf(true)}},
			{"boolfalse", []np.Index{np.ScalarOf(false)}}, {"suffix_bool", []np.Index{all, np.ScalarOf(true)}},
			{"slice_step", []np.Index{i, np.Slice{Step: np.Int(-2)}, np.Reverse()}},
			{"mask", []np.Index{m(np.FromSlice(mask, sh[0]))}}, {"mask2", []np.Index{m(np.FromSlice(mask2, sh[0], sh[1]))}},
			{"empty_adv", []np.Index{m(np.Zeros(np.Int64, 0, 1)), j}},
		} {
			index(p.name+"_"+tc.name, p.a, tc.ix...)
		}
	}
	a = ar(12, 3, 4)
	jf := m(np.Subtract(np.ScalarOf(int64(1)), f))
	jf = m(jf.AstypeWithOptions(np.Int64, np.CastOptions{Order: "F"}))
	index("uint_wrap", a, m(np.FromSlice([]uint64{^uint64(0)}, 1)))
	for _, tc := range []struct {
		name string
		ix   []np.Index
	}{
		{"ff", []np.Index{f, jf}}, {"fc", []np.Index{f, m(jf.Copy())}}, {"fb", []np.Index{f, m(np.FromSlice([]int64{0, 1, 2}, 3))}},
		{"single0", []np.Index{np.ScalarOf(int64(1)), np.ScalarOf(int64(2))}},
		{"ellipsis0", []np.Index{f, np.Ellipsis, jf}}, {"scalar_and_f", []np.Index{np.ScalarOf(int64(1)), f}},
		{"empty_oob", []np.Index{m(np.FromSlice([]int64{99}, 1)), m(np.Zeros(np.Int64, 0))}},
		{"maskempty", []np.Index{m(np.Zeros(np.Bool, 3, 0))}},
	} {
		index(tc.name, a, tc.ix...)
	}
	for _, tc := range []struct {
		name string
		a    *np.Array
	}{
		{"singleton", m(ar(6, 2, 1, 3).AstypeWithOptions(np.Int32, np.CastOptions{Order: "F"}))},
		{"broadcast", m(ar(3, 3).BroadcastTo(2, 1, 3))},
	} {
		ix := m(np.FromSlice([]int64{0, 1}, 2))
		index(tc.name+"_bool0", tc.a, np.ScalarOf(true))
		index(tc.name+"_prefix", tc.a, np.NewAxis, ix)
		index(tc.name+"_suffix", tc.a, ix, np.NewAxis)
	}
	for _, tc := range []struct {
		name string
		ix   *np.Array
	}{
		{"only", f}, {"gap", m(f.Slice(all, np.Reverse()))},
		{"singleton", m(m(np.Zeros(np.Int64, 2, 1, 3)).AstypeWithOptions(np.Int64, np.CastOptions{Order: "F"}))},
	} {
		src := ar(4, 4)
		index("index_"+tc.name, src, tc.ix)
		index("index_"+tc.name+"_newaxis", src, tc.ix, np.NewAxis)
	}
	for _, tc := range []struct {
		name string
		ix   []np.Index
	}{
		{"oob_empty_slice", []np.Index{m(np.FromSlice([]int64{99}, 1)), np.Range(0, 0)}},
		{"oob_scalar", []np.Index{np.AtIndex(99), m(np.Zeros(np.Int64, 0))}},
		{"bad_empty_mask", []np.Index{m(np.Zeros(np.Bool, 0, 5))}},
		{"uint_oob", []np.Index{m(np.FromSlice([]uint64{uint64(1) << 63}, 1))}},
		{"bad_index_type", []np.Index{m(np.FromSlice([]float64{1}, 1))}},
		{"two_ellipsis", []np.Index{np.Ellipsis, i, np.Ellipsis}},
	} {
		_, e := a.Index(tc.ix...)
		refcheck.Error(t, "ixdeep_"+tc.name, "IndexError", e, np.ErrIndex)
	}
}

func TestAdvancedIndexDescriptorAndAssignmentNumPy(t *testing.T) {
	m := refcheck.Require(t)
	ar := func(n int, sh ...int) *np.Array { return m(m(np.ArangeInt64(0, int64(n), 1)).Reshape(sh...)) }
	dm := refcheck.RequireValue[dtypes.Descriptor](t)
	raw := func(n string, a *np.Array) {
		t.Helper()
		refcheck.Bytes(t, "ixdeep_"+n+"_bytes", refcheck.RequireValue[[]byte](t)(a.ToBytes("C")))
		strides := []int64{}
		for _, d := range a.Strides() {
			strides = append(strides, int64(d))
		}
		refcheck.Check(t, "ixdeep_"+n+"_strides", m(np.FromSlice(strides, len(strides))))
		refcheck.Text(t, "ixdeep_"+n+"_dtype", dm(dtypes.FromArray(a)).String())
	}
	all := np.Slice{}
	ix := m(np.FromSlice([]int64{3, 1, 3}, 3))
	a := m(dm(dtypes.Parse(">i4")).Cast(ar(12, 3, 4), "unsafe"))
	y := m(a.Index(all, ix))
	raw("endian", y)
	refcheck.Check(t, "ixdeep_endian_values", m(y.Astype(np.Int32)))
	a = m(dm(dtypes.Parse(">M8[3h]")).Cast(ar(6, 2, 3), "unsafe"))
	ix = m(np.FromSlice([]int64{2, 0}, 2))
	y = m(a.Index(all, ix))
	raw("temporal", y)
	refcheck.Check(t, "ixdeep_temporal_ticks", m(m(y.DateTicks()).Astype(np.Int64)))
	a = m(np.FromFlexible([]any{"a", "b", "😀", "d", "é", "f"}, np.FixedUnicode(2, binary.BigEndian), 2, 3))
	raw("unicode", m(a.Index(all, ix)))
	s := refcheck.RequireValue[*records.Schema](t)(records.NewSchema([]records.FieldSpec{
		{Name: "code", Type: records.Numeric(np.Int16, binary.BigEndian)},
		{Name: "value", Type: records.Numeric(np.Float32, binary.LittleEndian)},
	}, false, 0))
	r := refcheck.RequireValue[*records.Array](t)(records.Zeros(s, 2, 3))
	col := m(r.NDArray().Field("code"))
	if e := col.Assign(nil, ar(6, 2, 3)); e != nil {
		t.Fatal(e)
	}
	col = m(r.NDArray().Field("value"))
	if e := col.Assign(nil, m(np.Add(ar(6, 2, 3), np.Scalar(.25)))); e != nil {
		t.Fatal(e)
	}
	raw("record", m(r.NDArray().Index(all, ix)))
	p := m(m(np.FromSlice([]uint32{0x7fa12345, 0x80000000, 0x7fcabcde, 0x3f800000}, 4)).View(np.Float32))
	p = m(p.Index(m(np.FromSlice([]int64{2, 0, 1}, 3))))
	refcheck.Check(t, "ixdeep_nan_payload", m(p.View(np.Uint32)))
	a = ar(24, 2, 3, 4)
	ix = m(np.FromSlice([]int64{2, 0, 2, 2}, 2, 2))
	if e := a.Assign([]np.Index{all, ix, np.AtIndex(1)}, m(np.FromSlice([]int64{100, 101, 102, 103}, 1, 2, 2))); e != nil {
		t.Fatal(e)
	}
	refcheck.Check(t, "ixdeep_assign_repeat", a)
	a = ar(6, 6)
	if e := a.Assign([]np.Index{m(np.FromSlice([]int64{2, 1, 3, 0}, 4))}, m(a.Slice(np.Range(0, 4)))); e != nil {
		t.Fatal(e)
	}
	refcheck.Check(t, "ixdeep_assign_alias", a)
	a = m(ar(24, 2, 3, 4).Transpose(1, 0, 2))
	add := refcheck.RequireValue[*np.UFunc](t)(np.LookupUFunc("add"))
	if e := add.AtIndices(a, []np.Index{ix, all, np.AtIndex(2)}, m(np.Add(ar(8, 2, 2, 2), np.ScalarOf(int64(1))))); e != nil {
		t.Fatal(e)
	}
	refcheck.Check(t, "ixdeep_at_repeat", a)
	a = ar(12, 3, 4)
	e := a.Assign([]np.Index{ix, all}, m(np.Ones(np.Float64, 3, 3)))
	refcheck.Error(t, "ixdeep_assign_bad_rhs", "ValueError", e, np.ErrShape)
	refcheck.Check(t, "ixdeep_assign_unchanged", a)
}
