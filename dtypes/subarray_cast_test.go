package dtypes

import (
	"encoding/binary"
	"fmt"
	"strconv"
	"strings"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"github.com/SamuelSupe/numpygo/records"
)

func TestSubarrayDTypeNumPy(t *testing.T) {
	m := refcheck.Require(t)
	dm := refcheck.RequireValue[Descriptor](t)
	types := []Descriptor{}
	for _, s := range []string{"i2", ">i2", "i8", "f4", "f8", "S2", "U3", "V0", "V4", "V8", "O"} {
		types = append(types, dm(Parse(s)))
	}
	for _, p := range []struct {
		dt string
		sh []int
	}{
		{"i2", []int{2}}, {">i2", []int{2}}, {"f4", []int{2}}, {"i8", []int{2}}, {"S2", []int{2}},
		{"U3", []int{2}}, {"V4", []int{2}}, {"i2", []int{1, 2}}, {"i2", []int{0}}, {"i2", []int{3}},
	} {
		types = append(types, dm(Subarray(dm(Parse(p.dt)), p.sh...)))
	}
	var desc func(Descriptor) string
	desc = func(d Descriptor) string {
		if base, sh, ok := d.Subdtype(); ok {
			parts := []string{}
			for _, n := range sh {
				parts = append(parts, strconv.Itoa(n))
			}
			return desc(base) + "(" + strings.Join(parts, ",") + ")"
		}
		return d.String()
	}
	flags := refcheck.RequireValue[[]bool](t)(np.Values[bool](refcheck.Input(t, "subdeep_can_cast")))
	raw := refcheck.RequireValue[[]uint8](t)(np.Values[uint8](refcheck.Input(t, "subdeep_promote")))
	promoted := strings.Split(string(raw), "\n")
	raw = refcheck.RequireValue[[]uint8](t)(np.Values[uint8](refcheck.Input(t, "subdeep_result_type")))
	resultTypes := strings.Split(string(raw), "\n")
	k, p := 0, 0
	for _, a := range types {
		for _, b := range types {
			if a.NDim() == 0 && b.NDim() == 0 {
				continue
			}
			t.Run(desc(a)+"-"+desc(b), func(t *testing.T) {
				for _, rule := range []string{"no", "equiv", "safe", "same_kind", "unsafe"} {
					got, e := CanCast(a, b, rule)
					if e != nil || got != flags[k] {
						t.Errorf("can_cast %s -> %s %s: got %v %v want %v", desc(a), desc(b), rule, got, e, flags[k])
					}
					k++
				}
				v, e := Promote(a, b)
				got := "!"
				if e == nil {
					got = desc(v)
				}
				if got != promoted[p] {
					t.Errorf("promotion %s -> %s: got %s want %s", desc(a), desc(b), got, promoted[p])
				}
				if got != resultTypes[p] {
					t.Errorf("result_type %s -> %s: got %s want %s", desc(a), desc(b), got, resultTypes[p])
				}
				p++
			})
		}
	}
	if k != len(flags) || p != len(promoted) || p != len(resultTypes) {
		t.Fatal("incomplete descriptor grid")
	}
	ar := func(n int, sh ...int) *np.Array {
		return m(m(m(np.ArangeInt64(0, int64(n), 1)).Astype(np.Int16)).Reshape(sh...))
	}
	for i, a := range []*np.Array{np.ScalarOf(int16(7)), ar(3, 3), m(ar(6, 2, 3).Slice(np.Slice{}, np.Reverse()))} {
		for j, tc := range []struct {
			dt string
			sh []int
		}{
			{"i4", []int{2}}, {"f4", []int{2, 3}}, {">i4", []int{1, 2}}, {"i8", []int{1}}, {"i2", []int{0}},
		} {
			prefix := fmt.Sprintf("subdeep_cast_%d_%d", i, j)
			d := dm(Subarray(dm(Parse(tc.dt)), tc.sh...))
			v := m(d.Cast(a, "safe"))
			refcheck.Bytes(t, prefix+"_bytes", refcheck.RequireValue[[]byte](t)(v.ToBytes("C")))
			sh := []int64{}
			for _, n := range v.Shape() {
				sh = append(sh, int64(n))
			}
			refcheck.Check(t, prefix+"_shape", m(np.FromSlice(sh, len(sh))))
			refcheck.Text(t, prefix+"_dtype", dm(FromArray(v)).String())
			if v.DType() != np.VoidDType {
				refcheck.Check(t, prefix+"_values", m(v.Astype(v.DType())))
			}
		}
	}
	d := dm(Subarray(dm(Parse("i2")), 2))
	a := m(np.FromSlice([]float64{1.8, -2.8}, 2))
	refcheck.Check(t, "subdeep_unsafe_cast", m(d.Cast(a, "unsafe")))
	_, e := d.Cast(a, "safe")
	refcheck.Error(t, "subdeep_safe_reject", "TypeError", e, np.ErrDType)
}

func TestStructuredCastNumPy(t *testing.T) {
	m := refcheck.Require(t)
	dm := refcheck.RequireValue[Descriptor](t)
	sm := refcheck.RequireValue[*records.Schema](t)
	rec := func(name, dt string, sh ...int) Descriptor {
		b := dm(Parse(dt))
		typ := refcheck.RequireValue[records.Type](t)(b.RecordType())
		return dm(New("VoidDType", Options{Schema: sm(records.NewSchema([]records.FieldSpec{{Name: name, Type: records.Subarray(typ, sh...)}}, false, 0))}))
	}
	multi := func(names []string, types []records.Type) Descriptor {
		spec := make([]records.FieldSpec, len(names))
		for i, n := range names {
			spec[i] = records.FieldSpec{Name: n, Type: types[i]}
		}
		return dm(New("VoidDType", Options{Schema: sm(records.NewSchema(spec, false, 0))}))
	}
	structs := []Descriptor{
		rec("a", "i2"), rec("b", "i4"), rec("a", ">i2"), rec("a", "i2", 2), rec("a", "i4", 3),
		rec("a", "i4", 1, 2), rec("a", "i2", 0),
		multi([]string{"a", "b"}, []records.Type{records.Numeric(np.Int16, nil), records.Numeric(np.Float32, nil)}),
		multi([]string{"x", "y"}, []records.Type{records.Numeric(np.Int32, nil), records.Numeric(np.Float64, nil)}),
		dm(Parse("i2")), dm(Parse("V0")), dm(Parse("V4")), dm(Parse("O")),
	}
	flags := refcheck.RequireValue[[]bool](t)(np.Values[bool](refcheck.Input(t, "subdeep_record_cast")))
	k := 0
	for i, a := range structs {
		for j, b := range structs {
			t.Run(fmt.Sprintf("cast_%d_%d", i, j), func(t *testing.T) {
				for _, rule := range []string{"no", "equiv", "safe", "same_kind", "unsafe"} {
					v, e := CanCast(a, b, rule)
					if e != nil || v != flags[k] {
						t.Errorf("%d->%d %s: got %v %v want %v", i, j, rule, v, e, flags[k])
					}
					k++
				}
			})
		}
	}
	for i, p := range []struct{ a, b []int }{
		{[]int{2}, []int{3}}, {[]int{3}, []int{2}}, {[]int{1}, []int{3}}, {[]int{2, 1}, []int{2, 3}},
		{[]int{2}, []int{2, 3}}, {[]int{3}, []int{2, 3}}, {[]int{2, 3}, []int{3}}, {[]int{2, 3}, []int{3, 2}},
		{[]int{2, 3}, []int{1, 2, 3}}, {[]int{2, 3}, []int{2, 1, 3}}, {[]int{2, 3}, nil}, {nil, []int{2, 3}},
		{[]int{1}, nil}, {[]int{0}, []int{2}}, {[]int{2}, []int{0}}, {[]int{1, 2}, []int{2, 1}},
	} {
		t.Run(fmt.Sprintf("shape_%d", i), func(t *testing.T) {
			src, dst := rec("a", ">i2", p.a...), rec("b", "i4", p.b...)
			a := m(src.Zeros(2))
			col := m(a.Field("a"))
			v := m(m(np.ArangeInt64(1, int64(col.Size()+1), 1)).Reshape(col.Shape()...))
			if e := col.Assign(nil, v); e != nil {
				t.Fatal(e)
			}
			x, e := dst.Cast(a, "unsafe")
			if i == 14 {
				// Empty destination storage must remain safe even though
				// the upstream reference cannot execute this conversion.
				if e != nil {
					t.Fatal(e)
				}
				if col := m(x.Field("b")); col.Size() != 0 {
					t.Fatal("nonempty zero-width field")
				}
				return
			}
			name := fmt.Sprintf("subdeep_record_shape_%d", i)
			if e != nil {
				refcheck.Case(t, name, nil, e)
				return
			}
			refcheck.Case(t, name, m(x.Field("b")), nil)
		})
	}
	a := m(multi([]string{"a", "b"}, []records.Type{records.Numeric(np.Int16, binary.BigEndian), records.Numeric(np.Float32, nil)}).Zeros(3))
	if e := m(a.Field("a")).Assign(nil, m(np.FromSlice([]int16{3, -9, 7}, 3))); e != nil {
		t.Fatal(e)
	}
	if e := m(a.Field("b")).Assign(nil, m(np.FromSlice([]float32{1.5, -2.25, 9}, 3))); e != nil {
		t.Fatal(e)
	}
	dst := multi([]string{"x", "y"}, []records.Type{records.Numeric(np.Int32, nil), records.Numeric(np.Float64, binary.BigEndian)})
	v := m(dst.Cast(m(a.Slice(np.Reverse())), "safe"))
	refcheck.Check(t, "subdeep_record_pos_x", m(v.Field("x")))
	refcheck.Check(t, "subdeep_record_pos_y", m(m(v.Field("y")).Astype(np.Float64)))
	dst = multi([]string{"x", "y"}, []records.Type{records.Numeric(np.Int32, nil), records.Subarray(records.Numeric(np.Float64, nil), 2)})
	v = m(dst.Cast(m(m(np.ArangeInt64(0, 3, 1)).Astype(np.Int16)), "unsafe"))
	refcheck.Check(t, "subdeep_scalar_record_x", m(v.Field("x")))
	refcheck.Check(t, "subdeep_scalar_record_y", m(v.Field("y")))
	one := m(rec("a", ">i2").Zeros(2))
	if e := m(one.Field("a")).Assign(nil, m(np.FromSlice([]int16{3, -7}, 2))); e != nil {
		t.Fatal(e)
	}
	refcheck.Check(t, "subdeep_record_scalar", m(dm(Parse("f8")).Cast(one, "unsafe")))
	plain := m(dm(Parse("V")).Cast(a, "no"))
	refcheck.Bytes(t, "subdeep_record_auto_void_bytes", refcheck.RequireValue[[]byte](t)(plain.ToBytes("C")))
	refcheck.Text(t, "subdeep_record_auto_void_fields", strings.Join(dm(FromArray(plain)).Names(), ","))
	n := m(rec("a", ">i2", 2).Zeros(2))
	if e := m(n.Field("a")).Assign(nil, m(np.FromSlice([]int16{256, 17, 3, -7}, 2, 2))); e != nil {
		t.Fatal(e)
	}
	for _, w := range []int{1, 2, 4, 6} {
		v = m(dm(Parse(fmt.Sprintf("V%d", w))).Cast(n, "unsafe"))
		refcheck.Bytes(t, fmt.Sprintf("subdeep_record_void_%d", w), refcheck.RequireValue[[]byte](t)(v.ToBytes("C")))
	}
	temporal := func(dt np.DType, unit string, order binary.ByteOrder) records.Type {
		return refcheck.RequireValue[records.Type](t)(records.Temporal(dt, unit, order))
	}
	inner := multi([]string{"coord", "time"}, []records.Type{
		records.Subarray(records.Numeric(np.Int16, binary.BigEndian), 2), temporal(np.DateTime64, "D", binary.BigEndian),
	})
	targetInner := multi([]string{"q", "ts"}, []records.Type{
		records.Subarray(records.Numeric(np.Float32, nil), 3), temporal(np.DateTime64, "h", nil),
	})
	outer := multi([]string{"point", "tag"}, []records.Type{records.Subarray(records.Struct(inner.Schema()), 2), records.FixedBytes(3)})
	target := multi([]string{"node", "label"}, []records.Type{records.Subarray(records.Struct(targetInner.Schema()), 2), records.FixedUnicode(4, nil)})
	n = m(outer.Zeros(2))
	coords := m(m(np.ArangeInt64(1, 9, 1)).Reshape(2, 2, 2))
	if e := m(n.Field("point", "coord")).Assign(nil, coords); e != nil {
		t.Fatal(e)
	}
	ticks := m(m(np.ArangeInt64(0, 4, 1)).Reshape(2, 2))
	if e := m(n.Field("point", "time")).Assign(nil, m(np.FromDateTicks(ticks, np.DateTime64, "D"))); e != nil {
		t.Fatal(e)
	}
	if e := m(n.Field("tag")).Assign(nil, m(np.FromFlexible([]any{"abc", "xy"}, np.FixedBytes(3), 2))); e != nil {
		t.Fatal(e)
	}
	v = m(target.Cast(m(n.Slice(np.Reverse())), "unsafe"))
	refcheck.Check(t, "subdeep_record_nested_q", m(v.Field("node", "q")))
	refcheck.Check(t, "subdeep_record_nested_time", m(m(v.Field("node", "ts")).DateTicks()))
	refcheck.Bytes(t, "subdeep_record_nested_label", refcheck.RequireValue[[]byte](t)(m(v.Field("label")).ToBytes("C")))
	n = m(dst.Zeros(3))
	if e := n.Assign([]np.Index{m(np.FromSlice([]int64{0, 0, 2}, 3))}, m(np.FromSlice([]int16{2, 7, 9}, 3))); e != nil {
		t.Fatal(e)
	}
	refcheck.Check(t, "subdeep_record_assign_x", m(n.Field("x")))
	refcheck.Check(t, "subdeep_record_assign_y", m(n.Field("y")))
	_, e := dm(Parse("f8")).Cast(a, "unsafe")
	refcheck.Error(t, "subdeep_multi_record_scalar", "TypeError", e, np.ErrDType)
	_, e = rec("x", "i4").Cast(a, "unsafe")
	refcheck.Error(t, "subdeep_bad_field_count", "TypeError", e, np.ErrDType)
}
