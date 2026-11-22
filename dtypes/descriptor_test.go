package dtypes

import (
	"encoding/binary"
	"errors"
	"fmt"
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"github.com/SamuelSupe/numpygo/records"
	"math"
	"strings"
	"testing"
)

func checkMetadata(t *testing.T, name string, d Descriptor) {
	t.Helper()
	m := d.Metadata()
	refcheck.Text(t, name+"_meta", fmt.Sprintf("%s|%s|%s|%c|%c|%c", m.Class, m.Name, m.Str, m.Kind, m.Char, m.ByteOrder))
	o := int64(0)
	if m.HasObject {
		o = 1
	}
	refcheck.Check(t, name+"_sizes", refcheck.Require(t)(np.FromSlice([]int64{int64(m.ItemSize), int64(m.Alignment), int64(m.Num), o}, 4)))
}
func referenceText(t *testing.T, name string) string {
	return string(refcheck.RequireValue[[]uint8](t)(np.Values[uint8](refcheck.Input(t, name))))
}
func TestDTypeClassesNumPy(t *testing.T) {
	names := []string{"BoolDType", "Int8DType", "ByteDType", "UInt8DType", "UByteDType", "Int16DType", "ShortDType", "UInt16DType", "UShortDType",
		"Int32DType", "IntDType", "UInt32DType", "UIntDType", "Int64DType", "LongDType", "UInt64DType", "ULongDType", "LongLongDType", "ULongLongDType",
		"Float16DType", "Float32DType", "Float64DType", "LongDoubleDType", "Complex64DType", "Complex128DType", "CLongDoubleDType", "ObjectDType",
		"BytesDType", "StrDType", "VoidDType", "DateTime64DType", "TimeDelta64DType", "StringDType"}
	for _, n := range names {
		t.Run(n, func(t *testing.T) {
			m := refcheck.Require(t)
			opt := Options{}
			switch n {
			case "BytesDType", "StrDType", "VoidDType":
				opt.Width = 3
			case "DateTime64DType":
				opt.Unit = "3h"
			case "TimeDelta64DType":
				opt.Unit = "2D"
			}
			d := refcheck.RequireValue[Descriptor](t)(New(n, opt))
			name := "dtype_class_" + n
			checkMetadata(t, name, d)
			a := m(d.Zeros(2))
			switch {
			case d.DType() == np.Object:
				refcheck.Check(t, name+"_zeros", m(a.Astype(np.Int64)))
			case d.DType().IsTemporal():
				refcheck.Check(t, name+"_zeros", m(a.DateTicks()))
			case d.DType().IsFlexible():
				refcheck.CheckFlexible(t, name+"_zeros", a, false)
			case d.DType() == np.StringDType:
				refcheck.CheckCharacter(t, name+"_zeros", a, false)
			default:
				refcheck.Check(t, name+"_zeros", a)
			}
		})
	}
	parse := refcheck.RequireValue[Descriptor](t)
	for i, s := range []string{"S0", "U0", "V0", ">i4", ">U3", ">M8[3h]", ">m8[D]", "M8", "m8"} {
		checkMetadata(t, fmt.Sprintf("dtype_extra_%d", i), parse(Parse(s)))
	}
	checkMetadata(t, "dtype_nullable", parse(New("StringDType", Options{String: np.StringDescriptor{HasNA: true, NA: nil, Strict: true}})))
	timeType := refcheck.RequireValue[records.Type](t)(records.Temporal(np.DateTime64, "D", nil))
	schema := refcheck.RequireValue[*records.Schema](t)(records.NewSchema([]records.FieldSpec{
		{Name: "a", Title: "title", Type: records.Numeric(np.Int16, binary.BigEndian)},
		{Name: "b", Type: timeType},
	}, true, 0))
	checkMetadata(t, "dtype_record", parse(New("VoidDType", Options{Schema: schema})))
}
func TestDTypePromotionCastingMatrixNumPy(t *testing.T) {
	specs := strings.Split(referenceText(t, "dtype_matrix_input"), "\n")
	codes := strings.Split(referenceText(t, "dtype_matrix_codes"), "\n")
	promote := refcheck.RequireValue[[]int64](t)(np.Values[int64](refcheck.Input(t, "dtype_matrix_promote")))
	cast := refcheck.RequireValue[[]bool](t)(np.Values[bool](refcheck.Input(t, "dtype_matrix_cast")))
	parse := refcheck.RequireValue[Descriptor](t)
	types := make([]Descriptor, len(specs))
	for i, s := range specs {
		types[i] = parse(Parse(s))
	}
	rules := []string{"no", "equiv", "safe", "same_kind", "unsafe"}
	for i, a := range types {
		for j, b := range types {
			t.Run(specs[i]+"_"+specs[j], func(t *testing.T) {
				p := i*len(types) + j
				v, e := Promote(a, b)
				if promote[p] < 0 {
					if !errors.Is(e, np.ErrDType) {
						t.Errorf("promotion error %v (want NumPy class code %d)", e, promote[p])
					}
				} else if e != nil {
					t.Errorf("promotion error %v; want %s", e, codes[promote[p]])
				} else {
					got := v.String() + "|" + string(v.Metadata().Char)
					if got != codes[promote[p]] {
						t.Errorf("promotion %q; want %q", got, codes[promote[p]])
					}
				}
				for k, r := range rules {
					x, e := CanCast(a, b, r)
					if e != nil || x != cast[p*len(rules)+k] {
						t.Errorf("cast %s: %v %v; want %v", r, x, e, cast[p*len(rules)+k])
					}
				}
			})
		}
	}
	d := func(name string, tpe records.Type) Descriptor {
		s := refcheck.RequireValue[*records.Schema](t)(records.NewSchema([]records.FieldSpec{{Name: name, Type: tpe}}, false, 0))
		return parse(New("VoidDType", Options{Schema: s}))
	}
	structs := []Descriptor{d("a", records.Numeric(np.Int16, nil)), d("b", records.Numeric(np.Int32, nil)), d("a", records.Numeric(np.Int16, binary.BigEndian)), d("a", records.Subarray(records.Numeric(np.Int16, nil), 2)), parse(Parse("V2")), parse(Parse("i2"))}
	td := []Descriptor{parse(New("StringDType", Options{})), parse(New("StringDType", Options{String: np.StringDescriptor{Strict: true}}))}
	for _, na := range []any{nil, math.NaN(), "NA"} {
		td = append(td, parse(New("StringDType", Options{String: np.StringDescriptor{HasNA: true, NA: na}})))
	}
	for _, p := range []struct {
		name  string
		types []Descriptor
	}{{"dtype_struct_cast", structs}, {"dtype_string_cast", td}} {
		flags := refcheck.RequireValue[[]bool](t)(np.Values[bool](refcheck.Input(t, p.name)))
		n := 0
		for i, a := range p.types {
			for j, b := range p.types {
				for _, r := range rules {
					x, e := CanCast(a, b, r)
					if e != nil || x != flags[n] {
						t.Errorf("%s %d->%d %s: %v %v; want %v", p.name, i, j, r, x, e, flags[n])
					}
					n++
				}
			}
		}
	}
}

func TestTemporalDescriptorCastsNumPy(t *testing.T) {
	m := refcheck.Require(t)
	parse := refcheck.RequireValue[Descriptor](t)
	for i, s := range []string{"M8[D]", "M8[3h]", "m8[D]", "m8[3h]", "m8[Y]"} {
		t.Run(s, func(t *testing.T) {
			d := parse(Parse(s))
			a := m(np.FromDateTicks(m(np.FromSlice([]int64{0, 1, -2, np.NaT}, 4)), d.DType(), d.Unit().String()))
			p := fmt.Sprintf("temporal_cast_%d_", i)
			for _, v := range []struct{ n, s string }{{"float", "f8"}, {"complex", "c16"}, {"int", "i8"}} {
				refcheck.Check(t, p+v.n, m(parse(Parse(v.s)).Cast(a, "unsafe")))
			}
			for _, v := range []struct{ n, s string }{{"string", "T"}, {"unicode", "U"}, {"bytes", "S32"}} {
				b := m(parse(Parse(v.s)).Cast(a, "unsafe"))
				refcheck.CheckCharacter(t, p+v.n, b, false)
			}
			for _, v := range []struct{ n, s string }{{"date_h", "M8[h]"}, {"delta_h", "m8[h]"}} {
				b := m(parse(Parse(v.s)).Cast(a, "unsafe"))
				refcheck.Check(t, p+v.n, m(b.DateTicks()))
			}
		})
	}
	for i, a := range []*np.Array{m(np.FromSlice([]float64{1.8, -2.8, math.NaN(), math.Inf(1)}, 4)), m(np.FromSlice([]complex128{1.8, -2.8, complex(math.NaN(), 0), complex(math.Inf(1), 0)}, 4))} {
		for _, p := range []struct{ n, s string }{{"date", "M8[D]"}, {"delta", "m8[3h]"}} {
			a := m(parse(Parse(p.s)).Cast(a, "unsafe"))
			refcheck.Check(t, fmt.Sprintf("temporal_from_%d_%s", i, p.n), m(a.DateTicks()))
		}
	}
	a := m(parse(Parse(">i2")).Cast(m(np.FromSlice([]int16{1, -7, 256}, 3)), "unsafe"))
	b := m(parse(Parse(">i4")).Cast(a, "safe"))
	refcheck.Bytes(t, "dtype_cast_endian", refcheck.RequireValue[[]byte](t)(b.ToBytes("C")))
	refcheck.Check(t, "dtype_cast_endian_values", b)
	refcheck.CheckFlexible(t, "dtype_cast_void", m(parse(Parse("V5")).Cast(a, "safe")), false)
	refcheck.CheckFlexible(t, "dtype_cast_void_auto", m(parse(Parse("V")).Cast(a, "safe")), false)
	delta := m(np.FromDateTicks(m(np.FromSlice([]int64{2, 3, -4}, 3)), np.TimeDelta64, "3h"))
	date := m(np.FromDateTicks(m(np.FromSlice([]int64{0, 1, -1}, 3)), np.DateTime64, "D"))
	c := m(np.Concatenate([]*np.Array{delta, m(np.FromSlice([]int16{5, -2}, 2))}, 0))
	refcheck.Check(t, "temporal_mixed_concat", m(c.DateTicks()))
	w := m(np.Where(m(np.FromSlice([]bool{true, false, true}, 3)), delta, m(np.FromSlice([]int16{7, 8, 9}, 3))))
	refcheck.Check(t, "temporal_mixed_where", m(w.DateTicks()))
	v, e := np.Concatenate([]*np.Array{date, delta}, 0)
	refcheck.Case(t, "temporal_mixed_concat_date", v, e)
}
func TestDLPackRegistryNumPy(t *testing.T) {
	parse := refcheck.RequireValue[Descriptor](t)
	r := NewDLPackRegistry()
	i, f := parse(Parse("i4")), parse(Parse("f4"))
	key := DLPackKey{42, 32}
	for range 2 {
		if e := r.RegisterDLPackDType(key, i); e != nil {
			t.Fatal(e)
		}
	}
	refcheck.Check(t, "dtype_dlpack_registered", np.ScalarOf(true))
	refcheck.Error(t, "dtype_dlpack_conflict", "ValueError", r.RegisterDLPackDType(key, f), np.ErrValue)
	refcheck.Error(t, "dtype_dlpack_export_conflict", "ValueError", r.RegisterDLPackDType(DLPackKey{43, 32}, f), np.ErrValue)
	refcheck.Error(t, "dtype_dlpack_bad_code", "ValueError", r.RegisterDLPackDType(DLPackKey{-1, 32}, i), np.ErrValue)
	refcheck.Error(t, "dtype_dlpack_bad_width", "ValueError", r.RegisterDLPackDType(DLPackKey{44, 16}, i), np.ErrValue)
	if got := refcheck.RequireValue[DLPackKey](t)(r.Export(f)); got != key {
		t.Fatalf("conflicting export not registered: %v", got)
	}
	if got := parse(r.Import(key)); got.DType() != np.Int32 {
		t.Fatal("import overwritten by conflict")
	}
	if got := parse(r.Import(DLPackKey{0, 32})); got.DType() != np.Int32 {
		t.Fatal("built-in import unavailable")
	}
}
