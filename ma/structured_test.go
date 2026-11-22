package ma

import (
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"github.com/SamuelSupe/numpygo/objects"
	"github.com/SamuelSupe/numpygo/records"
	"strings"
	"testing"
)

func TestStructuredMasks(t *testing.T) {
	m := refcheck.Require(t)
	schema := refcheck.RequireValue[*records.Schema](t)
	arr := refcheck.RequireValue[*records.Array](t)
	dt := schema(records.NewSchema([]records.FieldSpec{{Name: "a", Type: records.Numeric(np.Int32, nil)}, {Name: "v", Type: records.Subarray(records.Numeric(np.Float64, nil), 2)}}, false, 0))
	d := arr(records.FromRows(dt, [][]any{{int32(1), []float64{2, 3}}, {int32(4), []float64{5, 6}}}, 2))
	refcheck.Check(t, "final_record_flat", m(FlattenStructuredArray(d)))
	md := schema(MakeMaskDescr(dt))
	mask := arr(records.FromRows(md, [][]any{{false, []bool{true, false}}, {true, []bool{false, true}}}, 2))
	fdt := schema(records.NewSchema([]records.FieldSpec{{Name: "_data", Type: records.Struct(dt)}, {Name: "_mask", Type: records.Struct(md)}}, false, 0))
	r := refcheck.RequireValue[*records.Record](t)
	flex := arr(records.FromRows(fdt, [][]any{{r(d.At(0)), r(mask.At(0))}, {r(d.At(1)), r(mask.At(1))}}, 2))
	recovered := refcheck.RequireValue[*StructuredArray](t)(FromFlex(flex))
	f := refcheck.RequireValue[*Array](t)(FlattenMaskedStructuredArray(recovered))
	refcheck.Check(t, "final_record_fromflex_data", f.Data())
	refcheck.Check(t, "final_record_fromflex_mask", f.Mask())
	s := refcheck.RequireValue[*MaskedRecord](t)(recovered.At(0))
	v, vm, e := s.Field("a")
	if e != nil {
		t.Fatal(e)
	}
	b, bm, e := s.Field("v")
	if e != nil {
		t.Fatal(e)
	}
	floats := refcheck.RequireValue[[]float64](t)(b.(*np.Array).Float64s())
	bools := refcheck.RequireValue[[]bool](t)(np.Values[bool](bm.(*np.Array)))
	refcheck.Check(t, "final_record_mvoid", m(np.WrapSlice([]float64{float64(v.(int32)), floats[1]}, 2)))
	refcheck.Check(t, "final_record_mvoid_mask", m(np.WrapSlice([]bool{vm.(bool), bools[0]}, 2)))
	base := refcheck.RequireValue[*Array](t)(New(m(np.WrapSlice([]int64{1, 2, 3}, 3)), m(np.WrapSlice([]bool{false, true, false}, 3))))
	same := refcheck.RequireValue[*Array](t)(New(m(base.Data().Slice()), m(base.Mask().Slice())))
	p, q := Ids(base)
	p2, q2 := Ids(same)
	refcheck.Check(t, "final_record_ids", m(np.WrapSlice([]bool{p == p2, q == q2, p != 0, q != 0}, 4)))
	obj := refcheck.RequireValue[*objects.Array](t)(objects.FromSlice([]any{"red", "missing", "blue", "missing"}, 4))
	masked := refcheck.RequireValue[*ObjectArray](t)(MaskedObject(obj, "missing", false))
	refcheck.Check(t, "final_object_mask", masked.Mask)
	filled := refcheck.RequireValue[*objects.Array](t)(masked.Filled("NA"))
	strs := []string{}
	for _, v := range filled.Values() {
		strs = append(strs, v.(string))
	}
	refcheck.Text(t, "final_object_filled", strings.Join(strs, "|"))
}
