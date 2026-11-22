package ma

import (
	"slices"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/objects"
	"github.com/SamuelSupe/numpygo/records"
)

// Ids returns opaque data/mask address tokens. A nil mask is represented by
// zero instead of the address of Python's process-wide nomask singleton.
func Ids(a *Array) (uintptr, uintptr) {
	if !valid(a) {
		return 0, 0
	}
	m := GetMask(a)
	if m == nil {
		return a.data.DataIdentity(), 0
	}
	return a.data.DataIdentity(), m.DataIdentity()
}
func MakeMaskDescr(schema *records.Schema) (*records.Schema, error) {
	if schema == nil {
		return nil, np.ErrDType
	}
	return schema.MaskSchema()
}

type StructuredArray struct {
	Data, Mask *records.Array
	hard       bool
}

func sameMaskType(a, b *records.Schema) bool {
	if a == nil || b == nil {
		return false
	}
	x, y := a.Fields(), b.Fields()
	if len(x) != len(y) {
		return false
	}
	for i, f := range x {
		t := y[i].Type
		if f.Name != y[i].Name || !slices.Equal(f.Type.Shape, t.Shape) {
			return false
		}
		if f.Type.Kind == records.RecordKind {
			if t.Kind != records.RecordKind || !sameMaskType(f.Type.Schema, t.Schema) {
				return false
			}
		} else if t.Kind != records.NumericKind || t.DType != np.Bool {
			return false
		}
	}
	return true
}
func NewStructured(data, mask *records.Array) (*StructuredArray, error) {
	if data == nil {
		return nil, np.ErrDType
	}
	if mask == nil {
		s, e := MakeMaskDescr(data.Schema())
		if e != nil {
			return nil, e
		}
		mask, e = records.Zeros(s, data.Shape()...)
		if e != nil {
			return nil, e
		}
	}
	if !slices.Equal(data.Shape(), mask.Shape()) || !sameMaskType(data.Schema(), mask.Schema()) {
		return nil, ErrMask
	}
	return &StructuredArray{Data: data, Mask: mask}, nil
}

type MaskedRecord struct {
	Data, Mask *records.Record
	hard       bool
}

func (a *StructuredArray) At(indices ...int) (*MaskedRecord, error) {
	d, e := a.Data.At(indices...)
	if e != nil {
		return nil, e
	}
	m, e := a.Mask.At(indices...)
	if e != nil {
		return nil, e
	}
	return &MaskedRecord{Data: d, Mask: m, hard: a.hard}, nil
}
func (r *MaskedRecord) Field(name string) (value, mask any, err error) {
	v, e := r.Data.Field(name)
	if e != nil {
		return nil, nil, e
	}
	m, e := r.Mask.Field(name)
	return v, m, e
}
func (r *MaskedRecord) Set(name string, value, mask any) error {
	if r.hard {
		old, e := r.Mask.Field(name)
		if e != nil {
			return e
		}
		if b, ok := old.(bool); ok && b {
			return nil
		}
		if b, ok := mask.(bool); ok && b {
			return r.Mask.Set(name, true)
		}
		if _, ok := old.(bool); !ok {
			// Subarray hard masks require elementwise writes.
			return np.ErrDType
		}
	}
	if e := r.Data.Set(name, value); e != nil {
		return e
	}
	return r.Mask.Set(name, mask)
}
func FlattenStructuredArray(a *records.Array) (*np.Array, error) { return a.FlattenNumeric() }
func FlattenMaskedStructuredArray(a *StructuredArray) (*Array, error) {
	data, e := a.Data.FlattenNumeric()
	if e != nil {
		return nil, e
	}
	mask, e := a.Mask.FlattenNumeric()
	if e != nil {
		return nil, e
	}
	return New(data, mask)
}

// FromFlex returns structured data and mask views of NumPy's nested _data
// and _mask record layout. Additional fields are ignored.
func FromFlex(a *records.Array) (*StructuredArray, error) {
	d, e := a.Subrecords("_data")
	if e != nil {
		return nil, e
	}
	m, e := a.Subrecords("_mask")
	if e != nil {
		return nil, e
	}
	return NewStructured(d, m)
}
func FromFlexNumeric(a *records.Array) (*Array, error) {
	d, e := a.Field("_data")
	if e != nil {
		return nil, e
	}
	m, e := a.Field("_mask")
	if e != nil {
		return nil, e
	}
	data, e := d.NumericArray()
	if e != nil {
		return nil, e
	}
	mask, e := m.NumericArray()
	if e != nil {
		return nil, e
	}
	return New(data, mask)
}

type ObjectArray struct {
	Data      *objects.Array
	Mask      *np.Array
	FillValue any
}

func MaskedObject(a *objects.Array, value any, copyData bool) (*ObjectArray, error) {
	if a == nil {
		return nil, np.ErrDType
	}
	if copyData {
		var e error
		a, e = a.Copy()
		if e != nil {
			return nil, e
		}
	}
	m, e := a.EqualTo(value)
	if e != nil {
		return nil, e
	}
	return &ObjectArray{a, m, value}, nil
}
func (a *ObjectArray) Filled(fill any) (*objects.Array, error) {
	v := a.Data.Values()
	m, e := np.Values[bool](a.Mask)
	if e != nil {
		return nil, e
	}
	for i, x := range m {
		if x {
			v[i] = fill
		}
	}
	return objects.FromSlice(v, a.Data.Shape()...)
}
