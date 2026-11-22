package ma

import (
	"bytes"
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/dtypes"
	"github.com/SamuelSupe/numpygo/rec"
	"github.com/SamuelSupe/numpygo/records"
	"io"
)

// MaskedRecords is the explicit Go representation of the mrecords class.
type MaskedRecords = StructuredArray

func (a *StructuredArray) HardMask() bool { return a.hard }
func (a *StructuredArray) HardenMask()    { a.hard = true }
func (a *StructuredArray) SoftenMask()    { a.hard = false }
func (a *StructuredArray) Copy() (*StructuredArray, error) {
	d, e := a.Data.Copy()
	if e != nil {
		return nil, e
	}
	m, e := a.Mask.Copy()
	if e != nil {
		return nil, e
	}
	return &StructuredArray{Data: d, Mask: m, hard: a.hard}, nil
}
func (a *StructuredArray) View(indices ...np.Index) (*StructuredArray, error) {
	d, e := a.Data.Slice(indices...)
	if e != nil {
		return nil, e
	}
	m, e := a.Mask.Slice(indices...)
	if e != nil {
		return nil, e
	}
	return &StructuredArray{Data: d, Mask: m, hard: a.hard}, nil
}
func (a *StructuredArray) Field(name string) (*Array, error) {
	d, e := a.Data.NDArray().Field(name)
	if e != nil {
		return nil, e
	}
	m, e := a.Mask.NDArray().Field(name)
	if e != nil {
		return nil, e
	}
	if m.DType() != np.Bool {
		return nil, np.ErrDType
	}
	return New(d, m, Options{HardMask: a.hard})
}
func (a *StructuredArray) RecordMask() (*np.Array, error) {
	m, e := a.Mask.FlattenNumeric()
	if e != nil {
		return nil, e
	}
	v, e := np.All(m, np.SumOptions{Axes: []int{-1}})
	if e != nil {
		return nil, e
	}
	return v.Reshape(a.Data.Shape()...)
}
func maskedRecordList(data, mask *records.Record) ([]any, error) {
	vals, e := data.Values()
	if e != nil {
		return nil, e
	}
	masks, e := mask.Values()
	if e != nil {
		return nil, e
	}
	for i, v := range masks {
		switch m := v.(type) {
		case bool:
			if m {
				vals[i] = nil
			}
		case *records.Record:
			d, ok := vals[i].(*records.Record)
			if !ok {
				return nil, np.ErrDType
			}
			vals[i], e = maskedRecordList(d, m)
		case *np.Array:
			d, ok := vals[i].(*np.Array)
			if !ok {
				return nil, np.ErrDType
			}
			if m.DType() == np.RecordDType {
				x, err := records.FromNDArray(d)
				if err != nil {
					return nil, err
				}
				y, err := records.FromNDArray(m)
				if err != nil {
					return nil, err
				}
				a, err := NewStructured(x, y)
				if err != nil {
					return nil, err
				}
				vals[i], e = a.ToList()
			} else {
				a, err := New(d, m)
				if err != nil {
					return nil, err
				}
				vals[i], e = a.ToList()
			}
		default:
			return nil, np.ErrDType
		}
		if e != nil {
			return nil, e
		}
	}
	return vals, nil
}
func (a *StructuredArray) ToList() (any, error) {
	it, e := np.NDIndex(a.Data.Shape()...)
	if e != nil {
		return nil, e
	}
	v := make([]any, 0, a.Data.Size())
	for ix := range it {
		r, e := a.At(ix...)
		if e != nil {
			return nil, e
		}
		x, e := maskedRecordList(r.Data, r.Mask)
		if e != nil {
			return nil, e
		}
		v = append(v, x)
	}
	x, e := np.FromObjects(v, a.Data.Shape()...)
	if e != nil {
		return nil, e
	}
	return x.ToList()
}
func FromMaskedArrays(columns []*Array, opt rec.Options) (*StructuredArray, error) {
	data, mask := make([]*np.Array, len(columns)), make([]*np.Array, len(columns))
	for i, a := range columns {
		if !valid(a) {
			return nil, np.ErrDType
		}
		data[i], mask[i] = a.Data(), a.Mask()
	}
	d, e := rec.FromArrays(data, opt)
	if e != nil {
		return nil, e
	}
	s, e := d.Schema().MaskSchema()
	if e != nil {
		return nil, e
	}
	m, e := rec.FromArrays(mask, rec.Options{Schema: s, Shape: d.Shape()})
	if e != nil {
		return nil, e
	}
	return NewStructured(d, m)
}

// FromMaskedRecords accepts a declared schema and treats nil fields as masked.
func FromMaskedRecords(schema *records.Schema, rows [][]any, shape ...int) (*StructuredArray, error) {
	if shape == nil {
		shape = []int{len(rows)}
	}
	d, e := records.Zeros(schema, shape...)
	if e != nil {
		return nil, e
	}
	a, e := NewStructured(d, nil)
	if e != nil {
		return nil, e
	}
	if len(rows) != d.Size() {
		return nil, np.ErrShape
	}
	names := schema.Names()
	it, e := np.NDIndex(shape...)
	if e != nil {
		return nil, e
	}
	i := 0
	for ix := range it {
		if len(rows[i]) != len(names) {
			return nil, np.ErrShape
		}
		r, e := a.At(ix...)
		if e != nil {
			return nil, e
		}
		for j, v := range rows[i] {
			if v == nil {
				e = r.Mask.Set(names[j], true)
			} else {
				e = r.Set(names[j], v, false)
			}
			if e != nil {
				return nil, e
			}
		}
		i++
	}
	return a, nil
}
func FromMaskedText(r io.Reader, opt records.NamedTextOptions) (*StructuredArray, error) {
	b, e := io.ReadAll(r)
	if e != nil {
		return nil, e
	}
	d, e := records.GenFromText(bytes.NewReader(b), opt)
	if e != nil {
		return nil, e
	}
	a, e := NewStructured(d, nil)
	if e != nil {
		return nil, e
	}
	o := opt.Text
	if opt.Header && o.MaxRows > 0 {
		o.MaxRows++
	}
	rows, _, e := np.ReadTextRows(bytes.NewReader(b), o)
	if e != nil {
		return nil, e
	}
	if opt.Header {
		rows = rows[1:]
	}
	names := d.Schema().Names()
	it, e := np.NDIndex(d.Shape()...)
	if e != nil {
		return nil, e
	}
	i := 0
	for ix := range it {
		rec, e := a.At(ix...)
		if e != nil {
			return nil, e
		}
		for j, s := range rows[i] {
			missing := s == ""
			for _, m := range o.MissingValues {
				missing = missing || s == m
			}
			if missing {
				if e = rec.Mask.Set(names[j], true); e != nil {
					return nil, e
				}
			}
		}
		i++
	}
	return a, nil
}
func AddField(a *StructuredArray, name string, column *Array) (*StructuredArray, error) {
	if a == nil || !valid(column) {
		return nil, np.ErrDType
	}
	d, e := dtypes.FromArray(column.Data())
	if e != nil {
		return nil, e
	}
	t, e := d.RecordType()
	if e != nil {
		return nil, e
	}
	specs := make([]records.FieldSpec, 0, len(a.Data.Schema().Fields())+1)
	for _, f := range a.Data.Schema().Fields() {
		specs = append(specs, records.FieldSpec{Name: f.Name, Title: f.Title, Type: f.Type})
	}
	specs = append(specs, records.FieldSpec{Name: name, Type: t})
	s, e := records.NewSchema(specs, false, 0)
	if e != nil {
		return nil, e
	}
	dat, e := records.Zeros(s, a.Data.Shape()...)
	if e != nil {
		return nil, e
	}
	out, e := NewStructured(dat, nil)
	if e != nil {
		return nil, e
	}
	for _, f := range a.Data.Schema().Names() {
		for _, pair := range [][2]*records.Array{{out.Data, a.Data}, {out.Mask, a.Mask}} {
			x, e := pair[0].NDArray().Field(f)
			if e != nil {
				return nil, e
			}
			y, e := pair[1].NDArray().Field(f)
			if e != nil {
				return nil, e
			}
			if e = x.Assign(nil, y); e != nil {
				return nil, e
			}
		}
	}
	x, e := out.Data.NDArray().Field(name)
	if e != nil {
		return nil, e
	}
	y, e := out.Mask.NDArray().Field(name)
	if e != nil {
		return nil, e
	}
	if e = x.Assign(nil, column.Data()); e != nil {
		return nil, e
	}
	if e = y.Assign(nil, column.Mask()); e != nil {
		return nil, e
	}
	out.hard = a.hard
	return out, nil
}
