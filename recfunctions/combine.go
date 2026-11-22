package recfunctions

import (
	"fmt"
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/dtypes"
	"github.com/SamuelSupe/numpygo/ma"
	"github.com/SamuelSupe/numpygo/records"
	"slices"
)

// Array carries a data ndarray and an optional mask. Structured masks have
// matching field names and boolean leaves; plain arrays use a boolean mask.
// Unlike Python subclass dispatch, ownership and masking are explicit in Go.
type Array struct {
	Data, Mask *np.Array
}
type Options struct {
	UseMask   bool
	FillValue any // merge_arrays padding (default -1)
	Defaults  map[string]any
}

func ravel(a Array) (Array, error) {
	if a.Data.DType() == 0 {
		return Array{}, np.ErrDType
	}
	if a.Mask != nil {
		if !slices.Equal(a.Data.Shape(), a.Mask.Shape()) {
			return Array{}, np.ErrShape
		}
		if a.Data.DType() == np.RecordDType {
			d, e := records.FromNDArray(a.Data)
			if e != nil {
				return Array{}, e
			}
			m, e := records.FromNDArray(a.Mask)
			if e != nil {
				return Array{}, e
			}
			if _, e = ma.NewStructured(d, m); e != nil {
				return Array{}, e
			}
		} else if a.Mask.DType() != np.Bool {
			return Array{}, np.ErrDType
		}
	}
	var e error
	a.Data, e = np.Ravel(a.Data)
	if e != nil {
		return Array{}, e
	}
	if a.Mask != nil {
		a.Mask, e = np.Ravel(a.Mask)
	}
	return a, e
}
func selectField(a Array, name string) (Array, error) {
	d, e := a.Data.Field(name)
	if e != nil {
		return Array{}, e
	}
	var m *np.Array
	if a.Mask != nil {
		m, e = a.Mask.Field(name)
		if e != nil {
			return Array{}, e
		}
	}
	return Array{d, m}, nil
}
func fillTree(a *np.Array, val any) error {
	if a.DType() != np.RecordDType {
		return a.Fill(val)
	}
	s, e := schemaOf(a)
	if e != nil {
		return e
	}
	for _, f := range s.Fields() {
		v, e := a.Field(f.Name)
		if e != nil {
			return e
		}
		if e = fillTree(v, val); e != nil {
			return e
		}
	}
	return nil
}
func newMasked(s *records.Schema, n int) (Array, error) {
	d, e := records.Zeros(s, n)
	if e != nil {
		return Array{}, e
	}
	a, e := ma.NewStructured(d, nil)
	if e != nil {
		return Array{}, e
	}
	if e = fillTree(a.Mask.NDArray(), true); e != nil {
		return Array{}, e
	}
	return Array{a.Data.NDArray(), a.Mask.NDArray()}, nil
}
func defaultFill(a *np.Array) any {
	d := a.DType()
	switch {
	case d == np.Bool:
		return true
	case d.IsSigned() || d.IsUnsigned():
		return int64(999999)
	case d.IsFloat():
		return float64(1e20)
	case d.IsComplex():
		return complex(1e20, 0)
	case d.IsCharacter():
		return "N/A"
	case d.IsTemporal():
		return "NaT"
	}
	return 0
}

// fillMissing materializes deterministic values for the mask without
// altering the mask itself. Structured defaults can be scalar or a record.
func fillMissing(a Array, def any, defaults map[string]any) error {
	if a.Mask == nil {
		return nil
	}
	if a.Data.DType() == np.RecordDType {
		s, e := schemaOf(a.Data)
		if e != nil {
			return e
		}
		for _, f := range s.Fields() {
			v, e := selectField(a, f.Name)
			if e != nil {
				return e
			}
			x := def
			if z, ok := defaults[f.Name]; ok {
				x = z
			}
			if e = fillMissing(v, x, nil); e != nil {
				return e
			}
		}
		return nil
	}
	if def == nil {
		def = defaultFill(a.Data)
	}
	mask, e := np.Any(a.Mask)
	if e != nil {
		return e
	}
	any, e := mask.AtValue()
	if e != nil || !any.(bool) {
		return e
	}
	v, e := np.ZerosLike(a.Data)
	if e != nil {
		return e
	}
	if e = v.Fill(def); e != nil {
		return e
	}
	return np.CopyTo(a.Data, v, a.Mask)
}
func finish(a Array, opt Options, fill any) (Array, error) {
	if e := fillMissing(a, fill, opt.Defaults); e != nil {
		return Array{}, e
	}
	if !opt.UseMask {
		a.Mask = nil
	}
	return a, nil
}
func rangeArray(a Array, i, j int) (Array, error) {
	d, e := a.Data.Slice(np.Range(i, j))
	if e != nil {
		return Array{}, e
	}
	var m *np.Array
	if a.Mask != nil {
		m, e = a.Mask.Slice(np.Range(i, j))
	}
	return Array{d, m}, e
}
func copyField(dst, src Array, at int) error {
	n := src.Data.Shape()[0]
	v, e := rangeArray(dst, at, at+n)
	if e != nil {
		return e
	}
	if e = v.Data.Assign(nil, src.Data); e != nil {
		return e
	}
	if src.Mask == nil {
		return fillTree(v.Mask, false)
	}
	return v.Mask.Assign(nil, src.Mask)
}

type column struct {
	spec records.FieldSpec
	data Array
}

func columns(a Array, flatten bool) ([]column, error) {
	if a.Data.DType() != np.RecordDType {
		d, e := dtypes.FromArray(a.Data)
		if e != nil {
			return nil, e
		}
		t, e := d.RecordType()
		return []column{{records.FieldSpec{Type: t}, a}}, e
	}
	s, e := schemaOf(a.Data)
	if e != nil {
		return nil, e
	}
	out := []column{}
	for _, f := range s.Fields() {
		v, e := selectField(a, f.Name)
		if e != nil {
			return nil, e
		}
		if flatten && f.Type.Kind == records.RecordKind && len(f.Type.Shape) == 0 {
			c, e := columns(v, true)
			if e != nil {
				return nil, e
			}
			out = append(out, c...)
		} else {
			out = append(out, column{records.FieldSpec{Name: f.Name, Type: f.Type, Title: f.Title}, v})
		}
	}
	return out, nil
}
func mergeColumns(cols []column, opt Options, fill any) (Array, error) {
	spec := make([]records.FieldSpec, len(cols))
	n := 0
	for i, c := range cols {
		spec[i] = c.spec
		if spec[i].Name == "" {
			spec[i].Name = fmt.Sprintf("f%d", i)
		}
		n = max(n, c.data.Data.Shape()[0])
	}
	s, e := records.NewSchema(spec, false, 0)
	if e != nil {
		return Array{}, e
	}
	out, e := newMasked(s, n)
	if e != nil {
		return Array{}, e
	}
	if fill != nil {
		// merge_arrays supplies the padding value only for absent rows;
		// an explicitly masked source retains its original underlying value.
		if e = fillTree(out.Data, fill); e != nil {
			return Array{}, e
		}
	}
	for i, c := range cols {
		dst, e := selectField(out, spec[i].Name)
		if e != nil {
			return Array{}, e
		}
		if e = copyField(dst, c.data, 0); e != nil {
			return Array{}, e
		}
	}
	if fill != nil {
		if !opt.UseMask {
			out.Mask = nil
		}
		return out, nil
	}
	return finish(out, opt, nil)
}

func MergeArrays(arrays []Array, flatten bool, opt Options) (Array, error) {
	if len(arrays) == 0 {
		return Array{}, np.ErrValue
	}
	cols := []column{}
	for _, input := range arrays {
		a, e := ravel(input)
		if e != nil {
			return Array{}, e
		}
		// NumPy 2.5 rejects unequal-length multi-input flattened merges
		// when its padding iterator feeds the scalar field descriptors.
		if flatten && a.Data.Size() != arrays[0].Data.Size() {
			return Array{}, np.ErrValue
		}
		if !opt.UseMask {
			a.Mask = nil // NumPy's unmasked merge uses the raw input data.
		}
		c, e := columns(a, flatten)
		if e != nil {
			return Array{}, e
		}
		if !flatten && len(c) > 1 {
			s, e := schemaOf(a.Data)
			if e != nil {
				return Array{}, e
			}
			c = []column{{records.FieldSpec{Type: records.Struct(s)}, a}}
		}
		cols = append(cols, c...)
	}
	fill := opt.FillValue
	if fill == nil {
		fill = -1
	}
	return mergeColumns(cols, opt, fill)
}

func AppendFields(base Array, names []string, data []Array, dtype []dtypes.Descriptor, opt Options) (Array, error) {
	if len(names) != len(data) || len(data) == 0 || len(dtype) != 0 && len(dtype) != 1 && len(dtype) != len(data) {
		return Array{}, np.ErrShape
	}
	b, e := ravel(base)
	if e != nil {
		return Array{}, e
	}
	cols, e := columns(b, false)
	if e != nil {
		return Array{}, e
	}
	for i, input := range data {
		a, e := ravel(input)
		if e != nil {
			return Array{}, e
		}
		if len(dtype) != 0 {
			d := dtype[0]
			if len(dtype) != 1 {
				d = dtype[i]
			}
			a.Data, e = d.Cast(a.Data, "unsafe")
			if e != nil {
				return Array{}, e
			}
		}
		d, e := dtypes.FromArray(a.Data)
		if e != nil {
			return Array{}, e
		}
		t, e := d.RecordType()
		if e != nil {
			return Array{}, e
		}
		cols = append(cols, column{records.FieldSpec{Name: names[i], Type: t}, a})
	}
	// append_fields uses masked-array default fills for holes in the result;
	// the caller's fill_value only affects intermediate merged arrays.
	return mergeColumns(cols, opt, nil)
}
func RecAppendFields(base *np.Array, names []string, data []*np.Array, dtype []dtypes.Descriptor) (*records.Array, error) {
	a := make([]Array, len(data))
	for i, d := range data {
		a[i].Data = d
	}
	v, e := AppendFields(Array{Data: base}, names, a, dtype, Options{})
	if e != nil {
		return nil, e
	}
	return records.FromNDArray(v.Data)
}

func largestType(a, b records.Type, auto bool) (records.Type, error) {
	if !slices.Equal(a.Shape, b.Shape) {
		return records.Type{}, np.ErrDType
	}
	x, e := dtypes.FromRecordType(a)
	if e != nil {
		return records.Type{}, e
	}
	y, e := dtypes.FromRecordType(b)
	if e != nil {
		return records.Type{}, e
	}
	ok, e := dtypes.CanCast(x, y, "no")
	if e != nil {
		return records.Type{}, e
	}
	if ok {
		return a, nil
	}
	if !auto {
		return records.Type{}, np.ErrDType
	}
	// NumPy max(new_dtype, old_dtype) uses the dtype's safe-cast order,
	// not result_type, including incomparable integer families.
	xy, e := dtypes.CanCast(x, y, "safe")
	if e != nil {
		return records.Type{}, e
	}
	yx, e := dtypes.CanCast(y, x, "safe")
	if e != nil {
		return records.Type{}, e
	}
	if yx && !xy {
		return a, nil
	}
	return b, nil
}

func StackArrays(arrays []Array, autoconvert bool, opt Options) (Array, error) {
	if len(arrays) == 0 {
		return Array{}, np.ErrValue
	}
	if len(arrays) == 1 {
		return arrays[0], nil
	}
	spec := []records.FieldSpec{}
	byName := map[string]int{}
	input := make([][]column, len(arrays))
	total := 0
	for k, raw := range arrays {
		a, e := ravel(raw)
		if e != nil {
			return Array{}, e
		}
		c, e := columns(a, false)
		if e != nil {
			return Array{}, e
		}
		for i := range c {
			f := c[i].spec
			if f.Name == "" {
				f.Name = "f0"
				c[i].spec = f
			}
			j, ok := byName[f.Name]
			if !ok {
				byName[f.Name] = len(spec)
				spec = append(spec, f)
			} else {
				spec[j].Type, e = largestType(spec[j].Type, f.Type, autoconvert)
				if e != nil {
					return Array{}, e
				}
			}
		}
		input[k] = c
		if a.Data.Size() > int(^uint(0)>>1)-total {
			return Array{}, np.ErrShape
		}
		total += a.Data.Size()
	}
	s, e := records.NewSchema(spec, false, 0)
	if e != nil {
		return Array{}, e
	}
	out, e := newMasked(s, total)
	if e != nil {
		return Array{}, e
	}
	at := 0
	for k, cols := range input {
		for _, c := range cols {
			dst, e := selectField(out, c.spec.Name)
			if e != nil {
				return Array{}, e
			}
			if e = copyField(dst, c.data, at); e != nil {
				return Array{}, e
			}
		}
		at += arrays[k].Data.Size()
	}
	return finish(out, opt, nil)
}
