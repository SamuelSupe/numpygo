package recfunctions

import (
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/dtypes"
	"github.com/SamuelSupe/numpygo/records"
	"strconv"
)

type leaf struct {
	typ    records.Type
	offset int
}

func leaves(s *records.Schema, start int) ([]leaf, error) {
	out := []leaf{}
	for _, f := range s.Fields() {
		t := f.Type
		n, e := np.ElementCount(np.Bool, t.Shape...)
		if e != nil {
			return nil, e
		}
		t.Shape = nil
		w, e := t.Size()
		if e != nil {
			return nil, e
		}
		for i := 0; i < n; i++ {
			off := start + f.Offset + i*w
			if t.Kind == records.RecordKind {
				ls, e := leaves(t.Schema, off)
				if e != nil {
					return nil, e
				}
				out = append(out, ls...)
			} else {
				out = append(out, leaf{t, off})
			}
		}
	}
	return out, nil
}

func layout(t records.Type, offset int) (np.StructuredField, error) {
	s, e := records.NewSchema([]records.FieldSpec{{Name: "f", Type: t}}, false, 0)
	if e != nil {
		return np.StructuredField{}, e
	}
	f, e := s.FieldLayout("f")
	f.Offset = offset
	return f, e
}
func lastColumn(a *np.Array, i int) (*np.Array, error) {
	return a.Slice(np.Ellipsis, np.AtIndex(i))
}
func scalarDescriptorEqual(a, b dtypes.Descriptor) bool {
	return a.String() == b.String()
}

// StructuredToUnstructured appends a flattened field axis. A homogeneous
// constant-stride layout can be shared even with padding or reversed field
// offsets. Other layouts are converted using the requested casting rule.
func StructuredToUnstructured(a *np.Array, dtype *dtypes.Descriptor, copyData bool, casting string) (*np.Array, error) {
	s, e := schemaOf(a)
	if e != nil {
		return nil, e
	}
	ls, e := leaves(s, 0)
	if e != nil {
		return nil, e
	}
	if len(ls) == 0 {
		return nil, np.ErrValue
	}
	outType, e := dtypes.FromRecordType(ls[0].typ)
	if e != nil {
		return nil, e
	}
	if dtype == nil {
		for _, l := range ls[1:] {
			d, e := dtypes.FromRecordType(l.typ)
			if e != nil {
				return nil, e
			}
			outType, e = dtypes.Promote(outType, d)
			if e != nil {
				return nil, e
			}
		}
		// result_type always returns a native scalar descriptor.
		outType = outType.WithByteOrder(nil)
	} else {
		outType = *dtype
	}
	shape := append(a.Shape(), len(ls))
	step := outType.ItemSize()
	if len(ls) > 1 {
		step = ls[1].offset - ls[0].offset
	}
	canView := !copyData
	if casting == "" {
		casting = "unsafe"
	}
	for i, l := range ls {
		d, e := dtypes.FromRecordType(l.typ)
		if e != nil {
			return nil, e
		}
		ok, e := dtypes.CanCast(d, outType, casting)
		if e != nil {
			return nil, e
		}
		if !ok {
			return nil, np.ErrDType
		}
		canView = canView && scalarDescriptorEqual(d, outType) && (i == 0 || l.offset-ls[i-1].offset == step)
	}
	if canView {
		t, e := outType.RecordType()
		if e != nil {
			return nil, e
		}
		f, e := layout(t, ls[0].offset)
		if e != nil {
			return nil, e
		}
		return a.ViewLayout(f, shape, append(a.Strides(), step))
	}
	out, e := outType.Zeros(shape...)
	if e != nil {
		return nil, e
	}
	for i, l := range ls {
		f, e := layout(l.typ, l.offset)
		if e != nil {
			return nil, e
		}
		src, e := a.ViewLayout(f, a.Shape(), a.Strides())
		if e != nil {
			return nil, e
		}
		v, e := outType.Cast(src, casting)
		if e != nil {
			return nil, e
		}
		dst, e := lastColumn(out, i)
		if e != nil {
			return nil, e
		}
		if e = dst.Assign(nil, v); e != nil {
			return nil, e
		}
	}
	return out, nil
}

// UnstructuredToStructured consumes the last dimension in field order.
// As in NumPy, contiguous homogeneous packed inputs can share storage;
// strided inputs are first materialized in C order.
func UnstructuredToStructured(a *np.Array, dtype *records.Schema, names []string, align, copyData bool, casting string) (*np.Array, error) {
	shape := a.Shape()
	if len(shape) == 0 || shape[len(shape)-1] == 0 {
		return nil, np.ErrShape
	}
	n := shape[len(shape)-1]
	shape = shape[:len(shape)-1]
	srcType, e := dtypes.FromArray(a)
	if e != nil {
		return nil, e
	}
	if dtype == nil {
		t, e := srcType.RecordType()
		if e != nil {
			return nil, e
		}
		spec := make([]records.FieldSpec, n)
		if names != nil && len(names) != n {
			return nil, np.ErrShape
		}
		for i := range spec {
			name := "f" + strconv.Itoa(i)
			if names != nil {
				name = names[i]
			}
			spec[i] = records.FieldSpec{Name: name, Type: t}
		}
		dtype, e = records.NewSchema(spec, align, 0)
		if e != nil {
			return nil, e
		}
	} else if names != nil || align && !dtype.Aligned() {
		return nil, np.ErrValue
	}
	ls, e := leaves(dtype, 0)
	if e != nil {
		return nil, e
	}
	if len(ls) != n {
		return nil, np.ErrShape
	}
	if casting == "" {
		casting = "unsafe"
	}
	canView := !copyData && dtype.ItemSize() == n*a.ItemSize()
	for i, l := range ls {
		d, e := dtypes.FromRecordType(l.typ)
		if e != nil {
			return nil, e
		}
		ok, e := dtypes.CanCast(srcType, d, casting)
		if e != nil {
			return nil, e
		}
		if !ok {
			return nil, np.ErrDType
		}
		canView = canView && scalarDescriptorEqual(srcType, d) && l.offset == i*a.ItemSize()
	}
	if canView {
		a, e = np.AsContiguousArray(a)
		if e != nil {
			return nil, e
		}
		return a.ViewLayout(np.StructuredField{DType: np.RecordDType, Schema: dtype}, shape, a.Strides()[:len(shape)])
	}
	out, e := np.ZerosRecords(dtype, shape...)
	if e != nil {
		return nil, e
	}
	for i, l := range ls {
		f, e := layout(l.typ, l.offset)
		if e != nil {
			return nil, e
		}
		d, e := dtypes.FromRecordType(l.typ)
		if e != nil {
			return nil, e
		}
		src, e := lastColumn(a, i)
		if e != nil {
			return nil, e
		}
		v, e := d.Cast(src, casting)
		if e != nil {
			return nil, e
		}
		dst, e := out.ViewLayout(f, shape, out.Strides())
		if e != nil {
			return nil, e
		}
		if e = dst.Assign(nil, v); e != nil {
			return nil, e
		}
	}
	return out, nil
}

// ApplyAlongFields supplies the unstructured array and last axis to a Go
// reduction callback instead of relying on Python keyword dispatch.
func ApplyAlongFields(a *np.Array, f func(*np.Array, int) (*np.Array, error)) (*np.Array, error) {
	if f == nil {
		return nil, np.ErrValue
	}
	v, e := StructuredToUnstructured(a, nil, false, "")
	if e != nil {
		return nil, e
	}
	return f(v, -1)
}
