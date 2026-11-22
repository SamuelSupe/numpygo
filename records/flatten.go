package records

import (
	"math"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/objects"
)

func flatten(v any, out *[]any) error {
	switch x := v.(type) {
	case *Record:
		vals, e := x.Values()
		if e != nil {
			return e
		}
		for _, v := range vals {
			if e = flatten(v, out); e != nil {
				return e
			}
		}
	case *np.Array:
		it, e := np.NDIter(x, "C")
		if e != nil {
			return e
		}
		for v := range it {
			*out = append(*out, v)
		}
	case []any:
		for _, v := range x {
			if e := flatten(v, out); e != nil {
				return e
			}
		}
	default:
		*out = append(*out, v)
	}
	return nil
}
func leafCount(schema *Schema) int {
	n := 0
	for _, f := range schema.fields {
		count, _ := np.ElementCount(np.Bool, f.Type.Shape...)
		if f.Type.Kind == RecordKind {
			count *= leafCount(f.Type.Schema)
		}
		n += count
	}
	return n
}
func (a *Array) flattened() ([]any, []int, error) {
	width := leafCount(a.schema)
	shape := append(a.Shape(), width)
	// NumPy flatten_structured_array always has a leading record dimension
	// for a single structured scalar.
	if a.NDim() == 0 {
		shape = []int{1, width}
	}
	v := make([]any, 0, a.Size()*width)
	it, e := np.NDIndex(a.Shape()...)
	if e != nil {
		return nil, nil, e
	}
	for idx := range it {
		r, e := a.At(idx...)
		if e != nil {
			return nil, nil, e
		}
		if e = flatten(r, &v); e != nil {
			return nil, nil, e
		}
	}
	return v, shape, nil
}
func (a *Array) FlattenObjects() (*objects.Array, error) {
	v, shape, e := a.flattened()
	if e != nil {
		return nil, e
	}
	return objects.FromSlice(v, shape...)
}
func numericValue(v any) (any, np.DType, error) {
	switch x := v.(type) {
	case bool:
		return x, np.Bool, nil
	case int8:
		return int64(x), np.Int64, nil
	case int16:
		return int64(x), np.Int64, nil
	case int32:
		return int64(x), np.Int64, nil
	case int64:
		return x, np.Int64, nil
	case uint8:
		return int64(x), np.Int64, nil
	case uint16:
		return int64(x), np.Int64, nil
	case uint32:
		return int64(x), np.Int64, nil
	case uint64:
		if x <= math.MaxInt64 {
			return int64(x), np.Int64, nil
		}
		return x, np.Uint64, nil
	case np.Half:
		return x.Float64(), np.Float64, nil
	case float32:
		return float64(x), np.Float64, nil
	case float64:
		return x, np.Float64, nil
	case complex64:
		return complex128(x), np.Complex128, nil
	case complex128:
		return x, np.Complex128, nil
	}
	return nil, 0, np.ErrDType
}
func (a *Array) FlattenNumeric() (*np.Array, error) {
	values, shape, e := a.flattened()
	if e != nil {
		return nil, e
	}
	d := np.DType(0)
	for i, v := range values {
		x, t, e := numericValue(v)
		if e != nil {
			return nil, e
		}
		values[i] = x
		if d == 0 {
			d = t
		} else {
			d, e = np.PromoteTypes(d, t)
			if e != nil {
				return nil, e
			}
		}
	}
	if d == 0 {
		d = np.Float64
	}
	out, e := np.Zeros(d, len(values))
	if e != nil {
		return nil, e
	}
	for i, v := range values {
		if e = out.SetValue(v, i); e != nil {
			return nil, e
		}
	}
	return out.Reshape(shape...)
}
