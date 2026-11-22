package ma

import (
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/dtypes"
	"github.com/SamuelSupe/numpygo/records"
	"io"
	"iter"
	"slices"
)

func (a *Array) ItemSize() int      { return a.data.ItemSize() }
func (a *Array) NBytes() int        { return a.data.NBytes() }
func (a *Array) Strides() []int     { return a.data.Strides() }
func (a *Array) IsContiguous() bool { return a.data.IsCContiguous() }
func (a *Array) SharedMask() bool   { return a.shared && GetMask(a) != nil }
func (a *Array) BaseClass() string  { return "ndarray" }
func (a *Array) RecordMask() *np.Array {
	if m := GetMask(a); m != nil {
		return m
	}
	return np.ScalarOf(false)
}
func (a *Array) IDs() (uintptr, uintptr) { return Ids(a) }

// SetMask updates the underlying mask, preserving masked positions for
// hard masks. Nil clears the soft mask but retains an already explicit mask.
func (a *Array) SetMask(mask *np.Array) error {
	if !valid(a) {
		return np.ErrDType
	}
	if mask == nil {
		mask = np.ScalarOf(false)
	}
	if mask.DType() != np.Bool {
		if a.hard {
			return np.ErrDType
		}
		var e error
		mask, e = mask.Astype(np.Bool)
		if e != nil {
			return e
		}
	}
	if mask.Size() == 1 {
		var e error
		mask, e = mask.Reshape()
		if e != nil {
			return e
		}
	} else if !slices.Equal(mask.Shape(), a.Shape()) {
		var e error
		mask, e = np.Resize(mask, a.Shape()...)
		if e != nil {
			return e
		}
	}
	mask, e := mask.BroadcastTo(a.Shape()...)
	if e != nil {
		return e
	}
	if a.hard {
		mask, e = np.LogicalOr(a.mask, mask)
		if e != nil {
			return e
		}
	}
	if e = a.mask.Assign(nil, mask); e != nil {
		return e
	}
	a.explicitMask = true
	return nil
}
func (a *Array) UnshareMask() error {
	if !valid(a) {
		return np.ErrDType
	}
	if !a.shared {
		return nil
	}
	m, e := a.mask.Copy()
	if e != nil {
		return e
	}
	a.mask, a.shared = m, false
	return nil
}
func (a *Array) ShrinkMask() error {
	if !valid(a) {
		return np.ErrDType
	}
	if IsMasked(a) {
		return nil
	}
	m, e := np.Zeros(np.Bool, a.Shape()...)
	if e != nil {
		return e
	}
	a.mask, a.explicitMask, a.shared = m, false, false
	return nil
}

func (a *Array) Astype(dtype np.DType, opt np.CastOptions) (*Array, error) {
	if !valid(a) {
		return nil, np.ErrDType
	}
	d, e := a.data.AstypeWithOptions(dtype, opt)
	if e != nil {
		return nil, e
	}
	if d == a.data && opt.NoCopy {
		return a, nil
	}
	m, e := a.mask.AstypeWithOptions(np.Bool, np.CastOptions{Order: opt.Order})
	if e != nil {
		return nil, e
	}
	out := a.with(d, m)
	if dtype != a.DType() && a.customFill {
		f, e := np.Full(dtype, a.FillValue())
		if e != nil {
			return nil, e
		}
		v, e := f.AtValue()
		if e != nil {
			return nil, e
		}
		out.fill, out.customFill = &v, true
	}
	return out, nil
}
func (a *Array) ByteSwap(inplace bool) (*Array, error) {
	if !valid(a) {
		return nil, np.ErrDType
	}
	d, e := a.data.ByteSwap(inplace)
	if e != nil {
		return nil, e
	}
	if inplace {
		return a, nil
	}
	m, e := a.mask.Copy()
	if e != nil {
		return nil, e
	}
	return a.with(d, m), nil
}
func (a *Array) View() (*Array, error) {
	return transform(a, func(x *np.Array) (*np.Array, error) { return x.Slice() })
}
func (a *Array) Flatten() (*Array, error)              { return transform(a, np.Flatten) }
func (a *Array) Ravel() (*Array, error)                { return Ravel(a) }
func (a *Array) Reshape(shape ...int) (*Array, error)  { return Reshape(a, shape...) }
func (a *Array) Transpose(axes ...int) (*Array, error) { return Transpose(a, axes...) }
func (a *Array) MatrixTranspose() (*Array, error) {
	if a.NDim() < 2 {
		return nil, np.ErrShape
	}
	return SwapAxes(a, -1, -2)
}
func (a *Array) Real() (*Array, error) { return a.component(np.Real) }
func (a *Array) Imag() (*Array, error) { return a.component(np.Imag) }
func (a *Array) component(fn func(*np.Array) (*np.Array, error)) (*Array, error) {
	if !valid(a) {
		return nil, np.ErrDType
	}
	d, e := fn(a.data)
	if e != nil {
		return nil, e
	}
	// NumPy component extraction creates fresh mask metadata, with default
	// fill and soft mask even when the source is hard-masked.
	out, e := New(d, GetMask(a), Options{Copy: false})
	if e != nil {
		return nil, e
	}
	out.shared = false
	return out, nil
}
func (a *Array) Fill(value any) error { return a.data.Fill(value) }
func (a *Array) Item(flat ...int) (any, error) {
	if len(flat) > 1 {
		return a.data.AtValue(flat...)
	}
	if len(flat) == 0 {
		if a.Size() != 1 {
			return nil, np.ErrValue
		}
		flat = []int{0}
	}
	f, e := a.data.FlatIterator()
	if e != nil {
		return nil, e
	}
	v, e := f.Get(np.AtIndex(flat[0]))
	if e != nil {
		return nil, e
	}
	return v.AtValue()
}
func (a *Array) ToList(fill ...any) (any, error) {
	if len(fill) > 1 {
		return nil, np.ErrValue
	}
	if len(fill) == 1 && fill[0] != nil {
		d, e := a.Filled(fill[0])
		if e != nil {
			return nil, e
		}
		return d.ToList()
	}
	it, e := np.NDIndex(a.Shape()...)
	if e != nil {
		return nil, e
	}
	values := make([]any, 0, a.Size())
	for ix := range it {
		m, e := a.mask.AtValue(ix...)
		if e != nil {
			return nil, e
		}
		var v any
		if !m.(bool) {
			v, e = a.data.AtValue(ix...)
			if e != nil {
				return nil, e
			}
		}
		values = append(values, v)
	}
	d, e := np.FromObjects(values, a.Shape()...)
	if e != nil {
		return nil, e
	}
	return d.ToList()
}
func (a *Array) ToBytes(order string, fill ...any) ([]byte, error) {
	d, e := a.Filled(fill...)
	if e != nil {
		return nil, e
	}
	return d.ToBytes(order)
}

// ToFlex copies numerical data and mask into two structured fields. Unlike
// Filled or ToBytes, masked payloads are retained without replacing them.
func (a *Array) ToFlex() (*records.Array, error) {
	desc, e := dtypes.FromArray(a.data)
	if e != nil {
		return nil, e
	}
	t, e := desc.RecordType()
	if e != nil {
		return nil, e
	}
	schema, e := records.NewSchema([]records.FieldSpec{
		{Name: "_data", Type: t},
		{Name: "_mask", Type: records.Numeric(np.Bool, nil)},
	}, false, 0)
	if e != nil {
		return nil, e
	}
	out, e := records.Zeros(schema, a.Shape()...)
	if e != nil {
		return nil, e
	}
	for name, v := range map[string]*np.Array{"_data": a.data, "_mask": a.mask} {
		f, e := out.NDArray().Field(name)
		if e != nil {
			return nil, e
		}
		if e = f.Assign(nil, v); e != nil {
			return nil, e
		}
	}
	return out, nil
}

func (a *Array) SortInPlace(axis *int, endWith bool) error {
	if axis == nil && a.NDim() > 1 {
		return np.ErrAxis
	}
	v, e := Sort(a, axis, endWith)
	if e != nil {
		return e
	}
	if axis == nil && a.NDim() == 0 {
		v, e = Reshape(v)
		if e != nil {
			return e
		}
	}
	// MaskedArray.sort writes through the mask-aware assignment path,
	// including the special behavior of a hard mask.
	return a.Assign(nil, v)
}
func (a *Array) ResizeInPlace(_ ...int) error { return np.ErrValue }

// Partition and ArgPartition deliberately ignore the mask, corresponding
// to NumPy's warned ndarray partition path. The unspecified partition
// ordering is not reproduced; np.Partition returns a valid full ordering.
func (a *Array) PartitionInPlace(kth []int, axis *int) error {
	if !valid(a) {
		return np.ErrDType
	}
	v, e := np.Partition(a.data, kth, axis)
	if e != nil {
		return e
	}
	if !slices.Equal(v.Shape(), a.Shape()) {
		return np.ErrAxis
	}
	return a.data.Assign(nil, v)
}
func (a *Array) ArgPartition(kth []int, axis *int) (*np.Array, error) {
	if !valid(a) {
		return nil, np.ErrDType
	}
	return np.ArgPartition(a.data, kth, axis)
}
func (a *Array) SearchSorted(values *np.Array, right bool) (*np.Array, error) {
	return np.SearchSorted(a.data, values, right)
}

// NumPy refuses MaskedArray.tofile because raw file output loses the mask.
// Call ToFlex or Filled explicitly before writing.
func (a *Array) ToFile(_ io.Writer) error { return ErrMA }
func (a *Array) Len() (int, error) {
	if !valid(a) {
		return 0, np.ErrDType
	}
	if a.NDim() == 0 {
		return 0, np.ErrShape
	}
	return a.Shape()[0], nil
}
func (a *Array) Iter() (iter.Seq2[*Array, error], error) {
	n, e := a.Len()
	if e != nil {
		return nil, e
	}
	return func(yield func(*Array, error) bool) {
		for i := 0; i < n; i++ {
			v, e := a.Index(np.AtIndex(i))
			if !yield(v, e) || e != nil {
				return
			}
		}
	}, nil
}
