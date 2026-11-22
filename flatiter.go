package np

import (
	"iter"
	"slices"
)

// FlatIterator is a stateful C-order iterator over a shared ndarray.
// Get and Assign reset its cursor, as NumPy's flatiter does. Copy and
// Array do not change the cursor.
type FlatIterator struct {
	base, view *Array
	index      int
}

func NewFlatIterator(a *Array) (*FlatIterator, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	a.viewBase()
	return &FlatIterator{base: a, view: a}, nil
}
func (a *Array) FlatIterator() (*FlatIterator, error) { return NewFlatIterator(a) }
func (f *FlatIterator) Base() *Array                  { return f.base }
func (f *FlatIterator) Index() int                    { return f.index }
func (f *FlatIterator) Len() int                      { return f.view.Size() }
func (f *FlatIterator) Reset()                        { f.index = 0 }

func coordinates(flat int, shape []int) []int {
	out := make([]int, len(shape))
	for i := len(shape) - 1; i > 0; i-- {
		if shape[i] == 0 {
			return out
		}
		out[i] = flat % shape[i]
		flat /= shape[i]
	}
	if len(shape) > 0 {
		out[0] = flat
	}
	return out
}
func (f *FlatIterator) Coords() []int { return coordinates(f.index, f.view.shape) }
func (f *FlatIterator) Next() (any, bool, error) {
	if f.view.DType() == 0 {
		return nil, false, ErrDType
	}
	if f.index >= f.Len() {
		return nil, false, nil
	}
	v := f.view.rawAt(f.view.offset + flatOffset(f.index, f.view.shape, f.view.strides))
	f.index++
	return v, true, nil
}
func (f *FlatIterator) Iter() iter.Seq2[any, error] {
	return func(yield func(any, error) bool) {
		for {
			v, ok, e := f.Next()
			if e != nil {
				yield(nil, e)
				return
			}
			if !ok || !yield(v, nil) {
				return
			}
		}
	}
}
func (f *FlatIterator) Copy() (*Array, error) {
	if f.view.DType() == 0 {
		return nil, ErrDType
	}
	v, e := f.view.Copy()
	if e != nil {
		return nil, e
	}
	return v.Reshape(f.Len())
}

// Array shares C-contiguous storage and otherwise returns an independent
// C-order copy (including uniformly stepped or reversed arrays).
func (f *FlatIterator) Array() (*Array, error) {
	if f.view.IsCContiguous() {
		return f.view.Reshape(f.Len())
	}
	return f.Copy()
}

func flatSelection(a *Array, index Index, writing bool) (shape, positions []int, err error) {
	if a.DType() == 0 {
		return nil, nil, ErrDType
	}
	switch x := index.(type) {
	case AtIndex:
		i, e := normalizeIndex(int(x), a.Size())
		return nil, []int{i}, e
	case Slice:
		start, n, step, e := x.normalize(a.Size())
		if e != nil {
			return nil, nil, e
		}
		p := make([]int, n)
		for i := range p {
			p[i] = start + i*step
		}
		return []int{n}, p, nil
	case *Array:
		if x.DType() == Bool {
			if x.NDim() == 0 {
				v, e := x.AtValue()
				if e != nil {
					return nil, nil, e
				}
				if !v.(bool) {
					return []int{0}, nil, nil
				}
				if a.Size() == 0 {
					return nil, nil, ErrIndex
				}
				return nil, []int{0}, nil
			}
			if !writing || x.NDim() != 1 || x.Size() != a.Size() {
				return nil, nil, ErrIndex
			}
			p := []int{}
			x.walk(func(i, off int) {
				if x.elementAt(off).i != 0 {
					p = append(p, i)
				}
			})
			return []int{len(p)}, p, nil
		}
		if !x.DType().IsInteger() {
			return nil, nil, ErrIndex
		}
		p := make([]int, x.Size())
		var failure error
		x.walk(func(i, off int) {
			if failure == nil {
				p[i], failure = indexInteger(x, off, a.Size())
			}
		})
		return x.Shape(), p, failure
	default:
		if index == Ellipsis {
			return flatSelection(a, Slice{}, writing)
		}
		return nil, nil, ErrIndex
	}
}

// Get returns a scalar array or a detached gather, even for basic slices.
func (f *FlatIterator) Get(index Index) (*Array, error) {
	f.Reset()
	shape, p, e := flatSelection(f.view, index, false)
	if e != nil {
		return nil, e
	}
	out, e := f.view.zerosLike(shape)
	if e != nil {
		return nil, e
	}
	for i, j := range p {
		out.setElement(i, f.view.elementAt(f.view.offset+flatOffset(j, f.view.shape, f.view.strides)))
	}
	return out, nil
}

// Assign writes flattened RHS values cyclically in index order, with
// overlap-safe snapshot semantics. Empty RHS leaves the array unchanged.
func (f *FlatIterator) Assign(index Index, values *Array) error {
	f.Reset()
	if !f.view.Writable() {
		return ErrReadOnly
	}
	shape, p, e := flatSelection(f.view, index, true)
	if e != nil {
		return e
	}
	if values.DType() == 0 {
		return ErrDType
	}
	if len(shape) == 0 && values.Size() != 1 {
		return ErrValue
	}
	if values.Size() == 0 || len(p) == 0 {
		return nil
	}
	v, e := f.view.castLike(values)
	if e != nil {
		return e
	}
	for i, j := range p {
		f.view.setElement(f.view.offset+flatOffset(j, f.view.shape, f.view.strides), v.elementAt(i%v.Size()))
	}
	return nil
}

// IndexIterator and EnumerateIterator expose restart-free counterparts of
// the convenience NDIndex and NDEnumerate range functions.
type IndexIterator struct {
	shape       []int
	size, index int
}

func NewNDIndex(shape ...int) (*IndexIterator, error) {
	n, e := shapeSize(Bool, shape)
	if e != nil {
		return nil, e
	}
	return &IndexIterator{shape: slices.Clone(shape), size: n}, nil
}
func (i *IndexIterator) Next() ([]int, bool) {
	if i.index >= i.size {
		return nil, false
	}
	v := coordinates(i.index, i.shape)
	i.index++
	return v, true
}

type EnumerateIterator struct{ flat *FlatIterator }

func NewNDEnumerate(a *Array) (*EnumerateIterator, error) {
	f, e := NewFlatIterator(a)
	if e != nil {
		return nil, e
	}
	return &EnumerateIterator{f}, nil
}
func (it *EnumerateIterator) Iterator() *FlatIterator { return it.flat }
func (it *EnumerateIterator) Next() ([]int, any, bool, error) {
	ix := it.flat.Coords()
	v, ok, e := it.flat.Next()
	return ix, v, ok, e
}
