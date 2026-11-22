package ma

import "github.com/SamuelSupe/numpygo"

// MaskedIterator traverses logical C-order data and mask together, exposing
// both scalar payload and an explicit missing flag instead of a Python
// singleton. Its underlying iterators retain their independent cursors.
type MaskedIterator struct {
	a          *Array
	data, mask *np.FlatIterator
}

func (a *Array) FlatIterator() (*MaskedIterator, error) {
	if !valid(a) {
		return nil, np.ErrDType
	}
	d, e := a.data.FlatIterator()
	if e != nil {
		return nil, e
	}
	m, e := a.mask.FlatIterator()
	if e != nil {
		return nil, e
	}
	return &MaskedIterator{a: a, data: d, mask: m}, nil
}
func (f *MaskedIterator) Base() *Array                   { return f.a }
func (f *MaskedIterator) DataIterator() *np.FlatIterator { return f.data }
func (f *MaskedIterator) MaskIterator() *np.FlatIterator {
	if GetMask(f.a) == nil {
		return nil
	}
	return f.mask
}
func (f *MaskedIterator) Next() (value any, masked, ok bool, err error) {
	value, ok, err = f.data.Next()
	if err != nil || !ok {
		return
	}
	v, ok, err := f.mask.Next()
	if err != nil || !ok {
		return nil, false, ok, err
	}
	return value, v.(bool), true, nil
}
func (f *MaskedIterator) Get(index np.Index) (*Array, error) {
	d, e := f.data.Get(index)
	if e != nil {
		return nil, e
	}
	m, e := f.mask.Get(index)
	if e != nil {
		return nil, e
	}
	if GetMask(f.a) == nil {
		m = nil
	}
	out, e := New(d, m)
	if e != nil {
		return nil, e
	}
	out.shared = false
	return out, nil
}
func (f *MaskedIterator) Assign(index np.Index, values *Array) error {
	a := f.a
	if !valid(values) {
		return np.ErrDType
	}
	if !a.data.Writable() || !a.mask.Writable() {
		return np.ErrReadOnly
	}
	// Stage both writes, then commit together. This also preserves logical
	// selection and cyclic RHS behavior on non-contiguous views.
	d, e := a.data.Copy()
	if e != nil {
		return e
	}
	m, e := a.mask.Copy()
	if e != nil {
		return e
	}
	df, e := d.FlatIterator()
	if e != nil {
		return e
	}
	mf, e := m.FlatIterator()
	if e != nil {
		return e
	}
	if e = df.Assign(index, values.data); e != nil {
		return e
	}
	if e = mf.Assign(index, values.mask); e != nil {
		return e
	}
	v, e := New(d, m)
	if e != nil {
		return e
	}
	f.data.Reset()
	f.mask.Reset()
	return a.Assign(nil, v)
}
