package scalar

import (
	"github.com/SamuelSupe/numpygo"
	"io"
)

// Flags describes the immutable scalar's zero-dimensional value storage.
type Flags struct {
	CContiguous, FContiguous, OwnData, Writable, Aligned, Writeback bool
}

func (v Value) Flags() Flags          { return Flags{true, true, true, false, true, false} }
func (v Value) Base() *np.Array       { return nil }
func (v Value) Data() ([]byte, error) { return v.ToBytes() }
func (v Value) Flat() (*np.FlatIterator, error) {
	a, e := v.Array()
	if e != nil {
		return nil, e
	}
	return a.FlatIterator()
}
func (v Value) Dump(w io.Writer) error   { return v.a.Dump(w) }
func (v Value) Dumps() ([]byte, error)   { return v.a.Dumps() }
func (v Value) ToFile(w io.Writer) error { return v.a.ToFile(w) }
func (v Value) ToList() (any, error)     { return v.Item() }
func (v Value) ToDevice(device string) (Value, error) {
	if _, e := v.a.ToDevice(device); e != nil {
		return Value{}, e
	}
	return v.Copy()
}
func (v Value) All(o ...np.SumOptions) (*np.Array, error)    { return np.All(v.a, o...) }
func (v Value) Any(o ...np.SumOptions) (*np.Array, error)    { return np.Any(v.a, o...) }
func (v Value) ArgMax(o ...np.ArgOptions) (*np.Array, error) { return np.ArgMax(v.a, o...) }
func (v Value) ArgMin(o ...np.ArgOptions) (*np.Array, error) { return np.ArgMin(v.a, o...) }
func (v Value) Argsort() (*np.Array, error)                  { return np.Argsort(v.a, nil) }
func (v Value) Choose(choices []*np.Array, opt np.TakeOptions) (*np.Array, error) {
	if !v.DType().IsInteger() {
		return nil, np.ErrDType
	}
	return np.ChooseWithOptions(v.a, choices, opt)
}
func (v Value) Clip(lo, hi *np.Array) (*np.Array, error) { return np.Clip(v.a, lo, hi) }
func (v Value) Compress(condition *np.Array, axis *int) (*np.Array, error) {
	return np.Compress(condition, v.a, axis)
}
func (v Value) Cumprod(axis *int, d np.DType) (*np.Array, error) { return np.Cumprod(v.a, axis, d) }
func (v Value) Cumsum(axis *int, d np.DType) (*np.Array, error)  { return np.Cumsum(v.a, axis, d) }
func (v Value) Diagonal(offset, axis1, axis2 int) (*np.Array, error) {
	return np.Diagonal(v.a, offset, axis1, axis2)
}
func (v Value) Max(o ...np.SumOptions) (*np.Array, error)  { return np.Max(v.a, o...) }
func (v Value) Mean(o ...np.SumOptions) (*np.Array, error) { return np.Mean(v.a, o...) }
func (v Value) Min(o ...np.SumOptions) (*np.Array, error)  { return np.Min(v.a, o...) }
func (v Value) Nonzero() ([]*np.Array, error)              { return np.Nonzero(v.a) }
func (v Value) Prod(o ...np.SumOptions) (*np.Array, error) { return np.Prod(v.a, o...) }
func (v Value) Repeat(n int, axis *int) (*np.Array, error) { return np.Repeat(v.a, n, axis) }
func (v Value) Round(decimals int) (*np.Array, error)      { return np.Round(v.a, decimals) }
func (v Value) SearchSorted(values *np.Array, right bool) (*np.Array, error) {
	return np.SearchSorted(v.a, values, right)
}
func (v Value) Std(o ...np.VarOptions) (*np.Array, error) { return np.Std(v.a, o...) }
func (v Value) Sum(o ...np.SumOptions) (*np.Array, error) { return np.Sum(v.a, o...) }
func (v Value) SwapAxes(x, y int) (*np.Array, error)      { return np.SwapAxes(v.a, x, y) }
func (v Value) Take(indices *np.Array, axis *int, opt np.TakeOptions) (*np.Array, error) {
	return np.TakeWithOptions(v.a, indices, axis, opt)
}
func (v Value) Trace(offset, x, y int, d np.DType) (*np.Array, error) {
	return np.Trace(v.a, offset, x, y, d)
}
func (v Value) Var(o ...np.VarOptions) (*np.Array, error) { return np.Var(v.a, o...) }

// NumPy array scalar mutators validate a temporary ndarray and discard it.
// They cannot mutate the original immutable scalar.
func (v Value) Fill(value any) error {
	a, e := v.Array()
	if e != nil {
		return e
	}
	return a.Fill(value)
}
func (v Value) Put(indices, values *np.Array, mode string) error {
	a, e := v.Array()
	if e != nil {
		return e
	}
	return np.Put(a, indices, values, mode)
}
func (v Value) Resize(shape ...int) error {
	_, e := np.ElementCount(v.DType(), shape...)
	return e
}
func (v Value) SetFlags(write, align, writeback *bool) error           { return nil }
func (v Value) SetField(value *np.Array, d np.DType, offset int) error { return np.ErrDType }
func (v Value) Sort(axis int) error {
	_, e := np.Sort(v.a, np.Int(axis))
	return e
}
