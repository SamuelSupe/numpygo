package extended

import "github.com/SamuelSupe/numpygo"

type LongDouble = Float
type Float128 = Float
type CLongDouble = Complex
type Complex256 = Complex

// Array keeps the legacy extended package API on the same ndarray storage
// as the root extended dtypes. NDArray returns the original array header.
type Array struct{ nd *np.Array }

func FromNDArray(a *np.Array) (*Array, error) {
	if !a.DType().IsExtended() {
		return nil, np.ErrDType
	}
	return &Array{a}, nil
}
func (a *Array) NDArray() *np.Array { return a.nd }
func wrap(a *np.Array, e error) (*Array, error) {
	if e != nil {
		return nil, e
	}
	return FromNDArray(a)
}
func FromFloats(v []Float, shape ...int) (*Array, error)      { return wrap(np.FromSlice(v, shape...)) }
func FromComplexes(v []Complex, shape ...int) (*Array, error) { return wrap(np.FromSlice(v, shape...)) }
func FromStrings(v []string, shape ...int) (*Array, error) {
	f := make([]Float, len(v))
	for i, s := range v {
		x, e := Parse(s)
		if e != nil {
			return nil, e
		}
		f[i] = x
	}
	return FromFloats(f, shape...)
}
func Scalar(v Float) *Array          { a, _ := FromFloats([]Float{v}); return a }
func ComplexScalar(v Complex) *Array { a, _ := FromComplexes([]Complex{v}); return a }
func FromNumeric(a *np.Array) (*Array, error) {
	if !a.DType().IsNativeNumeric() && !a.DType().IsExtended() {
		return nil, np.ErrDType
	}
	d := np.Float128
	if a.DType().IsComplex() {
		d = np.Complex256
	}
	return wrap(a.Astype(d))
}
func (a *Array) Shape() []int    { return a.nd.Shape() }
func (a *Array) NDim() int       { return a.nd.NDim() }
func (a *Array) Size() int       { return a.nd.Size() }
func (a *Array) IsComplex() bool { return a.nd.DType() == np.Complex256 }
func (a *Array) DType() string   { return a.nd.DType().String() }
func (a *Array) Strides() []int  { return a.nd.Strides() }
func (a *Array) At(indices ...int) (Complex, error) {
	v, e := a.nd.AtValue(indices...)
	if e != nil {
		return Complex{}, e
	}
	if a.IsComplex() {
		return v.(Complex), nil
	}
	return Complex{Real: v.(Float)}, nil
}
func (a *Array) Set(value Complex, indices ...int) error {
	if !a.IsComplex() {
		if !value.Imag.IsZero() {
			return np.ErrDType
		}
		return a.nd.SetValue(value.Real, indices...)
	}
	return a.nd.SetValue(value, indices...)
}
func (a *Array) Values() []Complex {
	if a.IsComplex() {
		v, _ := np.Values[Complex](a.nd)
		return v
	}
	v, _ := np.Values[Float](a.nd)
	out := make([]Complex, len(v))
	for i, x := range v {
		out[i].Real = x
	}
	return out
}
func (a *Array) Bytes() []byte { b, _ := a.nd.ToBytes("C"); return b }
func (a *Array) ToNumeric() (*np.Array, error) {
	d := np.Float64
	if a.IsComplex() {
		d = np.Complex128
	}
	return a.nd.Astype(d)
}
func (a *Array) Slice(indices ...np.Index) (*Array, error) { return wrap(a.nd.Slice(indices...)) }
func (a *Array) Transpose(axes ...int) (*Array, error)     { return wrap(a.nd.Transpose(axes...)) }
func (a *Array) Reshape(shape ...int) (*Array, error)      { return wrap(a.nd.Reshape(shape...)) }
func (a *Array) BroadcastTo(shape ...int) (*Array, error)  { return wrap(a.nd.BroadcastTo(shape...)) }
func Add(a, b *Array) (*Array, error)                      { return wrap(np.Add(a.nd, b.nd)) }
func Subtract(a, b *Array) (*Array, error)                 { return wrap(np.Subtract(a.nd, b.nd)) }
func Multiply(a, b *Array) (*Array, error)                 { return wrap(np.Multiply(a.nd, b.nd)) }
func Divide(a, b *Array) (*Array, error)                   { return wrap(np.Divide(a.nd, b.nd)) }
func Sqrt(a *Array) (*Array, error)                        { return wrap(np.Sqrt(a.nd)) }
func Absolute(a *Array) (*Array, error)                    { return wrap(np.Absolute(a.nd)) }
func Conjugate(a *Array) (*Array, error)                   { return wrap(np.Conjugate(a.nd)) }
func Equal(a, b *Array) (*np.Array, error)                 { return np.Equal(a.nd, b.nd) }
func Sum(a *Array, axes []int, keepDims bool) (*Array, error) {
	return wrap(np.Sum(a.nd, np.SumOptions{Axes: axes, KeepDims: keepDims}))
}
func MatMul(a, b *Array) (*Array, error) {
	if a.NDim() != 2 || b.NDim() != 2 {
		return nil, np.ErrShape
	}
	return wrap(np.Dot(a.nd, b.nd))
}
