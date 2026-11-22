// Package objects provides explicit Go object arrays and broadcast callbacks.
// It does not embed Python, pickle, or its reference-counting object model.
package objects

import (
	"reflect"
	"slices"

	"github.com/SamuelSupe/numpygo"
)

type Array struct{ nd *np.Array }

func FromNDArray(a *np.Array) (*Array, error) {
	if a.DType() != np.Object {
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
func FromSlice(values []any, shape ...int) (*Array, error) {
	return wrap(np.FromObjects(values, shape...))
}
func Scalar(v any) *Array                                  { return &Array{np.ObjectScalar(v)} }
func FromNumeric(a *np.Array) (*Array, error)              { return wrap(a.Astype(np.Object)) }
func (a *Array) Shape() []int                              { return a.nd.Shape() }
func (a *Array) Size() int                                 { return a.nd.Size() }
func (a *Array) NDim() int                                 { return a.nd.NDim() }
func (a *Array) Writable() bool                            { return a.nd.Writable() }
func (a *Array) At(indices ...int) (any, error)            { return a.nd.AtValue(indices...) }
func (a *Array) Set(v any, indices ...int) error           { return a.nd.SetValue(v, indices...) }
func (a *Array) Values() []any                             { v, _ := a.nd.ObjectValues(); return v }
func (a *Array) Copy() (*Array, error)                     { return wrap(a.nd.Copy()) }
func (a *Array) BroadcastTo(shape ...int) (*Array, error)  { return wrap(a.nd.BroadcastTo(shape...)) }
func (a *Array) Slice(indices ...np.Index) (*Array, error) { return wrap(a.nd.Slice(indices...)) }
func (a *Array) Transpose(axes ...int) (*Array, error)     { return wrap(a.nd.Transpose(axes...)) }
func (a *Array) Reshape(shape ...int) (*Array, error)      { return wrap(a.nd.Reshape(shape...)) }
func (a *Array) EqualTo(v any) (*np.Array, error) {
	x := a.Values()
	flags := make([]bool, len(x))
	for i, z := range x {
		flags[i] = reflect.DeepEqual(z, v)
	}
	return np.WrapSlice(flags, a.Shape()...)
}

// UFunc is a typed replacement for np.frompyfunc's object callback contract.
// Inputs and outputs use Go values, with explicit arity and optional identity.
type UFunc struct {
	Name        string
	NIn, NOut   int
	Identity    any
	HasIdentity bool
	call        func(...any) ([]any, error)
}

func FromPyFunc(name string, nin, nout int, fn func(...any) ([]any, error)) (*UFunc, error) {
	if nin < 1 || nout < 1 || fn == nil {
		return nil, np.ErrValue
	}
	return &UFunc{Name: name, NIn: nin, NOut: nout, call: fn}, nil
}
func (u *UFunc) invoke(v ...any) ([]any, error) {
	if len(v) != u.NIn {
		return nil, np.ErrShape
	}
	r, e := u.call(v...)
	if e != nil {
		return nil, e
	}
	if len(r) != u.NOut {
		return nil, np.ErrShape
	}
	return r, nil
}
func (u *UFunc) Apply(args ...*Array) ([]*Array, error) {
	if len(args) != u.NIn {
		return nil, np.ErrShape
	}
	shapes := make([][]int, len(args))
	for i, a := range args {
		shapes[i] = a.Shape()
	}
	shape, e := np.BroadcastShapes(shapes...)
	if e != nil {
		return nil, e
	}
	inputs := make([][]any, len(args))
	for i, a := range args {
		v, e := a.BroadcastTo(shape...)
		if e != nil {
			return nil, e
		}
		inputs[i] = v.Values()
	}
	n, e := np.ElementCount(np.Int64, shape...)
	if e != nil {
		return nil, e
	}
	outputs := make([][]any, u.NOut)
	for i := range outputs {
		outputs[i] = make([]any, n)
	}
	in := make([]any, u.NIn)
	for j := 0; j < n; j++ {
		for i := range inputs {
			in[i] = inputs[i][j]
		}
		v, e := u.invoke(in...)
		if e != nil {
			return nil, e
		}
		for i, x := range v {
			outputs[i][j] = x
		}
	}
	r := make([]*Array, u.NOut)
	for i, v := range outputs {
		r[i], e = FromSlice(v, shape...)
		if e != nil {
			return nil, e
		}
	}
	return r, nil
}
func (u *UFunc) Reduce(a *Array, axis int, keepDims bool) (*Array, error) {
	if u.NIn != 2 || u.NOut != 1 {
		return nil, np.ErrValue
	}
	ax, e := np.NormalizeAxis(axis, a.NDim())
	if e != nil {
		return nil, e
	}
	shape := a.Shape()
	outer := append(slices.Clone(shape[:ax]), shape[ax+1:]...)
	it, e := np.NDIndex(outer...)
	if e != nil {
		return nil, e
	}
	var out []any
	for ix := range it {
		coord := append(slices.Clone(ix[:ax]), 0)
		coord = append(coord, ix[ax:]...)
		var acc any
		start := 0
		if u.HasIdentity {
			acc = u.Identity
		} else {
			if shape[ax] == 0 {
				return nil, np.ErrValue
			}
			acc, e = a.At(coord...)
			if e != nil {
				return nil, e
			}
			start = 1
		}
		for j := start; j < shape[ax]; j++ {
			coord[ax] = j
			v, e := a.At(coord...)
			if e != nil {
				return nil, e
			}
			r, e := u.invoke(acc, v)
			if e != nil {
				return nil, e
			}
			acc = r[0]
		}
		out = append(out, acc)
	}
	if keepDims {
		outer = append(slices.Clone(shape[:ax]), 1)
		outer = append(outer, shape[ax+1:]...)
	}
	return FromSlice(out, outer...)
}
func (u *UFunc) Accumulate(a *Array, axis int) (*Array, error) {
	if u.NIn != 2 || u.NOut != 1 {
		return nil, np.ErrValue
	}
	ax, e := np.NormalizeAxis(axis, a.NDim())
	if e != nil {
		return nil, e
	}
	out, e := FromSlice(make([]any, a.Size()), a.Shape()...)
	if e != nil {
		return nil, e
	}
	shape := a.Shape()
	outer := append(slices.Clone(shape[:ax]), shape[ax+1:]...)
	it, e := np.NDIndex(outer...)
	if e != nil {
		return nil, e
	}
	for ix := range it {
		coord := append(slices.Clone(ix[:ax]), 0)
		coord = append(coord, ix[ax:]...)
		var acc any
		for j := 0; j < shape[ax]; j++ {
			coord[ax] = j
			v, e := a.At(coord...)
			if e != nil {
				return nil, e
			}
			if j == 0 {
				acc = v
			} else {
				r, e := u.invoke(acc, v)
				if e != nil {
					return nil, e
				}
				acc = r[0]
			}
			if e = out.Set(acc, coord...); e != nil {
				return nil, e
			}
		}
	}
	return out, nil
}
func (u *UFunc) Outer(a, b *Array) (*Array, error) {
	if u.NIn != 2 || u.NOut != 1 {
		return nil, np.ErrValue
	}
	x, y := a.Shape(), b.Shape()
	sh1 := append(slices.Clone(x), make([]int, len(y))...)
	for i := len(x); i < len(sh1); i++ {
		sh1[i] = 1
	}
	sh2 := make([]int, len(x))
	for i := range sh2 {
		sh2[i] = 1
	}
	sh2 = append(sh2, y...)
	v, e := a.Reshape(sh1...)
	if e != nil {
		return nil, e
	}
	w, e := b.Reshape(sh2...)
	if e != nil {
		return nil, e
	}
	r, e := u.Apply(v, w)
	if e != nil {
		return nil, e
	}
	return r[0], nil
}
func (u *UFunc) At(a *Array, indices []int, b *Array) error {
	if u.NIn != 2 || u.NOut != 1 || a.NDim() != 1 || !a.Writable() {
		return np.ErrValue
	}
	values, e := b.BroadcastTo(len(indices))
	if e != nil {
		return e
	}
	v := values.Values()
	for i, j := range indices {
		x, e := a.At(j)
		if e != nil {
			return e
		}
		r, e := u.invoke(x, v[i])
		if e != nil {
			return e
		}
		if e = a.Set(r[0], j); e != nil {
			return e
		}
	}
	return nil
}
