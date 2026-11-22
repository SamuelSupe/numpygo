package np

import "slices"

// CallAll executes one or multiple-output ufuncs, consistently returning
// the output arrays as a slice.
func (u *UFunc) CallAll(args ...*Array) ([]*Array, error) {
	if u == nil || len(args) != u.NIn {
		return nil, ErrShape
	}
	if u.multi != nil {
		return u.multi(args...)
	}
	a, e := u.Call(args...)
	if e != nil {
		return nil, e
	}
	return []*Array{a}, nil
}

type MultiUFuncOptions struct {
	Out       []*Array
	Where     *Array
	DType     DType
	Casting   string
	Signature []DType
	Order     string
	Axes      [][]int
	Axis      *int
	KeepDims  *bool
}

// CallAllWithOptions evaluates selected elements only and snapshots all
// outputs before writing any caller-supplied view.
func (u *UFunc) CallAllWithOptions(opt MultiUFuncOptions, args ...*Array) ([]*Array, error) {
	if u == nil || len(args) != u.NIn || len(opt.Out) != 0 && len(opt.Out) != u.NOut {
		return nil, ErrShape
	}
	order, e := normalizeUFuncOrder(opt.Order)
	if e != nil || opt.DType != 0 && opt.Signature != nil {
		return nil, ErrValue
	}
	if opt.Signature != nil && len(opt.Signature) != u.NArgs() {
		return nil, ErrShape
	}
	opt.Order = order
	if u.NOut == 1 {
		var out *Array
		if len(opt.Out) > 0 {
			out = opt.Out[0]
		}
		v, e := u.CallWithOptions(UFuncOptions{Out: out, Where: opt.Where, DType: opt.DType, Casting: opt.Casting, Signature: opt.Signature, Order: opt.Order, Axes: opt.Axes, Axis: opt.Axis, KeepDims: opt.KeepDims}, args...)
		if e != nil {
			return nil, e
		}
		return []*Array{v}, nil
	}
	if opt.Axes != nil || opt.Axis != nil || opt.KeepDims != nil {
		return nil, ErrValue
	}
	rule := opt.Casting
	if rule == "" {
		rule = "same_kind"
	}
	typ := make([]DType, u.NArgs())
	shapes := [][]int{}
	for i, a := range args {
		if a.DType() == 0 {
			return nil, ErrDType
		}
		if rule == "no" && !a.NativeByteOrder() {
			return nil, ErrDType
		}
		typ[i] = a.DType()
		shapes = append(shapes, a.shape)
	}
	out := make([]*Array, u.NOut)
	copy(out, opt.Out)
	for i, a := range out {
		if a == nil {
			continue
		}
		if !a.Writable() {
			return nil, ErrReadOnly
		}
		if rule == "no" && !a.NativeByteOrder() {
			return nil, ErrDType
		}
		typ[u.NIn+i] = a.DType()
		shapes = append(shapes, a.shape)
	}
	if opt.Where != nil {
		if opt.Where.DType() != Bool {
			return nil, ErrDType
		}
		shapes = append(shapes, opt.Where.shape)
	}
	res := u.callResolveOptions(rule, opt.DType, opt.Signature)
	loop, e := u.ResolveDTypes(typ, res)
	if e != nil {
		return nil, e
	}
	shape, e := BroadcastShapes(shapes...)
	if e != nil {
		return nil, e
	}
	layoutInputs := slices.Clone(args)
	for _, a := range opt.Out {
		if a != nil {
			layoutInputs = append(layoutInputs, a)
		}
	}
	if opt.Where != nil {
		layoutInputs = append(layoutInputs, opt.Where)
	}
	axes := ufuncResultAxes(shape, layoutInputs, nil, 0, opt.Order)
	for i, a := range out {
		if a == nil {
			out[i], e = Zeros(loop[u.NIn+i], shape...)
			if e != nil {
				return nil, e
			}
		} else if !slices.Equal(a.shape, shape) {
			return nil, ErrShape
		}
	}
	args = slices.Clone(args)
	for i, a := range args {
		v, e := a.Astype(loop[i])
		if e != nil {
			return nil, e
		}
		args[i], e = v.BroadcastTo(shape...)
		if e != nil {
			return nil, e
		}
	}
	positions := []int{}
	if opt.Where != nil {
		w, e := opt.Where.BroadcastTo(shape...)
		if e != nil {
			return nil, e
		}
		w.walk(func(i, off int) {
			if w.elementAt(off).i != 0 {
				positions = append(positions, i)
			}
		})
		for i, a := range args {
			v, e := a.zerosLike([]int{len(positions)})
			if e != nil {
				return nil, e
			}
			for j, k := range positions {
				v.setElement(j, a.elementAt(a.offset+flatOffset(k, a.shape, a.strides)))
			}
			args[i] = v
		}
	}
	result, e := u.CallAll(args...)
	if e != nil {
		return nil, e
	}
	for i, v := range result {
		if v.DType() != loop[u.NIn+i] {
			return nil, ErrDType
		}
		if opt.Where == nil {
			if e = out[i].Assign(nil, v); e != nil {
				return nil, e
			}
		} else {
			dst := out[i]
			for j, k := range positions {
				dst.setElement(dst.offset+flatOffset(k, dst.shape, dst.strides), castElement(v.elementAt(j), v.DType(), dst.DType()))
			}
		}
	}
	for i, a := range out {
		if len(opt.Out) == 0 || opt.Out[i] == nil {
			out[i], e = packUFuncResult(a, axes)
			if e != nil {
				return nil, e
			}
		}
	}
	return out, nil
}

func ufuncMatMul(a, b *Array) (*Array, error) {
	if a.DType() == 0 || b.DType() == 0 {
		return nil, ErrDType
	}
	if a.NDim() == 0 || b.NDim() == 0 {
		return nil, ErrShape
	}
	av, bv := a.NDim() == 1, b.NDim() == 1
	var e error
	if av {
		a, e = a.Reshape(1, a.size)
		if e != nil {
			return nil, e
		}
	}
	if bv {
		b, e = b.Reshape(b.size, 1)
		if e != nil {
			return nil, e
		}
	}
	if a.shape[a.NDim()-1] != b.shape[b.NDim()-2] {
		return nil, ErrShape
	}
	r, e := Einsum("...ik,...kj->...ij", a, b)
	if e != nil {
		return nil, e
	}
	if av && bv {
		return Squeeze(r, -2, -1)
	}
	if av {
		return Squeeze(r, -2)
	}
	if bv {
		return Squeeze(r, -1)
	}
	return r, nil
}
func ufuncMatVec(a, b *Array) (*Array, error) {
	if a.NDim() < 2 || b.NDim() < 1 || a.shape[a.NDim()-1] != b.shape[b.NDim()-1] {
		return nil, ErrShape
	}
	return Einsum("...ij,...j->...i", a, b)
}
func ufuncVecMat(a, b *Array) (*Array, error) {
	if a.NDim() < 1 || b.NDim() < 2 || a.shape[a.NDim()-1] != b.shape[b.NDim()-2] {
		return nil, ErrShape
	}
	if a.DType().IsComplex() {
		v, e := Conjugate(a)
		if e != nil {
			return nil, e
		}
		a = v
	}
	return Einsum("...i,...ij->...j", a, b)
}
func ufuncVecDot(a, b *Array) (*Array, error) {
	if a.NDim() < 1 || b.NDim() < 1 || a.shape[a.NDim()-1] != b.shape[b.NDim()-1] {
		return nil, ErrShape
	}
	if a.DType().IsComplex() {
		v, e := Conjugate(a)
		if e != nil {
			return nil, e
		}
		a = v
	}
	return Einsum("...i,...i->...", a, b)
}
