package np

import "slices"

// gufuncPermutation maps a logical array to outer axes followed by the
// requested core axes, preserving the order of the remaining axes.
func gufuncPermutation(rank, ncore int, axes []int, explicit bool) ([]int, error) {
	if rank < ncore {
		return nil, ErrShape
	}
	if !explicit {
		p := make([]int, rank)
		for i := range p {
			p[i] = i
		}
		return p, nil
	}
	if len(axes) != ncore {
		return nil, ErrAxis
	}
	seen := make([]bool, rank)
	core := make([]int, ncore)
	for i, ax := range axes {
		ax, e := NormalizeAxis(ax, rank)
		if e != nil || seen[ax] {
			return nil, ErrAxis
		}
		seen[ax], core[i] = true, ax
	}
	p := make([]int, 0, rank)
	for ax, used := range seen {
		if !used {
			p = append(p, ax)
		}
	}
	return append(p, core...), nil
}

func (u *UFunc) callGufunc(opt UFuncOptions, args []*Array) (*Array, error) {
	if opt.Where != nil || opt.Axis != nil && opt.Axes != nil ||
		u.Name != "vecdot" && (opt.Axis != nil || opt.KeepDims != nil) {
		return nil, ErrValue
	}
	for _, a := range args {
		if a.DType() == 0 {
			return nil, ErrDType
		}
		if a.NDim() == 0 {
			return nil, ErrShape
		}
	}
	cores := []int{1, 1}
	outCore := 0
	switch u.Name {
	case "matmul":
		cores[0], cores[1] = min(2, args[0].NDim()), min(2, args[1].NDim())
		outCore = cores[0] + cores[1] - 2
	case "matvec":
		cores[0], outCore = 2, 1
	case "vecmat":
		cores[1], outCore = 2, 1
	case "vecdot":
		if opt.KeepDims != nil && *opt.KeepDims {
			outCore = 1
		}
	default:
		return nil, ErrValue
	}
	if opt.Axes != nil && len(opt.Axes) != 3 && !(u.Name == "vecdot" && len(opt.Axes) == 2) {
		return nil, ErrValue
	}
	mapped := make([]*Array, 2)
	for i, a := range args {
		var axes []int
		if opt.Axis != nil {
			axes = []int{*opt.Axis}
		} else if opt.Axes != nil {
			axes = opt.Axes[i]
		}
		p, e := gufuncPermutation(a.NDim(), cores[i], axes, opt.Axes != nil || opt.Axis != nil)
		if e != nil {
			return nil, e
		}
		mapped[i], e = a.Transpose(p...)
		if e != nil {
			return nil, e
		}
	}
	var outAxes []int
	explicitOutAxes := false
	if len(opt.Axes) == 3 {
		outAxes, explicitOutAxes = opt.Axes[2], true
	} else if opt.Axis != nil && outCore != 0 {
		outAxes, explicitOutAxes = []int{*opt.Axis}, true
	}
	rule := opt.Casting
	if rule == "" {
		rule = "same_kind"
	}
	d := []DType{args[0].DType(), args[1].DType(), 0}
	for _, a := range args {
		if rule == "no" && !a.NativeByteOrder() {
			return nil, ErrDType
		}
	}
	if opt.Out != nil {
		if !opt.Out.Writable() {
			return nil, ErrReadOnly
		}
		if rule == "no" && !opt.Out.NativeByteOrder() {
			return nil, ErrDType
		}
		d[2] = opt.Out.DType()
	}
	loop, e := u.ResolveDTypes(d, u.callResolveOptions(rule, opt.DType, opt.Signature))
	if e != nil {
		return nil, e
	}
	computation := slices.Clone(mapped)
	for i, a := range computation {
		if a.DType() != loop[i] {
			computation[i], e = a.Astype(loop[i])
			if e != nil {
				return nil, e
			}
		}
	}
	v, e := u.Call(computation...)
	if e != nil {
		return nil, e
	}
	if v.DType() != loop[2] {
		return nil, ErrDType
	}
	if u.Name == "vecdot" && outCore == 1 {
		v, e = ExpandDims(v, -1)
		if e != nil {
			return nil, e
		}
	}
	if opt.Out != nil {
		p, e := gufuncPermutation(opt.Out.NDim(), outCore, outAxes, explicitOutAxes)
		if e != nil {
			return nil, e
		}
		dst, e := opt.Out.Transpose(p...)
		if e != nil {
			return nil, e
		}
		// Only outer dimensions broadcast. Core dimensions must match
		// exactly, even when the computed result has length one.
		if !slices.Equal(dst.shape[dst.NDim()-outCore:], v.shape[v.NDim()-outCore:]) {
			return nil, ErrShape
		}
		b, e := BroadcastShapes(dst.shape[:dst.NDim()-outCore], v.shape[:v.NDim()-outCore])
		if e != nil || !slices.Equal(b, dst.shape[:dst.NDim()-outCore]) {
			return nil, ErrShape
		}
		if e = dst.Assign(nil, v); e != nil {
			return nil, e
		}
		return opt.Out, nil
	}
	p, e := gufuncPermutation(v.NDim(), outCore, outAxes, explicitOutAxes)
	if e != nil {
		return nil, e
	}
	v, e = packUFuncResult(v, ufuncResultAxes(v.shape, mapped, cores, outCore, ufuncAnyOrder(opt.Order, args)))
	if e != nil {
		return nil, e
	}
	// Allocation was in canonical core order; remapping axes changes only
	// its shape and strides. The returned array still owns its storage.
	shape, strides := v.Shape(), slices.Clone(v.strides)
	for i, ax := range p {
		v.shape[ax], v.strides[ax] = shape[i], strides[i]
	}
	return v, nil
}
