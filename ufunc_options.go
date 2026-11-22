package np

import "slices"

// UFuncOptions exposes the single-output numerical ufunc call contract.
// Where is a boolean array, broadcast together with the inputs and Out.
// A nil Out allocates zeroed storage (NumPy leaves false positions unspecified).
// Casting defaults to same_kind; DType explicitly selects the computation loop.
type UFuncOptions struct {
	Out, Where *Array
	DType      DType
	Casting    string
	// Signature specifies input and output loop dtypes; zero entries infer
	// the corresponding dtype. It is mutually exclusive with DType.
	Signature []DType
	// Order controls newly allocated results (K by default). An explicit
	// Out retains its own memory layout.
	Order string
	// Generalized ufunc core-axis remapping. Axis is a shortcut for vecdot.
	// KeepDims is only applicable to vecdot (nil means unspecified).
	Axes     [][]int
	Axis     *int
	KeepDims *bool
}

// resultType resolves the kernel on empty input without evaluating any
// values, including masked invalid powers or zero divisors.
func (u *UFunc) resultType(args []*Array) (DType, error) {
	if _, ok := nativeFloatOp(u); ok && len(args) == 2 {
		d := args[0].DType()
		if (d == Float32 || d == Float64) && args[1].DType() == d {
			return d, nil
		}
	}
	empty := make([]*Array, len(args))
	for i, a := range args {
		v, e := Zeros(a.DType(), 0)
		if e != nil {
			return 0, e
		}
		empty[i] = v
	}
	v, e := u.Call(empty...)
	if e != nil {
		return 0, e
	}
	return v.DType(), nil
}

// loopDTypes identifies the input types for a single-output numerical
// kernel. Comparisons and predicates return bool but do not cast values to
// bool. ldexp retains its integer exponent operand.
func (u *UFunc) loopDTypes(args []*Array, result DType, explicit bool) ([]DType, error) {
	types := make([]DType, len(args))
	for i := range types {
		types[i] = result
	}
	switch u.Name {
	case "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
		"logical_and", "logical_or", "logical_xor":
		if result != Bool {
			return nil, ErrDType
		}
		d, e := PromoteTypes(args[0].DType(), args[1].DType())
		if e != nil {
			return nil, e
		}
		types[0], types[1] = d, d
		// NumPy has exact mixed signed/unsigned 64-bit comparison loops.
		if d == Float64 && args[0].DType().IsInteger() && args[1].DType().IsInteger() {
			types[0], types[1] = args[0].DType(), args[1].DType()
		}
	case "isnan", "isinf", "isfinite", "logical_not", "signbit":
		if result != Bool {
			return nil, ErrDType
		}
		types[0] = args[0].DType()
	case "absolute":
		if args[0].DType().IsComplex() && !explicit {
			switch result {
			case Float32:
				types[0] = Complex64
			case Float64:
				types[0] = Complex128
			case Float128:
				types[0] = Complex256
			default:
				return nil, ErrDType
			}
		}
	case "ldexp":
		types[1] = args[1].DType()
	}
	return types, nil
}

// CallWithOptions supports dtype selection, checked casting, masked
// evaluation and snapshot-safe writes to strided/overlapping output arrays.
// The scalar numerical kernels remain independent of global error state.
func (u *UFunc) CallWithOptions(opt UFuncOptions, args ...*Array) (*Array, error) {
	if u == nil || len(args) != u.NIn {
		return nil, ErrShape
	}
	if u.NOut != 1 {
		return nil, ErrValue
	}
	order, e := normalizeUFuncOrder(opt.Order)
	if e != nil || opt.DType != 0 && opt.Signature != nil {
		return nil, ErrValue
	}
	if opt.Signature != nil && len(opt.Signature) != u.NArgs() {
		return nil, ErrShape
	}
	opt.Order = order
	if u.Signature() != "" {
		return u.callGufunc(opt, args)
	}
	if opt.Axes != nil || opt.Axis != nil || opt.KeepDims != nil {
		return nil, ErrValue
	}
	v, e := u.callElementwise(opt, args...)
	if e != nil || opt.Out != nil {
		return v, e
	}
	inputs := args
	if opt.Where != nil {
		inputs = append(slices.Clone(args), opt.Where)
	}
	return packUFuncResult(v, ufuncResultAxes(v.shape, inputs, nil, 0, u.elementwiseOrder(opt, args, v.shape)))
}

func (u *UFunc) callElementwise(opt UFuncOptions, args ...*Array) (*Array, error) {
	rule := opt.Casting
	if rule == "" {
		rule = "same_kind"
	}
	if _, e := CanCast(Bool, Bool, rule); e != nil {
		return nil, e
	}
	args = slices.Clone(args)
	shapes := make([][]int, 0, len(args)+2)
	for _, a := range args {
		if a.DType() == 0 {
			return nil, ErrDType
		}
		if rule == "no" && !a.NativeByteOrder() {
			return nil, ErrDType
		}
		shapes = append(shapes, a.shape)
	}
	if opt.Where != nil {
		if opt.Where.DType() != Bool {
			return nil, ErrDType
		}
		shapes = append(shapes, opt.Where.shape)
	}
	if opt.Out != nil {
		if !opt.Out.Writable() {
			return nil, ErrReadOnly
		}
		if rule == "no" && !opt.Out.NativeByteOrder() {
			return nil, ErrDType
		}
		shapes = append(shapes, opt.Out.shape)
	}
	shape, e := BroadcastShapes(shapes...)
	if e != nil {
		return nil, e
	}
	if opt.Out != nil && !slices.Equal(opt.Out.shape, shape) {
		return nil, ErrShape
	}
	for _, a := range args {
		if a.DType().IsFlexible() || a.DType() == StringDType || a.DType().IsTemporal() {
			if opt.Signature != nil {
				return nil, ErrDType
			}
			return u.callDescriptorOptions(opt, args, shape, rule)
		}
	}
	if opt.Out != nil && (opt.Out.DType().IsFlexible() || opt.Out.DType() == StringDType || opt.Out.DType().IsTemporal()) {
		if opt.Signature != nil {
			return nil, ErrDType
		}
		return u.callDescriptorOptions(opt, args, shape, rule)
	}

	d := DType(0)
	if opt.DType != 0 || opt.Signature != nil {
		dtypes := make([]DType, u.NArgs())
		for i, a := range args {
			dtypes[i] = a.DType()
		}
		if opt.Out != nil {
			dtypes[u.NIn] = opt.Out.DType()
		}
		loop, e := u.ResolveDTypes(dtypes, u.callResolveOptions(rule, opt.DType, opt.Signature))
		if e != nil {
			return nil, e
		}
		d = loop[u.NIn]
		for i, a := range args {
			if a.DType() != loop[i] {
				args[i], e = a.Astype(loop[i])
				if e != nil {
					return nil, e
				}
			}
		}
		rd, e := u.resultType(args)
		if e != nil || rd != d {
			return nil, ErrDType
		}
	} else {
		d, e = u.resultType(args)
		if e != nil {
			return nil, e
		}
	}
	if opt.DType == 0 && opt.Signature == nil && rule != "same_kind" && rule != "unsafe" {
		types, e := u.loopDTypes(args, d, false)
		if e != nil {
			return nil, e
		}
		for i, a := range args {
			ok, e := CanCast(a.DType(), types[i], rule)
			if e != nil {
				return nil, e
			}
			if !ok {
				return nil, ErrDType
			}
		}
		rd, e := u.resultType(args)
		if e != nil {
			return nil, e
		}
		if rd != d {
			return nil, ErrDType
		}
	}
	if opt.Out == nil && opt.Where == nil {
		return u.Call(args...)
	}
	out := opt.Out
	if out == nil {
		out, e = Zeros(d, shape...)
		if e != nil {
			return nil, e
		}
	} else {
		ok, e := CanCast(d, out.DType(), rule)
		if e != nil {
			return nil, e
		}
		if !ok {
			return nil, ErrDType
		}
	}
	for i, a := range args {
		args[i], e = a.BroadcastTo(shape...)
		if e != nil {
			return nil, e
		}
	}
	if tryNativeFloatOut(u, d, out, opt.Where, args) || tryMoreNativeOut(u, d, out, opt.Where, args) {
		return out, nil
	}
	if opt.Where == nil {
		v, e := u.Call(args...)
		if e != nil {
			return nil, e
		}
		if e = CopyTo(out, v, nil); e != nil {
			return nil, e
		}
		return out, nil
	}

	mask, e := opt.Where.BroadcastTo(shape...)
	if e != nil {
		return nil, e
	}
	// Gather all selected operands before writing output. In addition to
	// overlap safety this guarantees masked-off values are never evaluated.
	positions := make([]int, 0, mask.size)
	mask.walk(func(i, off int) {
		if mask.elementAt(off).i != 0 {
			positions = append(positions, i)
		}
	})
	for i, a := range args {
		compact, e := Zeros(a.DType(), len(positions))
		if e != nil {
			return nil, e
		}
		for j, k := range positions {
			compact.setElement(j, a.elementAt(a.offset+flatOffset(k, a.shape, a.strides)))
		}
		args[i] = compact
	}
	v, e := u.Call(args...)
	if e != nil {
		return nil, e
	}
	for j, k := range positions {
		out.setElement(out.offset+flatOffset(k, out.shape, out.strides), castElement(v.elementAt(v.offset+j*v.strides[0]), v.DType(), out.DType()))
	}
	return out, nil
}

// Descriptor-aware kernels retain widths, units and missing-value metadata
// even when compacting masked operands.
func (u *UFunc) callDescriptorOptions(opt UFuncOptions, args []*Array, shape []int, rule string) (*Array, error) {
	if rule == "no" || rule == "equiv" {
		var unit *DateUnit
		for _, a := range args {
			if a.DType().IsTemporal() {
				if unit != nil && *unit != a.data.unit {
					return nil, ErrDType
				}
				u := a.data.unit
				unit = &u
				if rule == "no" && !a.NativeByteOrder() {
					return nil, ErrDType
				}
			}
		}
	}
	if opt.Where == nil && opt.Out == nil && opt.DType == 0 {
		return u.Call(args...)
	}
	for i, a := range args {
		v, e := a.BroadcastTo(shape...)
		if e != nil {
			return nil, e
		}
		args[i] = v
	}
	var positions []int
	if opt.Where != nil {
		mask, e := opt.Where.BroadcastTo(shape...)
		if e != nil {
			return nil, e
		}
		positions = make([]int, 0, mask.size)
		mask.walk(func(i, off int) {
			if mask.elementAt(off).i != 0 {
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
	var v *Array
	var e error
	if u.Name == "multiply" && len(args) == 2 && (args[0].DType().IsCharacter() || args[1].DType().IsCharacter()) {
		if opt.Out == nil || !opt.Out.DType().IsCharacter() {
			return nil, ErrDType
		}
		v, e = repeatStrings(args[0], args[1], &opt.Out.data.flex)
	} else {
		v, e = u.Call(args...)
	}
	if e != nil {
		return nil, e
	}
	if opt.DType != 0 && opt.DType != v.DType() {
		return nil, ErrDType
	}
	out := opt.Out
	if out == nil {
		if opt.Where == nil {
			return v, nil
		}
		out, e = v.zerosLike(shape)
		if e != nil {
			return nil, e
		}
	} else {
		var ok bool
		if out.DType().IsFlexible() {
			ok, e = v.CanCastFlexible(out.data.flex, rule)
		} else if out.DType().IsTemporal() && v.DType() == out.DType() {
			ok, e = CanCastDateUnits(v.data.unit, out.data.unit, v.DType(), rule)
			if rule == "no" && v.NativeByteOrder() != out.NativeByteOrder() {
				ok = false
			}
		} else if out.DType() == StringDType && v.DType() == StringDType && v.data.stringNA != nil && out.data.stringNA == nil && rule != "unsafe" {
			ok = false
		} else {
			ok, e = CanCast(v.DType(), out.DType(), rule)
		}
		if e != nil {
			return nil, e
		}
		if !ok {
			return nil, ErrDType
		}
	}
	v, e = out.castLike(v)
	if e != nil {
		return nil, e
	}
	if opt.Where == nil {
		if e = out.Assign(nil, v); e != nil {
			return nil, e
		}
	} else {
		for j, k := range positions {
			out.setElement(out.offset+flatOffset(k, out.shape, out.strides), v.elementAt(j))
		}
	}
	return out, nil
}
