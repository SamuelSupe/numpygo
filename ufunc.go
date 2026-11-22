package np

import "slices"

// UFunc exposes numerical kernel metadata and the common application,
// reduce, accumulate, outer and in-place indexed forms without Python dispatch.
type UFunc struct {
	Name        string
	NIn, NOut   int
	Identity    any
	HasIdentity bool
	unary       func(*Array) (*Array, error)
	binary      func(*Array, *Array) (*Array, error)
	multi       func(...*Array) ([]*Array, error)
	nativeOp    uint8
	nativeExtra uint8
}

func LookupUFunc(name string) (*UFunc, error) {
	if n, ok := ufuncAliases[name]; ok {
		name = n
	}
	binary := map[string]func(*Array, *Array) (*Array, error){
		"add": Add, "subtract": Subtract, "multiply": Multiply, "divide": Divide, "floor_divide": FloorDivide, "remainder": Remainder, "power": Power,
		"minimum": Minimum, "maximum": Maximum, "fmin": FMin, "fmax": FMax,
		"equal": Equal, "not_equal": NotEqual, "less": Less, "less_equal": LessEqual, "greater": Greater, "greater_equal": GreaterEqual,
		"logical_and": LogicalAnd, "logical_or": LogicalOr, "logical_xor": LogicalXor,
		"bitwise_and": BitwiseAnd, "bitwise_or": BitwiseOr, "bitwise_xor": BitwiseXor,
		"left_shift": LeftShift, "right_shift": RightShift, "gcd": GCD, "lcm": LCM,
		"float_power": FloatPower, "fmod": Fmod, "hypot": Hypot, "arctan2": Arctan2, "copysign": Copysign,
		"logaddexp": LogAddExp, "logaddexp2": LogAddExp2, "heaviside": Heaviside, "nextafter": Nextafter, "ldexp": Ldexp,
		"matmul": ufuncMatMul, "matvec": ufuncMatVec, "vecmat": ufuncVecMat, "vecdot": ufuncVecDot,
	}
	if f, ok := binary[name]; ok {
		u := &UFunc{Name: name, NIn: 2, NOut: 1, binary: f, nativeExtra: nativeExtraOp(name)}
		if o, ok := nativeFloatOpName(name); ok {
			u.nativeOp = uint8(o) + 1
		}
		u.Identity = ufuncReference[name].identity
		u.HasIdentity = u.Identity != nil
		return u, nil
	}
	unary := map[string]func(*Array) (*Array, error){
		"sqrt": Sqrt, "exp": Exp, "exp2": Exp2, "expm1": Expm1, "log": Log, "log2": Log2, "log10": Log10, "log1p": Log1p,
		"reciprocal": Reciprocal, "square": Square, "sin": Sin, "cos": Cos, "tan": Tan, "absolute": Absolute, "negative": Negative, "positive": Positive,
		"arcsin": Arcsin, "arccos": Arccos, "arctan": Arctan, "sinh": Sinh, "cosh": Cosh, "tanh": Tanh,
		"arcsinh": Arcsinh, "arccosh": Arccosh, "arctanh": Arctanh, "cbrt": Cbrt, "degrees": Degrees, "radians": Radians,
		"rint": Rint, "floor": Floor, "ceil": Ceil, "trunc": Trunc, "fabs": Fabs, "conjugate": Conjugate, "sign": Sign,
		"logical_not": LogicalNot, "invert": Invert, "isnan": IsNaN, "isinf": IsInf, "isfinite": IsFinite, "isnat": IsNaT, "signbit": Signbit, "spacing": Spacing,
		"bitwise_count": BitwiseCount, "deg2rad": Radians, "rad2deg": Degrees,
	}
	if f, ok := unary[name]; ok {
		return &UFunc{Name: name, NIn: 1, NOut: 1, unary: f, nativeExtra: nativeExtraOp(name)}, nil
	}
	switch name {
	case "divmod":
		return &UFunc{Name: name, NIn: 2, NOut: 2, multi: func(a ...*Array) ([]*Array, error) { q, r, e := Divmod(a[0], a[1]); return []*Array{q, r}, e }}, nil
	case "frexp":
		return &UFunc{Name: name, NIn: 1, NOut: 2, multi: func(a ...*Array) ([]*Array, error) { q, r, e := Frexp(a[0]); return []*Array{q, r}, e }}, nil
	case "modf":
		return &UFunc{Name: name, NIn: 1, NOut: 2, multi: func(a ...*Array) ([]*Array, error) { q, r, e := Modf(a[0]); return []*Array{q, r}, e }}, nil
	}
	return nil, ErrValue
}
func (u *UFunc) Call(args ...*Array) (*Array, error) {
	if u == nil || len(args) != u.NIn {
		return nil, ErrShape
	}
	if u.NIn == 1 && u.unary != nil {
		return u.unary(args[0])
	}
	if u.NIn == 2 && u.binary != nil {
		return u.binary(args[0], args[1])
	}
	return nil, ErrValue
}
func (u *UFunc) Reduce(a *Array, options ...SumOptions) (*Array, error) {
	if u == nil || u.NIn != 2 || u.NOut != 1 || u.Signature() != "" {
		return nil, ErrValue
	}
	if len(options) == 0 && a.NDim() > 0 {
		options = []SumOptions{{Axes: []int{0}}}
	}
	switch u.Name {
	case "add":
		return Sum(a, options...)
	case "multiply":
		return Prod(a, options...)
	case "minimum":
		return Min(a, options...)
	case "maximum":
		return Max(a, options...)
	case "logical_and":
		return All(a, options...)
	case "logical_or":
		return Any(a, options...)
	}
	if len(options) > 1 {
		return nil, ErrAxis
	}
	var opt ReduceOptions
	if len(options) == 1 {
		opt.Axes = options[0].Axes
		opt.AllAxes = options[0].Axes == nil
		opt.KeepDims = options[0].KeepDims
		opt.DType = options[0].DType
	}
	return u.ReduceWithOptions(a, opt)
}
func (u *UFunc) Accumulate(a *Array, axis int, dtype DType) (*Array, error) {
	if u == nil || u.NIn != 2 || u.NOut != 1 || u.Signature() != "" {
		return nil, ErrValue
	}
	switch u.Name {
	case "add":
		return Cumsum(a, &axis, dtype)
	case "multiply":
		return Cumprod(a, &axis, dtype)
	}
	return u.accumulate(a, axis, dtype)
}
func (u *UFunc) Outer(a, b *Array) (*Array, error) {
	return u.OuterWithOptions(a, b, UFuncOptions{})
}
func (u *UFunc) OuterWithOptions(a, b *Array, opt UFuncOptions) (*Array, error) {
	if u == nil || u.NIn != 2 || u.NOut != 1 || u.Signature() != "" {
		return nil, ErrValue
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
	return u.CallWithOptions(opt, v, w)
}
func (u *UFunc) At(a *Array, indices []int, b *Array) error {
	ix := make([]int64, len(indices))
	for i, k := range indices {
		ix[i] = int64(k)
	}
	ind, e := FromSlice(ix, len(ix))
	if e != nil {
		return e
	}
	return u.AtIndices(a, []Index{ind}, b)
}

// AtIndices applies unary or binary ufuncs in-place in logical index order.
// Gather all indices and the RHS before mutation; repeated sites are updated
// sequentially rather than collapsed into a buffered fancy assignment.
func (u *UFunc) AtIndices(a *Array, indices []Index, b *Array) error {
	if u == nil || u.NOut != 1 || u.Signature() != "" || u.NIn < 1 || u.NIn > 2 {
		return ErrValue
	}
	if !a.Writable() {
		return ErrReadOnly
	}
	var shape, offsets []int
	var e error
	advanced := false
	for _, x := range indices {
		if _, ok := x.(*Array); ok {
			advanced = true
			break
		}
	}
	if advanced {
		var p *advancedSelection
		p, e = a.advancedLayout(indices)
		if e == nil {
			shape, offsets = p.shape, p.offsets
		}
	} else {
		var v *Array
		v, e = a.Slice(indices...)
		if e == nil {
			shape = v.shape
			offsets = make([]int, v.size)
			v.walk(func(i, off int) { offsets[i] = off })
		}
	}
	if e != nil {
		return e
	}
	var x *Array
	if u.NIn == 2 {
		if b.DType() == 0 {
			return ErrDType
		}
		x, e = b.BroadcastTo(shape...)
		if e != nil {
			return e
		}
		x, e = x.Copy()
		if e != nil {
			return e
		}
	} else if b != nil {
		return ErrShape
	}
	for i, off := range offsets {
		v := &Array{data: a.data, offset: off, size: 1, readOnly: false, base: a.viewBase()}
		args := []*Array{v}
		if u.NIn == 2 {
			args = append(args, &Array{data: x.data, offset: i * x.contiguousStep(), size: 1, base: x.viewBase()})
		}
		r, e := u.Call(args...)
		if e != nil {
			return e
		}
		value, e := r.AtValue()
		if e != nil {
			return e
		}
		el, e := a.convertValue(value)
		if e != nil {
			return e
		}
		a.setElement(off, el)
	}
	return nil
}
