package np

import (
	"math"
	"slices"
)

// ReduceOptions controls the binary ufunc reduction contract. Nil Axes
// defaults to axis zero. AllAxes selects NumPy's axis=None. Initial is a typed
// Go scalar; a nil Initial uses the ufunc identity or the first element.
type ReduceOptions struct {
	Axes       []int
	AllAxes    bool
	KeepDims   bool
	DType      DType
	Where, Out *Array
	Initial    any
}

type reduceKernel func(element, element) (element, error)

func (u *UFunc) reductionType(a *Array, d DType) (DType, error) {
	if u == nil || u.NIn != 2 || u.NOut != 1 || u.Signature() != "" || !a.DType().IsNativeNumeric() && !a.DType().IsExtended() {
		return 0, ErrDType
	}
	if d == 0 {
		if u.Name == "add" || u.Name == "multiply" {
			d = sumDType(a.DType())
		} else {
			var e error
			d, e = u.resultType([]*Array{a, a})
			if e != nil {
				return 0, e
			}
		}
	}
	v, e := Zeros(d, 0)
	if e != nil {
		return 0, e
	}
	rd, e := u.resultType([]*Array{v, v})
	if e != nil {
		return 0, e
	}
	if rd != d {
		return 0, ErrDType
	}
	return d, nil
}

// AccumulateInto writes a cumulative reduction into a supplied output view.
func (u *UFunc) AccumulateInto(a *Array, axis int, dtype DType, out *Array) (*Array, error) {
	if out == nil {
		return u.Accumulate(a, axis, dtype)
	}
	if !out.Writable() {
		return nil, ErrReadOnly
	}
	if !slices.Equal(a.shape, out.shape) {
		return nil, ErrShape
	}
	if dtype == 0 && (u.Name == "add" || u.Name == "multiply") {
		dtype = out.DType()
	}
	v, e := u.Accumulate(a, axis, dtype)
	if e != nil {
		return nil, e
	}
	if e = out.Assign(nil, v); e != nil {
		return nil, e
	}
	return out, nil
}

func (u *UFunc) reductionKernel(d DType) (reduceKernel, bool, error) {
	ops := map[string]op{"add": addOp, "subtract": subOp, "multiply": mulOp, "divide": divOp, "floor_divide": floorOp, "remainder": remOp, "power": powOp,
		"bitwise_and": andOp, "bitwise_or": orOp, "bitwise_xor": xorOp}
	if o, ok := ops[u.Name]; ok {
		reorder := o == addOp || o == mulOp || o >= andOp
		return func(x, y element) (element, error) { return arithmeticScalar(x, y, d, o) }, reorder, nil
	}
	if d == Float128 && (u.Name == "hypot" || u.Name == "logaddexp" || u.Name == "logaddexp2") {
		f := extendedRealBinary[u.Name]
		return func(x, y element) (element, error) {
			return element{extra: f(extendedValue(x, d).Real, extendedValue(y, d).Real)}, nil
		}, true, nil
	}
	switch u.Name {
	case "minimum", "maximum", "fmin", "fmax":
		maximum := u.Name == "maximum" || u.Name == "fmax"
		skip := u.Name == "fmin" || u.Name == "fmax"
		return func(x, y element) (element, error) { return selectExtrema(x, y, d, maximum, skip), nil }, true, nil
	case "logical_and", "logical_or", "logical_xor":
		return func(x, y element) (element, error) {
			a, b := x.truth(d), y.truth(d)
			switch u.Name {
			case "logical_and":
				return boolElement(a && b), nil
			case "logical_or":
				return boolElement(a || b), nil
			default:
				return boolElement(a != b), nil
			}
		}, true, nil
	case "gcd", "lcm":
		return func(x, y element) (element, error) {
			a, b := absInteger(x.i, d), absInteger(y.i, d)
			g := integerGCD(a, b)
			if u.Name == "lcm" && g != 0 {
				g = (a / g) * b
			}
			return element{i: g}, nil
		}, true, nil
	case "hypot":
		return func(x, y element) (element, error) {
			return element{z: complex(math.Hypot(real(x.z), real(y.z)), 0)}, nil
		}, true, nil
	case "logaddexp", "logaddexp2":
		return func(x, y element) (element, error) {
			a, b := real(x.z), real(y.z)
			if u.Name == "logaddexp2" {
				if a == b {
					return element{z: complex(a+1, 0)}, nil
				}
				return element{z: complex(math.Max(a, b)+math.Log1p(math.Exp2(-math.Abs(a-b)))/math.Ln2, 0)}, nil
			}
			if a == b {
				return element{z: complex(a+math.Ln2, 0)}, nil
			}
			return element{z: complex(math.Max(a, b)+math.Log1p(math.Exp(-math.Abs(a-b))), 0)}, nil
		}, true, nil
	default:
		return nil, false, ErrValue
	}
}

func (u *UFunc) ReduceWithOptions(a *Array, opt ReduceOptions) (*Array, error) {
	if u == nil || u.NIn != 2 {
		return nil, ErrValue
	}
	if a.DType().IsTemporal() {
		return u.reduceTemporal(a, opt)
	}
	if !a.DType().IsNativeNumeric() && !a.DType().IsExtended() {
		return nil, ErrDType
	}
	axes := reduceAxes(a, opt)
	// NumPy accepts axis 0/-1 for scalar reductions.
	if a.NDim() == 0 && len(axes) == 1 && (axes[0] == 0 || axes[0] == -1) {
		axes = []int{}
	}
	r, _, e := reductionAny(a, []SumOptions{{Axes: axes, KeepDims: opt.KeepDims}})
	if e != nil {
		return nil, e
	}
	requested := opt.DType
	if requested == 0 && opt.Out != nil && (u.Name == "add" || u.Name == "multiply") {
		requested = opt.Out.DType()
	}
	d, e := u.reductionType(a, requested)
	if e != nil {
		return nil, e
	}
	acc := d
	if d == Float16 && (u.Name == "add" || u.Name == "multiply") {
		acc = Float32
	}
	step, reorder, e := u.reductionKernel(acc)
	if e != nil {
		return nil, e
	}
	if !reorder && len(r.reduceAxes) > 1 {
		return nil, ErrValue
	}
	start := opt.Initial
	if start == nil && u.HasIdentity {
		start = u.Identity
	}
	if opt.Where != nil && start == nil {
		return nil, ErrValue
	}
	var seed element
	have := start != nil
	if have {
		from := dtypeOf(start)
		if from == 0 {
			return nil, ErrDType
		}
		seed = castElement(castElement(elementFrom(start), from, d), d, acc)
	}
	out, e := Zeros(d, r.shape...)
	if e != nil {
		return nil, e
	}
	if opt.Out != nil {
		if !opt.Out.Writable() {
			return nil, ErrReadOnly
		}
		if !slices.Equal(opt.Out.shape, r.shape) {
			return nil, ErrShape
		}
	}
	var mask *Array
	var mr reductionLayout
	if opt.Where != nil {
		if opt.Where.DType() != Bool {
			return nil, ErrDType
		}
		mask, e = opt.Where.BroadcastTo(a.shape...)
		if e != nil {
			return nil, e
		}
		mr = r.companion(mask, opt.KeepDims)
	}
	for i := 0; i < out.size; i++ {
		base := r.base(a, i)
		mbase := 0
		if mask != nil {
			mbase = mr.base(mask, i)
		}
		at := func(j int) (element, bool) {
			if mask != nil && mask.elementAt(mr.offset(mbase, j)).i == 0 {
				return element{}, false
			}
			return castElement(castElement(a.elementAt(r.offset(base, j)), a.DType(), d), d, acc), true
		}
		z, e := u.fold(r.count, at, seed, have, acc, step, r.pairwise)
		if e != nil {
			return nil, e
		}
		out.setElement(i, castElement(z, acc, d))
	}
	if opt.Out != nil {
		if e = CopyTo(opt.Out, out, nil); e != nil {
			return nil, e
		}
		return opt.Out, nil
	}
	return out, nil
}

func reduceAxes(a *Array, opt ReduceOptions) []int {
	if opt.AllAxes {
		return nil
	}
	if opt.Axes == nil && a.NDim() > 0 {
		return []int{0}
	}
	return opt.Axes
}

func (u *UFunc) reduceTemporal(a *Array, opt ReduceOptions) (*Array, error) {
	d := a.DType()
	add := u.Name == "add" && d == TimeDelta64
	maximum := u.Name == "maximum" || u.Name == "fmax"
	skip := u.Name == "fmin" || u.Name == "fmax"
	if !add && !maximum && u.Name != "minimum" && u.Name != "fmin" {
		return nil, ErrDType
	}
	if opt.DType != 0 && opt.DType != d {
		return nil, ErrDType
	}
	axes := reduceAxes(a, opt)
	if a.NDim() == 0 && len(axes) == 1 && (axes[0] == 0 || axes[0] == -1) {
		axes = []int{}
	}
	r, _, e := reductionAny(a, []SumOptions{{Axes: axes, KeepDims: opt.KeepDims}})
	if e != nil {
		return nil, e
	}
	var mask *Array
	var mr reductionLayout
	if opt.Where != nil {
		if opt.Where.DType() != Bool {
			return nil, ErrDType
		}
		if !add && opt.Initial == nil {
			return nil, ErrValue
		}
		mask, e = opt.Where.BroadcastTo(a.shape...)
		if e != nil {
			return nil, e
		}
		mr = r.companion(mask, opt.KeepDims)
	}
	seed, have := int64(0), add
	if opt.Initial != nil {
		v, err := convertTemporal(opt.Initial, d, a.data.unit)
		if err != nil {
			return nil, err
		}
		seed, have = temporalTicks(v), true
	}
	out, e := a.zerosLike(r.shape)
	if e != nil {
		return nil, e
	}
	for i := 0; i < out.size; i++ {
		base := r.base(a, i)
		mb := 0
		if mask != nil {
			mb = mr.base(mask, i)
		}
		z, seen := seed, have
		for j := 0; j < r.count; j++ {
			if mask != nil && mask.elementAt(mr.offset(mb, j)).i == 0 {
				continue
			}
			v := temporalTicks(a.elementAt(r.offset(base, j)))
			if !seen {
				z, seen = v, true
				continue
			}
			if add {
				if z == NaT || v == NaT {
					z = NaT
				} else {
					z += v
				}
			} else if z == NaT || v == NaT {
				if !skip {
					z = NaT
				} else if z == NaT {
					z = v
				}
			} else if maximum && v > z || !maximum && v < z {
				z = v
			}
		}
		if !seen {
			return nil, ErrValue
		}
		out.setElement(i, temporalElement(d, z, a.data.unit))
	}
	if opt.Out != nil {
		if !slices.Equal(opt.Out.shape, out.shape) {
			return nil, ErrShape
		}
		if e = CopyTo(opt.Out, out, nil); e != nil {
			return nil, e
		}
		return opt.Out, nil
	}
	return out, nil
}

func (u *UFunc) fold(n int, at func(int) (element, bool), seed element, have bool, d DType, step reduceKernel, pairwise bool) (element, error) {
	if u.Name == "add" && have {
		z := sumGroup(n, func(j int) element { v, _ := at(j); return v }, d, pairwise)
		return roundElement(addScalar(seed, z, d), d), nil
	}
	z := seed
	for j := 0; j < n; j++ {
		v, ok := at(j)
		if !ok {
			continue
		}
		if !have {
			z = v
			have = true
			continue
		}
		x, e := step(z, v)
		if e != nil {
			return element{}, e
		}
		z = roundElement(x, d)
	}
	if !have {
		return element{}, ErrValue
	}
	return z, nil
}

func (u *UFunc) accumulate(a *Array, axis int, dtype DType) (*Array, error) {
	d, e := u.reductionType(a, dtype)
	if e != nil {
		return nil, e
	}
	ax, e := NormalizeAxis(axis, a.NDim())
	if e != nil {
		return nil, e
	}
	step, _, e := u.reductionKernel(d)
	if e != nil {
		return nil, e
	}
	out, e := Zeros(d, a.shape...)
	if e != nil {
		return nil, e
	}
	r, _, e := reductionAny(a, []SumOptions{{Axes: []int{ax}}})
	if e != nil {
		return nil, e
	}
	groups, e := shapeSize(d, r.shape)
	if e != nil {
		return nil, e
	}
	or := r.companion(out, false)
	for i := 0; i < groups; i++ {
		base, obase := r.base(a, i), or.base(out, i)
		var z element
		for j := 0; j < r.count; j++ {
			v := castElement(a.elementAt(r.offset(base, j)), a.DType(), d)
			if j == 0 {
				z = v
			} else {
				z, e = step(z, v)
				if e != nil {
					return nil, e
				}
				z = roundElement(z, d)
			}
			out.setElement(or.offset(obase, j), z)
		}
	}
	return out, nil
}

// ReduceAt reduces consecutive slices along one axis. A repeated or
// decreasing next index selects just the indexed element.
func (u *UFunc) ReduceAt(a *Array, indices []int, axis int, dtype DType, out *Array) (*Array, error) {
	d, e := u.reductionType(a, dtype)
	if e != nil {
		return nil, e
	}
	ax, e := NormalizeAxis(axis, a.NDim())
	if e != nil {
		return nil, e
	}
	for _, i := range indices {
		if i < 0 || i >= a.shape[ax] {
			return nil, ErrIndex
		}
	}
	acc := d
	if d == Float16 && (u.Name == "add" || u.Name == "multiply") {
		acc = Float32
	}
	step, _, e := u.reductionKernel(acc)
	if e != nil {
		return nil, e
	}
	shape := slices.Clone(a.shape)
	shape[ax] = len(indices)
	result, e := Zeros(d, shape...)
	if e != nil {
		return nil, e
	}
	if out != nil {
		if !out.Writable() {
			return nil, ErrReadOnly
		}
		if !slices.Equal(out.shape, shape) {
			return nil, ErrShape
		}
	}
	r, _, e := reductionAny(a, []SumOptions{{Axes: []int{ax}}})
	if e != nil {
		return nil, e
	}
	or := r.companion(result, false)
	groups, e := shapeSize(d, r.shape)
	if e != nil {
		return nil, e
	}
	for i := 0; i < groups; i++ {
		base, obase := r.base(a, i), or.base(result, i)
		for k, start := range indices {
			end := a.shape[ax]
			if k+1 < len(indices) {
				end = max(start+1, indices[k+1])
			}
			at := func(j int) (element, bool) {
				return castElement(castElement(a.elementAt(r.offset(base, start+j)), a.DType(), d), d, acc), true
			}
			z, e := u.fold(end-start, at, element{}, false, acc, step, r.pairwise)
			if e != nil {
				return nil, e
			}
			result.setElement(obase+k*result.strides[ax], castElement(z, acc, d))
		}
	}
	if out != nil {
		if e = CopyTo(out, result, nil); e != nil {
			return nil, e
		}
		return out, nil
	}
	return result, nil
}
