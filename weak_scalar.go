package np

import (
	"errors"
	"math"
	"math/big"
	"slices"
)

// ErrOverflow corresponds to a Python-style weak integer that cannot be
// represented in the selected numerical loop. Strong typed casts retain
// their existing explicit wrapping behavior.
var ErrOverflow = errors.New("scalar outside dtype range")

// WeakScalar opts into NumPy 2.x Python-scalar promotion. Ordinary Go
// scalars and ScalarOf remain strongly typed. Integer values can be *big.Int.
type WeakScalar struct{ Value any }

func Weak(value any) WeakScalar { return WeakScalar{value} }

type weakValue struct {
	kind    int // bool, integer, real, complex
	integer *big.Int
	z       complex128
}

func dtypeKind(d DType) int {
	switch {
	case d.IsComplex():
		return 3
	case d.IsFloat():
		return 2
	case d.IsInteger():
		return 1
	default:
		return 0
	}
}
func defaultWeakDType(kind int) DType { return []DType{Bool, Int64, Float64, Complex128}[kind] }
func readWeak(w WeakScalar) (weakValue, error) {
	if v, ok := w.Value.(*big.Int); ok {
		if v == nil {
			return weakValue{}, ErrDType
		}
		return weakValue{kind: 1, integer: new(big.Int).Set(v)}, nil
	}
	d := dtypeOf(w.Value)
	if d == 0 {
		return weakValue{}, ErrDType
	}
	e := elementFrom(w.Value)
	v := weakValue{kind: dtypeKind(d)}
	if d.IsSigned() {
		v.integer = big.NewInt(int64(e.i))
	} else if d.IsUnsigned() {
		v.integer = new(big.Int).SetUint64(e.i)
	} else {
		v.z = e.number(d)
	}
	return v, nil
}
func weakPromotion(strong DType, kind int) DType {
	if strong == 0 {
		return defaultWeakDType(kind)
	}
	if kind <= dtypeKind(strong) {
		return strong
	}
	if kind == 3 && (strong == Float16 || strong == Float32) {
		return Complex64
	}
	return defaultWeakDType(kind)
}
func integerBounds(d DType) (*big.Int, *big.Int) {
	bits := uint(d.ItemSize() * 8)
	if d.IsSigned() {
		max := new(big.Int).Lsh(big.NewInt(1), bits-1)
		min := new(big.Int).Neg(new(big.Int).Set(max))
		return min, max.Sub(max, big.NewInt(1))
	}
	return big.NewInt(0), new(big.Int).Sub(new(big.Int).Lsh(big.NewInt(1), bits), big.NewInt(1))
}
func (v weakValue) array(d DType, rule string) (*Array, error) {
	if d.ItemSize() == 0 {
		return nil, ErrDType
	}
	if v.kind > dtypeKind(d) && rule != "unsafe" {
		return nil, ErrDType
	}
	if v.integer != nil && d.IsInteger() {
		lo, hi := integerBounds(d)
		if v.integer.Cmp(lo) < 0 || v.integer.Cmp(hi) > 0 {
			return nil, ErrOverflow
		}
		if d.IsSigned() {
			return Full(d, v.integer.Int64())
		}
		return Full(d, v.integer.Uint64())
	}
	z := v.z
	if v.integer != nil {
		x, _ := new(big.Float).SetInt(v.integer).Float64()
		if math.IsInf(x, 0) {
			return nil, ErrOverflow
		}
		z = complex(x, 0)
	}
	if d == Bool {
		return Full(d, z != 0)
	}
	if d.IsComplex() {
		return Full(d, z)
	}
	return Full(d, real(z))
}

// CallValues accepts explicit Weak(...) Python-style scalar operands, *Array
// values and ordinary (strong) Go scalars. It uses the same ufunc options
// and output/masking semantics as CallWithOptions.
func (u *UFunc) CallValues(opt UFuncOptions, values ...any) (*Array, error) {
	if u == nil || len(values) != u.NIn {
		return nil, ErrShape
	}
	if opt.DType != 0 && opt.Signature != nil {
		return nil, ErrValue
	}
	args := make([]*Array, len(values))
	weak := make([]*weakValue, len(values))
	strong := DType(0)
	kind := 0
	for i, x := range values {
		if w, ok := x.(WeakScalar); ok {
			v, e := readWeak(w)
			if e != nil {
				return nil, e
			}
			weak[i] = &v
			kind = max(kind, v.kind)
			continue
		}
		a, ok := x.(*Array)
		if !ok {
			var e error
			a, e = Full(dtypeOf(x), x)
			if e != nil {
				return nil, e
			}
		}
		if a.DType() == 0 {
			return nil, ErrDType
		}
		args[i] = a
	}
	if !slices.ContainsFunc(weak, func(w *weakValue) bool { return w != nil }) {
		return u.CallWithOptions(opt, args...)
	}
	for _, a := range args {
		if a == nil {
			continue
		}
		var e error
		if strong == 0 {
			strong = a.DType()
		} else if strong, e = PromoteTypes(strong, a.DType()); e != nil {
			return nil, e
		}
	}
	base := weakPromotion(strong, kind)
	logical := u.Name == "logical_and" || u.Name == "logical_or" || u.Name == "logical_xor" || u.Name == "logical_not"
	for i, w := range weak {
		if w == nil {
			continue
		}
		d := base
		if u.Name == "ldexp" || logical {
			d = defaultWeakDType(w.kind)
		}
		var e error
		args[i], e = Zeros(d)
		if e != nil {
			return nil, e
		}
	}
	result := opt.DType
	var types []DType
	var e error
	if opt.Signature != nil {
		dtypes := make([]DType, u.NArgs())
		for i, a := range args {
			dtypes[i] = a.DType()
		}
		if opt.Out != nil {
			dtypes[u.NIn] = opt.Out.DType()
		}
		loop, e := u.ResolveDTypes(dtypes, u.callResolveOptions(opt.Casting, 0, opt.Signature))
		if e != nil {
			return nil, e
		}
		result, types = loop[u.NIn], loop[:u.NIn]
	} else {
		if result == 0 {
			result, e = u.resultType(args)
			if e != nil {
				return nil, e
			}
		}
		types, e = u.loopDTypes(args, result, opt.DType != 0)
		if e != nil {
			return nil, e
		}
	}
	for i, w := range weak {
		if w == nil {
			continue
		}
		// Logical ufuncs evaluate Python scalars in their own default
		// dtype rather than inheriting the other operand's numeric dtype.
		if logical {
			types[i] = defaultWeakDType(w.kind)
		}
		args[i], e = w.array(types[i], opt.Casting)
		if e == ErrOverflow && result == Bool && len(args) == 2 && weak[1-i] == nil && args[1-i].DType().IsInteger() && w.integer != nil {
			if v, ok := u.outOfRangeComparison(args[1-i].DType(), w.integer, i == 0); ok {
				return constantComparison(v, args[1-i], opt)
			}
		}
		if e != nil {
			return nil, e
		}
	}
	return u.CallWithOptions(opt, args...)
}

func (u *UFunc) outOfRangeComparison(d DType, scalar *big.Int, scalarFirst bool) (bool, bool) {
	lo, hi := integerBounds(d)
	order := 0
	if scalar.Cmp(lo) < 0 {
		order = 1
	} else if scalar.Cmp(hi) > 0 {
		order = -1
	} else {
		return false, false
	}
	if scalarFirst {
		order = -order
	}
	switch u.Name {
	case "equal":
		return false, true
	case "not_equal":
		return true, true
	case "less", "less_equal":
		return order < 0, true
	case "greater", "greater_equal":
		return order > 0, true
	default:
		return false, false
	}
}
func constantComparison(value bool, a *Array, opt UFuncOptions) (*Array, error) {
	if opt.DType != 0 && opt.DType != Bool {
		return nil, ErrDType
	}
	rule := opt.Casting
	if rule == "" {
		rule = "same_kind"
	}
	if _, e := CanCast(Bool, Bool, rule); e != nil {
		return nil, e
	}
	shapes := [][]int{a.shape}
	if opt.Where != nil {
		if opt.Where.DType() != Bool {
			return nil, ErrDType
		}
		shapes = append(shapes, opt.Where.shape)
	}
	if opt.Out != nil {
		shapes = append(shapes, opt.Out.shape)
	}
	shape, e := BroadcastShapes(shapes...)
	if e != nil {
		return nil, e
	}
	out := opt.Out
	if out == nil {
		out, e = Zeros(Bool, shape...)
		if e != nil {
			return nil, e
		}
	} else {
		if !out.Writable() {
			return nil, ErrReadOnly
		}
		if !slices.Equal(shape, out.shape) {
			return nil, ErrShape
		}
		ok, e := CanCast(Bool, out.DType(), rule)
		if e != nil {
			return nil, e
		}
		if !ok {
			return nil, ErrDType
		}
	}
	if e = CopyTo(out, ScalarOf(value), opt.Where); e != nil {
		return nil, e
	}
	return out, nil
}
