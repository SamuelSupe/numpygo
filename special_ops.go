package np

import (
	"math"
	"reflect"
	"strings"

	"github.com/SamuelSupe/numpygo/internal/x87"
)

func extendedArithmetic(x, y element, d DType, o op) (element, error) {
	a, b := extendedValue(x, d), extendedValue(y, d)
	var z x87.Complex
	if d == Float128 {
		switch o {
		case addOp:
			z.Real = a.Real.Add(b.Real)
		case subOp:
			z.Real = a.Real.Sub(b.Real)
		case mulOp:
			z.Real = a.Real.Mul(b.Real)
		case divOp:
			z.Real = a.Real.Div(b.Real)
		case floorOp:
			z.Real, _ = a.Real.Divmod(b.Real)
		case remOp:
			_, z.Real = a.Real.Divmod(b.Real)
		case powOp:
			z.Real = a.Real.Pow(b.Real)
		default:
			return element{}, ErrDType
		}
	} else {
		switch o {
		case addOp:
			z = a.Add(b)
		case subOp:
			z = a.Sub(b)
		case mulOp:
			z = a.Mul(b)
		case divOp:
			z = a.Div(b)
		case powOp:
			z = a.Pow(b)
		default:
			return element{}, ErrDType
		}
	}
	return extendedElement(z, d), nil
}

func extendedVariance(a *Array, skip, sqrt bool, opt VarOptions) (*Array, error) {
	d := a.DType()
	if !d.IsExtended() {
		d = opt.DType
		v, e := a.Astype(d)
		if e != nil {
			return nil, e
		}
		a = v
	}
	outType := realDType(d)
	if opt.DType != 0 && opt.DType != outType {
		return nil, ErrDType
	}
	p, e := prepareVariance(a, opt)
	if e != nil {
		return nil, e
	}
	r := p.r
	out, e := Zeros(outType, r.shape...)
	if e != nil {
		return nil, e
	}
	for i := 0; i < out.size; i++ {
		base := r.base(a, i)
		keep := p.maskAt(i)
		mu, count := p.mean(a, i, skip, d)
		center := extendedValue(mu, d)
		var sum x87.Float
		for j := 0; j < r.count; j++ {
			x := a.elementAt(r.offset(base, j))
			if !keep(j) || skip && nanElement(x, d) {
				continue
			}
			delta := extendedValue(x, d).Sub(center)
			sq := delta.Real.Mul(delta.Real)
			if d == Complex256 {
				sq = sq.Add(delta.Imag.Mul(delta.Imag))
			}
			sum = sum.Add(sq)
		}
		den := math.Max(float64(count)-opt.DDOF, 0)
		sum = sum.Div(x87.FromFloat64(den))
		if count == 0 || skip && den == 0 {
			sum = x87.NaN()
		}
		if sqrt {
			sum = sum.Sqrt()
		}
		out.setElement(i, element{extra: sum})
	}
	return reductionOutput(opt.Out, out)
}

func specialCumulative(a *Array, axis int, d DType, product, skip bool) (*Array, error) {
	if a.DType() != d {
		v, e := a.Astype(d)
		if e != nil {
			return nil, e
		}
		return cumulative(v, &axis, d, product, skip)
	}
	if d == RecordDType || d == DateTime64 || d == TimeDelta64 && product {
		return nil, ErrDType
	}
	out, e := a.zerosLike(a.shape)
	if e != nil {
		return nil, e
	}
	if out.size == 0 {
		return out, nil
	}
	r, _, e := reductionAny(a, []SumOptions{{Axes: []int{axis}}})
	if e != nil {
		return nil, e
	}
	strides := append(append([]int(nil), out.strides[:axis]...), out.strides[axis+1:]...)
	o := addOp
	identity := uint64(0)
	if product {
		o = mulOp
		identity = 1
	}
	for i := 0; i < out.size/a.shape[axis]; i++ {
		base, obase := r.base(a, i), flatOffset(i, r.shape, strides)
		z := castElement(element{i: identity}, Int64, d)
		if d == TimeDelta64 {
			z = temporalElement(d, 0, a.data.unit)
		}
		have := false
		for j := 0; j < r.count; j++ {
			x := a.elementAt(r.offset(base, j))
			if !skip || !nanElement(x, d) {
				switch d {
				case TimeDelta64:
					p, q := temporalTicks(z), temporalTicks(x)
					if p == NaT || q == NaT {
						p = NaT
					} else {
						p += q
					}
					z = temporalElement(d, p, a.data.unit)
				case Object:
					if !have {
						z = x
					} else {
						v, err := objectBinary(z.extra, x.extra, o)
						if err != nil {
							return nil, err
						}
						z = element{extra: v}
					}
				default:
					z, e = extendedArithmetic(z, x, d, o)
					if e != nil {
						return nil, e
					}
				}
				have = true
			}
			out.setElement(obase+j*out.strides[axis], z)
		}
	}
	return out, nil
}
func extendedLess(x, y element, d DType) bool {
	a, b := extendedValue(x, d), extendedValue(y, d)
	c, ordered := a.Real.Compare(b.Real)
	if !ordered {
		return false
	}
	if c != 0 {
		return c < 0
	}
	if d == Complex256 {
		c, ordered = a.Imag.Compare(b.Imag)
		return ordered && c < 0
	}
	return false
}
func objectNumeric(a, b any) (element, element, DType, bool) {
	ad, bd := dtypeOf(a), dtypeOf(b)
	if ad == 0 || bd == 0 || ad.IsTemporal() || bd.IsTemporal() || ad.IsFlexible() || bd.IsFlexible() {
		return element{}, element{}, 0, false
	}
	d, e := PromoteTypes(ad, bd)
	if e != nil {
		return element{}, element{}, 0, false
	}
	return castElement(elementFrom(a), ad, d), castElement(elementFrom(b), bd, d), d, true
}
func objectBinary(a, b any, o op) (any, error) {
	if x, y, d, ok := objectNumeric(a, b); ok {
		if o == divOp && (d.IsInteger() || d == Bool) {
			x, y, d = castElement(x, d, Float64), castElement(y, d, Float64), Float64
		}
		if (o == divOp || o == floorOp || o == remOp) && !y.truth(d) {
			return nil, ErrValue
		}
		z, e := arithmeticScalar(x, y, d, o)
		if e != nil {
			return nil, e
		}
		return scalarFromElement(roundElement(z, d), d), nil
	}
	if x, ok := a.(string); ok {
		if y, ok := b.(string); ok && o == addOp {
			return x + y, nil
		}
		if o == mulOp && dtypeOf(b).IsInteger() {
			n := int64(elementFrom(b).i)
			if n <= 0 || len(x) == 0 {
				return "", nil
			}
			if n > int64(maxInt/len(x)) {
				return nil, ErrShape
			}
			return strings.Repeat(x, int(n)), nil
		}
	}
	if _, ok := b.(string); ok && o == mulOp && dtypeOf(a).IsInteger() {
		return objectBinary(b, a, o)
	}
	return nil, ErrDType
}
func objectEqual(a, b any) bool {
	if x, y, d, ok := objectNumeric(a, b); ok {
		return equalElement(x, y, d)
	}
	return reflect.DeepEqual(a, b)
}
func objectLess(a, b any) bool {
	if x, y, d, ok := objectNumeric(a, b); ok {
		return lessElement(x, y, d)
	}
	x, xok := a.(string)
	y, yok := b.(string)
	return xok && yok && x < y
}
func objectOrderable(a, b any) bool {
	if _, _, d, ok := objectNumeric(a, b); ok {
		return !d.IsComplex()
	}
	_, x := a.(string)
	_, y := b.(string)
	return x && y
}
func specialArithmetic(a, b *Array, o op) (*Array, error) {
	ad, bd := a.DType(), b.DType()
	if o == mulOp && (ad == StringDType || bd == StringDType) {
		return stringRepeat(a, b)
	}
	if ad == StringDType || bd == StringDType {
		if o != addOp {
			return nil, ErrDType
		}
		a, b, e := stringPair(a, b)
		if e != nil {
			return nil, e
		}
		r, e := binaryMapCore(a, b, StringDType, StringDType, stringAdd)
		if e == nil {
			r.setStringDescriptor(a.stringDescriptor())
		}
		return r, e
	}
	if (ad.IsFlexible() || bd.IsFlexible()) && ad != Object && bd != Object {
		return flexAdd(a, b, o)
	}
	if ad.IsTemporal() || bd.IsTemporal() {
		return temporalArithmetic(a, b, o)
	}
	d, e := PromoteTypes(ad, bd)
	if e != nil {
		return nil, e
	}
	if d == Object {
		a, e = a.Astype(Object)
		if e != nil {
			return nil, e
		}
		b, e = b.Astype(Object)
		if e != nil {
			return nil, e
		}
		return binaryMapCore(a, b, Object, Object, func(x, y element) (element, error) {
			v, e := objectBinary(x.extra, y.extra, o)
			return element{extra: v}, e
		})
	}
	if !d.IsExtended() || o > powOp || d.IsComplex() && (o == floorOp || o == remOp) {
		return nil, ErrDType
	}
	return binaryMapCore(a, b, d, d, func(x, y element) (element, error) { return extendedArithmetic(x, y, d, o) })
}
func specialComparison(a, b *Array, ordered bool, fn func(element, element, DType) bool) (*Array, error) {
	ad, bd := a.DType(), b.DType()
	if ad == StringDType || bd == StringDType {
		a, b, e := stringPair(a, b)
		if e != nil {
			return nil, e
		}
		return binaryMapCore(a, b, StringDType, Bool, func(x, y element) (element, error) {
			if stringNaN(x) || stringNaN(y) {
				return boolElement(false), nil
			}
			if ordered && (stringBadOrder(x) || stringBadOrder(y)) {
				return element{}, ErrValue
			}
			return boolElement(fn(x, y, StringDType)), nil
		})
	}
	if (ad.IsFlexible() || bd.IsFlexible()) && ad != Object && bd != Object {
		return flexCompare(a, b, ordered, fn)
	}
	if ad.IsTemporal() || bd.IsTemporal() {
		if ad != bd {
			return nil, ErrDType
		}
		u, e := commonDateUnit(a.data.unit, b.data.unit, ad == DateTime64)
		if e != nil {
			return nil, e
		}
		a, e = a.CastTemporal(ad, u.String())
		if e != nil {
			return nil, e
		}
		b, e = b.CastTemporal(bd, u.String())
		if e != nil {
			return nil, e
		}
		return temporalMap(a, b, u, Bool, func(x, y element) (element, error) { return boolElement(fn(x, y, ad)), nil })
	}
	if ad == RecordDType || bd == RecordDType {
		if ordered || ad != bd {
			return nil, ErrDType
		}
		if a.HasObject() || b.HasObject() {
			if !a.sameDescriptor(b) {
				if _, e := PromoteRecordSchemas(a.data.schema, b.data.schema); e != nil {
					return nil, e
				}
			}
			return binaryMapCore(a, b, 0, Bool, func(x, y element) (element, error) {
				r := decodeRecordElement(a.data.schema, x)
				s := decodeRecordElement(b.data.schema, y)
				c, ok := r.(interface{ EqualRecord(any) (bool, error) })
				if !ok {
					return element{}, ErrDType
				}
				v, e := c.EqualRecord(s)
				return boolElement(fn(boolElement(v), boolElement(true), Bool)), e
			})
		}
		if !a.sameDescriptor(b) {
			if _, e := PromoteRecordSchemas(a.data.schema, b.data.schema); e != nil {
				return nil, e
			}
			s, ok := a.data.schema.(interface {
				EqualScalarOther([]byte, StructuredSchema, []byte) (bool, error)
			})
			if !ok {
				return nil, ErrDType
			}
			return binaryMapCore(a, b, 0, Bool, func(x, y element) (element, error) {
				r, e := s.EqualScalarOther(x.extra.([]byte), b.data.schema, y.extra.([]byte))
				return boolElement(fn(boolElement(r), boolElement(true), Bool)), e
			})
		}
		return binaryMapCore(a, b, 0, Bool, func(x, y element) (element, error) {
			r, e := a.data.schema.EqualScalar(x.extra.([]byte), y.extra.([]byte))
			return boolElement(fn(boolElement(r), boolElement(true), Bool)), e
		})
	}
	d, e := PromoteTypes(ad, bd)
	if e != nil {
		return nil, e
	}
	if d == Object {
		a, e = a.Astype(Object)
		if e != nil {
			return nil, e
		}
		b, e = b.Astype(Object)
		if e != nil {
			return nil, e
		}
	}
	return binaryMapCore(a, b, d, Bool, func(x, y element) (element, error) {
		if d == Object && ordered && !objectOrderable(x.extra, y.extra) {
			return element{}, ErrDType
		}
		return boolElement(fn(x, y, d)), nil
	})
}
func mapSpecialUnary(a *Array, d DType, fn func(element) (element, error)) (*Array, error) {
	var out *Array
	var e error
	if d == a.DType() {
		out, e = a.zerosLike(a.shape)
	} else {
		out, e = Zeros(d, a.shape...)
	}
	if e != nil {
		return nil, e
	}
	a.walk(func(i, off int) {
		if e != nil {
			return
		}
		v, err := fn(a.elementAt(off))
		if err != nil {
			e = err
			return
		}
		out.setElement(i, v)
	})
	if e != nil {
		return nil, e
	}
	return out, nil
}
func specialUnary(a *Array, name string) (*Array, error) {
	d := a.DType()
	if d == StringDType {
		if name == "isnan" {
			return mapSpecialUnary(a, Bool, func(x element) (element, error) { return boolElement(stringNaN(x)), nil })
		}
		return nil, ErrDType
	}
	if d.IsFlexible() {
		return nil, ErrDType
	}
	if d.IsTemporal() {
		switch name {
		case "isnan":
			return IsNaT(a)
		case "isinf":
			return Zeros(Bool, a.shape...)
		case "isfinite":
			v, e := IsNaT(a)
			if e != nil {
				return nil, e
			}
			return LogicalNot(v)
		case "negative", "absolute":
			if d != TimeDelta64 {
				return nil, ErrDType
			}
			return mapSpecialUnary(a, d, func(e element) (element, error) {
				v := temporalTicks(e)
				if v != NaT && (name == "negative" || v < 0) {
					v = -v
				}
				return temporalElement(d, v, a.data.unit), nil
			})
		}
	}
	if name == "logical_not" {
		return mapSpecialUnary(a, Bool, func(x element) (element, error) { return boolElement(!x.truth(d)), nil })
	}
	if !d.IsExtended() {
		return nil, ErrDType
	}
	out := d
	switch name {
	case "absolute":
		out = Float128
	case "isnan", "isinf", "isfinite", "signbit":
		out = Bool
	}
	if name == "signbit" && d == Complex256 {
		return nil, ErrDType
	}
	return mapSpecialUnary(a, out, func(x element) (element, error) {
		z := extendedValue(x, d)
		switch name {
		case "negative":
			z.Real = z.Real.Neg()
			z.Imag = z.Imag.Neg()
		case "conjugate":
			z = z.Conj()
		case "absolute":
			if d == Complex256 {
				z.Real = z.Abs()
			} else {
				z.Real = z.Real.Abs()
			}
		case "reciprocal":
			if d == Float128 {
				z.Real = x87.FromInt64(1).Div(z.Real)
			} else {
				z = (x87.Complex{Real: x87.FromInt64(1)}).Div(z)
			}
		case "isnan":
			return boolElement(z.Real.IsNaN() || z.Imag.IsNaN()), nil
		case "isinf":
			return boolElement(z.Real.IsInf() || z.Imag.IsInf()), nil
		case "isfinite":
			return boolElement(z.Real.IsFinite() && z.Imag.IsFinite()), nil
		case "signbit":
			return boolElement(z.Real.Signbit()), nil
		default:
			return element{}, ErrDType
		}
		return extendedElement(z, out), nil
	})
}

func specialReduce(a *Array, options []SumOptions, kind string, skip bool) (*Array, error) {
	d := a.DType()
	if d == StringDType {
		switch kind {
		case "sum", "min", "max", "all", "any", "count":
		default:
			return nil, ErrDType
		}
	}
	if d.IsFlexible() && kind != "all" && kind != "any" && kind != "count" {
		return nil, ErrDType
	}
	if d == RecordDType {
		return nil, ErrDType
	}
	if d == DateTime64 && (kind == "sum" || kind == "prod" || kind == "mean") {
		return nil, ErrDType
	}
	if d == TimeDelta64 && kind == "prod" {
		return nil, ErrDType
	}
	r, opt, e := reductionAny(a, options)
	if e != nil {
		return nil, e
	}
	outType := d
	if kind == "all" || kind == "any" {
		outType = Bool
	} else if kind == "count" {
		outType = Int64
	} else if opt.DType != 0 {
		outType = opt.DType
	}
	if outType != d && kind != "all" && kind != "any" && kind != "count" {
		if kind == "min" || kind == "max" {
			return nil, ErrDType
		}
		v, e := a.Astype(outType)
		if e != nil {
			return nil, e
		}
		switch kind {
		case "sum":
			if skip {
				return NanSum(v, options...)
			}
			return Sum(v, options...)
		case "prod":
			if skip {
				return NanProd(v, options...)
			}
			return Prod(v, options...)
		case "mean":
			if skip {
				return NanMean(v, options...)
			}
			return Mean(v, options...)
		}
	}
	var out *Array
	if outType == d {
		out, e = a.zerosLike(r.shape)
	} else {
		out, e = Zeros(outType, r.shape...)
	}
	if e != nil {
		return nil, e
	}
	for i := 0; i < out.size; i++ {
		base := r.base(a, i)
		var v element
		have := false
		count := 0
		switch kind {
		case "all":
			v = boolElement(true)
		case "sum", "mean", "prod":
			identity := int64(0)
			if kind == "prod" {
				identity = 1
			}
			if d == StringDType {
				v = element{extra: ""}
			} else if d.IsTemporal() {
				v = temporalElement(d, 0, a.data.unit)
			} else {
				v = castElement(element{i: uint64(identity)}, Int64, d)
			}
		}
		for j := 0; j < r.count; j++ {
			x := a.elementAt(r.offset(base, j))
			if skip && nanElement(x, d) {
				continue
			}
			count++
			switch kind {
			case "all":
				v = boolElement(v.i != 0 && x.truth(d))
			case "any":
				v = boolElement(v.i != 0 || x.truth(d))
			case "count":
				if x.truth(d) {
					v.i++
				}
			case "min", "max":
				if !have {
					v = x
					have = true
				} else {
					if d == Object && !objectOrderable(v.extra, x.extra) {
						return nil, ErrDType
					}
					if d == StringDType && (stringBadOrder(v) || stringBadOrder(x)) {
						return nil, ErrValue
					}
					v = selectExtrema(v, x, d, kind == "max", skip || d == StringDType && kind == "min")
				}
			case "sum", "mean", "prod":
				o := addOp
				if kind == "prod" {
					o = mulOp
				}
				if d == TimeDelta64 {
					p, q := temporalTicks(v), temporalTicks(x)
					if p == NaT || q == NaT {
						p = NaT
					} else {
						p += q
					}
					v = temporalElement(d, p, a.data.unit)
				} else if d == StringDType {
					v, e = stringAdd(v, x)
					if e != nil {
						return nil, e
					}
				} else if d == Object {
					if !have && kind != "prod" {
						v = x
					} else {
						z, e := objectBinary(v.extra, x.extra, o)
						if e != nil {
							return nil, e
						}
						v = element{extra: z}
					}
					have = true
				} else {
					v, e = extendedArithmetic(v, x, d, o)
					if e != nil {
						return nil, e
					}
				}
			}
		}
		if (kind == "min" || kind == "max") && !have {
			if r.count == 0 {
				return nil, ErrValue
			}
			v = a.elementAt(base)
		}
		if kind == "mean" {
			if d == TimeDelta64 {
				p := temporalTicks(v)
				if p != NaT && count > 0 {
					p /= int64(count)
				} else {
					p = NaT
				}
				v = temporalElement(d, p, a.data.unit)
			} else if d == Object {
				if count == 0 {
					v = element{extra: math.NaN()}
				} else {
					x, e := objectBinary(v.extra, float64(count), divOp)
					if e != nil {
						return nil, e
					}
					v = element{extra: x}
				}
			} else {
				q := castElement(element{i: uint64(count)}, Int64, d)
				v, e = extendedArithmetic(v, q, d, divOp)
				if e != nil {
					return nil, e
				}
			}
		}
		out.setElement(i, v)
	}
	return out, nil
}
