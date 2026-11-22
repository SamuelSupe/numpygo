package np

import (
	"math"
	"math/big"
	"math/cmplx"
	"slices"

	"github.com/SamuelSupe/numpygo/internal/x87"
)

// ArangeTyped avoids converting exact integer/extended endpoints through
// binary64. All three arguments are explicit typed Go numerical scalars.
func ArangeTyped(start, stop, step any, dtype DType) (*Array, error) {
	sd, ed, td := dtypeOf(start), dtypeOf(stop), dtypeOf(step)
	if sd == 0 || ed == 0 || td == 0 {
		return nil, ErrDType
	}
	d, e := PromoteTypes(sd, ed)
	if e != nil {
		return nil, e
	}
	d, e = PromoteTypes(d, td)
	if e != nil {
		return nil, e
	}
	if dtype == 0 {
		dtype = d
		if d.IsInteger() {
			dtype, _ = PromoteTypes(d, Int64)
		} else if d.IsFloat() {
			dtype, _ = PromoteTypes(d, Float64)
		}
	}
	if sd.IsInteger() && ed.IsInteger() && td.IsInteger() {
		asInt := func(v any, t DType) *big.Int {
			x := elementFrom(v).i
			if t.IsSigned() {
				return big.NewInt(int64(x))
			}
			return new(big.Int).SetUint64(x)
		}
		x, y, s := asInt(start, sd), asInt(stop, ed), asInt(step, td)
		if s.Sign() == 0 {
			return nil, ErrValue
		}
		delta := new(big.Int).Sub(y, x)
		if delta.Sign() != s.Sign() {
			return Zeros(dtype, 0)
		}
		n, rem := new(big.Int), new(big.Int)
		n.QuoRem(delta, s, rem)
		if rem.Sign() != 0 {
			n.Add(n, big.NewInt(1))
		}
		if !n.IsInt64() || n.Int64() > int64(maxInt/max(1, dtype.ItemSize())) {
			return nil, ErrShape
		}
		out, e := Zeros(dtype, int(n.Int64()))
		if e != nil {
			return nil, e
		}
		if dtype.IsInteger() {
			// Modulo conversion is independent of the magnitude/sign of
			// the input scalar, including the entire uint64 range.
			mask := new(big.Int).Sub(new(big.Int).Lsh(big.NewInt(1), 64), big.NewInt(1))
			first := new(big.Int).And(x, mask).Uint64()
			increment := new(big.Int).And(s, mask).Uint64()
			for i := 0; i < out.size; i++ {
				out.setElement(i, element{i: first + uint64(i)*increment})
			}
			return out, nil
		}
		first := castElement(elementFrom(start), sd, dtype)
		next, er := x87.Parse(new(big.Int).Add(x, s).String())
		if er != nil {
			return nil, er
		}
		second := castElement(element{extra: next}, Float128, dtype)
		increment, er := arithmeticScalar(second, first, dtype, subOp)
		if er != nil {
			return nil, er
		}
		for i := 0; i < out.size; i++ {
			j := castElement(element{i: uint64(i)}, Int64, dtype)
			out.setElement(i, addScalar(first, multiplyScalar(j, increment, dtype), dtype))
		}
		return out, nil
	}
	if d.IsExtended() && !d.IsComplex() || dtype == Float128 {
		asFloat := func(v any, t DType) x87.Float {
			return extendedValue(castElement(elementFrom(v), t, Float128), Float128).Real
		}
		x, y, s := asFloat(start, sd), asFloat(stop, ed), asFloat(step, td)
		if s.IsZero() || !s.IsFinite() || !x.IsFinite() || !y.IsFinite() {
			return nil, ErrValue
		}
		n := y.Sub(x).Div(s).Round("ceil")
		if n.Signbit() || n.IsZero() {
			return Zeros(dtype, 0)
		}
		if !n.IsFinite() || !n.Rat().Num().IsInt64() || n.Rat().Num().Int64() > int64(maxInt/max(1, dtype.ItemSize())) {
			return nil, ErrShape
		}
		out, e := Zeros(dtype, int(n.Rat().Num().Int64()))
		if e != nil {
			return nil, e
		}
		increment := x.Add(s).Sub(x)
		for i := 0; i < out.size; i++ {
			v := x.Add(increment.Mul(x87.FromInt64(int64(i))))
			out.setElement(i, castElement(element{extra: v}, Float128, dtype))
		}
		return out, nil
	}
	if d.IsComplex() || dtype.IsComplex() {
		return nil, ErrDType
	}
	return Arange(real(elementFrom(start).number(sd)), real(elementFrom(stop).number(ed)), real(elementFrom(step).number(td)), dtype)
}

type SpaceOptions struct {
	Num      int
	Axis     int
	DType    DType
	Endpoint *bool // nil retains the endpoint
}

// LinspaceArray broadcasts array endpoints and returns the spacing in the
// inferred calculation dtype, even when the requested samples are integers.
func LinspaceArray(start, stop *Array, opt SpaceOptions) (*Array, *Array, error) {
	if opt.Num < 0 {
		return nil, nil, ErrValue
	}
	d, e := PromoteTypes(start.DType(), stop.DType())
	if e != nil {
		return nil, nil, e
	}
	if d.IsInteger() || d == Bool {
		d = Float64
	}
	if !d.IsFloat() && !d.IsComplex() {
		return nil, nil, ErrDType
	}
	x, e := start.Astype(d)
	if e != nil {
		return nil, nil, e
	}
	y, e := stop.Astype(d)
	if e != nil {
		return nil, nil, e
	}
	shape, e := BroadcastShapes(x.shape, y.shape)
	if e != nil {
		return nil, nil, e
	}
	x, e = x.BroadcastTo(shape...)
	if e != nil {
		return nil, nil, e
	}
	y, e = y.BroadcastTo(shape...)
	if e != nil {
		return nil, nil, e
	}
	ax, e := NormalizeAxis(opt.Axis, len(shape)+1)
	if e != nil {
		return nil, nil, e
	}
	sh := slices.Insert(slices.Clone(shape), 0, opt.Num)
	rt := opt.DType
	if rt == 0 {
		rt = d
	}
	if !rt.IsNativeNumeric() && !rt.IsExtended() {
		return nil, nil, ErrDType
	}
	out, e := Zeros(rt, sh...)
	if e != nil {
		return nil, nil, e
	}
	step := Scalar(math.NaN())
	end := opt.Endpoint == nil || *opt.Endpoint
	div := opt.Num
	if end {
		div--
	}
	delta, e := Subtract(y, x)
	if e != nil {
		return nil, nil, e
	}
	if div > 0 {
		den, e := Full(d, float64(div))
		if e != nil {
			return nil, nil, e
		}
		step, e = Divide(delta, den)
		if e != nil {
			return nil, nil, e
		}
	}
	otherStrides := out.strides[1:]
	x.walk(func(i, off int) {
		base := flatOffset(i, shape, otherStrides)
		first := x.elementAt(off)
		last := y.elementAt(y.offset + flatOffset(i, shape, y.strides))
		diff := delta.elementAt(i)
		for j := 0; j < opt.Num; j++ {
			v := first
			if div > 0 {
				n := castElement(element{z: complex(float64(j), 0)}, Float64, d)
				s := step.elementAt(i)
				var inc element
				if !s.truth(d) {
					fraction := castElement(element{z: complex(float64(j)/float64(div), 0)}, Float64, d)
					inc = multiplyScalar(diff, fraction, d)
				} else {
					inc = multiplyScalar(s, n, d)
				}
				v = roundElement(addScalar(first, roundElement(inc, d), d), d)
			}
			if end && opt.Num > 1 && j == opt.Num-1 {
				v = last
			}
			if rt.IsInteger() {
				if d.IsExtended() {
					z := extendedValue(v, d)
					z.Real = z.Real.Round("floor")
					v = extendedElement(z, d)
				} else {
					v.z = complex(math.Floor(real(v.z)), imag(v.z))
				}
			}
			out.setElement(base+j*out.strides[0], castElement(v, d, rt))
		}
	})
	if ax != 0 {
		out, e = MoveAxis(out, 0, ax)
		if e != nil {
			return nil, nil, e
		}
	}
	return out, step, nil
}

func LogspaceArray(start, stop, base *Array, opt SpaceOptions) (*Array, error) {
	shape, e := BroadcastShapes(start.shape, stop.shape, base.shape)
	if e != nil {
		return nil, e
	}
	x, e := start.BroadcastTo(shape...)
	if e != nil {
		return nil, e
	}
	y, e := stop.BroadcastTo(shape...)
	if e != nil {
		return nil, e
	}
	b, e := base.BroadcastTo(shape...)
	if e != nil {
		return nil, e
	}
	calc := opt
	calc.DType = 0
	exponents, _, e := LinspaceArray(x, y, calc)
	if e != nil {
		return nil, e
	}
	b, e = ExpandDims(b, opt.Axis)
	if e != nil {
		return nil, e
	}
	result, e := Power(b, exponents)
	if e != nil || opt.DType == 0 {
		return result, e
	}
	return result.Astype(opt.DType)
}

// GeomspaceArray chooses the shortest complex logarithmic path, and preserves
// real negative sequences when both endpoints have the same sign.
func GeomspaceArray(start, stop *Array, opt SpaceOptions) (*Array, error) {
	if opt.Num < 0 {
		return nil, ErrValue
	}
	d, e := PromoteTypes(start.DType(), stop.DType())
	if e != nil {
		return nil, e
	}
	// NumPy includes a strong scalar of the requested dtype (float64
	// when omitted) when choosing the logarithmic calculation dtype.
	requested := opt.DType
	if requested == 0 {
		requested = Float64
	}
	d, e = PromoteTypes(d, requested)
	if e != nil {
		return nil, e
	}
	if d.IsInteger() || d == Bool {
		d = Float64
	}
	if !d.IsFloat() && !d.IsComplex() {
		return nil, ErrDType
	}
	shape, e := BroadcastShapes(start.shape, stop.shape)
	if e != nil {
		return nil, e
	}
	x, e := start.BroadcastTo(shape...)
	if e != nil {
		return nil, e
	}
	y, e := stop.BroadcastTo(shape...)
	if e != nil {
		return nil, e
	}
	ax, e := NormalizeAxis(opt.Axis, len(shape)+1)
	if e != nil {
		return nil, e
	}
	rt := opt.DType
	if rt == 0 {
		rt = d
	}
	out, e := Zeros(rt, append([]int{opt.Num}, shape...)...)
	if e != nil {
		return nil, e
	}
	end := opt.Endpoint == nil || *opt.Endpoint
	div := opt.Num
	if end {
		div--
	}
	pi := x87.MustParse("3.141592653589793238462643383279502884197")
	twoPi := pi.Add(pi)
	for i := 0; i < x.size; i++ {
		px := x.elementAt(x.offset + flatOffset(i, shape, x.strides))
		qy := y.elementAt(y.offset + flatOffset(i, shape, y.strides))
		if !d.IsExtended() && !rt.IsExtended() && (rt.IsFloat() || rt.IsComplex()) {
			p, q := px.number(x.DType()), qy.number(y.DType())
			if p == 0 || q == 0 {
				return nil, ErrValue
			}
			// Keep the extended calculation for extreme/nonfinite endpoints.
			// Normal binary64 inputs do not need software x87 arithmetic.
			pa, qa := cmplx.Abs(p), cmplx.Abs(q)
			if pa >= 1e-300 && pa <= 1e300 && qa >= 1e-300 && qa <= 1e300 {
				geomspaceNativeLane(out, i, p, q, d.IsComplex(), opt.Num, div, end)
				continue
			}
		}
		p := extendedValue(castElement(px, x.DType(), Complex256), Complex256)
		q := extendedValue(castElement(qy, y.DType(), Complex256), Complex256)
		if p.Real.IsZero() && p.Imag.IsZero() || q.Real.IsZero() && q.Imag.IsZero() {
			return nil, ErrValue
		}
		var a, b x87.Complex
		negative := false
		if d.IsComplex() {
			a, b = p.Log(), q.Log()
			delta := b.Imag.Sub(a.Imag)
			if v, ok := delta.Compare(pi); ok && v > 0 {
				b.Imag = b.Imag.Sub(twoPi)
			}
			if v, ok := delta.Compare(pi.Neg()); ok && v < 0 {
				b.Imag = b.Imag.Add(twoPi)
			}
		} else {
			negative = p.Real.Signbit() && q.Real.Signbit()
			if p.Real.Signbit() != q.Real.Signbit() {
				a.Real, b.Real = x87.NaN(), x87.NaN()
			} else {
				a.Real, b.Real = p.Real.Abs().Log(), q.Real.Abs().Log()
			}
		}
		for j := 0; j < opt.Num; j++ {
			z := p
			if j != 0 && div > 0 {
				t := x87.FromInt64(int64(j)).Div(x87.FromInt64(int64(div)))
				log := a.Add(b.Sub(a).Mul(x87.Complex{Real: t}))
				if d.IsComplex() {
					z = log.Exp()
				} else {
					z.Real = log.Real.Exp()
					if negative {
						z.Real = z.Real.Neg()
					}
				}
			}
			if end && opt.Num > 1 && j == opt.Num-1 {
				z = q
			}
			out.setElement(j*out.strides[0]+i, castElement(extendedElement(z, Complex256), Complex256, rt))
		}
	}
	if ax != 0 {
		return MoveAxis(out, 0, ax)
	}
	return out, nil
}

func geomspaceNativeLane(out *Array, i int, p, q complex128, isComplex bool, n, div int, end bool) {
	var a, b complex128
	rotation := complex(1, 0)
	negative := false
	if isComplex {
		// Rotate the start to the positive real axis. This also preserves
		// exactly zero components for a collinear imaginary sequence.
		abs := cmplx.Abs(p)
		rotation = complex(real(p)/abs, imag(p)/abs)
		a = complex(math.Log(abs), 0)
		b = cmplx.Log(q / rotation)
	} else {
		negative = real(p) < 0 && real(q) < 0
		if math.Signbit(real(p)) != math.Signbit(real(q)) {
			a, b = complex(math.NaN(), 0), complex(math.NaN(), 0)
		} else {
			a, b = complex(math.Log(math.Abs(real(p))), 0), complex(math.Log(math.Abs(real(q))), 0)
		}
	}
	for j := 0; j < n; j++ {
		z := p
		if j != 0 && div > 0 {
			t := float64(j) / float64(div)
			if isComplex {
				z = cmplx.Exp(a+(b-a)*complex(t, 0)) * rotation
			} else {
				v := math.Exp(real(a) + (real(b)-real(a))*t)
				if negative {
					v = -v
				}
				z = complex(v, 0)
			}
		}
		if end && n > 1 && j == n-1 {
			z = q
		}
		out.setElement(j*out.strides[0]+i, castElement(element{z: z}, Complex128, out.DType()))
	}
}
