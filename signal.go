package np

import (
	"math"
	"math/cmplx"
)

func Bartlett(n int) (*Array, error) {
	return window(n, func(i, m float64) float64 { return 2 / (m - 1) * ((m-1)/2 - math.Abs(i-(m-1)/2)) })
}
func Blackman(n int) (*Array, error) {
	return window(n, func(i, m float64) float64 {
		t := 2 * math.Pi * i / (m - 1)
		return .42 - .5*math.Cos(t) + .08*math.Cos(2*t)
	})
}
func Hamming(n int) (*Array, error) {
	return window(n, func(i, m float64) float64 { return .54 - .46*math.Cos(2*math.Pi*i/(m-1)) })
}
func Hanning(n int) (*Array, error) {
	return window(n, func(i, m float64) float64 { return .5 - .5*math.Cos(2*math.Pi*i/(m-1)) })
}
func Kaiser(n int, beta float64) (*Array, error) {
	den := besselI0(beta)
	return window(n, func(i, m float64) float64 { z := (2*i)/(m-1) - 1; return besselI0(beta*math.Sqrt(1-z*z)) / den })
}
func window(n int, f func(float64, float64) float64) (*Array, error) {
	n = max(n, 0)
	if _, err := ElementCount(Float64, n); err != nil {
		return nil, err
	}
	v := make([]float64, n)
	if n == 1 {
		v[0] = 1
	} else {
		for i := range v {
			v[i] = f(float64(i), float64(n))
		}
	}
	return WrapFloat64(v, n)
}

// Geomspace constructs geometric samples with scalar endpoints.
func Geomspace(start, stop complex128, num int, endpoint bool, dtype DType) (*Array, error) {
	if start == 0 || stop == 0 {
		return nil, ErrValue
	}
	out, err := Zeros(dtype, num)
	if err != nil {
		return nil, err
	}
	div := num
	if endpoint {
		div--
	}
	for i := 0; i < num; i++ {
		z := start
		if div > 0 {
			t := float64(i) / float64(div)
			if !dtype.IsComplex() && imag(start) == 0 && imag(stop) == 0 {
				x, y := real(start), real(stop)
				if math.Signbit(x) != math.Signbit(y) {
					z = complex(math.NaN(), 0)
				} else {
					z = complex(math.Copysign(math.Exp((1-t)*math.Log(math.Abs(x))+t*math.Log(math.Abs(y))), x), 0)
				}
			} else {
				z = cmplx.Exp(complex(1-t, 0)*cmplx.Log(start) + complex(t, 0)*cmplx.Log(stop))
			}
		}
		if i == 0 {
			z = start
		}
		if endpoint && num > 1 && i == num-1 {
			z = stop
		}
		out.setElement(i, castElement(element{z: z}, Complex128, dtype))
	}
	return out, nil
}

type SignalMode string

const (
	FullMode  SignalMode = "full"
	SameMode  SignalMode = "same"
	ValidMode SignalMode = "valid"
)

func Convolve(a, b *Array, mode SignalMode) (*Array, error)  { return signal(a, b, mode, false) }
func Correlate(a, b *Array, mode SignalMode) (*Array, error) { return signal(a, b, mode, true) }
func signal(a, b *Array, mode SignalMode, correlate bool) (*Array, error) {
	if a.DType() == 0 || b.DType() == 0 || a.NDim() != 1 || b.NDim() != 1 || a.size == 0 || b.size == 0 {
		return nil, ErrShape
	}
	d, err := PromoteTypes(a.DType(), b.DType())
	if err != nil {
		return nil, err
	}
	if a.size > maxInt-b.size {
		return nil, ErrShape
	}
	n, m := a.size, b.size
	if mode == "" {
		if correlate {
			mode = ValidMode
		} else {
			mode = FullMode
		}
	}
	start, length := 0, n+m-1
	switch mode {
	case FullMode:
	case ValidMode:
		start = min(n, m) - 1
		length = max(n, m) - min(n, m) + 1
	case SameMode:
		start = (min(n, m) - 1) / 2
		if correlate && n < m && min(n, m)%2 == 0 {
			start++
		}
		length = max(n, m)
	default:
		return nil, ErrValue
	}
	out, err := Zeros(d, length)
	if err != nil {
		return nil, err
	}
	x, err := a.Astype(d)
	if err != nil {
		return nil, err
	}
	y, err := b.Astype(d)
	if err != nil {
		return nil, err
	}
	for p := 0; p < length; p++ {
		k := p + start
		var total element
		for j := max(0, k-(m-1)); j <= min(k, n-1); j++ {
			ix := x.elementAt(j)
			idx := k - j
			if correlate {
				idx = m - 1 - idx
			}
			iy := y.elementAt(idx)
			if correlate && d.IsComplex() {
				iy.z = cmplx.Conj(iy.z)
			}
			total = addScalar(total, multiplyScalar(ix, iy, d), d)
		}
		out.setElement(p, total)
	}
	return out, nil
}

type BitOrder string

const (
	BigBits    BitOrder = "big"
	LittleBits BitOrder = "little"
)

func Packbits(a *Array, axis *int, order BitOrder) (*Array, error) {
	if a.DType() == 0 || (!a.DType().IsInteger() && a.DType() != Bool) {
		return nil, ErrDType
	}
	if order == "" {
		order = BigBits
	}
	if order != BigBits && order != LittleBits {
		return nil, ErrValue
	}
	if axis == nil {
		var err error
		a, err = Ravel(a)
		if err != nil {
			return nil, err
		}
		axis = Int(0)
	}
	ax, err := NormalizeAxis(*axis, a.NDim())
	if err != nil {
		return nil, err
	}
	shape := a.Shape()
	n := shape[ax]
	shape[ax] = (n + 7) / 8
	out, err := Zeros(Uint8, shape...)
	if err != nil {
		return nil, err
	}
	if out.size == 0 {
		return out, nil
	}
	r, _, _ := reduction(a, []SumOptions{{Axes: []int{ax}}})
	os := append(append([]int(nil), out.strides[:ax]...), out.strides[ax+1:]...)
	for i := 0; i < a.size/n; i++ {
		base, ob := r.base(a, i), flatOffset(i, r.shape, os)
		for j := 0; j < n; j++ {
			if !a.elementAt(base + j*a.strides[ax]).truth(a.DType()) {
				continue
			}
			bit := j % 8
			if order == BigBits {
				bit = 7 - bit
			}
			p := ob + (j/8)*out.strides[ax]
			out.setElement(p, element{i: out.elementAt(p).i | 1<<bit})
		}
	}
	return out, nil
}
func Unpackbits(a *Array, axis *int, count *int, order BitOrder) (*Array, error) {
	if a.DType() != Uint8 {
		return nil, ErrDType
	}
	if order == "" {
		order = BigBits
	}
	if order != BigBits && order != LittleBits {
		return nil, ErrValue
	}
	if axis == nil {
		var err error
		a, err = Ravel(a)
		if err != nil {
			return nil, err
		}
		axis = Int(0)
	}
	ax, err := NormalizeAxis(*axis, a.NDim())
	if err != nil {
		return nil, err
	}
	shape := a.Shape()
	n := shape[ax]
	if n > maxInt/8 {
		return nil, ErrShape
	}
	available := n * 8
	length := available
	if count != nil {
		length = *count
		if length < 0 {
			length += available
		}
		if length < 0 {
			return nil, ErrValue
		}
	}
	shape[ax] = length
	out, err := Zeros(Uint8, shape...)
	if err != nil {
		return nil, err
	}
	if out.size == 0 || a.size == 0 {
		return out, nil
	}
	r, _, _ := reduction(a, []SumOptions{{Axes: []int{ax}}})
	os := append(append([]int(nil), out.strides[:ax]...), out.strides[ax+1:]...)
	for i := 0; i < a.size/n; i++ {
		base, ob := r.base(a, i), flatOffset(i, r.shape, os)
		for j := 0; j < min(length, available); j++ {
			v := a.elementAt(base + (j/8)*a.strides[ax]).i
			bit := j % 8
			if order == BigBits {
				bit = 7 - bit
			}
			out.setElement(ob+j*out.strides[ax], element{i: (v >> bit) & 1})
		}
	}
	return out, nil
}

func SortComplex(a *Array) (*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	d := Complex128
	if a.DType() == Complex64 || a.DType() == Int8 || a.DType() == Int16 || a.DType() == Uint8 || a.DType() == Uint16 {
		d = Complex64
	}
	x, err := a.Astype(d)
	if err != nil {
		return nil, err
	}
	return Sort(x, Int(-1))
}
func Unwrap(a *Array, period, discont float64, axis int) (*Array, error) {
	if a.DType() == 0 || a.DType().IsComplex() || !(period > 0) {
		return nil, ErrValue
	}
	ax, err := NormalizeAxis(axis, a.NDim())
	if err != nil {
		return nil, err
	}
	d := a.DType()
	if !d.IsFloat() {
		d = Float64
	}
	n := a.shape[ax]
	out, err := Zeros(d, a.shape...)
	if err != nil {
		return nil, err
	}
	if a.size == 0 {
		return out, nil
	}
	half := period / 2
	if discont == 0 {
		discont = half
	}
	r, _, _ := reduction(a, []SumOptions{{Axes: []int{ax}}})
	os := append(append([]int(nil), out.strides[:ax]...), out.strides[ax+1:]...)
	for i := 0; i < a.size/n; i++ {
		base, ob := r.base(a, i), flatOffset(i, r.shape, os)
		prev := real(a.elementAt(base).number(a.DType()))
		out.setElement(ob, element{z: complex(prev, 0)})
		var correction float64
		for j := 1; j < n; j++ {
			x := real(a.elementAt(base + j*a.strides[ax]).number(a.DType()))
			delta := x - prev
			mod := math.Mod(delta+half, period)
			if mod < 0 {
				mod += period
			}
			mod -= half
			if mod == -half && delta > 0 {
				mod = half
			}
			if math.Abs(delta) >= discont {
				correction += mod - delta
			}
			out.setElement(ob+j*out.strides[ax], element{z: complex(x+correction, 0)})
			prev = x
		}
	}
	return out, nil
}
