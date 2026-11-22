package linalg

import (
	"math"
	"math/cmplx"
	"slices"
	"strconv"
	"strings"

	"github.com/SamuelSupe/numpygo"
)

// MatrixPower raises square matrices to an integer exponent with broadcasting
// across their leading dimensions. Negative exponents use the inverse.
func MatrixPower(a *np.Array, exponent int) (*np.Array, error) {
	if a.DType() == 0 || a.NDim() < 2 {
		return nil, np.ErrShape
	}
	shape := a.Shape()
	n := shape[len(shape)-1]
	if shape[len(shape)-2] != n {
		return nil, np.ErrShape
	}
	if exponent == 0 {
		eye, err := np.Identity(n, a.DType())
		if err != nil {
			return nil, err
		}
		out, err := eye.BroadcastTo(shape...)
		if err != nil {
			return nil, err
		}
		return out.Copy()
	}
	if exponent == 1 {
		return a.Copy()
	}
	p := uint(exponent)
	if exponent < 0 {
		var err error
		a, err = Inv(a)
		if err != nil {
			return nil, err
		}
		p = 0 - p
	}
	var out *np.Array
	for p != 0 {
		if p&1 != 0 {
			if out == nil {
				out = a
			} else {
				var err error
				out, err = MatMul(out, a)
				if err != nil {
					return nil, err
				}
			}
		}
		p >>= 1
		if p != 0 {
			var err error
			a, err = MatMul(a, a)
			if err != nil {
				return nil, err
			}
		}
	}
	return out, nil
}

// Lstsq solves the least-squares problem for a rank-two A and a vector or matrix B.
// Nil rcond uses eps*max(m,n), negative rcond uses eps.
func Lstsq(a, b *np.Array, rcond *float64) (x, residuals *np.Array, rank int64, s *np.Array, err error) {
	d, err := linearDType(a.DType(), b.DType())
	if err != nil {
		return nil, nil, 0, nil, err
	}
	if a.NDim() != 2 || (b.NDim() != 1 && b.NDim() != 2) {
		return nil, nil, 0, nil, np.ErrShape
	}
	as, bs := a.Shape(), b.Shape()
	m, n := as[0], as[1]
	if bs[0] != m {
		return nil, nil, 0, nil, np.ErrShape
	}
	k := min(m, n)
	cols := 1
	if len(bs) == 2 {
		cols = bs[1]
	}
	a, err = a.Astype(d)
	if err != nil {
		return nil, nil, 0, nil, err
	}
	b, err = b.Astype(d)
	if err != nil {
		return nil, nil, 0, nil, err
	}
	u, s, vh, err := SVD(a, SVDOptions{Reduced: true})
	if err != nil {
		return nil, nil, 0, nil, err
	}
	sv, _ := s.Complex128s()
	uv, _ := u.Complex128s()
	vv, _ := vh.Complex128s()
	bv, _ := b.Complex128s()
	eps := 2.220446049250313e-16
	// NumPy's LAPACK least-squares calculation uses double precision internally.
	cutoff := eps * float64(max(m, n))
	if rcond != nil {
		cutoff = *rcond
		if cutoff < 0 {
			cutoff = eps
		}
	}
	threshold := 0.0
	if k > 0 {
		threshold = cutoff * real(sv[0])
	}
	for _, z := range sv {
		if real(z) > threshold {
			rank++
		}
	}
	xshape := []int{n}
	if len(bs) == 2 {
		xshape = append(xshape, cols)
	}
	size, err := np.ElementCount(d, xshape...)
	if err != nil {
		return nil, nil, 0, nil, err
	}
	out := make([]complex128, size)
	for j := 0; j < cols; j++ {
		for h := 0; h < k; h++ {
			sigma := real(sv[h])
			if sigma <= threshold {
				continue
			}
			var w complex128
			for i := 0; i < m; i++ {
				w += cmplx.Conj(uv[i*k+h]) * bv[i*cols+j]
			}
			w /= complex(sigma, 0)
			for i := 0; i < n; i++ {
				out[i*cols+j] += cmplx.Conj(vv[h*n+i]) * w
			}
		}
	}
	x, err = asComplexArray(out, d, xshape)
	if err != nil {
		return nil, nil, 0, nil, err
	}
	var rv []float64
	if m > n && rank == int64(n) {
		rv = make([]float64, cols)
		av, _ := a.Complex128s()
		for j := 0; j < cols; j++ {
			for i := 0; i < m; i++ {
				z := -bv[i*cols+j]
				for h := 0; h < n; h++ {
					z += av[i*n+h] * out[h*cols+j]
				}
				rv[j] += real(z)*real(z) + imag(z)*imag(z)
			}
		}
	}
	residuals, err = np.WrapSlice(rv, len(rv))
	if err != nil {
		return nil, nil, 0, nil, err
	}
	if d == np.Float32 || d == np.Complex64 {
		residuals, err = residuals.Astype(np.Float32)
	}
	return x, residuals, rank, s, err
}

// NormOptions accepts the standard matrix and vector orders as strings, e.g.
// "fro", "nuc", "2", "-1", "inf". Empty Ord uses NumPy's default.
type NormOptions struct {
	Ord      string
	Axes     []int
	KeepDims bool
}

func Norm(a *np.Array, options ...NormOptions) (*np.Array, error) {
	if !a.DType().IsNativeNumeric() && !a.DType().IsExtended() {
		return nil, np.ErrDType
	}
	if len(options) > 1 {
		return nil, np.ErrShape
	}
	var opt NormOptions
	if len(options) == 1 {
		opt = options[0]
	}
	ord := strings.ToLower(opt.Ord)
	if a.DType().IsExtended() {
		if ord != "" && ord != "fro" && ord != "f" && !(ord == "2" && (len(opt.Axes) == 1 || opt.Axes == nil && a.NDim() == 1)) {
			return extendedOrderedNorm(a, opt, ord)
		}
		if (ord == "fro" || ord == "f") && !(len(opt.Axes) == 2 || opt.Axes == nil && a.NDim() == 2) {
			return nil, np.ErrShape
		}
		var sq *np.Array
		var e error
		if a.DType() == np.Complex256 {
			r, err := np.Real(a)
			if err != nil {
				return nil, err
			}
			i, err := np.Imag(a)
			if err != nil {
				return nil, err
			}
			rr, err := np.Multiply(r, r)
			if err != nil {
				return nil, err
			}
			ii, err := np.Multiply(i, i)
			if err != nil {
				return nil, err
			}
			sq, e = np.Add(rr, ii)
		} else {
			sq, e = np.Multiply(a, a)
		}
		if e != nil {
			return nil, e
		}
		s, e := np.Sum(sq, np.SumOptions{Axes: opt.Axes, KeepDims: opt.KeepDims})
		if e != nil {
			return nil, e
		}
		return np.Sqrt(s)
	}
	shape := a.Shape()
	dtype := np.Float64
	if a.DType() == np.Float16 {
		dtype = np.Float16
	} else if a.DType() == np.Float32 || a.DType() == np.Complex64 {
		dtype = np.Float32
	}
	axes := slices.Clone(opt.Axes)
	if axes == nil {
		if ord == "" {
			v, _ := a.Complex128s()
			z := arrayL2(v, dtype)
			out := np.Scalar(z)
			if opt.KeepDims {
				ones := make([]int, a.NDim())
				for i := range ones {
					ones[i] = 1
				}
				var err error
				out, err = out.Reshape(ones...)
				if err != nil {
					return nil, err
				}
			}
			return out.Astype(dtype)
		}
		if a.NDim() != 1 && a.NDim() != 2 {
			return nil, np.ErrShape
		}
		axes = make([]int, a.NDim())
		for i := range axes {
			axes[i] = i
		}
	}
	if len(axes) != 1 && len(axes) != 2 {
		return nil, np.ErrAxis
	}
	seen := make([]bool, a.NDim())
	for i, ax := range axes {
		v, err := np.NormalizeAxis(ax, a.NDim())
		if err != nil || seen[v] {
			return nil, np.ErrAxis
		}
		axes[i] = v
		seen[v] = true
	}
	var order, batch []int
	for i, d := range shape {
		if !seen[i] {
			order = append(order, i)
			batch = append(batch, d)
		}
	}
	order = append(order, axes...)
	transpose, err := a.Transpose(order...)
	if err != nil {
		return nil, err
	}
	vals, _ := transpose.Complex128s()
	count, err := np.ElementCount(dtype, batch...)
	if err != nil {
		return nil, err
	}
	out := make([]float64, count)
	if len(axes) == 1 {
		n := shape[axes[0]]
		for i := range out {
			out[i], err = vectorOrder(vals[i*n:(i+1)*n], ord, dtype)
			if err != nil {
				return nil, err
			}
		}
	} else {
		m, n := shape[axes[0]], shape[axes[1]]
		for i := range out {
			out[i], err = matrixOrder(vals[i*m*n:(i+1)*m*n], m, n, ord, dtype)
			if err != nil {
				return nil, err
			}
		}
	}
	if opt.KeepDims {
		batch = slices.Clone(shape)
		for _, axis := range axes {
			batch[axis] = 1
		}
	}
	r, err := np.WrapFloat64(out, batch...)
	if err != nil {
		return nil, err
	}
	return r.Astype(dtype)
}

func extendedOrderedNorm(a *np.Array, opt NormOptions, ord string) (*np.Array, error) {
	axes := slices.Clone(opt.Axes)
	if axes == nil {
		switch a.NDim() {
		case 1:
			axes = []int{0}
		case 2:
			axes = []int{0, 1}
		default:
			return nil, np.ErrShape
		}
	}
	for i, v := range axes {
		x, e := np.NormalizeAxis(v, a.NDim())
		if e != nil {
			return nil, e
		}
		axes[i] = x
	}
	if len(axes) == 2 && axes[0] == axes[1] {
		return nil, np.ErrAxis
	}
	abs, e := np.Absolute(a)
	if e != nil {
		return nil, e
	}
	o := np.SumOptions{Axes: axes, KeepDims: opt.KeepDims}
	if len(axes) == 1 {
		switch ord {
		case "inf", "+inf":
			return np.Max(abs, o)
		case "-inf":
			return np.Min(abs, o)
		case "1":
			return np.Sum(abs, o)
		case "0":
			n, e := np.CountNonzero(abs, o)
			if e != nil {
				return nil, e
			}
			return n.Astype(np.Float128)
		default:
			p, e := strconv.ParseFloat(ord, 64)
			if e != nil || math.IsNaN(p) || p == 0 {
				return nil, np.ErrValue
			}
			power, e := np.Power(abs, np.ScalarOf(p))
			if e != nil {
				return nil, e
			}
			s, e := np.Sum(power, o)
			if e != nil {
				return nil, e
			}
			one, e := np.Full(np.Float128, 1.)
			if e != nil {
				return nil, e
			}
			recrip, e := np.Divide(one, np.ScalarOf(p))
			if e != nil {
				return nil, e
			}
			return np.Power(s, recrip)
		}
	}
	if len(axes) != 2 {
		return nil, np.ErrShape
	}
	if ord == "2" || ord == "-2" || ord == "nuc" {
		return nil, np.ErrDType
	}
	sumAxis, extAxis := axes[0], axes[1]
	if ord == "inf" || ord == "+inf" || ord == "-inf" {
		sumAxis, extAxis = axes[1], axes[0]
	} else if ord != "1" && ord != "-1" {
		return nil, np.ErrValue
	}
	s, e := np.Sum(abs, np.SumOptions{Axes: []int{sumAxis}, KeepDims: true})
	if e != nil {
		return nil, e
	}
	o = np.SumOptions{Axes: []int{extAxis}, KeepDims: true}
	var v *np.Array
	if strings.HasPrefix(ord, "-") {
		v, e = np.Min(s, o)
	} else {
		v, e = np.Max(s, o)
	}
	if e != nil || opt.KeepDims {
		return v, e
	}
	return np.Squeeze(v, axes...)
}

func normRound(x float64, d np.DType) float64 {
	switch d {
	case np.Float16:
		return np.HalfFromFloat64(x).Float64()
	case np.Float32:
		return float64(float32(x))
	default:
		return x
	}
}

// The public NumPy norms sum squares in the result precision. Unlike the
// scaled norm used inside numerical factorizations, their intermediate
// squares can overflow or underflow.
func arrayL2(v []complex128, d np.DType) float64 {
	var total float64
	for _, z := range v {
		x, y := real(z), imag(z)
		total = normRound(total+normRound(normRound(x*x, d)+normRound(y*y, d), d), d)
	}
	return math.Sqrt(total)
}

func vectorOrder(v []complex128, ord string, d np.DType) (float64, error) {
	switch ord {
	case "", "2":
		return arrayL2(v, d), nil
	case "0":
		n := 0
		for _, z := range v {
			if z != 0 {
				n++
			}
		}
		return float64(n), nil
	case "inf", "+inf", "-inf":
		if len(v) == 0 {
			return 0, np.ErrValue
		}
		out := cmplx.Abs(v[0])
		for _, z := range v[1:] {
			if ord == "-inf" {
				out = math.Min(out, cmplx.Abs(z))
			} else {
				out = math.Max(out, cmplx.Abs(z))
			}
		}
		return out, nil
	default:
		p, err := strconv.ParseFloat(ord, 64)
		if err != nil || math.IsNaN(p) {
			return 0, np.ErrValue
		}
		total := 0.0
		for _, z := range v {
			total = normRound(total+normRound(math.Pow(normRound(cmplx.Abs(z), d), p), d), d)
		}
		return math.Pow(total, 1/p), nil
	}
}
func matrixOrder(v []complex128, m, n int, ord string, d np.DType) (float64, error) {
	switch ord {
	case "", "fro", "f":
		return arrayL2(v, d), nil
	case "1", "-1", "inf", "+inf", "-inf":
		cols := ord == "1" || ord == "-1"
		groups, items := m, n
		if cols {
			groups, items = n, m
		}
		if groups == 0 {
			return 0, np.ErrValue
		}
		out := 0.0
		if ord == "-1" || ord == "-inf" {
			out = math.Inf(1)
		}
		for i := 0; i < groups; i++ {
			var sum float64
			for j := 0; j < items; j++ {
				if cols {
					sum += cmplx.Abs(v[j*n+i])
				} else {
					sum += cmplx.Abs(v[i*n+j])
				}
			}
			if ord == "-1" || ord == "-inf" {
				out = math.Min(out, sum)
			} else {
				out = math.Max(out, sum)
			}
		}
		return out, nil
	case "nuc", "2", "-2":
		array, err := np.WrapComplex128(v, m, n)
		if err != nil {
			return 0, err
		}
		_, s, _, err := SVD(array, SVDOptions{ValuesOnly: true})
		if err != nil {
			return 0, err
		}
		val, _ := s.Float64s()
		if ord == "nuc" {
			var total float64
			for _, z := range val {
				total += z
			}
			return total, nil
		}
		if len(val) == 0 {
			return 0, np.ErrValue
		}
		if ord == "2" {
			return val[0], nil
		}
		return val[len(val)-1], nil
	default:
		return 0, np.ErrValue
	}
}
