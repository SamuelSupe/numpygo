package np

import "math"

type CovOptions struct {
	ColumnVariables    bool
	Bias               bool
	DDOF               *int
	FWeights, AWeights *Array
	DType              DType
}

// CovWithOptions treats rows as variables by default. AWeights are relative
// importance weights; FWeights are nonnegative integral observation counts.
func CovWithOptions(m, y *Array, opt CovOptions) (*Array, error) {
	prepare := func(a *Array) (*Array, error) {
		if a.DType() == 0 || a.NDim() > 2 || !a.DType().IsNativeNumeric() && !a.DType().IsExtended() {
			return nil, ErrDType
		}
		if a.NDim() < 2 {
			return a.Reshape(1, a.size)
		}
		if opt.ColumnVariables && a.shape[0] != 1 {
			return a.Transpose()
		}
		return a, nil
	}
	x, e := prepare(m)
	if e != nil {
		return nil, e
	}
	d := opt.DType
	if d == 0 {
		d, e = PromoteTypes(x.DType(), Float64)
		if e != nil {
			return nil, e
		}
		if y != nil {
			d, e = PromoteTypes(d, y.DType())
			if e != nil {
				return nil, e
			}
		}
	}
	if !d.IsFloat() && !d.IsComplex() {
		return nil, ErrDType
	}
	if x.DType() != d {
		x, e = x.Astype(d)
		if e != nil {
			return nil, e
		}
	}
	if y != nil {
		z, e := prepare(y)
		if e != nil {
			return nil, e
		}
		if z.DType() != d {
			z, e = z.Astype(d)
			if e != nil {
				return nil, e
			}
		}
		x, e = Concatenate([]*Array{x, z}, 0)
		if e != nil {
			return nil, e
		}
	}
	rows, n := x.shape[0], x.shape[1]
	if rows == 0 {
		return Zeros(d, 0, 0)
	}
	ddof := 1.
	if opt.Bias {
		ddof = 0
	}
	if opt.DDOF != nil {
		ddof = float64(*opt.DDOF)
	}
	weights := func(a *Array, frequency bool) (*Array, error) {
		if a == nil {
			return nil, nil
		}
		if a.NDim() != 1 || a.size != n {
			return nil, ErrShape
		}
		if a.DType().IsComplex() || !a.DType().IsNativeNumeric() && !a.DType().IsExtended() {
			return nil, ErrDType
		}
		w, e := a.Astype(Float64)
		if e != nil {
			return nil, e
		}
		for i := 0; i < n; i++ {
			v := real(w.elementAt(i).z)
			if math.IsNaN(v) || math.IsInf(v, 0) || v < 0 || frequency && v != math.Floor(v) {
				return nil, ErrValue
			}
		}
		return w, nil
	}
	f, e := weights(opt.FWeights, true)
	if e != nil {
		return nil, e
	}
	w, e := weights(opt.AWeights, false)
	if e != nil {
		return nil, e
	}
	aw := w
	if f != nil {
		if w == nil {
			w = f
		} else {
			w, e = Multiply(f, w)
			if e != nil {
				return nil, e
			}
		}
	}
	mean, e := Average(x, w, SumOptions{Axes: []int{1}, KeepDims: true})
	if e != nil {
		return nil, e
	}
	factor := float64(n) - ddof
	if w != nil {
		s, e := Sum(w)
		if e != nil {
			return nil, e
		}
		v := real(s.elementAt(0).z)
		factor = v - ddof
		if aw != nil && ddof != 0 {
			wa, e := Multiply(w, aw)
			if e != nil {
				return nil, e
			}
			s, e := Sum(wa)
			if e != nil {
				return nil, e
			}
			factor = v - ddof*real(s.elementAt(0).z)/v
		}
	}
	if mean.DType() != d {
		mean, e = mean.Astype(d)
		if e != nil {
			return nil, e
		}
	}
	center, e := Subtract(x, mean)
	if e != nil {
		return nil, e
	}
	right := center
	if center.DType().IsComplex() {
		right, e = Conjugate(center)
		if e != nil {
			return nil, e
		}
	}
	if w != nil {
		right, e = Multiply(right, w)
		if e != nil {
			return nil, e
		}
	}
	out, e := Tensordot(center, right, []int{1}, []int{1})
	if e != nil {
		return nil, e
	}
	if factor < 0 {
		factor = 0
	}
	den, e := Full(out.DType(), factor)
	if e != nil {
		return nil, e
	}
	out, e = Divide(out, den)
	if e != nil {
		return nil, e
	}
	if rows == 1 {
		return out.Reshape()
	}
	return out, nil
}

func CorrcoefWithOptions(m, y *Array, opt CovOptions) (*Array, error) {
	c, e := CovWithOptions(m, y, opt)
	if e != nil {
		return nil, e
	}
	if c.NDim() == 0 {
		return Divide(c, c)
	}
	n := c.shape[0]
	out, e := Zeros(c.DType(), n, n)
	if e != nil {
		return nil, e
	}
	if c.DType().IsExtended() {
		diag, e := Diagonal(c, 0, 0, 1)
		if e != nil {
			return nil, e
		}
		diag, e = Real(diag)
		if e != nil {
			return nil, e
		}
		s, e := Sqrt(diag)
		if e != nil {
			return nil, e
		}
		col, e := s.Reshape(n, 1)
		if e != nil {
			return nil, e
		}
		den, e := Multiply(col, s)
		if e != nil {
			return nil, e
		}
		return Divide(c, den)
	}
	diag := make([]float64, n)
	for i := range diag {
		diag[i] = math.Sqrt(real(c.elementAt(i*n + i).z))
	}
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			z := c.elementAt(i*n+j).z / complex(diag[i]*diag[j], 0)
			r, im := math.Max(-1, math.Min(1, real(z))), math.Max(-1, math.Min(1, imag(z)))
			out.setElement(i*n+j, element{z: complex(r, im)})
		}
	}
	return out, nil
}
