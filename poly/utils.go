package poly

import "github.com/SamuelSupe/numpygo"

// TrimSequence removes exactly zero trailing coefficients, preserving dtype
// and underlying storage. Empty arrays and one zero are retained.
func TrimSequence(a *np.Array) (*np.Array, error) {
	if !a.DType().IsNativeNumeric() && !a.DType().IsExtended() {
		return nil, np.ErrDType
	}
	if a.NDim() != 1 {
		return nil, np.ErrShape
	}
	if a.Size() <= 1 {
		return a, nil
	}
	z, e := np.Zeros(a.DType())
	if e != nil {
		return nil, e
	}
	eq, e := np.Equal(a, z)
	if e != nil {
		return nil, e
	}
	flags, e := np.Values[bool](eq)
	if e != nil {
		return nil, e
	}
	n := len(flags)
	for n > 1 && flags[n-1] {
		n--
	}
	if n == len(flags) {
		return a, nil
	}
	return a.Slice(np.Range(0, n))
}

// AsSeries constructs detached, nonempty one-dimensional coefficient arrays
// using NumPy's common inexact numerical dtype. Scalars become length one;
// trim removes exact trailing zeros before conversion.
func AsSeries(arrays []*np.Array, trim bool) ([]*np.Array, error) {
	out := make([]*np.Array, len(arrays))
	for i, a := range arrays {
		if a.DType() == 0 || a.Size() == 0 || a.NDim() > 1 {
			return nil, np.ErrShape
		}
		var e error
		if a.NDim() == 0 {
			a, e = a.Reshape(1)
			if e != nil {
				return nil, e
			}
		}
		if trim {
			a, e = TrimSequence(a)
			if e != nil {
				return nil, e
			}
		}
		out[i] = a
	}
	d, e := np.CommonType(out...)
	if e != nil {
		return nil, np.ErrValue
	}
	for i, a := range out {
		out[i], e = a.Astype(d)
		if e != nil {
			return nil, e
		}
	}
	return out, nil
}

// TrimCoefficients retains through the final coefficient with magnitude > tol.
// NumPy's comparison deliberately excludes NaNs rather than using !(abs <= tol).
func TrimCoefficients(a *np.Array, tol float64) (*np.Array, error) {
	if tol < 0 {
		return nil, np.ErrValue
	}
	v, e := AsSeries([]*np.Array{a}, true)
	if e != nil {
		return nil, e
	}
	a = v[0]
	mag, e := np.Absolute(a)
	if e != nil {
		return nil, e
	}
	cmp, e := np.LookupUFunc("greater")
	if e != nil {
		return nil, e
	}
	keep, e := cmp.CallValues(np.UFuncOptions{}, mag, np.Weak(tol))
	if e != nil {
		return nil, e
	}
	flags, e := np.Values[bool](keep)
	if e != nil {
		return nil, e
	}
	n := len(flags)
	for n > 0 && !flags[n-1] {
		n--
	}
	if n == 0 {
		first, e := a.Slice(np.Range(0, 1))
		if e != nil {
			return nil, e
		}
		zero, e := np.Zeros(a.DType())
		if e != nil {
			return nil, e
		}
		return np.Multiply(first, zero)
	}
	view, e := a.Slice(np.Range(0, n))
	if e != nil {
		return nil, e
	}
	return view.Copy()
}

// GetDomain returns extrema for a real vector, or the opposite corners of
// the complex vector's axis-aligned bounding rectangle.
func GetDomain(x *np.Array) (*np.Array, error) {
	v, e := AsSeries([]*np.Array{x}, false)
	if e != nil {
		return nil, e
	}
	x = v[0]
	if x.DType().IsComplex() {
		re, e := np.Real(x)
		if e != nil {
			return nil, e
		}
		im, e := np.Imag(x)
		if e != nil {
			return nil, e
		}
		extrema := make([]float64, 4)
		for j, a := range []*np.Array{re, im} {
			lo, e := np.Min(a)
			if e != nil {
				return nil, e
			}
			hi, e := np.Max(a)
			if e != nil {
				return nil, e
			}
			l, e := lo.Complex128s()
			if e != nil {
				return nil, e
			}
			h, e := hi.Complex128s()
			if e != nil {
				return nil, e
			}
			extrema[2*j], extrema[2*j+1] = real(l[0]), real(h[0])
		}
		return np.FromComplex128([]complex128{complex(extrema[0], extrema[2]), complex(extrema[1], extrema[3])}, 2)
	}
	lo, e := np.Min(x)
	if e != nil {
		return nil, e
	}
	hi, e := np.Max(x)
	if e != nil {
		return nil, e
	}
	return np.Stack([]*np.Array{lo, hi}, 0)
}

// MapParameters returns the two scalar arrays defining the affine map
// from old to new. Explicit typed endpoints preserve ndarray promotion,
// including complex intervals and fixed-width integer arithmetic.
func MapParameters(old, new *np.Array) (off, scale *np.Array, err error) {
	if old.NDim() != 1 || new.NDim() != 1 || old.Size() != 2 || new.Size() != 2 {
		return nil, nil, np.ErrShape
	}
	a, e := old.Slice(np.AtIndex(0))
	if e != nil {
		return nil, nil, e
	}
	b, e := old.Slice(np.AtIndex(1))
	if e != nil {
		return nil, nil, e
	}
	c, e := new.Slice(np.AtIndex(0))
	if e != nil {
		return nil, nil, e
	}
	d, e := new.Slice(np.AtIndex(1))
	if e != nil {
		return nil, nil, e
	}
	delta, e := np.Subtract(b, a)
	if e != nil {
		return nil, nil, e
	}
	n, e := np.Subtract(d, c)
	if e != nil {
		return nil, nil, e
	}
	scale, e = np.Divide(n, delta)
	if e != nil {
		return nil, nil, e
	}
	left, e := np.Multiply(b, c)
	if e != nil {
		return nil, nil, e
	}
	right, e := np.Multiply(a, d)
	if e != nil {
		return nil, nil, e
	}
	n, e = np.Subtract(left, right)
	if e != nil {
		return nil, nil, e
	}
	off, e = np.Divide(n, delta)
	return off, scale, e
}

func MapDomain(x, old, new *np.Array) (*np.Array, error) {
	off, scale, e := MapParameters(old, new)
	if e != nil {
		return nil, e
	}
	v, e := np.Multiply(x, scale)
	if e != nil {
		return nil, e
	}
	return np.Add(off, v)
}

// FormatFloat applies the current ndarray print options to a float64
// coefficient. parens wraps only scientific notation.
func FormatFloat(value float64, parens bool) string {
	return polynomialFloat(value, parens, np.GetPrintOptions())
}
