// Package scimath implements NumPy's domain-sensitive numerical functions
// (numpy.lib.scimath, also exported as numpy.emath).
package scimath

import "github.com/SamuelSupe/numpygo"

func numeric(a *np.Array) bool {
	d := a.DType()
	return d.IsNativeNumeric() || d.IsExtended()
}

// needsComplex checks for real components outside the chosen real domain.
// In particular, already complex non-real components need no conversion.
func needsComplex(a *np.Array, absGT1 bool) (bool, error) {
	if !numeric(a) {
		return false, np.ErrDType
	}
	r, e := np.IsReal(a)
	if e != nil {
		return false, e
	}
	var outside *np.Array
	if absGT1 {
		mag, e := np.Absolute(a)
		if e != nil {
			return false, e
		}
		one, e := np.FromSlice([]int8{1})
		if e != nil {
			return false, e
		}
		outside, e = np.Greater(mag, one)
		if e != nil {
			return false, e
		}
	} else {
		re, e := np.Real(a)
		if e != nil {
			return false, e
		}
		zero, e := np.FromSlice([]int8{0})
		if e != nil {
			return false, e
		}
		outside, e = np.Less(re, zero)
		if e != nil {
			return false, e
		}
	}
	v, e := np.LogicalAnd(r, outside)
	if e != nil {
		return false, e
	}
	v, e = np.Any(v)
	if e != nil {
		return false, e
	}
	x, e := v.AtValue()
	if e != nil {
		return false, e
	}
	return x.(bool), nil
}
func promote(a *np.Array, absGT1 bool) (*np.Array, error) {
	yes, e := needsComplex(a, absGT1)
	if e != nil {
		return nil, e
	}
	if !yes {
		return a, nil
	}
	// NumPy's scimath explicitly chooses csingle for these six legacy
	// types; half and extended precision instead select cdouble.
	d := np.Complex128
	switch a.DType() {
	case np.Float32, np.Int8, np.Int16, np.Uint8, np.Uint16, np.Complex64:
		d = np.Complex64
	}
	return a.Astype(d)
}
func unary(a *np.Array, absGT1 bool, f func(*np.Array) (*np.Array, error)) (*np.Array, error) {
	v, e := promote(a, absGT1)
	if e != nil {
		return nil, e
	}
	return f(v)
}
func Sqrt(a *np.Array) (*np.Array, error)    { return unary(a, false, np.Sqrt) }
func Log(a *np.Array) (*np.Array, error)     { return unary(a, false, np.Log) }
func Log2(a *np.Array) (*np.Array, error)    { return unary(a, false, np.Log2) }
func Log10(a *np.Array) (*np.Array, error)   { return unary(a, false, np.Log10) }
func Arccos(a *np.Array) (*np.Array, error)  { return unary(a, true, np.Arccos) }
func Arcsin(a *np.Array) (*np.Array, error)  { return unary(a, true, np.Arcsin) }
func Arctanh(a *np.Array) (*np.Array, error) { return unary(a, true, np.Arctanh) }

// Logn returns log(x)/log(n) with domain-sensitive promotion of both operands.
func Logn(n, x *np.Array) (*np.Array, error) {
	a, e := Log(x)
	if e != nil {
		return nil, e
	}
	b, e := Log(n)
	if e != nil {
		return nil, e
	}
	return np.Divide(a, b)
}

// Power promotes a negative real base to complex and converts negative
// integral exponents to floating point, like NumPy's x * 1.0 weak-scalar rule.
func Power(x, p *np.Array) (*np.Array, error) {
	a, e := promote(x, false)
	if e != nil {
		return nil, e
	}
	neg, e := needsComplex(p, false)
	if e != nil {
		return nil, e
	}
	if neg {
		u, e := np.LookupUFunc("multiply")
		if e != nil {
			return nil, e
		}
		p, e = u.CallValues(np.UFuncOptions{}, p, np.Weak(1.0))
		if e != nil {
			return nil, e
		}
	}
	return np.Power(a, p)
}
