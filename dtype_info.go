package np

import (
	"math"
	"strings"

	"github.com/SamuelSupe/numpygo/internal/x87"
)

// CanCast reports NumPy 2.x's strong dtype casting rules. The library
// supports native-order fixed-width dtypes, so "no" and "equiv" coincide.
func CanCast(from, to DType, rule string) (bool, error) {
	if from.ItemSize() == 0 || to.ItemSize() == 0 {
		return false, ErrDType
	}
	switch rule {
	case "no", "equiv":
		return from == to, nil
	case "unsafe":
		return true, nil
	case "", "safe", "same_kind":
	default:
		return false, ErrValue
	}
	if to == Object {
		return true, nil
	}
	if from == Object {
		return false, nil
	}
	if from == StringDType || to == StringDType {
		if from == to {
			return true, nil
		}
		if to == StringDType {
			if from.IsNativeNumeric() || from.IsExtended() || from.IsTemporal() {
				return true, nil
			}
			return rule == "same_kind" && from.IsFlexible(), nil
		}
		return rule == "same_kind" && (to.IsFlexible() || to == Bool), nil
	}
	if from.IsFlexible() || to.IsFlexible() {
		if from == to {
			return true, nil
		}
		if to == UnicodeDType && from == BytesDType {
			return true, nil
		}
		if to.IsCharacter() && !from.IsFlexible() {
			_, e := FlexibleStringWidth(from)
			return e == nil, nil
		}
		return false, nil
	}
	if from.IsTemporal() || to.IsTemporal() || from == RecordDType || to == RecordDType {
		if to == TimeDelta64 && (from == Bool || from.IsInteger()) {
			return rule == "same_kind" || from == Bool || from.IsSigned() || from.ItemSize() < 8, nil
		}
		return from == to, nil
	}
	if from == to || from == Bool {
		return true, nil
	}
	if rule == "same_kind" {
		kind := func(d DType) int {
			switch {
			case d.IsUnsigned():
				return 1
			case d.IsSigned():
				return 2
			case d.IsFloat():
				return 3
			case d.IsComplex():
				return 4
			}
			return 0
		}
		return kind(from) <= kind(to), nil
	}
	if to.IsInteger() && from.IsInteger() {
		f, t := from.ItemSize(), to.ItemSize()
		if from.IsSigned() && to.IsUnsigned() {
			return false, nil
		}
		if from.IsUnsigned() && to.IsSigned() {
			return f < t, nil
		}
		return f <= t, nil
	}
	f, t := from, to
	if to.IsComplex() {
		if to == Complex64 {
			t = Float32
		} else if to == Complex256 {
			t = Float128
		} else {
			t = Float64
		}
	}
	if from.IsComplex() {
		if !to.IsComplex() {
			return false, nil
		}
		if from == Complex64 {
			f = Float32
		} else if from == Complex256 {
			f = Float128
		} else {
			f = Float64
		}
	}
	if !t.IsFloat() {
		return false, nil
	}
	if f.IsFloat() {
		return f.ItemSize() <= t.ItemSize(), nil
	}
	if f.IsInteger() {
		// NumPy classifies 64-bit integers to float64 as safe even though
		// floating point cannot exactly represent every such integer.
		limit := 8
		if t == Float16 {
			limit = 1
		} else if t == Float32 {
			limit = 2
		}
		return f.ItemSize() <= limit, nil
	}
	return false, nil
}

// IsDType accepts numeric dtype names and NumPy's built-in kind strings.
// Multiple kind strings are combined with OR, like NumPy's tuple argument.
func IsDType(d DType, kinds ...string) (bool, error) {
	if d.ItemSize() == 0 {
		return false, ErrDType
	}
	result := false
	for _, kind := range kinds {
		var match bool
		switch kind {
		case "bool":
			match = d == Bool
		case "signed integer":
			match = d.IsSigned()
		case "unsigned integer":
			match = d.IsUnsigned()
		case "integral":
			match = d.IsInteger()
		case "real floating":
			match = d.IsFloat()
		case "complex floating":
			match = d.IsComplex()
		case "numeric":
			match = d.IsInteger() || d.IsFloat() || d.IsComplex()
		default:
			other, err := ParseDType(kind)
			if err != nil {
				return false, ErrValue
			}
			match = d == other
		}
		result = result || match
	}
	return result, nil
}

type IntegerInfo struct {
	Bits     int
	Min, Max any
}

func IInfo(d DType) (IntegerInfo, error) {
	if !d.IsInteger() {
		return IntegerInfo{}, ErrDType
	}
	n := d.ItemSize() * 8
	if d.IsSigned() {
		maximum := uint64(1)<<(n-1) - 1
		minimum := -(int64(maximum)) - 1
		return IntegerInfo{n, minimum, int64(maximum)}, nil
	}
	return IntegerInfo{n, uint64(0), ^uint64(0) >> (64 - n)}, nil
}

type FloatInfo struct {
	Bits, Mantissa, Exponent, MinExp, MaxExp, Precision        int
	Eps, EpsNeg, Min, Max, Tiny, SmallestSubnormal, Resolution float64
}

// ExtendedFInfo returns the exact x87 limits without narrowing them to float64.
func ExtendedFInfo() ExtendedFloatInfo { return x87.Info() }

func FInfo(d DType) (FloatInfo, error) {
	if d == Complex64 {
		d = Float32
	}
	if d == Complex128 {
		d = Float64
	}
	var f FloatInfo
	switch d {
	case Float16:
		f = FloatInfo{Bits: 16, Mantissa: 10, Exponent: 5, MinExp: -14, MaxExp: 16, Precision: 3}
	case Float32:
		f = FloatInfo{Bits: 32, Mantissa: 23, Exponent: 8, MinExp: -126, MaxExp: 128, Precision: 6}
	case Float64:
		f = FloatInfo{Bits: 64, Mantissa: 52, Exponent: 11, MinExp: -1022, MaxExp: 1024, Precision: 15}
	default:
		return f, ErrDType
	}
	f.Eps = math.Ldexp(1, -f.Mantissa)
	f.EpsNeg = f.Eps / 2
	f.Max = math.Ldexp(2-f.Eps, f.MaxExp-1)
	f.Min = -f.Max
	f.Tiny = math.Ldexp(1, f.MinExp)
	f.SmallestSubnormal = math.Ldexp(1, f.MinExp-f.Mantissa)
	r := math.Pow10(-f.Precision)
	if d == Float16 {
		r = HalfFromFloat64(r).Float64()
	} else if d == Float32 {
		r = float64(float32(r))
	}
	f.Resolution = r
	return f, nil
}

func CommonType(arrays ...*Array) (DType, error) {
	precision := 0
	complexType := false
	for _, a := range arrays {
		d := a.DType()
		switch d {
		case Float16:
		case Float32, Complex64:
			precision = max(precision, 1)
		case Float64, Complex128:
			precision = max(precision, 2)
		case Float128, Complex256:
			precision = 3
		default:
			if d.IsInteger() {
				precision = max(precision, 2)
			} else {
				return 0, ErrDType
			}
		}
		complexType = complexType || d.IsComplex()
	}
	if complexType {
		if precision == 3 {
			return Complex256, nil
		}
		if precision < 2 {
			return Complex64, nil
		}
		return Complex128, nil
	}
	return []DType{Float16, Float32, Float64, Float128}[precision], nil
}

// MinScalarType returns the smallest NumPy fixed-width scalar dtype that
// accommodates a value. Non-scalar arrays retain their existing dtype.
func MinScalarType(value any) (DType, error) {
	if a, ok := value.(*Array); ok {
		if a.DType() == 0 {
			return 0, ErrDType
		}
		if a.NDim() != 0 {
			return a.DType(), nil
		}
		v, e := a.AtValue()
		if e != nil {
			return 0, e
		}
		value = v
	}
	d := dtypeOf(value)
	if d == 0 {
		return 0, ErrDType
	}
	e := elementFrom(value)
	if d == Bool {
		return d, nil
	}
	if d.IsInteger() {
		if d.IsSigned() && int64(e.i) < 0 {
			v := int64(e.i)
			for _, t := range []DType{Int8, Int16, Int32, Int64} {
				f, _ := IInfo(t)
				if v >= f.Min.(int64) {
					return t, nil
				}
			}
		}
		for _, t := range []DType{Uint8, Uint16, Uint32, Uint64} {
			f, _ := IInfo(t)
			if e.i <= f.Max.(uint64) {
				return t, nil
			}
		}
	}
	if d.IsFloat() {
		x := math.Abs(real(e.z))
		if d == Float16 || math.IsInf(x, 0) || math.IsNaN(x) || x < 65000 {
			return Float16, nil
		}
		if d == Float32 || x < 3.4e38 {
			return Float32, nil
		}
		return Float64, nil
	}
	if d.IsComplex() {
		if d == Complex64 || math.Abs(real(e.z)) < 3.4e38 && math.Abs(imag(e.z)) < 3.4e38 {
			return Complex64, nil
		}
		return Complex128, nil
	}
	return 0, ErrDType
}

// IsSubdtype implements the numeric portion of NumPy's scalar type hierarchy.
// Concrete types are matched by dtype name, abstract kinds by NumPy class name.
func IsSubdtype(d DType, kind string) (bool, error) {
	if d.ItemSize() == 0 {
		return false, ErrDType
	}
	switch kind {
	case "generic":
		return true, nil
	case "number":
		return d.IsFloat() || d.IsComplex() || d.IsInteger(), nil
	case "inexact":
		return d.IsFloat() || d.IsComplex(), nil
	case "integer":
		return d.IsInteger(), nil
	case "signedinteger":
		return d.IsSigned(), nil
	case "unsignedinteger":
		return d.IsUnsigned(), nil
	case "floating":
		return d.IsFloat(), nil
	case "complexfloating":
		return d.IsComplex(), nil
	case "bool_":
		return d == Bool, nil
	}
	t, e := ParseDType(kind)
	if e != nil {
		return false, e
	}
	return d == t, nil
}

func MinTypeCode(typechars, typeset, defaultCode string) (string, error) {
	if typeset == "" {
		typeset = "GDFgdf"
	}
	if defaultCode == "" {
		defaultCode = "d"
	}
	const rank = "GDFgdfQqLlIiHhBb?"
	if strings.ContainsRune(typechars, 'F') && strings.ContainsRune(typechars, 'd') && strings.ContainsRune(typeset, 'F') && strings.ContainsRune(typeset, 'd') {
		return "D", nil
	}
	best := len(rank)
	found := false
	for _, c := range typechars {
		if strings.ContainsRune(typeset, c) {
			i := strings.IndexRune(rank, c)
			if i < 0 {
				return "", ErrValue
			}
			best = min(best, i)
			found = true
		}
	}
	if !found {
		return defaultCode, nil
	}
	return string(rank[best]), nil
}

func TypeName(code string) (string, error) {
	names := map[string]string{"S1": "character", "?": "bool", "b": "signed char", "B": "unsigned char", "h": "short", "H": "unsigned short", "i": "integer", "I": "unsigned integer", "l": "long integer", "L": "unsigned long integer", "q": "long long integer", "Q": "unsigned long long integer", "f": "single precision", "d": "double precision", "g": "long precision", "F": "complex single precision", "D": "complex double precision", "G": "complex long double precision", "S": "string", "U": "unicode", "V": "void", "O": "object"}
	v, ok := names[code]
	if !ok {
		return "", ErrValue
	}
	return v, nil
}
