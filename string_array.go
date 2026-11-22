package np

import (
	"math"
	"strconv"
	"strings"
	"unicode/utf8"
)

// FromStrings creates a variable-length UTF-8 ndarray. Unlike fixed-width
// Unicode, it retains trailing NULs and never truncates an assignment.
func FromStrings(values []string, shape ...int) (*Array, error) {
	a, err := Zeros(StringDType, shape...)
	if err != nil {
		return nil, err
	}
	if len(values) != a.Size() {
		return nil, ErrShape
	}
	for i, s := range values {
		if !utf8.ValidString(s) {
			return nil, ErrValue
		}
		a.data.raw.([]any)[i] = s
	}
	return a, nil
}

// FromStringValues supports NumPy's coerce=False descriptor via strict=true.
func FromStringValues(values []any, strict bool, shape ...int) (*Array, error) {
	return FromStringDescriptor(values, StringDescriptor{Strict: strict}, shape...)
}

// StringDescriptor carries the variable-length dtype's coercion and missing
// sentinel. HasNA distinguishes an omitted sentinel from an explicit nil.
// The Go equivalent accepts immutable scalar sentinels (nil, string, bool,
// signed/unsigned integers and floats), including NaN.
type StringDescriptor struct {
	Strict bool
	HasNA  bool
	NA     any
}
type stringNull struct{ value any }

func stringNaN(e element) bool {
	n, ok := e.extra.(stringNull)
	if !ok {
		return false
	}
	switch x := n.value.(type) {
	case float64:
		return math.IsNaN(x)
	case float32:
		return math.IsNaN(float64(x))
	}
	return false
}
func stringNonTextNull(e element) bool {
	n, ok := e.extra.(stringNull)
	if !ok {
		return false
	}
	_, text := n.value.(string)
	return !text
}
func stringBadOrder(e element) bool { return stringNonTextNull(e) && !stringNaN(e) }
func stringAdd(x, y element) (element, error) {
	if stringBadOrder(x) || stringBadOrder(y) {
		return element{}, ErrValue
	}
	if stringNaN(x) {
		return x, nil
	}
	if stringNaN(y) {
		return y, nil
	}
	return element{extra: flexKey(x) + flexKey(y)}, nil
}
func stringNullEqual(a, b any) bool {
	if stringNaN(element{extra: stringNull{a}}) && stringNaN(element{extra: stringNull{b}}) {
		return true
	}
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	switch a.(type) {
	case string, bool, int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64, float32, float64:
		return a == b
	}
	return false
}
func (d StringDescriptor) validate() error {
	if !d.HasNA {
		return nil
	}
	switch x := d.NA.(type) {
	case string:
		if !utf8.ValidString(x) {
			return ErrValue
		}
	case nil, bool, int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64, float32, float64:
	default:
		return ErrDType
	}
	return nil
}
func (d StringDescriptor) String() string {
	parts := []string{}
	if d.HasNA {
		s := stringScalarRepr(d.NA)
		parts = append(parts, "na_object="+s)
	}
	if d.Strict {
		parts = append(parts, "coerce=False")
	}
	return "StringDType(" + strings.Join(parts, ", ") + ")"
}
func stringScalarRepr(v any) string {
	if s, ok := v.(string); ok {
		return flexibleRepr(s)
	}
	s, e := stringValue(v, false)
	if e != nil {
		return flexibleRepr(v)
	}
	return s
}
func (a *Array) StringDescriptor() (StringDescriptor, error) {
	if a.DType() != StringDType {
		return StringDescriptor{}, ErrDType
	}
	return a.stringDescriptor(), nil
}
func (a *Array) stringDescriptor() StringDescriptor {
	d := StringDescriptor{Strict: a.data.strictString}
	if a.data.stringNA != nil {
		d.HasNA = true
		d.NA = a.data.stringNA.value
	}
	return d
}
func (a *Array) setStringDescriptor(d StringDescriptor) {
	a.data.strictString = d.Strict
	a.data.stringNA = nil
	if d.HasNA {
		a.data.stringNA = &stringNull{d.NA}
	}
}
func sameStringDescriptor(a, b StringDescriptor) bool {
	return a.Strict == b.Strict && a.HasNA == b.HasNA && (!a.HasNA || stringNullEqual(a.NA, b.NA))
}
func mergeStringDescriptors(arrays ...*Array) (StringDescriptor, error) {
	var d StringDescriptor
	for _, a := range arrays {
		if a.DType() != StringDType {
			continue
		}
		f := a.stringDescriptor()
		d.Strict = d.Strict || f.Strict
		if f.HasNA {
			if d.HasNA && !stringNullEqual(f.NA, d.NA) {
				return d, ErrDType
			}
			d.HasNA, d.NA = true, f.NA
		}
	}
	return d, nil
}

// PromoteStringDescriptors resolves coercion and sentinel compatibility for
// string kernels with variable-length operands.
func PromoteStringDescriptors(arrays ...*Array) (StringDescriptor, error) {
	return mergeStringDescriptors(arrays...)
}
func (a *Array) convertString(v any) (element, error) {
	if a.data.stringNA != nil && stringNullEqual(a.data.stringNA.value, v) {
		return element{extra: *a.data.stringNA}, nil
	}
	s, e := stringValue(v, a.data.strictString)
	return element{extra: s}, e
}

// FromStringDescriptor constructs a nullable ndarray. Missing slots retain
// the sentinel even when equal text produced by a string kernel does not.
func FromStringDescriptor(values []any, d StringDescriptor, shape ...int) (*Array, error) {
	if e := d.validate(); e != nil {
		return nil, e
	}
	a, err := Zeros(StringDType, shape...)
	if err != nil {
		return nil, err
	}
	if len(values) != a.Size() {
		return nil, ErrShape
	}
	a.setStringDescriptor(d)
	for i, v := range values {
		s, err := a.convertString(v)
		if err != nil {
			return nil, err
		}
		a.setElement(i, s)
	}
	return a, nil
}

func (a *Array) StringStrict() (bool, error) {
	if a.DType() != StringDType {
		return false, ErrDType
	}
	return a.data.strictString, nil
}

// CastStrings explicitly selects the variable-string coercion descriptor.
func (a *Array) CastStrings(strict bool) (*Array, error) {
	return a.castStrings(strict)
}
func (a *Array) CastStringDescriptor(d StringDescriptor) (*Array, error) {
	return a.castStringDescriptor(d)
}

func stringValue(v any, strict bool) (string, error) {
	var s string
	switch x := v.(type) {
	case string:
		s = x
	case TextScalar:
		s = string(x)
	default:
		if strict {
			return "", ErrValue
		}
		switch x := v.(type) {
		case nil:
			s = "None"
		case BytesScalar:
			s = string(x)
		case TimeDelta:
			if x.IsNaT() {
				s = "NaT"
			} else {
				s = strconv.FormatInt(x.Ticks, 10)
			}
		default:
			var err error
			s, err = flexibleNumericText(v)
			if err != nil {
				return "", err
			}
		}
	}
	if !utf8.ValidString(s) {
		return "", ErrValue
	}
	return s, nil
}

func (a *Array) castStrings(strict bool) (*Array, error) {
	return a.castStringDescriptor(StringDescriptor{Strict: strict})
}
func (a *Array) castStringDescriptor(d StringDescriptor) (*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	if e := d.validate(); e != nil {
		return nil, e
	}
	r, err := Zeros(StringDType, a.shape...)
	if err != nil {
		return nil, err
	}
	r.setStringDescriptor(d)
	sameNA := a.DType() == StringDType && a.stringDescriptor().HasNA == d.HasNA && (!d.HasNA || stringNullEqual(a.data.stringNA.value, d.NA))
	a.walk(func(i, off int) {
		if err != nil {
			return
		}
		if sameNA {
			r.setElement(i, a.elementAt(off))
			return
		}
		v := a.rawAt(off)
		if s, ok := v.(BytesScalar); ok {
			v = string(s)
		}
		var s element
		s, err = r.convertString(v)
		if err == nil {
			r.setElement(i, s)
		}
	})
	if err != nil {
		return nil, err
	}
	return r, nil
}

// CharacterValues returns the logical strings in C order for S, U and
// StringDType. S strings contain uninterpreted bytes, U and StringDType UTF-8.
func (a *Array) CharacterValues() ([]string, error) {
	if !a.DType().IsCharacter() && a.DType() != StringDType {
		return nil, ErrDType
	}
	v := make([]string, a.size)
	var err error
	a.walk(func(i, off int) {
		e := a.elementAt(off)
		if a.DType() == StringDType && stringNonTextNull(e) {
			err = ErrDType
		}
		v[i] = flexKey(e)
	})
	if err != nil {
		return nil, err
	}
	return v, nil
}

// StringValues returns exact logical Go scalars and the physical missing
// mask in C order, including computed strings equal to a string sentinel.
func (a *Array) StringValues() ([]any, []bool, error) {
	if a.DType() != StringDType {
		return nil, nil, ErrDType
	}
	v := make([]any, a.size)
	missing := make([]bool, a.size)
	a.walk(func(i, off int) {
		x := a.elementAt(off)
		v[i] = a.rawAt(off)
		_, missing[i] = x.extra.(stringNull)
	})
	return v, missing, nil
}

// FromStringSlots materializes explicit text and missing slots without
// reinterpreting computed text equal to the string sentinel as missing.
func FromStringSlots(values []string, missing []bool, d StringDescriptor, shape ...int) (*Array, error) {
	if e := d.validate(); e != nil {
		return nil, e
	}
	a, e := FromStrings(values, shape...)
	if e != nil {
		return nil, e
	}
	if missing != nil && len(missing) != a.size {
		return nil, ErrShape
	}
	a.setStringDescriptor(d)
	for i, m := range missing {
		if m {
			if !d.HasNA {
				return nil, ErrDType
			}
			a.setElement(i, element{extra: *a.data.stringNA})
		}
	}
	return a, nil
}

func stringPair(a, b *Array) (*Array, *Array, error) {
	if (a.DType() != StringDType && a.DType() != UnicodeDType) ||
		(b.DType() != StringDType && b.DType() != UnicodeDType) {
		return nil, nil, ErrDType
	}
	d, err := mergeStringDescriptors(a, b)
	if err != nil {
		return nil, nil, err
	}
	a, err = a.castStringDescriptor(d)
	if err != nil {
		return nil, nil, err
	}
	b, err = b.castStringDescriptor(d)
	return a, b, err
}

// RepeatStrings is the numpy.strings.multiply contract; unlike the direct
// fixed-width multiply ufunc, it infers the result width from the output.
func RepeatStrings(a, b *Array) (*Array, error) {
	return stringRepeat(a, b)
}

// stringRepeat uses the actual broadcasted output lengths for the fixed
// character descriptor. Negative repeat counts produce an empty string.
func stringRepeat(a, b *Array) (*Array, error) {
	return repeatStrings(a, b, nil)
}

func repeatStrings(a, b *Array, fixedOut *FlexibleDType) (*Array, error) {
	if b.DType().IsCharacter() || b.DType() == StringDType {
		a, b = b, a
	}
	d := a.DType()
	if (!d.IsCharacter() && d != StringDType) || (!b.DType().IsInteger() && b.DType() != Bool) {
		return nil, ErrDType
	}
	if fixedOut != nil && fixedOut.Kind != d {
		return nil, ErrDType
	}
	r, err := binaryMapCore(a, b, 0, StringDType, func(x, y element) (element, error) {
		if d == StringDType && stringNonTextNull(x) {
			if stringNaN(x) {
				return x, nil
			}
			return element{}, ErrDType
		}
		s := flexKey(x)
		if b.DType().IsSigned() && int64(y.i) <= 0 || y.i == 0 || s == "" {
			return element{extra: ""}, nil
		}
		count := y.i
		if fixedOut != nil {
			n := len(s)
			if d == UnicodeDType {
				n = utf8.RuneCountInString(s)
			}
			if fixedOut.Length == 0 {
				return element{extra: ""}, nil
			}
			count = min(count, uint64(1+(fixedOut.Length-1)/n))
		}
		if count > uint64(maxInt/len(s)) {
			return element{}, ErrShape
		}
		return element{extra: strings.Repeat(s, int(count))}, nil
	})
	if err == nil && d == StringDType {
		r.setStringDescriptor(a.stringDescriptor())
	}
	if err != nil || d == StringDType {
		return r, err
	}
	if fixedOut == nil && r.size == 0 {
		return nil, ErrValue
	}
	v := make([]string, r.size)
	for i, x := range r.data.raw.([]any) {
		v[i] = x.(string)
	}
	width := 1
	for _, s := range v {
		n := len(s)
		if d == UnicodeDType {
			n = utf8.RuneCountInString(s)
		}
		width = max(width, n)
	}
	if fixedOut != nil {
		width = fixedOut.Length
		if d == UnicodeDType {
			values := make([]any, len(v))
			for i, s := range v {
				values[i] = TextScalar(s)
			}
			return FromFlexible(values, *fixedOut, r.shape...)
		}
	}
	if d == UnicodeDType {
		return FromUnicodeStrings(v, width, r.shape...)
	}
	return FromByteStrings(v, width, r.shape...)
}
