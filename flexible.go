package np

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/big"
	"slices"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/SamuelSupe/numpygo/internal/x87"
)

// BytesScalar, TextScalar and VoidScalar are the Go values of fixed-width
// byte-string, UCS-4 Unicode and opaque ndarray elements.
type BytesScalar string
type TextScalar string
type VoidScalar []byte

func (b BytesScalar) Len() int      { return len(b) }
func (s TextScalar) Len() int       { return utf8.RuneCountInString(string(s)) }
func (s TextScalar) String() string { return strings.TrimRight(string(s), "\x00") }

// FlexibleDType carries the item-specific width absent from the DType enum.
// Length is bytes for S/V and Unicode code points for U, including zero.
type FlexibleDType struct {
	Kind   DType
	Length int
	Order  binary.ByteOrder
}

func FixedBytes(length int) FlexibleDType { return FlexibleDType{Kind: BytesDType, Length: length} }
func FixedUnicode(length int, order binary.ByteOrder) FlexibleDType {
	return FlexibleDType{Kind: UnicodeDType, Length: length, Order: order}
}
func FixedVoid(length int) FlexibleDType { return FlexibleDType{Kind: VoidDType, Length: length} }
func (d FlexibleDType) width() (int, error) {
	if !d.Kind.IsFlexible() || d.Length < 0 {
		return 0, ErrDType
	}
	if d.Kind == UnicodeDType {
		if d.Length > maxInt/4 {
			return 0, ErrShape
		}
		return d.Length * 4, nil
	}
	return d.Length, nil
}
func (d FlexibleDType) ItemSize() int { n, _ := d.width(); return n }
func (d FlexibleDType) byteOrder() binary.ByteOrder {
	if d.Order == nil {
		return binary.NativeEndian
	}
	return d.Order
}
func (d FlexibleDType) native() FlexibleDType { d.Order = binary.NativeEndian; return d }
func (d FlexibleDType) Equal(e FlexibleDType) bool {
	if d.Kind != e.Kind || d.Length != e.Length {
		return false
	}
	return d.Kind != UnicodeDType || d.byteOrder().Uint16([]byte{1, 0}) == e.byteOrder().Uint16([]byte{1, 0})
}
func (d FlexibleDType) String() string {
	switch d.Kind {
	case BytesDType:
		return fmt.Sprintf("|S%d", d.Length)
	case VoidDType:
		return fmt.Sprintf("|V%d", d.Length)
	case UnicodeDType:
		endian := "<"
		if d.byteOrder().Uint16([]byte{1, 0}) != 1 {
			endian = ">"
		}
		return fmt.Sprintf("%sU%d", endian, d.Length)
	default:
		return "invalid"
	}
}
func ParseFlexibleDType(s string) (FlexibleDType, error) {
	if len(s) < 2 {
		return FlexibleDType{}, ErrDType
	}
	var o binary.ByteOrder = binary.NativeEndian
	switch s[0] {
	case '<':
		o = binary.LittleEndian
		s = s[1:]
	case '>':
		o = binary.BigEndian
		s = s[1:]
	case '|', '=':
		s = s[1:]
	}
	if len(s) < 2 {
		return FlexibleDType{}, ErrDType
	}
	n, e := strconv.Atoi(s[1:])
	if e != nil || n < 0 {
		return FlexibleDType{}, ErrDType
	}
	var d FlexibleDType
	switch s[0] {
	case 'S':
		d = FixedBytes(n)
	case 'U':
		d = FixedUnicode(n, o)
	case 'V':
		d = FixedVoid(n)
	default:
		return d, ErrDType
	}
	_, e = d.width()
	return d, e
}
func (a *Array) FlexibleType() (FlexibleDType, error) {
	if !a.DType().IsFlexible() {
		return FlexibleDType{}, ErrDType
	}
	return a.data.flex, nil
}
func (a *Array) flexBytes(off int) []byte {
	start := off * a.strideUnit()
	return a.data.buffer[start : start+a.ItemSize()]
}

// FromFlexibleBuffer shares the actual fixed-width storage, including its
// Unicode byte order and zero-width items. Shape is in logical elements.
func FromFlexibleBuffer(buffer []byte, d FlexibleDType, shape []int, readonly bool) (*Array, error) {
	width, e := d.width()
	if e != nil {
		return nil, e
	}
	n, e := shapeSizeWidth(max(width, 1), shape)
	if e != nil {
		return nil, e
	}
	if len(buffer) != n*width {
		return nil, ErrShape
	}
	if buffer == nil {
		buffer = make([]byte, 0)
	}
	d.Order = d.byteOrder()
	a := owned(&storage{dtype: d.Kind, flex: d, buffer: buffer, order: d.byteOrder()}, shape, n)
	a.readOnly = readonly
	if readonly {
		a.owner = false
	}
	return a, nil
}
func ZerosFlexible(d FlexibleDType, shape ...int) (*Array, error) {
	w, e := d.width()
	if e != nil {
		return nil, e
	}
	n, e := shapeSizeWidth(max(w, 1), shape)
	if e != nil {
		return nil, e
	}
	return FromFlexibleBuffer(make([]byte, n*w), d, shape, false)
}
func FromFlexible(values []any, d FlexibleDType, shape ...int) (*Array, error) {
	a, e := ZerosFlexible(d, shape...)
	if e != nil {
		return nil, e
	}
	if len(values) != a.size {
		return nil, ErrShape
	}
	for i, v := range values {
		z, e := a.convertValue(v)
		if e != nil {
			return nil, e
		}
		a.setElement(i, z)
	}
	return a, nil
}
func FromByteStrings(values []string, length int, shape ...int) (*Array, error) {
	v := make([]any, len(values))
	for i, x := range values {
		v[i] = BytesScalar(x)
	}
	return FromFlexible(v, FixedBytes(length), shape...)
}
func FromUnicodeStrings(values []string, length int, shape ...int) (*Array, error) {
	v := make([]any, len(values))
	for i, x := range values {
		v[i] = TextScalar(x)
	}
	return FromFlexible(v, FixedUnicode(length, nil), shape...)
}

// flexElement retains raw bytes together with the descriptor so gathers,
// copies and strided writes preserve padding and even opaque bit patterns.
type flexElement struct {
	raw   []byte
	dtype FlexibleDType
}

func decodeFlexible(raw []byte, d FlexibleDType) any {
	switch d.Kind {
	case BytesDType:
		return BytesScalar(strings.TrimRight(string(raw), "\x00"))
	case UnicodeDType:
		end := d.Length
		o := d.byteOrder()
		for end > 0 && o.Uint32(raw[4*(end-1):]) == 0 {
			end--
		}
		r := make([]rune, end)
		for i := range r {
			r[i] = rune(o.Uint32(raw[i*4:]))
		}
		return TextScalar(string(r))
	case VoidDType:
		return VoidScalar(slices.Clone(raw))
	}
	return nil
}
func flexScalar(e element) any {
	if x, ok := e.extra.(flexElement); ok {
		return decodeFlexible(x.raw, x.dtype)
	}
	return e.extra
}
func flexibleRepr(v any) string {
	if x, ok := v.(VoidScalar); ok {
		var b strings.Builder
		b.WriteString("b'")
		for _, c := range x {
			fmt.Fprintf(&b, "\\x%02X", c)
		}
		b.WriteByte('\'')
		return b.String()
	}
	var s string
	isBytes := false
	switch x := v.(type) {
	case BytesScalar:
		s = string(x)
		isBytes = true
	case TextScalar:
		s = string(x)
	case string:
		s = x
	default:
		return fmt.Sprint(v)
	}
	quote := byte('\'')
	if strings.ContainsRune(s, '\'') && !strings.ContainsRune(s, '"') {
		quote = '"'
	}
	var b strings.Builder
	if isBytes {
		b.WriteByte('b')
	}
	b.WriteByte(quote)
	write := func(r rune) {
		switch r {
		case '\\':
			b.WriteString("\\\\")
		case '\t':
			b.WriteString("\\t")
		case '\n':
			b.WriteString("\\n")
		case '\r':
			b.WriteString("\\r")
		default:
			if r == rune(quote) {
				b.WriteByte('\\')
				b.WriteByte(quote)
			} else if isBytes && (r < 32 || r >= 127) || !isBytes && !unicode.IsPrint(r) {
				if r <= 0xff {
					fmt.Fprintf(&b, "\\x%02x", r)
				} else if r <= 0xffff {
					fmt.Fprintf(&b, "\\u%04x", r)
				} else {
					fmt.Fprintf(&b, "\\U%08x", r)
				}
			} else {
				b.WriteRune(r)
			}
		}
	}
	if isBytes {
		for i := 0; i < len(s); i++ {
			write(rune(s[i]))
		}
	} else {
		for _, r := range s {
			write(r)
		}
	}
	b.WriteByte(quote)
	return b.String()
}
func flexKey(e element) string {
	switch v := flexScalar(e).(type) {
	case BytesScalar:
		return string(v)
	case TextScalar:
		return string(v)
	case VoidScalar:
		return string(v)
	case string:
		return v
	case stringNull:
		s, _ := v.value.(string)
		return s
	}
	return ""
}
func flexOrderKey(e element, d DType) string {
	if d == UnicodeDType {
		if x, ok := e.extra.(flexElement); ok {
			n := x.dtype.Length
			o := x.dtype.byteOrder()
			for n > 0 && o.Uint32(x.raw[4*(n-1):]) == 0 {
				n--
			}
			key := make([]byte, n*4)
			for i := 0; i < n; i++ {
				binary.BigEndian.PutUint32(key[4*i:], o.Uint32(x.raw[4*i:]))
			}
			return string(key)
		}
	}
	return flexKey(e)
}
func flexTruth(e element, d DType) bool {
	if d == VoidDType {
		v, ok := e.extra.(flexElement)
		if ok {
			for _, b := range v.raw {
				if b != 0 {
					return true
				}
			}
			return false
		}
		if b, ok := e.extra.(VoidScalar); ok {
			for _, x := range b {
				if x != 0 {
					return true
				}
			}
		}
		return false
	}
	return flexKey(e) != ""
}
func ascii(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] >= 128 {
			return false
		}
	}
	return true
}
func formatNativeFloat(x float64, bits int) string {
	switch {
	case math.IsNaN(x):
		return "nan"
	case math.IsInf(x, 1):
		return "inf"
	case math.IsInf(x, -1):
		return "-inf"
	}
	style := byte('f')
	if x != 0 && (math.Abs(x) < 1e-4 || math.Abs(x) >= 1e16) {
		style = 'e'
	}
	s := strconv.FormatFloat(x, style, -1, bits)
	if style == 'f' && !strings.Contains(s, ".") {
		s += ".0"
	}
	return s
}
func formatHalf(h Half) string {
	x := h.Float64()
	if math.IsNaN(x) || math.IsInf(x, 0) {
		return formatNativeFloat(x, 64)
	}
	// Find the shortest decimal significand that round-trips to binary16.
	for digits := 1; digits <= 5; digits++ {
		s := strconv.FormatFloat(x, 'g', digits, 64)
		y, e := strconv.ParseFloat(s, 64)
		if e == nil && HalfFromFloat64(y) == h {
			return formatNativeFloat(y, 64)
		}
	}
	return formatNativeFloat(x, 64)
}
func formatExtended(x x87.Float) string {
	s := x.String()
	if x.IsFinite() && !strings.ContainsAny(s, ".eE") {
		s += ".0"
	}
	return s
}
func formatComplexParts(r, i string, reZero, negativeReal, negativeImag bool) string {
	trim := func(s string) string { return strings.TrimSuffix(s, ".0") }
	r, i = trim(r), trim(i)
	if reZero && !negativeReal {
		return i + "j"
	}
	sign := "+"
	if negativeImag || strings.HasPrefix(i, "-") {
		sign = "-"
		i = strings.TrimPrefix(i, "-")
	}
	return "(" + r + sign + i + "j)"
}
func flexibleNumericText(v any) (string, error) {
	switch x := v.(type) {
	case DateTime:
		return x.AsString("")
	case TimeDelta:
		if x.IsNaT() {
			return "NaT", nil
		}
		names := map[string]string{"Y": "years", "M": "months", "W": "weeks", "D": "days", "h": "hours", "m": "minutes", "s": "seconds", "ms": "milliseconds", "us": "microseconds", "ns": "nanoseconds", "ps": "picoseconds", "fs": "femtoseconds", "as": "attoseconds", "generic": "generic time units"}
		n := new(big.Int).Mul(big.NewInt(x.Ticks), big.NewInt(x.Unit.Step))
		return n.String() + " " + names[x.Unit.Name], nil
	case bool:
		if x {
			return "True", nil
		}
		return "False", nil
	case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64:
		return fmt.Sprint(x), nil
	case Half:
		return formatHalf(x), nil
	case float32:
		return formatNativeFloat(float64(x), 32), nil
	case float64:
		return formatNativeFloat(x, 64), nil
	case x87.Float:
		return formatExtended(x), nil
	case complex64:
		z := complex128(x)
		return formatComplexParts(formatNativeFloat(real(z), 32), formatNativeFloat(imag(z), 32), real(z) == 0, math.Signbit(real(z)), math.Signbit(imag(z))), nil
	case complex128:
		return formatComplexParts(formatNativeFloat(real(x), 64), formatNativeFloat(imag(x), 64), real(x) == 0, math.Signbit(real(x)), math.Signbit(imag(x))), nil
	case x87.Complex:
		return formatComplexParts(formatExtended(x.Real), formatExtended(x.Imag), x.Real.IsZero(), x.Real.Signbit(), x.Imag.Signbit()), nil
	}
	return "", ErrDType
}
func encodeFlexible(v any, d FlexibleDType) (flexElement, error) {
	w, e := d.width()
	if e != nil {
		return flexElement{}, e
	}
	b := make([]byte, w)
	if value, ok := v.(flexElement); ok {
		if value.dtype.Equal(d) {
			copy(b, value.raw)
			return flexElement{b, d}, nil
		}
		v = decodeFlexible(value.raw, value.dtype)
	}
	if d.Kind == VoidDType {
		switch x := v.(type) {
		case VoidScalar:
			copy(b, x)
		case BytesScalar:
			copy(b, string(x))
		case []byte:
			copy(b, x)
		default:
			return flexElement{}, ErrDType
		}
		return flexElement{b, d}, nil
	}
	var s string
	byteInput := false
	switch x := v.(type) {
	case BytesScalar:
		s = string(x)
		byteInput = true
	case []byte:
		s = string(x)
		byteInput = true
	case TextScalar:
		s = string(x)
	case string:
		s = x
	default:
		s, e = flexibleNumericText(v)
		if e != nil {
			return flexElement{}, e
		}
	}
	if d.Kind == BytesDType {
		if !byteInput && !ascii(s) {
			return flexElement{}, ErrValue
		}
		copy(b, s)
	} else {
		if byteInput && !ascii(s) {
			return flexElement{}, ErrValue
		}
		if !byteInput && !utf8.ValidString(s) {
			return flexElement{}, ErrValue
		}
		i := 0
		o := d.byteOrder()
		for _, r := range s {
			if i == d.Length {
				break
			}
			o.PutUint32(b[i*4:], uint32(r))
			i++
		}
	}
	return flexElement{b, d}, nil
}

// FlexibleStringWidth is NumPy's default fixed-width numeric-to-character
// cast capacity, not the length of any particular scalar.
func FlexibleStringWidth(d DType) (int, error) {
	switch d {
	case Bool:
		return 5, nil
	case Int8:
		return 4, nil
	case Int16:
		return 6, nil
	case Int32:
		return 11, nil
	case Int64:
		return 21, nil
	case Uint8:
		return 3, nil
	case Uint16:
		return 5, nil
	case Uint32:
		return 10, nil
	case Uint64:
		return 20, nil
	case Float16, Float32, Float64:
		return 32, nil
	case Float128:
		return 48, nil
	case Complex64, Complex128:
		return 64, nil
	case Complex256:
		return 96, nil
	}
	return 0, ErrDType
}
func flexibleDescriptorFor(arrays []*Array, d DType) (FlexibleDType, error) {
	result := FlexibleDType{Kind: d, Order: binary.NativeEndian}
	got := false
	for _, a := range arrays {
		n := 0
		if a.DType().IsFlexible() {
			f := a.data.flex
			if d == VoidDType {
				if f.Kind != VoidDType || got && result.Length != f.Length {
					return result, ErrDType
				}
			} else if !f.Kind.IsCharacter() {
				return result, ErrDType
			}
			n = f.Length
		} else {
			if d == VoidDType {
				return result, ErrDType
			}
			var e error
			if a.DType().IsTemporal() {
				n, e = temporalStringWidth(a.DType(), a.data.unit)
			} else {
				n, e = FlexibleStringWidth(a.DType())
			}
			if e != nil {
				return result, e
			}
		}
		result.Length = max(result.Length, n)
		got = true
	}
	_, e := result.width()
	return result, e
}

// CastFlexible converts actual scalar values to an explicit width and order.
// Incompatible or non-ASCII byte/Unicode conversions fail before returning.
func (a *Array) CastFlexible(d FlexibleDType) (*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	out, e := ZerosFlexible(d, a.shape...)
	if e != nil {
		return nil, e
	}
	a.walk(func(i, off int) {
		if e != nil {
			return
		}
		var v any = a.rawAt(off)
		if a.DType() == StringDType {
			if _, missing := a.elementAt(off).extra.(stringNull); missing {
				var err error
				v, err = stringValue(v, false)
				if err != nil {
					e = err
					return
				}
			}
		}
		if a.DType().IsFlexible() {
			v = a.elementAt(off).extra
		}
		z, err := encodeFlexible(v, d)
		if err != nil {
			e = err
			return
		}
		out.setElement(i, element{extra: z})
	})
	if e != nil {
		return nil, e
	}
	return out, nil
}

// CanCastFlexible tests actual widths in addition to dtype families.
func (a *Array) CanCastFlexible(d FlexibleDType, rule string) (bool, error) {
	if a.DType() == 0 {
		return false, ErrDType
	}
	if _, e := d.width(); e != nil {
		return false, e
	}
	if rule == "unsafe" {
		return true, nil
	}
	from := a.DType()
	if rule == "no" || rule == "equiv" {
		if !from.IsFlexible() {
			return false, nil
		}
		if rule == "no" {
			return a.data.flex.Equal(d), nil
		}
		return from == d.Kind && a.data.flex.Length == d.Length, nil
	}
	if rule != "safe" && rule != "same_kind" && rule != "" {
		return false, ErrValue
	}
	ok, e := CanCast(from, d.Kind, rule)
	if e != nil || !ok {
		return ok, e
	}
	if rule == "same_kind" {
		return true, nil
	}
	if from.IsFlexible() {
		return a.data.flex.Length <= d.Length && (from != VoidDType || a.data.flex.Length == d.Length), nil
	}
	n, e := FlexibleStringWidth(from)
	if e != nil {
		return false, nil
	}
	return n <= d.Length, nil
}
func (a *Array) CastFlexibleWithOptions(d FlexibleDType, opt CastOptions) (*Array, error) {
	rule := opt.Casting
	if rule == "" {
		rule = "unsafe"
	}
	ok, e := a.CanCastFlexible(d, rule)
	if e != nil {
		return nil, e
	}
	if !ok {
		return nil, ErrDType
	}
	order := opt.Order
	switch order {
	case "", "K":
		order = "K"
	case "A":
		order = "C"
		if a.IsFContiguous() && !a.IsCContiguous() {
			order = "F"
		}
	case "C", "F":
	default:
		return nil, ErrValue
	}
	if opt.NoCopy && a.DType().IsFlexible() && a.data.flex.Equal(d) && (order == "K" || order == "C" && a.IsCContiguous() || order == "F" && a.IsFContiguous()) {
		return a, nil
	}
	if order == "C" || a.NDim() < 2 || order == "K" && a.IsCContiguous() {
		return a.CastFlexible(d)
	}
	axes, inverse := layoutAxes(a, order)
	v, e := a.Transpose(axes...)
	if e != nil {
		return nil, e
	}
	v, e = v.CastFlexible(d)
	if e != nil {
		return nil, e
	}
	shape, strides := make([]int, len(inverse)), make([]int, len(inverse))
	for i, ax := range inverse {
		shape[i], strides[i] = v.shape[ax], v.strides[ax]
	}
	v.shape, v.strides = shape, strides
	return v, nil
}

// ViewFlexible reinterprets storage without conversion, including packed
// fields whose strides are not multiples of their item width.
func (a *Array) ViewFlexible(d FlexibleDType) (*Array, error) {
	if a.DType() == 0 || a.rawReferenceStorage() {
		return nil, ErrDType
	}
	w, e := d.width()
	if e != nil {
		return nil, e
	}
	old := a.ItemSize()
	sh, st := a.Shape(), a.Strides()
	if old != w {
		if len(sh) == 0 || old == 0 || w == 0 {
			return nil, ErrShape
		}
		last := len(sh) - 1
		if sh[last] > 1 && st[last] != old || sh[last]*old%w != 0 {
			return nil, ErrShape
		}
		sh[last] = sh[last] * old / w
		st[last] = w
	}
	n, e := shapeSizeWidth(max(w, 1), sh)
	if e != nil {
		return nil, e
	}
	off := a.offset * a.strideUnit()
	if n == 0 {
		off = 0
	}
	return &Array{data: &storage{dtype: d.Kind, flex: d, buffer: rawStorageBytes(a.data), order: d.byteOrder(), byteIndexed: true}, shape: sh, strides: st, offset: off, size: n, readOnly: a.readOnly, base: a.viewBase()}, nil
}
func flexCompare(a, b *Array, ordered bool, fn func(element, element, DType) bool) (*Array, error) {
	d := a.DType()
	if d != b.DType() || !d.IsFlexible() {
		return nil, ErrDType
	}
	if d == VoidDType && (ordered || a.ItemSize() != b.ItemSize()) {
		return nil, ErrDType
	}
	return binaryMapCore(a, b, 0, Bool, func(x, y element) (element, error) { return boolElement(fn(x, y, d)), nil })
}
func flexAdd(a, b *Array, o op) (*Array, error) {
	d := a.DType()
	if d != b.DType() || !d.IsCharacter() || o != addOp {
		return nil, ErrDType
	}
	s := a.data.flex.Length
	if s > maxInt-b.data.flex.Length {
		return nil, ErrShape
	}
	f := FlexibleDType{Kind: d, Length: s + b.data.flex.Length, Order: binary.NativeEndian}
	shape, e := BroadcastShapes(a.shape, b.shape)
	if e != nil {
		return nil, e
	}
	a, e = a.BroadcastTo(shape...)
	if e != nil {
		return nil, e
	}
	b, e = b.BroadcastTo(shape...)
	if e != nil {
		return nil, e
	}
	out, e := ZerosFlexible(f, shape...)
	if e != nil {
		return nil, e
	}
	for i := 0; i < out.size; i++ {
		x, y := a.elementAt(a.offset+flatOffset(i, shape, a.strides)), b.elementAt(b.offset+flatOffset(i, shape, b.strides))
		s := flexKey(x) + flexKey(y)
		var v any = TextScalar(s)
		if d == BytesDType {
			v = BytesScalar(s)
		}
		z, e := encodeFlexible(v, f)
		if e != nil {
			return nil, e
		}
		out.setElement(i, element{extra: z})
	}
	return out, nil
}
