package np

import (
	"encoding/binary"
	"math"
	"math/big"
	"slices"
	"strconv"
	"strings"
)

func temporalValue(d DType, ticks int64, u DateUnit) any {
	if d == DateTime64 {
		return DateTime{ticks, u}
	}
	return TimeDelta{ticks, u}
}
func temporalTicks(e element) int64 {
	switch v := e.extra.(type) {
	case DateTime:
		return v.Ticks
	case TimeDelta:
		return v.Ticks
	}
	return int64(e.i)
}
func temporalElement(d DType, ticks int64, u DateUnit) element {
	return element{extra: temporalValue(d, ticks, u)}
}
func temporalStringWidth(d DType, u DateUnit) (int, error) {
	if d == TimeDelta64 {
		return 21, nil
	}
	if d != DateTime64 {
		return 0, ErrDType
	}
	width := map[string]int{"Y": 22, "M": 25, "W": 28, "D": 28, "h": 32, "m": 35, "s": 38, "ms": 42, "us": 45, "ns": 48, "ps": 51, "fs": 54, "as": 57}[u.Name]
	if width == 0 {
		return 0, ErrValue
	}
	return width, nil
}
func (a *Array) DateUnit() (DateUnit, error) {
	if !a.DType().IsTemporal() {
		return DateUnit{}, ErrDType
	}
	return a.data.unit, nil
}

// FromTemporalBuffer shares raw int64 ticks with a temporal descriptor.
// Unlike a scalar constructor, raw storage may contain generic-unit ticks.
func FromTemporalBuffer(buffer []byte, d DType, unit string, shape []int, readonly bool, order binary.ByteOrder) (*Array, error) {
	if !d.IsTemporal() {
		return nil, ErrDType
	}
	u, e := ParseDateUnit(unit)
	if e != nil {
		return nil, e
	}
	n, e := shapeSize(d, shape)
	if e != nil {
		return nil, e
	}
	if len(buffer) != n*8 {
		return nil, ErrShape
	}
	if buffer == nil {
		buffer = make([]byte, 0)
	}
	if order == nil {
		order = binary.NativeEndian
	}
	a := owned(&storage{dtype: d, buffer: buffer, order: order, unit: u}, shape, n)
	a.readOnly = readonly
	return a, nil
}

// ParseTemporalDescriptor parses the date/time leaf notation in an NPY
// header (for example >M8[3h]), including the generic unit.
func ParseTemporalDescriptor(desc string) (DType, DateUnit, binary.ByteOrder, error) {
	var u DateUnit
	if len(desc) < 3 || (desc[1:3] != "M8" && desc[1:3] != "m8") {
		return 0, u, nil, ErrDType
	}
	d := DateTime64
	if desc[1] == 'm' {
		d = TimeDelta64
	}
	unit := "generic"
	if len(desc) > 3 {
		if len(desc) < 6 || desc[3] != '[' || desc[len(desc)-1] != ']' {
			return 0, u, nil, ErrDType
		}
		unit = desc[4 : len(desc)-1]
	}
	u, e := ParseDateUnit(unit)
	if e != nil {
		return 0, u, nil, e
	}
	_, order, e := dtypeDescriptor(desc[:1] + "i8")
	return d, u, order, e
}

// FromDateTicks reinterprets an int64 ndarray, retaining its storage, shape,
// negative strides and writability. Units are explicit metadata.
func FromDateTicks(ticks *Array, d DType, unit string) (*Array, error) {
	if ticks.DType() != Int64 || !d.IsTemporal() {
		return nil, ErrDType
	}
	u, e := ParseDateUnit(unit)
	if e != nil {
		return nil, e
	}
	if d == DateTime64 && u.Name == "generic" {
		valid := true
		ticks.walk(func(_, off int) {
			if ticks.elementAt(off).i != uint64(1)<<63 {
				valid = false
			}
		})
		if !valid {
			return nil, ErrValue
		}
	}
	return ticks.retag(d, u), nil
}
func (a *Array) retag(d DType, u DateUnit) *Array {
	s := a.data
	data := &storage{dtype: d, raw: s.raw, buffer: s.buffer, order: s.order, mapping: s.mapping, unit: u, byteIndexed: s.byteIndexed}
	return &Array{data: data, shape: a.Shape(), strides: slices.Clone(a.strides), offset: a.offset, size: a.size, readOnly: a.readOnly, base: a.viewBase()}
}
func (a *Array) DateTicks() (*Array, error) {
	if !a.DType().IsTemporal() {
		return nil, ErrDType
	}
	return a.retag(Int64, DateUnit{}), nil
}
func FromDateStrings(values []string, unit string, shape ...int) (*Array, error) {
	n, e := shapeSize(DateTime64, shape)
	if e != nil {
		return nil, e
	}
	if len(values) != n {
		return nil, ErrShape
	}
	u, e := ParseDateUnit(unit)
	if e != nil || u.Name == "generic" {
		return nil, ErrValue
	}
	ticks := make([]int64, n)
	for i, s := range values {
		v, e := ParseDateTime(s, unit)
		if e != nil {
			return nil, e
		}
		ticks[i] = v.Ticks
	}
	s := makeStorage(DateTime64, n)
	s.unit = u
	s.raw = ticks
	return owned(s, shape, n), nil
}
func (v TimeDelta) Cast(unit string) (TimeDelta, error) {
	u, e := ParseDateUnit(unit)
	if e != nil {
		return TimeDelta{}, e
	}
	if v.IsNaT() || v.Unit == u || v.Unit.Name == "generic" {
		return TimeDelta{v.Ticks, u}, nil
	}
	f, ok := durationScale(v.Unit)
	if !ok {
		return TimeDelta{}, ErrValue
	}
	t, ok := durationScale(u)
	if !ok || calendarUnit(v.Unit) != calendarUnit(u) {
		return TimeDelta{}, ErrDType
	}
	q := new(big.Int).Mul(big.NewInt(v.Ticks), f)
	q.Div(q, t)
	if !q.IsInt64() || q.Int64() == NaT {
		return TimeDelta{}, ErrValue
	}
	return TimeDelta{q.Int64(), u}, nil
}
func calendarUnit(u DateUnit) bool { return u.Name == "Y" || u.Name == "M" }
func durationScale(u DateUnit) (*big.Int, bool) {
	if calendarUnit(u) {
		s := big.NewInt(u.Step)
		if u.Name == "Y" {
			s.Mul(s, big.NewInt(12))
		}
		return s, true
	}
	s, e := unitScale(u)
	return s, e == nil
}
func convertTemporal(v any, d DType, u DateUnit) (element, error) {
	var ticks int64
	switch x := v.(type) {
	case DateTime:
		if d != DateTime64 {
			return element{}, ErrDType
		}
		y, e := x.Cast(u.String())
		if e != nil {
			return element{}, e
		}
		ticks = y.Ticks
	case TimeDelta:
		if d != TimeDelta64 {
			return element{}, ErrDType
		}
		y, e := x.Cast(u.String())
		if e != nil {
			return element{}, e
		}
		ticks = y.Ticks
	case string:
		if d == TimeDelta64 {
			if strings.EqualFold(strings.TrimSpace(x), "NaT") {
				ticks = NaT
			} else {
				v, e := strconv.ParseInt(strings.TrimSpace(x), 10, 64)
				if e != nil {
					return element{}, ErrValue
				}
				ticks = v
			}
		} else {
			y, e := ParseDateTime(x, u.String())
			if e != nil {
				return element{}, e
			}
			ticks = y.Ticks
		}
	case TextScalar:
		return convertTemporal(string(x), d, u)
	case BytesScalar:
		return convertTemporal(string(x), d, u)
	default:
		f := dtypeOf(v)
		if !f.IsInteger() && f != Bool {
			return element{}, ErrDType
		}
		ticks = int64(elementFrom(v).i)
	}
	if d == DateTime64 && u.Name == "generic" && ticks != NaT {
		return element{}, ErrValue
	}
	return temporalElement(d, ticks, u), nil
}
func (a *Array) CastTemporal(d DType, unit string) (*Array, error) {
	if a.DType() == 0 || !d.IsTemporal() {
		return nil, ErrDType
	}
	u, e := ParseDateUnit(unit)
	if e != nil {
		return nil, e
	}
	n, e := shapeSize(d, a.shape)
	if e != nil {
		return nil, e
	}
	s := makeStorage(d, n)
	s.unit = u
	out := owned(s, a.shape, n)
	a.walk(func(i, off int) {
		if e != nil {
			return
		}
		v := a.rawAt(off)
		if a.DType().IsTemporal() && a.DType() != d {
			v = temporalTicks(a.elementAt(off))
		}
		if a.DType().IsFloat() || a.DType().IsComplex() {
			v = int64(castElement(a.elementAt(off), a.DType(), Int64).i)
		}
		if td, ok := v.(TimeDelta); ok && d == TimeDelta64 && calendarUnit(td.Unit) != calendarUnit(u) && td.Unit.Name != "generic" && u.Name != "generic" && !td.IsNaT() {
			td, err := castCalendarDelta(td, u)
			if err != nil {
				e = err
				return
			}
			v = td
		}
		x, err := convertTemporal(v, d, u)
		if err != nil {
			e = err
			return
		}
		out.setElement(i, x)
	})
	if e != nil {
		return nil, e
	}
	return out, nil
}

// NumPy's explicit unsafe timedelta cast uses the 400-year Gregorian
// average for nonlinear months and years, distinct from unit promotion.
func castCalendarDelta(v TimeDelta, to DateUnit) (TimeDelta, error) {
	scale := func(u DateUnit) (*big.Int, error) {
		f, ok := durationScale(u)
		if !ok {
			return nil, ErrValue
		}
		if calendarUnit(u) {
			day, e := unitScale(DateUnit{"D", 1})
			if e != nil {
				return nil, e
			}
			return f.Mul(f, day.Mul(day, big.NewInt(146097))), nil
		}
		return f.Mul(f, big.NewInt(4800)), nil
	}
	f, e := scale(v.Unit)
	if e != nil {
		return TimeDelta{}, e
	}
	t, e := scale(to)
	if e != nil {
		return TimeDelta{}, e
	}
	n := new(big.Int).Mul(big.NewInt(v.Ticks), f)
	n.Div(n, t)
	if !n.IsInt64() || n.Int64() == NaT {
		return TimeDelta{}, ErrValue
	}
	return TimeDelta{n.Int64(), to}, nil
}
func commonDateUnit(a, b DateUnit, allowCalendarMix bool) (DateUnit, error) {
	if a.Name == "generic" {
		return b, nil
	}
	if b.Name == "generic" {
		return a, nil
	}
	if a == b {
		return a, nil
	}
	if calendarUnit(a) != calendarUnit(b) {
		if !allowCalendarMix {
			return DateUnit{}, ErrDType
		}
		if !calendarUnit(a) {
			a, b = b, a
		}
		// Calendar boundaries have integral days. Keep the GCD of the
		// requested fixed tick and one day (weeks retain week resolution).
		f, _ := durationScale(DateUnit{b.Name, 1})
		day, _ := durationScale(DateUnit{"D", 1})
		if f == nil || day == nil {
			return DateUnit{}, ErrValue
		}
		n := new(big.Int).Quo(day, f)
		if n.Sign() == 0 {
			n.SetInt64(1)
		}
		g := new(big.Int).GCD(nil, nil, n, big.NewInt(b.Step))
		return DateUnit{b.Name, g.Int64()}, nil
	}
	order := []string{"Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns", "ps", "fs", "as"}
	fine := a.Name
	if slices.Index(order, b.Name) > slices.Index(order, a.Name) {
		fine = b.Name
	}
	f, _ := durationScale(a)
	g, _ := durationScale(b)
	if f == nil || g == nil {
		return DateUnit{}, ErrValue
	}
	h := new(big.Int).GCD(nil, nil, f, g)
	base, _ := durationScale(DateUnit{fine, 1})
	h.Quo(h, base)
	if !h.IsInt64() || h.Sign() <= 0 {
		return DateUnit{}, ErrValue
	}
	return DateUnit{fine, h.Int64()}, nil
}

// PromoteDateUnits returns a common temporal resolution; calendar unit
// mixing is legal for datetime64 and illegal for nonlinear timedelta64.
func PromoteDateUnits(a, b DateUnit, allowCalendarMix bool) (DateUnit, error) {
	if _, e := ParseDateUnit(a.String()); e != nil || a.Step <= 0 {
		return DateUnit{}, ErrValue
	}
	if _, e := ParseDateUnit(b.String()); e != nil || b.Step <= 0 {
		return DateUnit{}, ErrValue
	}
	return commonDateUnit(a, b, allowCalendarMix)
}

// CanCastDateUnits implements safe and same_kind resolution checks for
// temporal descriptors of the same dtype. Calendar-to-linear conversion
// is allowed for datetimes but not nonlinear timedelta arithmetic.
func CanCastDateUnits(from, to DateUnit, d DType, rule string) (bool, error) {
	if !d.IsTemporal() {
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
	if from == to || from.Name == "generic" {
		return true, nil
	}
	if to.Name == "generic" {
		return false, nil
	}
	if calendarUnit(from) != calendarUnit(to) {
		return d == DateTime64 && (rule == "same_kind" || calendarUnit(from)), nil
	}
	if rule == "same_kind" {
		return true, nil
	}
	f, ok := durationScale(from)
	if !ok {
		return false, ErrValue
	}
	t, ok := durationScale(to)
	if !ok || t.Sign() == 0 {
		return false, ErrValue
	}
	return new(big.Int).Mod(f, t).Sign() == 0, nil
}
func temporalMap(a, b *Array, unit DateUnit, d DType, fn func(element, element) (element, error)) (*Array, error) {
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
	var out *Array
	if d.IsTemporal() {
		n, err := shapeSize(d, shape)
		if err != nil {
			return nil, err
		}
		s := makeStorage(d, n)
		s.unit = unit
		out = owned(s, shape, n)
	} else {
		out, e = Zeros(d, shape...)
		if e != nil {
			return nil, e
		}
	}
	for i := 0; i < out.size; i++ {
		x := a.elementAt(a.offset + flatOffset(i, shape, a.strides))
		y := b.elementAt(b.offset + flatOffset(i, shape, b.strides))
		z, err := fn(x, y)
		if err != nil {
			return nil, err
		}
		out.setElement(i, z)
	}
	return out, nil
}
func temporalArithmetic(a, b *Array, o op) (*Array, error) {
	ad, bd := a.DType(), b.DType()
	if ad.IsTemporal() && bd.IsTemporal() {
		u, e := commonDateUnit(a.data.unit, b.data.unit, ad == DateTime64 || bd == DateTime64)
		if e != nil {
			return nil, e
		}
		// Calendar durations cannot be added to a fixed-day datetime.
		if ad != bd && calendarUnit(a.data.unit) != calendarUnit(b.data.unit) {
			td := a
			if bd == TimeDelta64 {
				td = b
			}
			if calendarUnit(td.data.unit) {
				return nil, ErrDType
			}
		}
		d := TimeDelta64
		if o == addOp {
			if ad == DateTime64 && bd == DateTime64 {
				return nil, ErrDType
			}
			if ad == DateTime64 || bd == DateTime64 {
				d = DateTime64
			}
		} else if o == subOp {
			if ad == TimeDelta64 && bd == DateTime64 {
				return nil, ErrDType
			}
			if ad == DateTime64 && bd == TimeDelta64 {
				d = DateTime64
			}
		} else if (o == divOp || o == floorOp || o == remOp) && ad == TimeDelta64 && bd == TimeDelta64 {
			if o == divOp {
				d = Float64
			} else if o == floorOp {
				d = Int64
			}
		} else {
			return nil, ErrDType
		}
		a, e = a.CastTemporal(ad, u.String())
		if e != nil {
			return nil, e
		}
		b, e = b.CastTemporal(bd, u.String())
		if e != nil {
			return nil, e
		}
		return temporalMap(a, b, u, d, func(x, y element) (element, error) {
			p, q := temporalTicks(x), temporalTicks(y)
			if d == Float64 {
				if p == NaT || q == NaT {
					return element{z: complex(math.NaN(), 0)}, nil
				}
				return element{z: complex(float64(p)/float64(q), 0)}, nil
			}
			if d == Int64 {
				if p == NaT || q == NaT || q == 0 {
					return element{}, nil
				}
				v, e := integerOp(uint64(p), uint64(q), Int64, floorOp)
				return element{i: v}, e
			}
			if p == NaT || q == NaT || o == remOp && q == 0 {
				return temporalElement(d, NaT, u), nil
			}
			switch o {
			case addOp:
				return temporalElement(d, p+q, u), nil
			case subOp:
				return temporalElement(d, p-q, u), nil
			case remOp:
				r, e := integerOp(uint64(p), uint64(q), Int64, remOp)
				return temporalElement(d, int64(r), u), e
			}
			return element{}, ErrDType
		})
	}
	// Integer arithmetic with dates/timedeltas uses the temporal unit;
	// multiplication/division by real scalars is allowed only for durations.
	td, other := a, b
	reversed := false
	if !ad.IsTemporal() {
		td, other = b, a
		reversed = true
	}
	d, u := td.DType(), td.data.unit
	od := other.DType()
	if !od.IsNativeNumeric() || od.IsComplex() {
		return nil, ErrDType
	}
	if o == addOp || o == subOp {
		if !od.IsInteger() && od != Bool || reversed && d == DateTime64 && o == subOp {
			return nil, ErrDType
		}
		delta, e := other.CastTemporal(TimeDelta64, u.String())
		if e != nil {
			return nil, e
		}
		if reversed {
			return temporalArithmetic(delta, td, o)
		}
		return temporalArithmetic(td, delta, o)
	}
	if d != TimeDelta64 || !(o == mulOp || o == divOp || o == floorOp) || reversed && o != mulOp {
		return nil, ErrDType
	}
	return temporalMap(td, other, u, d, func(x, y element) (element, error) {
		v := temporalTicks(x)
		if v == NaT {
			return temporalElement(d, NaT, u), nil
		}
		var z int64
		if od.IsInteger() || od == Bool {
			p := int64(y.i)
			if o == mulOp {
				z = v * p
			} else if p == 0 {
				z = NaT
			} else {
				z = v / p
			}
		} else {
			f := real(y.number(od))
			if o == mulOp {
				f = float64(v) * f
			} else {
				f = float64(v) / f
			}
			if math.IsNaN(f) || math.IsInf(f, 0) || f >= 0x1p63 || f < -0x1p63 {
				z = NaT
			} else {
				z = int64(f)
			}
		}
		return temporalElement(d, z, u), nil
	})
}
func IsNaT(a *Array) (*Array, error) {
	if !a.DType().IsTemporal() {
		return nil, ErrDType
	}
	out, e := Zeros(Bool, a.shape...)
	if e != nil {
		return nil, e
	}
	a.walk(func(i, off int) { out.setElement(i, boolElement(temporalTicks(a.elementAt(off)) == NaT)) })
	return out, nil
}
