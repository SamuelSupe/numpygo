package np

import (
	"fmt"
	"math"
	"math/big"
	"strconv"
	"strings"
	"time"
)

const NaT int64 = math.MinInt64

type DateUnit struct {
	Name string
	Step int64
}

func ParseDateUnit(text string) (DateUnit, error) {
	if text == "" {
		text = "generic"
	}
	i := 0
	for i < len(text) && text[i] >= '0' && text[i] <= '9' {
		i++
	}
	step := int64(1)
	if i > 0 {
		var e error
		step, e = strconv.ParseInt(text[:i], 10, 64)
		if e != nil || step <= 0 {
			return DateUnit{}, ErrValue
		}
	}
	name := text[i:]
	switch name {
	case "generic", "Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns", "ps", "fs", "as":
		return DateUnit{name, step}, nil
	}
	return DateUnit{}, ErrValue
}
func (u DateUnit) String() string {
	if u.Step <= 1 {
		return u.Name
	}
	return strconv.FormatInt(u.Step, 10) + u.Name
}
func (u DateUnit) DateTimeData() (string, int64) { return u.Name, u.Step }
func unitScale(u DateUnit) (*big.Int, error) {
	var seconds int64 = 1
	power := 18
	switch u.Name {
	case "W":
		seconds = 7 * 86400
	case "D":
		seconds = 86400
	case "h":
		seconds = 3600
	case "m":
		seconds = 60
	case "s":
	case "ms":
		power = 15
	case "us":
		power = 12
	case "ns":
		power = 9
	case "ps":
		power = 6
	case "fs":
		power = 3
	case "as":
		power = 0
	default:
		return nil, ErrValue
	}
	if u.Step <= 0 {
		return nil, ErrValue
	}
	r := new(big.Int).Exp(big.NewInt(10), big.NewInt(int64(power)), nil)
	r.Mul(r, big.NewInt(seconds))
	r.Mul(r, big.NewInt(u.Step))
	return r, nil
}

type DateTime struct {
	Ticks int64
	Unit  DateUnit
}
type TimeDelta struct {
	Ticks int64
	Unit  DateUnit
}

func NewTimeDelta(ticks int64, unit string) (TimeDelta, error) {
	u, e := ParseDateUnit(unit)
	return TimeDelta{ticks, u}, e
}
func NewDateTime(ticks int64, unit string) (DateTime, error) {
	u, e := ParseDateUnit(unit)
	if e == nil && u.Name == "generic" && ticks != NaT {
		e = ErrValue
	}
	return DateTime{ticks, u}, e
}
func (v DateTime) IsNaT() bool  { return v.Ticks == NaT }
func (v TimeDelta) IsNaT() bool { return v.Ticks == NaT }

var attoPerSecond = big.NewInt(1_000_000_000_000_000_000)

func dateToAttos(t time.Time) *big.Int {
	r := new(big.Int).Mul(big.NewInt(t.Unix()), attoPerSecond)
	return r.Add(r, big.NewInt(int64(t.Nanosecond())*1_000_000_000))
}
func floorDiv(x, y int64) int64 {
	q, r := x/y, x%y
	if r < 0 {
		q--
	}
	return q
}
func fromAttos(value *big.Int, u DateUnit) (DateTime, error) {
	if u.Name == "Y" || u.Name == "M" {
		seconds := new(big.Int).Div(value, attoPerSecond)
		if !seconds.IsInt64() {
			return DateTime{}, ErrValue
		}
		t := time.Unix(seconds.Int64(), 0).UTC()
		n := int64(t.Year() - 1970)
		if u.Name == "M" {
			n = n*12 + int64(t.Month()-1)
		}
		return DateTime{floorDiv(n, u.Step), u}, nil
	}
	s, e := unitScale(u)
	if e != nil {
		return DateTime{}, e
	}
	q := new(big.Int).Div(value, s)
	if !q.IsInt64() || q.Int64() == NaT {
		return DateTime{}, ErrValue
	}
	return DateTime{q.Int64(), u}, nil
}
func (v DateTime) attos() (*big.Int, error) {
	if v.IsNaT() {
		return nil, ErrValue
	}
	if v.Unit.Name == "Y" || v.Unit.Name == "M" {
		n := new(big.Int).Mul(big.NewInt(v.Ticks), big.NewInt(v.Unit.Step))
		if !n.IsInt64() {
			return nil, ErrValue
		}
		y, m := n.Int64(), int64(0)
		if v.Unit.Name == "M" {
			y = floorDiv(y, 12)
			m = n.Int64() - y*12
		}
		// Calendar conversion is deliberately limited to the ISO year domain.
		if y < -1970 || y > 8029 {
			return nil, ErrValue
		}
		return dateToAttos(time.Date(int(y)+1970, time.Month(m+1), 1, 0, 0, 0, 0, time.UTC)), nil
	}
	scale, e := unitScale(v.Unit)
	if e != nil {
		return nil, e
	}
	return new(big.Int).Mul(big.NewInt(v.Ticks), scale), nil
}
func (v DateTime) Cast(unit string) (DateTime, error) {
	u, e := ParseDateUnit(unit)
	if e != nil {
		return DateTime{}, e
	}
	if v.IsNaT() {
		return DateTime{NaT, u}, nil
	}
	if v.Unit == u {
		return v, nil
	}
	a, e := v.attos()
	if e != nil {
		return DateTime{}, e
	}
	return fromAttos(a, u)
}

// ParseDateTime accepts ISO calendar dates and time strings, including up to 18
// fractional second digits, and rounds down to the requested unit.
func ParseDateTime(text, unit string) (DateTime, error) {
	u, e := ParseDateUnit(unit)
	if e != nil {
		return DateTime{}, e
	}
	if strings.EqualFold(text, "nat") {
		return DateTime{NaT, u}, nil
	}
	text = strings.TrimSpace(strings.TrimSuffix(text, "Z"))
	text = strings.ReplaceAll(text, " ", "T")
	fraction := ""
	if i := strings.IndexByte(text, '.'); i >= 0 {
		fraction = text[i+1:]
		text = text[:i]
	}
	if len(fraction) > 18 {
		return DateTime{}, ErrValue
	}
	layout, guess := "2006", "Y"
	switch len(text) {
	case 4:
	case 7:
		layout, guess = "2006-01", "M"
	case 10:
		layout, guess = time.DateOnly, "D"
	case 13:
		layout, guess = "2006-01-02T15", "h"
	case 16:
		layout, guess = "2006-01-02T15:04", "m"
	case 19:
		layout, guess = "2006-01-02T15:04:05", "s"
	default:
		return DateTime{}, ErrValue
	}
	t, e := time.ParseInLocation(layout, text, time.UTC)
	if e != nil {
		return DateTime{}, ErrValue
	}
	a := dateToAttos(t)
	if fraction != "" {
		f, ok := new(big.Int).SetString(fraction+strings.Repeat("0", 18-len(fraction)), 10)
		if !ok || f.Sign() < 0 {
			return DateTime{}, ErrValue
		}
		a.Add(a, f)
		guess = []string{"ms", "us", "ns", "ps", "fs", "as"}[(len(fraction)-1)/3]
	}
	if u.Name == "generic" {
		u = DateUnit{guess, 1}
	}
	return fromAttos(a, u)
}

func (v DateTime) AsString(unit string) (string, error) {
	if v.IsNaT() {
		return "NaT", nil
	}
	if unit == "" {
		unit = v.Unit.Name
	}
	u, e := ParseDateUnit(unit)
	if e != nil {
		return "", e
	}
	a, e := v.attos()
	if e != nil {
		return "", e
	}
	if u.Name == "Y" || u.Name == "M" {
		c, e := fromAttos(a, u)
		if e != nil {
			return "", e
		}
		a, e = c.attos()
		if e != nil {
			return "", e
		}
	} else {
		scale, e := unitScale(u)
		if e != nil {
			return "", e
		}
		a.Mul(new(big.Int).Div(a, scale), scale)
	}
	seconds, rem := new(big.Int), new(big.Int)
	seconds.DivMod(a, attoPerSecond, rem)
	if !seconds.IsInt64() {
		return "", ErrValue
	}
	t := time.Unix(seconds.Int64(), 0).UTC()
	switch u.Name {
	case "Y":
		return t.Format("2006"), nil
	case "M":
		return t.Format("2006-01"), nil
	case "W", "D":
		return t.Format(time.DateOnly), nil
	case "h":
		return t.Format("2006-01-02T15"), nil
	case "m":
		return t.Format("2006-01-02T15:04"), nil
	case "s":
		return t.Format("2006-01-02T15:04:05"), nil
	default:
		digits := map[string]int{"ms": 3, "us": 6, "ns": 9, "ps": 12, "fs": 15, "as": 18}[u.Name]
		f := fmt.Sprintf("%018d", rem.Int64())
		return t.Format("2006-01-02T15:04:05") + "." + f[:digits], nil
	}
}
