package datetime

import (
	"math"
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/SamuelSupe/numpygo"
)

// BusDayCalendar stores a Monday-first weekmask and sorted unique holidays
// that are not already excluded by the weekmask.
type BusDayCalendar struct {
	week        [7]bool
	holidays    []int64
	daysPerWeek int64
}

func NewBusDayCalendar(weekmask string, holidays []int64) (*BusDayCalendar, error) {
	if weekmask == "" {
		weekmask = "1111100"
	}
	var week [7]bool
	if len(weekmask) == 7 && !strings.ContainsAny(weekmask, " \t") {
		for i, c := range weekmask {
			if c != '0' && c != '1' {
				return nil, np.ErrValue
			}
			week[i] = c == '1'
		}
	} else {
		for _, d := range strings.Fields(weekmask) {
			i := slices.Index([]string{"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"}, d)
			if i < 0 {
				return nil, np.ErrValue
			}
			week[i] = true
		}
	}
	n := int64(0)
	for _, v := range week {
		if v {
			n++
		}
	}
	if n == 0 {
		return nil, np.ErrValue
	}
	c := &BusDayCalendar{week: week, daysPerWeek: n}
	for _, d := range holidays {
		if d != NaT && week[weekday(d)] {
			c.holidays = append(c.holidays, d)
		}
	}
	slices.Sort(c.holidays)
	c.holidays = slices.Compact(c.holidays)
	return c, nil
}
func (c *BusDayCalendar) Weekmask() [7]bool { return c.week }
func (c *BusDayCalendar) Holidays() []int64 { return slices.Clone(c.holidays) }
func defaultCalendar(c *BusDayCalendar) *BusDayCalendar {
	if c != nil {
		return c
	}
	d, _ := NewBusDayCalendar("", nil)
	return d
}
func weekday(day int64) int { return int((day%7 + 10) % 7) }
func (c *BusDayCalendar) valid(day int64) bool {
	if day == NaT || !c.week[weekday(day)] {
		return false
	}
	_, found := slices.BinarySearch(c.holidays, day)
	return !found
}
func (c *BusDayCalendar) count(start, end int64) (int64, error) {
	if start == NaT || end == NaT {
		return 0, np.ErrValue
	}
	sign := int64(1)
	// For reversed intervals NumPy counts starting from the later endpoint
	// down to but not including the earlier endpoint.
	if start > end {
		if start == math.MaxInt64 || end == math.MaxInt64 {
			return 0, np.ErrValue
		}
		start, end = end+1, start+1
		sign = -1
	}
	if start < 0 && end > math.MaxInt64+start {
		return 0, np.ErrValue
	}
	diff := end - start
	total := diff / 7 * c.daysPerWeek
	for i := int64(0); i < diff%7; i++ {
		if c.week[weekday(start+i)] {
			total++
		}
	}
	l := sort.Search(len(c.holidays), func(i int) bool { return c.holidays[i] >= start })
	r := sort.Search(len(c.holidays), func(i int) bool { return c.holidays[i] >= end })
	return sign * (total - int64(r-l)), nil
}

func IsBusDay(a *Array, calendar *BusDayCalendar) (*np.Array, error) {
	c := defaultCalendar(calendar)
	x, e := a.Cast("D")
	if e != nil {
		return nil, e
	}
	ticks, e := np.Values[int64](x.Ticks)
	if e != nil {
		return nil, e
	}
	out := make([]bool, len(ticks))
	for i, d := range ticks {
		out[i] = c.valid(d)
	}
	return np.FromSlice(out, x.Ticks.Shape()...)
}
func BusdayCount(begin, end *Array, calendar *BusDayCalendar) (*np.Array, error) {
	c := defaultCalendar(calendar)
	a, e := begin.Cast("D")
	if e != nil {
		return nil, e
	}
	b, e := end.Cast("D")
	if e != nil {
		return nil, e
	}
	return np.Vectorize(np.Int64, func(v ...any) (any, error) { return c.count(v[0].(int64), v[1].(int64)) }, a.Ticks, b.Ticks)
}

func stepDay(day, amount int64) (int64, error) {
	if amount > 0 && day > math.MaxInt64-amount || amount < 0 && day < math.MinInt64+1-amount {
		return 0, np.ErrValue
	}
	return day + amount, nil
}
func (c *BusDayCalendar) roll(day int64, mode string) (int64, error) {
	if mode == "" {
		mode = "raise"
	}
	switch mode {
	case "raise", "nat", "forward", "following", "backward", "preceding", "modifiedfollowing", "modifiedpreceding":
	default:
		return 0, np.ErrValue
	}
	if day == NaT {
		if mode == "nat" {
			return NaT, nil
		}
		return 0, np.ErrValue
	}
	if c.valid(day) {
		return day, nil
	}
	if mode == "raise" {
		return 0, np.ErrValue
	}
	if mode == "nat" {
		return NaT, nil
	}
	dir := int64(1)
	if mode == "backward" || mode == "preceding" || mode == "modifiedpreceding" {
		dir = -1
	}
	search := func(step int64) (int64, error) {
		d := day
		for {
			var e error
			d, e = stepDay(d, step)
			if e != nil {
				return 0, e
			}
			if c.valid(d) {
				return d, nil
			}
		}
	}
	r, e := search(dir)
	if e != nil {
		return 0, e
	}
	if mode == "modifiedfollowing" || mode == "modifiedpreceding" {
		if day > math.MaxInt64/86400 || day < math.MinInt64/86400 || r > math.MaxInt64/86400 || r < math.MinInt64/86400 {
			return 0, np.ErrValue
		}
		a, b := time.Unix(day*86400, 0).UTC(), time.Unix(r*86400, 0).UTC()
		if a.Month() != b.Month() || a.Year() != b.Year() {
			return search(-dir)
		}
	}
	return r, nil
}
func (c *BusDayCalendar) offset(day, offset int64, mode string) (int64, error) {
	day, e := c.roll(day, mode)
	if e != nil || day == NaT {
		return day, e
	}
	if offset == math.MinInt64 {
		return 0, np.ErrValue
	}
	dir := int64(1)
	if offset < 0 {
		offset = -offset
		dir = -1
	}
	for offset > 0 {
		weeks := offset / c.daysPerWeek
		if weeks > 0 {
			if weeks > math.MaxInt64/7 {
				return 0, np.ErrValue
			}
			d, e := stepDay(day, dir*weeks*7)
			if e != nil {
				return 0, e
			}
			var n int64
			if dir > 0 {
				if d == math.MaxInt64 || day == math.MaxInt64 {
					return 0, np.ErrValue
				}
				n, e = c.count(day+1, d+1)
			} else {
				n, e = c.count(d, day)
			}
			if e != nil {
				return 0, e
			}
			offset -= n
			day = d
		} else {
			day, e = stepDay(day, dir)
			if e != nil {
				return 0, e
			}
			if c.valid(day) {
				offset--
			}
		}
	}
	return day, nil
}
func BusdayOffset(dates *Array, offsets *np.Array, roll string, calendar *BusDayCalendar) (*Array, error) {
	c := defaultCalendar(calendar)
	if offsets.DType() != np.Int64 {
		return nil, np.ErrDType
	}
	d, e := dates.Cast("D")
	if e != nil {
		return nil, e
	}
	ticks, e := np.Vectorize(np.Int64, func(v ...any) (any, error) { return c.offset(v[0].(int64), v[1].(int64), roll) }, d.Ticks, offsets)
	if e != nil {
		return nil, e
	}
	return FromTicks(ticks, "D")
}
