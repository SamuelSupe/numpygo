package datetime_test

import (
	"strconv"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/datetime"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
)

func TestDatetimeTicksAndUnits(t *testing.T) {
	must := refcheck.Require(t)
	check := func(n string, a *np.Array) { refcheck.Check(t, "datetime_"+n, a) }
	txt := func(n, s string) { check(n, must(np.FromSlice([]byte(s), len(s)))) }
	for i, v := range [][2]string{
		{"2024", "Y"}, {"2024-03", "M"}, {"1969-12-31", "D"}, {"1969-12-31T23:59:59.987654321", "ns"},
		{"2024-02-29T13:14:15.123456", "us"}, {"1970-01-08", "W"},
		{"1970-01-01T00:00:00.000001234567", "ps"}, {"1969-12-31T23:59:59.999999999999999999", "as"},
		{"2024-02-29", "3D"}, {"2024-02-29T13:14:15", "5h"}, {"NaT", "D"},
	} {
		d, e := datetime.Parse(v[0], v[1])
		if e != nil {
			t.Fatal(v, e)
		}
		n := strconv.Itoa(i)
		check("parse_"+n, np.ScalarOf(d.Ticks))
		s, e := d.AsString("")
		if e != nil {
			t.Fatal(v, e)
		}
		txt("format_"+n, s)
		u, step := d.Unit.DateTimeData()
		txt("unit_"+n, u)
		check("step_"+n, np.ScalarOf(step))
	}
	for i, v := range []struct {
		ticks int64
		unit  string
	}{{17, "h"}, {-12, "3D"}, {0, "ms"}, {datetime.NaT, "ns"}} {
		d, e := datetime.TimeDelta64(v.ticks, v.unit)
		if e != nil {
			t.Fatal(e)
		}
		n := strconv.Itoa(i)
		check("timedelta_"+n, np.ScalarOf(d.Ticks))
		check("timedelta_nat_"+n, np.ScalarOf(d.IsNaT()))
	}
	for i, v := range []struct {
		ticks int64
		unit  string
	}{{0, "s"}, {-7, "D"}, {32768, "ms"}, {datetime.NaT, "us"}} {
		d, e := datetime.DateTime64(v.ticks, v.unit)
		if e != nil {
			t.Fatal(e)
		}
		n := strconv.Itoa(i)
		check("scalar_ticks_"+n, np.ScalarOf(d.Ticks))
		s, e := d.AsString("")
		if e != nil {
			t.Fatal(e)
		}
		txt("scalar_format_"+n, s)
	}
	a, e := datetime.FromStrings([]string{"2024-01-05", "2024-01-06", "NaT", "1969-12-31"}, "D", 2, 2)
	if e != nil {
		t.Fatal(e)
	}
	check("array_ticks", a.Ticks)
	check("isnat", must(datetime.IsNaT(a)))
	b, e := a.Cast("h")
	if e != nil {
		t.Fatal(e)
	}
	check("cast_hours", b.Ticks)
	b, e = a.Cast("M")
	if e != nil {
		t.Fatal(e)
	}
	check("cast_months", b.Ticks)
	_, e = datetime.Parse("2023-02-29", "D")
	refcheck.Error(t, "datetime_bad_date", "ValueError", e, np.ErrValue)
	_, e = datetime.DateTime64(1, "invalid")
	refcheck.Error(t, "datetime_bad_unit", "TypeError", e, np.ErrValue)
}

func TestBusinessDays(t *testing.T) {
	in := func(n string) *np.Array { return refcheck.Input(t, "datetime_"+n) }
	must := refcheck.Require(t)
	check := func(n string, a *np.Array) { refcheck.Check(t, "datetime_"+n, a) }
	holidays, e := datetime.FromStrings([]string{"2024-01-01", "2024-01-06", "2024-01-01", "2024-01-15", "NaT"}, "D", 5)
	if e != nil {
		t.Fatal(e)
	}
	h, e := np.Values[int64](holidays.Ticks)
	if e != nil {
		t.Fatal(e)
	}
	c, e := datetime.NewBusDayCalendar("Mon Tue Wed Thu Fri", h)
	if e != nil {
		t.Fatal(e)
	}
	week := c.Weekmask()
	check("calendar_weekmask", must(np.FromSlice(week[:], 7)))
	hd := c.Holidays()
	check("calendar_holidays", must(np.FromSlice(hd, len(hd))))
	date := func(n string) *datetime.Array {
		a, e := datetime.FromTicks(in(n), "D")
		if e != nil {
			t.Fatal(e)
		}
		return a
	}
	check("is_busday", must(datetime.IsBusDay(date("days"), c)))
	begin, end := date("begin"), date("end")
	check("busday_count", must(datetime.BusdayCount(begin, end, c)))
	starts, offsets := date("starts"), in("offsets")
	for _, roll := range []string{"forward", "backward", "following", "preceding", "modifiedfollowing", "modifiedpreceding", "nat"} {
		a, e := datetime.BusdayOffset(starts, offsets, roll, c)
		if e != nil {
			t.Fatal(roll, e)
		}
		check("busday_offset_"+roll, a.Ticks)
	}
	hDates, e := datetime.FromStrings([]string{"2024-01-06", "2024-01-10"}, "D", 2)
	if e != nil {
		t.Fatal(e)
	}
	h, e = np.Values[int64](hDates.Ticks)
	if e != nil {
		t.Fatal(e)
	}
	other, e := datetime.NewBusDayCalendar("Mon Wed Sat", h)
	if e != nil {
		t.Fatal(e)
	}
	check("custom_count", must(datetime.BusdayCount(begin, end, other)))
	a, e := datetime.BusdayOffset(starts, offsets, "forward", other)
	if e != nil {
		t.Fatal(e)
	}
	check("custom_offset", a.Ticks)
	nat, e := datetime.FromStrings([]string{"NaT"}, "D", 1)
	if e != nil {
		t.Fatal(e)
	}
	_, e = datetime.BusdayCount(nat, end, c)
	refcheck.Error(t, "datetime_busday_count_nat", "ValueError", e, np.ErrValue)
	invalid, e := datetime.FromStrings([]string{"2024-01-07"}, "D", 1)
	if e != nil {
		t.Fatal(e)
	}
	_, e = datetime.BusdayOffset(invalid, np.ScalarOf(int64(1)), "raise", nil)
	refcheck.Error(t, "datetime_busday_offset_raise", "ValueError", e, np.ErrValue)
	_, e = datetime.NewBusDayCalendar("0000000", nil)
	refcheck.Error(t, "datetime_calendar_empty_week", "ValueError", e, np.ErrValue)
}
