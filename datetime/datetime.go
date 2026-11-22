// Package datetime exposes the shared NumPy-style calendar scalar kernels and
// the legacy tick-array adapter.
package datetime

import "github.com/SamuelSupe/numpygo"

const NaT = np.NaT

type Unit = np.DateUnit
type Value = np.DateTime
type Duration = np.TimeDelta

func ParseUnit(text string) (Unit, error)                    { return np.ParseDateUnit(text) }
func DateTime64(ticks int64, unit string) (Value, error)     { return np.NewDateTime(ticks, unit) }
func TimeDelta64(ticks int64, unit string) (Duration, error) { return np.NewTimeDelta(ticks, unit) }
func Parse(text, unit string) (Value, error)                 { return np.ParseDateTime(text, unit) }

type Array struct {
	Ticks *np.Array
	Unit  Unit
}

// NDArray returns a datetime64 view sharing the original int64 ticks.
func (a *Array) NDArray() (*np.Array, error) {
	return np.FromDateTicks(a.Ticks, np.DateTime64, a.Unit.String())
}
func FromNDArray(a *np.Array) (*Array, error) {
	if a.DType() != np.DateTime64 {
		return nil, np.ErrDType
	}
	u, e := a.DateUnit()
	if e != nil {
		return nil, e
	}
	t, e := a.DateTicks()
	if e != nil {
		return nil, e
	}
	return &Array{t, u}, nil
}

func FromTicks(ticks *np.Array, unit string) (*Array, error) {
	if ticks.DType() != np.Int64 {
		return nil, np.ErrDType
	}
	u, e := ParseUnit(unit)
	if e != nil {
		return nil, e
	}
	return &Array{ticks, u}, nil
}
func FromStrings(values []string, unit string, shape ...int) (*Array, error) {
	u, e := ParseUnit(unit)
	if e != nil || u.Name == "generic" {
		return nil, np.ErrValue
	}
	ticks := make([]int64, len(values))
	for i, s := range values {
		v, e := Parse(s, unit)
		if e != nil {
			return nil, e
		}
		ticks[i] = v.Ticks
	}
	a, e := np.FromSlice(ticks, shape...)
	if e != nil {
		return nil, e
	}
	return &Array{a, u}, nil
}
func (a *Array) Cast(unit string) (*Array, error) {
	u, e := ParseUnit(unit)
	if e != nil {
		return nil, e
	}
	if a.Unit == u {
		return a, nil
	}
	ticks, e := np.Values[int64](a.Ticks)
	if e != nil {
		return nil, e
	}
	for i, v := range ticks {
		r, e := (Value{Ticks: v, Unit: a.Unit}).Cast(unit)
		if e != nil {
			return nil, e
		}
		ticks[i] = r.Ticks
	}
	out, e := np.FromSlice(ticks, a.Ticks.Shape()...)
	if e != nil {
		return nil, e
	}
	return &Array{out, u}, nil
}
func IsNaT(a *Array) (*np.Array, error) {
	if a == nil || a.Ticks.DType() != np.Int64 {
		return nil, np.ErrDType
	}
	return np.Equal(a.Ticks, np.ScalarOf(NaT))
}
