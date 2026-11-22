package np_test

import (
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"math/big"
	"testing"
)

func TestWeakScalarGrid(t *testing.T) {
	lookup := refcheck.RequireValue[*np.UFunc](t)
	scalars := []struct {
		tag string
		v   any
	}{
		{"bool", true}, {"small", 3}, {"negative", -1}, {"wide", 256},
		{"u63", new(big.Int).Lsh(big.NewInt(1), 63)}, {"big", new(big.Int).Lsh(big.NewInt(1), 64)},
		{"float", .5}, {"complex", complex(1, 2)},
	}
	for _, name := range []string{"add", "subtract", "multiply", "divide", "floor_divide", "power", "hypot", "minimum", "equal", "less", "greater_equal", "logical_and"} {
		u := lookup(np.LookupUFunc(name))
		for _, d := range behaviorDTypes {
			a := refcheck.Input(t, "behavior_weak_input_"+d.String())
			for _, v := range scalars {
				for _, side := range []string{"left", "right"} {
					t.Run(name+"/"+d.String()+"/"+v.tag+"/"+side, func(t *testing.T) {
						var x, y any = a, np.Weak(v.v)
						if side == "left" {
							x, y = y, x
						}
						got, e := u.CallValues(np.UFuncOptions{}, x, y)
						refcheck.Case(t, "behavior_weak_"+name+"_"+d.String()+"_"+v.tag+"_"+side, got, e)
					})
				}
			}
		}
	}
}
func TestWeakScalarOptions(t *testing.T) {
	m := refcheck.Require(t)
	lookup := refcheck.RequireValue[*np.UFunc](t)
	add := lookup(np.LookupUFunc("add"))
	a := m(np.WrapSlice([]int8{127, 1}, 2))
	refcheck.Check(t, "behavior_weak_dtype", m(add.CallValues(np.UFuncOptions{DType: np.Int16}, a, np.Weak(256))))
	b := m(np.WrapSlice([]float32{1, 2}, 2))
	refcheck.Check(t, "behavior_weak_float32", m(add.CallValues(np.UFuncOptions{}, b, np.Weak(.25))))
	refcheck.Check(t, "behavior_weak_strong", m(add.CallValues(np.UFuncOptions{}, b, float64(.25))))
	out := m(np.WrapSlice([]bool{true, false}, 2))
	v := m(lookup(np.LookupUFunc("less")).CallValues(np.UFuncOptions{Out: out, Where: m(np.WrapSlice([]bool{false, true}, 2))}, a, np.Weak(100)))
	if v != out {
		t.Fatal("out replaced")
	}
	refcheck.Check(t, "behavior_weak_compare_mask", out)
}
