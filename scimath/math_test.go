package scimath

import (
	"fmt"
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"strings"
	"testing"
)

func fixtureText(t *testing.T, n string) string {
	t.Helper()
	return string(refcheck.RequireValue[[]uint8](t)(np.Values[uint8](refcheck.Input(t, n))))
}
func fixtureType(t *testing.T, n string) np.DType {
	t.Helper()
	return refcheck.RequireValue[np.DType](t)(np.ParseDType(fixtureText(t, n+"_dtype")))
}
func sciInput(t *testing.T, n string) *np.Array {
	t.Helper()
	d := fixtureType(t, n)
	if d.IsExtended() {
		return refcheck.ExtendedInput(t, n, d)
	}
	return refcheck.Input(t, n)
}
func sciCheck(t *testing.T, n string, a *np.Array) {
	t.Helper()
	d := fixtureType(t, n)
	if d.IsExtended() {
		ulps := int64(8)
		if d.IsComplex() {
			ulps = 32
		}
		refcheck.CheckExtended(t, n, a, d, ulps)
	} else {
		refcheck.Check(t, n, a)
	}
}

var scientificUnary = []struct {
	name string
	f    func(*np.Array) (*np.Array, error)
}{{"sqrt", Sqrt}, {"log", Log}, {"log2", Log2}, {"log10", Log10}, {"arcsin", Arcsin}, {"arccos", Arccos}, {"arctanh", Arctanh}}

func TestScimathDTypeDomainsNumPy(t *testing.T) {
	types := strings.Split(fixtureText(t, "scimath_types"), "|")
	for i, d := range types {
		modes := []string{"inside", "outside"}
		if strings.HasPrefix(d, "complex") {
			modes = append(modes, "nonreal")
		}
		for _, mode := range modes {
			t.Run(d+"_"+mode, func(t *testing.T) {
				m := refcheck.Require(t)
				name := fmt.Sprintf("scimath_%d_%s", i, mode)
				a := sciInput(t, name+"_input")
				if mode != "nonreal" {
					a = m(a.Slice(np.Reverse()))
				}
				for _, op := range scientificUnary {
					t.Run(op.name, func(t *testing.T) {
						sciCheck(t, name+"_"+op.name, refcheck.Require(t)(op.f(a)))
					})
				}
			})
		}
	}
}

func TestScimathBinaryDTypeMatrixNumPy(t *testing.T) {
	types := strings.Split(fixtureText(t, "scimath_types"), "|")
	a, b := make([]*np.Array, len(types)), make([]*np.Array, len(types))
	for i := range types {
		a[i] = sciInput(t, fmt.Sprintf("scimath_binary_a_%d", i))
		b[i] = sciInput(t, fmt.Sprintf("scimath_binary_b_%d", i))
	}
	for i, x := range types {
		for j, y := range types {
			t.Run(x+"_"+y, func(t *testing.T) {
				m := refcheck.Require(t)
				sciCheck(t, fmt.Sprintf("scimath_power_%d_%d", i, j), m(Power(a[i], b[j])))
				sciCheck(t, fmt.Sprintf("scimath_logn_%d_%d", i, j), m(Logn(b[j], a[i])))
			})
		}
	}
}

func TestScimathBranchesNumPy(t *testing.T) {
	m := refcheck.Require(t)
	for _, n := range []string{"half", "extended", "extended_positive", "signed_min", "branch", "empty", "scalar"} {
		t.Run(n, func(t *testing.T) {
			key := "scimath_edge_" + n
			a := sciInput(t, key+"_input")
			for _, op := range scientificUnary {
				t.Run(op.name, func(t *testing.T) {
					m := refcheck.Require(t)
					b := m(op.f(a))
					sciCheck(t, key+"_"+op.name, b)
					if n == "branch" {
						re := m(np.Signbit(m(np.Real(b))))
						im := m(np.Signbit(m(np.Imag(b))))
						v := m(np.Stack([]*np.Array{re, im}, 1))
						refcheck.Check(t, "scimath_branch_sign_"+op.name, v)
					}
				})
			}
		})
	}
	a := m(np.FromStrings([]string{"-1", "2"}, 2))
	_, e := Sqrt(a)
	refcheck.Error(t, "scimath_bad_string", "UFuncTypeError", e, np.ErrDType)
	_, e = Power(m(np.Ones(np.Float64, 2)), m(np.Ones(np.Float64, 3)))
	refcheck.Error(t, "scimath_bad_broadcast", "ValueError", e, np.ErrShape)
}
