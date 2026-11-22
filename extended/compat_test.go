package extended

import (
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"testing"
)

func compare(t *testing.T, name string, a *Array) {
	t.Helper()
	m := refcheck.Require(t)
	raw := a.Bytes()
	rows := len(raw) / 16
	b := make([]uint8, 0, rows*10)
	for i := 0; i < rows; i++ {
		b = append(b, raw[i*16:i*16+10]...)
	}
	refcheck.Check(t, "final_extended_"+name, m(np.WrapSlice(b, rows, 10)))
	shape := make([]int64, a.NDim())
	for i, d := range a.Shape() {
		shape[i] = int64(d)
	}
	refcheck.Check(t, "final_extended_"+name+"_shape", m(np.WrapSlice(shape, len(shape))))
}
func TestExtendedNumPy(t *testing.T) {
	m := refcheck.RequireValue[*Array](t)
	p := refcheck.RequireValue[Float](t)
	v := []string{"1.0000000000000000001", "9223372036854775809", "1e4000", "1e-4000", "-0", "1e-4951", "-1.25", "3.5"}
	a := m(FromStrings(v, len(v)))
	compare(t, "parse", a)
	compare(t, "alias", a)
	info := Info()
	compare(t, "info", m(FromFloats([]Float{info.Eps, info.Tiny, info.Max, info.SmallestSubnormal}, 4)))
	x := m(FromStrings([]string{"1.0000000000000000001", "3.125", "-7", "1e3000"}, 2, 2))
	y := m(FromStrings([]string{"0.0000000000000000001", "2.25"}, 2))
	for _, tc := range []struct {
		name string
		fn   func(*Array, *Array) (*Array, error)
	}{{"add", Add}, {"sub", Subtract}, {"mul", Multiply}, {"div", Divide}} {
		t.Run(tc.name, func(t *testing.T) { compare(t, tc.name, m(tc.fn(x, y))) })
	}
	next := []Float{FromInt64(0).Nextafter(FromInt64(2)), FromInt64(1).Nextafter(FromInt64(2)), FromInt64(-1).Nextafter(FromInt64(2))}
	compare(t, "next", m(FromFloats(next, 3)))
	compare(t, "sqrt", m(Sqrt(m(FromStrings([]string{"2", "9", "1e3000"}, 3)))))
	compare(t, "sum", m(Sum(x, []int{0}, false)))
	compare(t, "matmul", m(MatMul(x, m(FromStrings([]string{"1", "2", "3", "4"}, 2, 2)))))
	zview := m(x.Transpose())
	if e := zview.Set(Complex{Real: p(Parse("11.000000000000000001"))}, 0, 1); e != nil {
		t.Fatal(e)
	}
	compare(t, "view", x)
	z := m(FromComplexes([]Complex{{Real: p(Parse("1.0000000000000000001")), Imag: p(Parse("2.25"))}, {Real: p(Parse("-3.125")), Imag: p(Parse("4.5"))}}, 2))
	w := m(FromComplexes([]Complex{{Real: FromInt64(2), Imag: FromInt64(-1)}, {Real: FromInt64(3), Imag: FromInt64(4)}}, 2))
	compare(t, "complex", z)
	compare(t, "complex_alias", w)
	compare(t, "cadd", m(Add(z, w)))
	compare(t, "cmul", m(Multiply(z, w)))
	compare(t, "cdiv", m(Divide(z, w)))
	compare(t, "conj", m(Conjugate(z)))
	compare(t, "abs", m(Absolute(z)))
	special := []Float{NaN(), Inf(false), Inf(true), Zero(true)}
	b := make([]bool, 0, 12)
	for _, f := range []func(Float) bool{Float.IsNaN, Float.IsInf, Float.Signbit} {
		for _, x := range special {
			b = append(b, f(x))
		}
	}
	refcheck.Check(t, "final_extended_special", refcheck.Require(t)(np.WrapSlice(b, 3, 4)))
}
