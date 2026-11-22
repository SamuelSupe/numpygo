package scalar

import (
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/extended"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"math"
	"strconv"
	"testing"
)

func TestNumericScalarClassesNumPy(t *testing.T) {
	m := refcheck.Require(t)
	vm := refcheck.RequireValue[Value](t)
	check := func(n string, a *np.Array) { t.Helper(); refcheck.Check(t, "numeric_class_scalar_"+n, a) }
	text := func(n, s string) { t.Helper(); refcheck.Text(t, "numeric_class_scalar_"+n, s) }
	for i, v := range []Value{Of(int16(-7)), Of(uint64(math.MaxUint64)), Of(float32(3.25)), Of(complex64(1 + 2i)), Of(true)} {
		p := strconv.Itoa(i) + "_"
		check(p+"value", m(v.Array()))
		check(p+"meta", m(np.FromSlice([]int64{int64(v.NDim()), int64(v.Size()), int64(v.ItemSize()), int64(v.NBytes())}, 4)))
		for n, s := range map[string]Value{"real": vm(v.Real()), "imag": vm(v.Imag()), "conj": vm(v.Conjugate()), "copy": vm(v.Copy()), "T": vm(v.Transpose()), "transpose": vm(v.Transpose()), "squeeze": vm(v.Squeeze()), "byteswap": vm(v.ByteSwap())} {
			check(p+n, m(s.Array()))
		}
		check(p+"item", m(np.Full(v.DType(), refcheck.RequireValue[any](t)(v.Item()))))
		text(p+"device", v.Device())
		if len(v.Shape()) != 0 || len(v.Strides()) != 0 {
			t.Fatal("scalar shape/strides")
		}
		check(p+"reshape", m(v.Reshape(1, 1)))
		check(p+"ravel", m(v.Ravel()))
		check(p+"flatten", m(v.Flatten()))
		refcheck.Bytes(t, "numeric_class_scalar_"+p+"bytes", refcheck.RequireValue[[]byte](t)(v.ToBytes()))
		if !v.DType().IsComplex() {
			check(p+"cast", m(vm(v.Astype(np.Float64)).Array()))
		}
	}
	check("view", m(vm(Of(int16(-7)).View(np.Uint16)).Array()))
	check("getfield", m(vm(Of(float32(3.25)).GetField(np.Int32, 0)).Array()))
	orig := Of(int16(-7))
	detached := m(orig.Array())
	if e := detached.SetValue(int16(88)); e != nil {
		t.Fatal(e)
	}
	check("detached", m(orig.Array()))
	for i, d := range []np.DType{np.Int8, np.Int16, np.Int32, np.Int64, np.Uint8, np.Uint16, np.Uint32, np.Uint64, np.Int64, np.Uint64} {
		for j := 0; j < 3; j++ {
			p := "integer_" + strconv.Itoa(i) + "_" + strconv.Itoa(j)
			v := vm(New(refcheck.Input(t, "numeric_class_scalar_"+p+"_numerator")))
			if v.DType() != d {
				t.Fatal("dtype")
			}
			check(p+"_numerator", m(vm(v.Numerator()).Array()))
			bits := refcheck.RequireValue[int](t)(v.BitCount())
			den := refcheck.RequireValue[int](t)(v.Denominator())
			is := refcheck.RequireValue[bool](t)(v.IsInteger())
			flag := int64(0)
			if is {
				flag = 1
			}
			check(p+"_properties", m(np.FromSlice([]int64{int64(den), int64(bits), flag}, 3)))
		}
	}
	makeValue := func(d np.DType, s string) Value {
		if d == np.Float128 {
			return Of(refcheck.RequireValue[np.ExtendedFloat](t)(extended.Parse(s)))
		}
		var x float64
		switch s {
		case "nan":
			x = math.NaN()
		case "inf":
			x = math.Inf(1)
		case "-inf":
			x = math.Inf(-1)
		default:
			x = refcheck.RequireValue[float64](t)(strconv.ParseFloat(s, 64))
		}
		return vm(Of(x).Astype(d))
	}
	for i, d := range []np.DType{np.Float16, np.Float32, np.Float64, np.Float128} {
		for j, s := range []string{"0", "-0", "0.1", "-3.25", "4", "1.0000000000000000001"} {
			v := makeValue(d, s)
			a, b, e := v.AsIntegerRatio()
			if e != nil {
				t.Fatal(e)
			}
			text("ratio_"+strconv.Itoa(i)+"_"+strconv.Itoa(j), a.String()+"/"+b.String())
			check("is_integer_"+strconv.Itoa(i)+"_"+strconv.Itoa(j), np.ScalarOf(refcheck.RequireValue[bool](t)(v.IsInteger())))
		}
		for _, s := range []string{"nan", "inf", "-inf"} {
			v := makeValue(d, s)
			_, _, e := v.AsIntegerRatio()
			target, kind := np.ErrOverflow, "OverflowError"
			if s == "nan" {
				target, kind = np.ErrValue, "ValueError"
			}
			refcheck.Error(t, "numeric_class_scalar_ratio_"+strconv.Itoa(i)+"_"+s, kind, e, target)
			check("is_integer_"+strconv.Itoa(i)+"_"+s, np.ScalarOf(refcheck.RequireValue[bool](t)(v.IsInteger())))
		}
	}
	for i, x := range []float64{0, math.Copysign(0, -1), .1, -3.25, math.SmallestNonzeroFloat64, math.Float64frombits(1 << 52), math.MaxFloat64, math.Inf(1), math.Inf(-1), math.NaN()} {
		s := refcheck.RequireValue[string](t)(Of(x).Hex())
		text("hex_"+strconv.Itoa(i), s)
		refcheck.CheckExact(t, "numeric_class_scalar_fromhex_"+strconv.Itoa(i), m(vm(FromHex(s)).Array()))
	}
	for i, s := range []string{"  -0x1.8p+2  ", "a.8", "0x1", "-0x0p0", "0X1.FP3"} {
		refcheck.CheckExact(t, "numeric_class_scalar_hex_parse_"+strconv.Itoa(i), m(vm(FromHex(s)).Array()))
	}
	_, e := FromHex("xzz")
	refcheck.Error(t, "numeric_class_scalar_hex_bad", "ValueError", e, np.ErrValue)
	_, e = FromHex("0x1p2000")
	refcheck.Error(t, "numeric_class_scalar_hex_over", "OverflowError", e, np.ErrOverflow)
}
