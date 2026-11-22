package np_test

import (
	"encoding/binary"
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"testing"
)

func TestDTypeCastBoundaries(t *testing.T) {
	dtypes := []np.DType{np.Bool, np.Int8, np.Int16, np.Int32, np.Int64, np.Uint8, np.Uint16, np.Uint32, np.Uint64, np.Float16, np.Float32, np.Float64, np.Complex64, np.Complex128}
	for _, src := range dtypes {
		for _, dst := range dtypes {
			t.Run(src.String()+"_"+dst.String(), func(t *testing.T) {
				must := refcheck.Require(t)
				a := must(refcheck.Input(t, "param_cast_input_"+src.String()).Slice(np.Reverse()))
				refcheck.CheckExact(t, "param_cast_"+src.String()+"_"+dst.String(), must(a.Astype(dst)))
			})
		}
		if !src.IsFloat() && !src.IsComplex() {
			continue
		}
		for _, dst := range []np.DType{np.Bool, np.Int8, np.Int16, np.Int32, np.Int64, np.Uint8, np.Uint16, np.Uint64, np.Float16, np.Float32, np.Float64, np.Complex64, np.Complex128} {
			t.Run("extreme_"+src.String()+"_"+dst.String(), func(t *testing.T) {
				a := refcheck.Input(t, "param_cast_extreme_"+src.String())
				refcheck.CheckExact(t, "param_cast_extreme_"+src.String()+"_"+dst.String(), refcheck.Require(t)(a.Astype(dst)))
			})
		}
		t.Run("extreme_"+src.String()+"_uint32_scalar", func(t *testing.T) {
			a := refcheck.Input(t, "param_cast_extreme_"+src.String())
			refcheck.CheckExact(t, "param_cast_extreme_"+src.String()+"_uint32_scalar", refcheck.Require(t)(a.Astype(np.Uint32)))
		})
	}
}

func TestAstypeOptions(t *testing.T) {
	must := refcheck.Require(t)
	base := must(np.Arange(0, 24, 1, np.Int64))
	base = must(base.Reshape(2, 3, 4))
	for _, kind := range []string{"c", "f", "perm", "reverse"} {
		a := base
		switch kind {
		case "f":
			a = must(a.Transpose(2, 1, 0))
		case "perm":
			a = must(a.Transpose(1, 2, 0))
		case "reverse":
			a = must(a.Slice(np.Reverse(), np.Slice{}, np.Reverse()))
		}
		for _, order := range []string{"C", "F", "A", "K"} {
			v := must(a.AstypeWithOptions(np.Float32, np.CastOptions{Order: order}))
			if v.Base() != nil {
				t.Fatal("astype copy must own its allocation")
			}
			refcheck.CheckLayout(t, "param_cast_layout_"+kind+"_"+order, v)
		}
		if must(a.AstypeWithOptions(np.Int64, np.CastOptions{NoCopy: true})) != a {
			t.Fatal("K order no-copy should reuse original array")
		}
		b := must(a.AstypeWithOptions(np.Int64, np.CastOptions{}))
		if b == a {
			t.Fatal("default copy reused source")
		}
		old, _ := a.AtValue(0, 0, 0)
		if e := b.SetValue(int64(999), 0, 0, 0); e != nil {
			t.Fatal(e)
		}
		got, _ := a.AtValue(0, 0, 0)
		if got != old {
			t.Fatal("copy shares source")
		}
	}
	_, e := base.AstypeWithOptions(np.Int8, np.CastOptions{Casting: "safe"})
	refcheck.Error(t, "param_cast_safe", "TypeError", e, np.ErrDType)
	_, e = base.AstypeWithOptions(np.Int64, np.CastOptions{Order: "invalid"})
	refcheck.Error(t, "param_cast_order", "ValueError", e, np.ErrValue)
	_, e = base.AstypeWithOptions(np.Int64, np.CastOptions{Casting: "invalid"})
	refcheck.Error(t, "param_cast_rule", "ValueError", e, np.ErrValue)
	foreign := binary.ByteOrder(binary.BigEndian)
	if binary.NativeEndian.Uint16([]byte{1, 0}) == 256 {
		foreign = binary.LittleEndian
	}
	buf := make([]byte, 6)
	for i, v := range []uint16{1, 65534, 300} {
		foreign.PutUint16(buf[i*2:], v)
	}
	x := must(np.FromBuffer(buf, np.Int16, 3, 0, foreign))
	out := must(x.AstypeWithOptions(np.Int16, np.CastOptions{Casting: "equiv", NoCopy: true}))
	if out == x || !out.NativeByteOrder() {
		t.Fatal("byte-order cast must use native copy")
	}
	refcheck.Check(t, "param_cast_endian_equiv", out)
	_, e = x.AstypeWithOptions(np.Int16, np.CastOptions{Casting: "no"})
	refcheck.Error(t, "param_cast_endian_no", "TypeError", e, np.ErrDType)
}
