package np_test

import (
	"math"
	"math/cmplx"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/fft"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"github.com/SamuelSupe/numpygo/linalg"
	"github.com/SamuelSupe/numpygo/random"
)

func nativeValues[T np.Number](t *testing.T, a *np.Array) *np.Array {
	t.Helper()
	v, e := np.Values[T](a)
	if e != nil {
		t.Fatal(e)
	}
	return refcheck.Require(t)(np.WrapSlice(v, a.Shape()...))
}
func TestNativeMatrixCopy(t *testing.T) {
	m := refcheck.Require(t)
	for _, d := range []string{"bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float16", "float32", "float64", "complex64", "complex128"} {
		t.Run(d, func(t *testing.T) {
			m := refcheck.Require(t)
			a := refcheck.Input(t, "native_copy_input_"+d)
			v := m(m(a.Slice(np.Reverse(), np.Reverse())).Transpose())
			cp := m(v.Astype(v.DType()))
			refcheck.CheckExact(t, "native_copy_"+d, cp)
			if !cp.IsCContiguous() || np.SharesMemory(a, cp) {
				t.Fatal("Astype must be a detached C-order array")
			}
			f := m(np.EmptyLike(v, v.DType(), "F"))
			if e := f.Assign(nil, v); e != nil {
				t.Fatal(e)
			}
			refcheck.CheckLayout(t, "native_assign_f_"+d, f)
			refcheck.CheckExact(t, "native_assign_f_"+d, f)
			// Exercise direct same-order copy of a disjoint F-order source.
			f2 := m(np.EmptyLike(f, f.DType(), "F"))
			if e := f2.Assign(nil, f); e != nil {
				t.Fatal(e)
			}
			refcheck.CheckExact(t, "native_assign_f_"+d, f2)
			switch v.DType() {
			case np.Uint64:
				refcheck.CheckExact(t, "native_copy_"+d, nativeValues[uint64](t, v))
			case np.Complex128:
				refcheck.CheckExact(t, "native_copy_"+d, nativeValues[complex128](t, v))
			case np.Float64:
				refcheck.CheckExact(t, "native_copy_"+d, nativeValues[float64](t, v))
			}
			refcheck.CheckExact(t, "native_copy_input_"+d, a)
		})
	}
	a := refcheck.Input(t, "native_overlap_input")
	if e := m(a.Transpose()).Assign(nil, a); e != nil {
		t.Fatal(e)
	}
	refcheck.CheckExact(t, "native_overlap_transpose", a)
	// Deliberate overlap in a writable destination retains the documented
	// C-order last-write result even with an independent RHS.
	raw := []int64{0, 0}
	b := m(np.WrapSlice(raw, 2))
	v := m(np.AsStrided(b, []int{3, 2}, []int{0, 8}, true))
	if e := v.Assign(nil, m(np.WrapSlice([]int64{1, 2, 3, 4, 5, 6}, 3, 2))); e != nil {
		t.Fatal(e)
	}
	if raw[0] != 5 || raw[1] != 6 {
		t.Fatalf("overlapping destination wrote %v", raw)
	}
	for _, d := range []string{"float32", "float64"} {
		a, b := refcheck.Input(t, "native_arithmetic_input_"+d), refcheck.Input(t, "native_arithmetic_rhs_"+d)
		v := m(m(a.Slice(np.Reverse(), np.Reverse())).Transpose())
		for _, tc := range []struct {
			name string
			fn   func(*np.Array, *np.Array) (*np.Array, error)
		}{
			{"add", np.Add}, {"sub", np.Subtract}, {"mul", np.Multiply}, {"div", np.Divide},
		} {
			refcheck.CheckExact(t, "native_"+tc.name+"_"+d, m(tc.fn(v, b)))
		}
	}
}

func TestNativeRNG(t *testing.T) {
	m := refcheck.Require(t)
	for _, tc := range []struct {
		name string
		g    *random.Generator
	}{
		{"PCG64", random.NewPCG64(642)}, {"PCG64DXSM", random.NewPCG64DXSM(642)},
	} {
		g := tc.g
		check := func(name string, a *np.Array) { refcheck.CheckExact(t, "native_rng_"+name+"_"+tc.name, a) }
		check("first", m(g.UnsignedIntegers(0, 1<<32, np.Uint32, false, 1)))
		check("reject", m(g.Integers(-17, 1<<31-16, 2051)))
		check("double", m(g.Random(17)))
		check("full32", m(g.UnsignedIntegers(0, 1<<32, np.Uint64, false, 5)))
		check("wide", m(g.UnsignedIntegers(0, 1<<32+1, np.Uint64, false, 9)))
		check("full_i64", m(g.IntegersDType(math.MinInt64, math.MaxInt64, np.Int64, true, 11)))
		check("full_u64", m(g.UnsignedIntegers(0, math.MaxUint64, np.Uint64, true, 11)))
		check("fixed", m(g.IntegersDType(4, 4, np.Int64, true, 3)))
		check("raw", m(g.RandomRaw(13)))
		check("float32", m(g.RandomDType(np.Float32, 13)))
		check("tail", m(g.Random(11)))
	}
	u := uint64(math.MaxUint64)
	g := random.PhiloxFromKey([2]uint64{17, 88}, [4]uint64{u, u, u, u})
	refcheck.CheckExact(t, "native_philox_carry", m(g.RandomRaw(13)))
	refcheck.CheckExact(t, "native_philox_cache", m(g.UnsignedIntegers(0, 1<<32, np.Uint32, false, 3)))
	refcheck.CheckExact(t, "native_philox_tail", m(g.RandomRaw(9)))
	refcheck.CheckExact(t, "native_philox_cache_tail", m(g.UnsignedIntegers(0, 1<<32, np.Uint32, false, 3)))
}

func TestNativePowerOfTwoFFT(t *testing.T) {
	m := refcheck.Require(t)
	in := func(name string) *np.Array { return refcheck.Input(t, "native_"+name) }
	check := func(name string, a *np.Array, n int) {
		t.Helper()
		key := "native_" + name
		if a.DType() != np.Float32 && a.DType() != np.Complex64 {
			refcheck.Check(t, key, a)
			return
		}
		refcheck.Metadata(t, key, a)
		refcheck.Check(t, key+"_oracle", a)
		got, e := a.Complex128s()
		if e != nil {
			t.Fatal(e)
		}
		want, e := in(name).Complex128s()
		if e != nil {
			t.Fatal(e)
		}
		maxMag, maxErr := 0., 0.
		for i, v := range want {
			maxMag = max(maxMag, cmplx.Abs(v))
			maxErr = max(maxErr, cmplx.Abs(got[i]-v))
		}
		bound := 4 * 0x1p-23 * math.Log2(float64(n)) * maxMag
		if !(maxErr <= bound) {
			t.Errorf("%s: error %g > %g", key, maxErr, bound)
		}
		t.Logf("%s: single-precision max error %.3g, bound %.3g", key, maxErr, bound)
	}
	for _, d := range []string{"float32", "float64"} {
		a := in("fft_real_input_" + d)
		check("rfft_512_"+d, m(fft.RFFT(m(a.Transpose()), fft.Options{N: np.Int(512), Axis: np.Int(0), Norm: fft.Ortho})), 512)
		check("rfft_1024_"+d, m(fft.RFFT(m(a.Slice(np.Slice{}, np.Reverse())), fft.Options{N: np.Int(1024), Norm: fft.Forward})), 1024)
		z := in("ifft_real_input_" + d)
		check("irfft_512_"+d, m(fft.IRFFT(z, fft.Options{N: np.Int(512)})), 512)
		check("irfft_1024_"+d, m(fft.IRFFT(m(m(z.Slice(np.Slice{}, np.Reverse())).Transpose()), fft.Options{N: np.Int(1024), Axis: np.Int(0), Norm: fft.Ortho})), 1024)
	}
	for _, d := range []string{"complex64", "complex128"} {
		a := in("fft_complex_input_" + d)
		check("fft_512_"+d, m(fft.FFT(m(a.Slice(np.Slice{}, np.Reverse())), fft.Options{N: np.Int(512)})), 512)
		check("ifft_1024_"+d, m(fft.IFFT(m(a.Transpose()), fft.Options{N: np.Int(1024), Axis: np.Int(0), Norm: fft.Ortho})), 1024)
	}
	refcheck.Check(t, "native_rfft_nonfinite", m(fft.RFFT(in("fft_nonfinite_input"))))
}

func TestNativeComplexKernels(t *testing.T) {
	m := refcheck.Require(t)
	in := func(name string) *np.Array { return refcheck.Input(t, "native_"+name) }
	for _, d := range []string{"complex64", "complex128"} {
		a, b := in("matmul_a_"+d), in("matmul_b_"+d)
		refcheck.Check(t, "native_matmul_"+d, m(linalg.MatMul(m(a.Slice(np.Reverse(), np.Reverse())), m(b.Slice(np.Reverse(), np.Reverse())))))
		refcheck.Check(t, "native_tensordot_"+d, m(np.Tensordot(a, b, []int{1}, []int{0})))
		refcheck.CheckExact(t, "native_matmul_a_"+d, a)
		refcheck.CheckExact(t, "native_matmul_b_"+d, b)
	}
	for _, name := range []string{"mixed", "scale"} {
		refcheck.Check(t, "native_matmul_"+name, m(linalg.MatMul(in("matmul_"+name+"_a"), in("matmul_"+name+"_b"))))
	}
	a, b := in("solve_matrix"), in("solve_rhs")
	v := m(a.Slice(np.Reverse()))
	refcheck.Check(t, "native_solve", m(linalg.Solve(v, m(b.Slice(np.Reverse())))))
	s, l, e := linalg.Slogdet(v)
	if e != nil {
		t.Fatal(e)
	}
	refcheck.Check(t, "native_sign", s)
	refcheck.Check(t, "native_logdet", l)
	refcheck.Check(t, "native_chol_lower", m(linalg.Cholesky(a)))
	refcheck.Check(t, "native_chol_upper", m(linalg.Cholesky(a, true)))
	refcheck.CheckExact(t, "native_solve_matrix", a)
}
