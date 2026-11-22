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

func TestHotSolveAndFactorization(t *testing.T) {
	m := refcheck.Require(t)
	input := func(name string) *np.Array { return refcheck.Input(t, "hot_"+name) }
	check := func(name string, a *np.Array) { refcheck.Check(t, "hot_"+name, a) }
	for _, d := range []string{"float32", "float64", "complex64", "complex128"} {
		a, b := input("solve_a_"+d), input("solve_b_"+d)
		aa := m(a.Slice(np.Slice{}, np.NewAxis, np.Reverse(), np.Reverse()))
		bb := m(b.Slice(np.NewAxis, np.Slice{}, np.Reverse(), np.Slice{}))
		check("solve_repeat_"+d, m(linalg.Solve(aa, bb)))
		aa = m(a.Slice(np.NewAxis, np.Ellipsis))
		bb = m(b.Slice(np.Slice{}, np.NewAxis, np.Ellipsis))
		check("solve_alternate_"+d, m(linalg.Solve(aa, bb)))
		refcheck.CheckExact(t, "hot_solve_a_"+d, a)
		refcheck.CheckExact(t, "hot_solve_b_"+d, b)
	}
	for _, d := range []string{"float32", "float64", "int64"} {
		a := input("chol_input_" + d)
		v := m(a.Slice(np.Reverse()))
		check("chol_lower_"+d, m(linalg.Cholesky(v)))
		check("chol_upper_"+d, m(linalg.Cholesky(m(v.Transpose(0, 2, 1)), true)))
		refcheck.CheckExact(t, "hot_chol_input_"+d, a)
		a = input("det_input_" + d)
		v = m(a.Slice(np.Reverse()))
		s, l, e := linalg.Slogdet(v)
		if e != nil {
			t.Fatal(e)
		}
		check("det_sign_"+d, s)
		check("det_log_"+d, l)
		check("det_"+d, m(linalg.Det(v)))
		refcheck.CheckExact(t, "hot_det_input_"+d, a)
	}
	s, l, e := linalg.Slogdet(m(np.Zeros(np.Float64, 2, 0, 0)))
	if e != nil {
		t.Fatal(e)
	}
	check("det_empty_sign", s)
	check("det_empty_log", l)
	_, e = linalg.Cholesky(m(np.WrapFloat64([]float64{1, 2, 2, 1}, 2, 2)))
	refcheck.Error(t, "hot_chol_not_pd", "LinAlgError", e, linalg.ErrNotPositiveDefinite)
}

func TestHotFFT(t *testing.T) {
	m := refcheck.Require(t)
	input := func(name string) *np.Array { return refcheck.Input(t, "hot_"+name) }
	check := func(name string, a *np.Array, n int) {
		t.Helper()
		key := "hot_" + name
		if a.DType() != np.Float32 && a.DType() != np.Complex64 {
			refcheck.Check(t, key, a)
			return
		}
		// The low-precision NumPy loop rounds intermediate FFT operations.
		// Require metadata and a scale-aware forward-error bound against it,
		// plus the standard strict tolerance against its binary64 oracle
		// rounded with the actual single-precision normalization.
		refcheck.Metadata(t, key, a)
		refcheck.Check(t, key+"_double_oracle", a)
		got, e := a.Complex128s()
		if e != nil {
			t.Fatal(e)
		}
		want, e := input(name).Complex128s()
		if e != nil {
			t.Fatal(e)
		}
		maxMag, maxErr := 0., 0.
		for i, z := range want {
			maxMag = max(maxMag, cmplx.Abs(z))
			maxErr = max(maxErr, cmplx.Abs(got[i]-z))
		}
		bound := 4 * 0x1p-23 * math.Log2(float64(n)) * maxMag
		if !(maxErr <= bound) {
			t.Errorf("%s: max error %g exceeds scaled FFT bound %g", key, maxErr, bound)
		}
		t.Logf("%s: NumPy single-precision max error %.3g, bound %.3g", key, maxErr, bound)
	}
	for _, d := range []string{"float32", "float64"} {
		a := input("fft_input_" + d)
		check("rfft_crop_"+d, m(fft.RFFT(m(a.Transpose()), fft.Options{N: np.Int(257), Axis: np.Int(0), Norm: fft.Ortho})), 257)
		check("rfft_pad_"+d, m(fft.RFFT(m(a.Slice(np.Slice{}, np.Reverse())), fft.Options{N: np.Int(514), Norm: fft.Forward})), 514)
		z := input("irfft_input_" + d)
		check("irfft_odd_"+d, m(fft.IRFFT(z, fft.Options{N: np.Int(257)})), 257)
		check("irfft_even_"+d, m(fft.IRFFT(z, fft.Options{N: np.Int(514), Norm: fft.Ortho})), 514)
		check("fft_real_"+d, m(fft.FFT(a, fft.Options{N: np.Int(1009)})), 1009)
		refcheck.CheckExact(t, "hot_fft_input_"+d, a)
	}
	for _, d := range []string{"complex64", "complex128"} {
		a := input("fft_input_" + d)
		check("fft_complex_"+d, m(fft.FFT(m(m(a.Slice(np.Slice{}, np.Reverse())).Transpose()), fft.Options{N: np.Int(257), Axis: np.Int(0), Norm: fft.Forward})), 257)
		check("ifft_complex_"+d, m(fft.IFFT(a, fft.Options{N: np.Int(257), Norm: fft.Ortho})), 257)
		refcheck.CheckExact(t, "hot_fft_input_"+d, a)
	}
	// Explicit out may overlap the source. A cropped prime transform
	// must gather every input before writing the returned output view.
	overlap := input("fft_input_complex128")
	dst := m(overlap.Slice(np.Slice{}, np.Range(0, 257)))
	result := m(fft.IFFT(overlap, fft.Options{N: np.Int(257), Norm: fft.Ortho, Out: dst}))
	if result != dst {
		t.Fatal("FFT did not return the provided out view")
	}
	check("ifft_complex_complex128", result, 257)
	a := input("fft_long_input")
	check("fft_long", m(fft.FFT(a)), 4093)
	check("rfft_long", m(fft.RFFT(a)), 4093)
	check("irfft_long", m(fft.IRFFT(input("irfft_long_input"), fft.Options{N: np.Int(4093)})), 4093)
	a = m(input("rfft_layout_input").Transpose())
	refcheck.CheckLayout(t, "hot_rfft_layout_0", m(fft.RFFT(a, fft.Options{Axis: np.Int(0)})))
	refcheck.CheckLayout(t, "hot_rfft_layout_1", m(fft.RFFT(a, fft.Options{Axis: np.Int(1)})))
}

func TestHotStridedArithmeticAndAssignment(t *testing.T) {
	m := refcheck.Require(t)
	for _, d := range []string{"float32", "float64"} {
		a, b := refcheck.Input(t, "hot_arithmetic_input_"+d), refcheck.Input(t, "hot_arithmetic_rhs_"+d)
		v := m(m(a.Slice(np.Reverse(), np.Reverse())).Transpose())
		for _, tc := range []struct {
			name string
			fn   func(*np.Array, *np.Array) (*np.Array, error)
		}{
			{"add", np.Add}, {"subtract", np.Subtract}, {"multiply", np.Multiply}, {"divide", np.Divide},
		} {
			refcheck.CheckExact(t, "hot_"+tc.name+"_"+d, m(tc.fn(v, b)))
		}
		refcheck.CheckExact(t, "hot_astype_"+d, m(v.Astype(v.DType())))
		c := m(a.Copy())
		dst := m(m(c.Transpose()).Slice(np.Reverse(), np.Slice{}))
		if e := dst.Assign(nil, m(a.Transpose())); e != nil {
			t.Fatal(e)
		}
		refcheck.CheckExact(t, "hot_assign_"+d, c)
	}
	// Distinct handles may alias the same Go slice without sharing storage metadata.
	raw := make([]float64, 12)
	for i := range raw {
		raw[i] = float64(i)
	}
	a, b := m(np.WrapFloat64(raw[2:10], 8)), m(np.WrapFloat64(raw[:8], 8))
	if e := a.Assign(nil, b); e != nil {
		t.Fatal(e)
	}
	refcheck.CheckExact(t, "hot_assign_overlap", m(np.WrapFloat64(raw, 12)))
}

func TestHotRandomStreams(t *testing.T) {
	m := refcheck.Require(t)
	check := func(name string, a *np.Array) { refcheck.CheckExact(t, "hot_"+name, a) }
	for _, tc := range []struct {
		name string
		g    *random.Generator
	}{
		{"PCG64", random.NewPCG64(123)}, {"PCG64DXSM", random.NewPCG64DXSM(123)},
		{"MT19937", random.NewMT19937(123)}, {"Philox", random.NewPhilox(123)}, {"SFC64", random.NewSFC64(123)},
	} {
		g, name := tc.g, tc.name
		check("raw_"+name, m(g.RandomRaw(1251)))
		check("uint32_"+name, m(g.UnsignedIntegers(0, 1<<32, np.Uint32, false, 3)))
		check("raw_cached_"+name, m(g.RandomRaw(9)))
		check("random_after_"+name, m(g.Random(11)))
		check("uint32_after_"+name, m(g.UnsignedIntegers(0, 1<<32, np.Uint32, false, 5)))
	}
	for _, d := range []np.DType{np.Bool, np.Int8, np.Int16, np.Int32, np.Int64, np.Uint8, np.Uint16, np.Uint32, np.Uint64} {
		g := random.NewPCG64(456)
		width := d.ItemSize() * 8
		var a *np.Array
		switch {
		case d == np.Bool:
			a = m(g.UnsignedIntegers(0, 1, d, true, 1029))
		case d.IsSigned():
			h := int64(1)<<(width-1) - 1
			l := -h - 1
			a = m(g.IntegersDType(l, h, d, true, 1029))
		default:
			a = m(g.UnsignedIntegers(0, ^uint64(0)>>(64-width), d, true, 1029))
		}
		check("integers_full_"+d.String(), a)
		check("integers_after_"+d.String(), m(g.RandomRaw(7)))
		g = random.NewPCG64(456)
		lo, hi := int64(0), int64(53)
		if d.IsSigned() {
			lo = -11
		}
		if d == np.Bool {
			hi = 1
		}
		check("integers_bounded_"+d.String(), m(g.IntegersDType(lo, hi, d, false, 1029)))
		check("bounded_after_"+d.String(), m(g.RandomRaw(7)))
	}
	rs := random.NewRandomState(456)
	check("legacy_integers", m(rs.IntegersDType(-55, 73, np.Int16, false, 1705)))
	check("legacy_after", m(rs.RandomSample(7)))
	g := random.NewPCG64(789)
	lo := m(np.WrapSlice([]int16{-9, -3}, 2, 1))
	hi := m(np.WrapSlice([]int16{15, 20, 45, 99}, 1, 4))
	check("integers_array", m(g.IntegersArray(lo, hi, random.IntegerOptions{Size: []int{3, 2, 4}, DType: np.Int16})))
	check("integer_array_after", m(g.RandomRaw(9)))
	g = random.NewPCG64(123)
	mu := m(refcheck.Input(t, "hot_normal_broadcast_input").Slice(np.Slice{}, np.Reverse()))
	sig := refcheck.Input(t, "hot_normal_broadcast_sigma")
	refcheck.Check(t, "hot_normal_broadcast", m(g.Sample("normal", []int{5, 3, 4}, mu, sig)))
	check("poisson_broadcast", m(g.Sample("poisson", []int{2, 3, 4}, mu)))
	trials := m(np.WrapSlice([]int64{2, 5, 9}, 3, 1))
	prob := m(np.WrapFloat64([]float64{.2, .4, .6, .8}, 1, 4))
	check("binomial_broadcast", m(g.Sample("binomial", []int{2, 3, 4}, trials, prob)))
	_, e := g.Sample("normal", nil, mu, m(np.Multiply(sig, np.Scalar(-1.))))
	refcheck.Error(t, "hot_normal_invalid", "ValueError", e, np.ErrValue)
	check("broadcast_stream_after", m(g.RandomRaw(9)))
}
