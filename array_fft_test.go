package np_test

import (
	"fmt"
	"math"
	"math/cmplx"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/fft"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
)

func TestArrayFFTCasts(t *testing.T) {
	m := refcheck.Require(t)
	dtypes := []np.DType{np.Float32, np.Float64, np.Complex64, np.Complex128}
	for _, d := range dtypes {
		t.Run(d.String(), func(t *testing.T) {
			m := refcheck.Require(t)
			tag := "arrayfft_cast_" + d.String()
			a := refcheck.Input(t, tag)
			v := m(m(a.Slice(np.Reverse(), np.Reverse())).Transpose())
			for _, dst := range dtypes {
				if dst == d {
					continue
				}
				key := tag + "_" + dst.String()
				for _, case_ := range []struct {
					name string
					x    *np.Array
				}{{"packed", a}, {"view", v}} {
					out := m(case_.x.Astype(dst))
					refcheck.CheckExact(t, key+"_"+case_.name, out)
					if np.SharesMemory(out, case_.x) || !out.IsCContiguous() {
						t.Fatal("conversion must be a detached C-order result")
					}
					if dst == np.Complex128 {
						z, e := case_.x.Complex128s()
						if e != nil {
							t.Fatal(e)
						}
						refcheck.CheckExact(t, key+"_"+case_.name, m(np.WrapComplex128(z, case_.x.Shape()...)))
					}
				}
			}
			if d != np.Complex128 {
				b := m(m(a.Slice(np.Range(0, 1), np.Slice{})).BroadcastTo(4, 35, 37))
				z, e := b.Complex128s()
				if e != nil {
					t.Fatal(e)
				}
				refcheck.CheckExact(t, tag+"_complex128_broadcast", m(np.WrapComplex128(z, b.Shape()...)))
				refcheck.CheckExact(t, tag+"_complex128_broadcast", m(b.Astype(np.Complex128)))
			}
			refcheck.CheckExact(t, tag, a)
		})
	}
	// Empty typed arrays must not access storage element zero.
	for _, d := range dtypes {
		a := m(np.Zeros(d, 3, 0))
		b := m(a.Astype(np.Float64))
		if b.Size() != 0 || b.DType() != np.Float64 {
			t.Fatal("empty cast metadata")
		}
	}
}

func TestArrayFFTBroadcast(t *testing.T) {
	m := refcheck.Require(t)
	for _, d := range []np.DType{np.Float32, np.Float64} {
		tag := "arrayfft_lane_"
		a := refcheck.Input(t, tag+d.String())
		flat := m(a.Reshape(207))
		x := m(flat.Slice(np.Range(1, 70)))
		mat := m(m(flat.Slice(np.Range(1, 205))).Reshape(3, 68))
		col := refcheck.Input(t, tag+"col_"+d.String())
		row := m(a.Slice(np.AtIndex(0)))
		var scalar *np.Array
		if d == np.Float32 {
			scalar = np.ScalarOf(float32(-1.25))
		} else {
			scalar = np.ScalarOf(float64(-1.25))
		}
		for _, op := range []struct {
			name string
			f    func(*np.Array, *np.Array) (*np.Array, error)
		}{{"add", np.Add}, {"sub", np.Subtract}, {"mul", np.Multiply}, {"div", np.Divide}} {
			for _, tc := range []struct {
				name string
				x, y *np.Array
			}{{"right", x, scalar}, {"left", scalar, x}, {"col", mat, col}, {"row", a, row}} {
				refcheck.CheckExact(t, tag+op.name+"_"+tc.name+"_"+d.String(), m(op.f(tc.x, tc.y)))
			}
		}
		refcheck.CheckExact(t, tag+d.String(), a)
	}
}

func TestArrayFFTTransforms(t *testing.T) {
	m := refcheck.Require(t)
	in := func(name string) *np.Array { return refcheck.Input(t, "arrayfft_"+name) }
	a := in("real_long")
	for _, n := range []int{4096, 65536, 4093} {
		t.Run(fmt.Sprint(n), func(t *testing.T) {
			m := refcheck.Require(t)
			x := m(a.Slice(np.Range(0, n)))
			key := fmt.Sprintf("arrayfft_rfft_%d", n)
			refcheck.Check(t, key, m(fft.RFFT(x, fft.Options{Norm: fft.Ortho})))
			refcheck.Check(t, fmt.Sprintf("arrayfft_irfft_%d", n), m(fft.IRFFT(refcheck.Input(t, key), fft.Options{N: np.Int(n), Norm: fft.Ortho})))
		})
	}
	z := in("complex_input")
	refcheck.Check(t, "arrayfft_fft_transposed", m(fft.FFT(m(z.Transpose()), fft.Options{Axis: np.Int(0), Norm: fft.Ortho})))
	refcheck.Check(t, "arrayfft_ifft_reverse", m(fft.IFFT(m(z.Slice(np.Slice{}, np.Reverse())), fft.Options{Norm: fft.Forward})))
	refcheck.CheckExact(t, "arrayfft_complex_input", z)
	refcheck.CheckExact(t, "arrayfft_real_long", a)
	c, x := in("complex32_input"), in("real32_input")
	// Accept NumPy's float32 internal rounding at its transform scale,
	// whilst also checking an independent double-precision oracle rounded
	// to the requested result dtype.
	for _, tc := range []struct {
		name string
		y    *np.Array
	}{
		{"fft32", m(fft.FFT(c))}, {"ifft32", m(fft.IFFT(c))},
		{"rfft32", m(fft.RFFT(x))}, {"irfft32", m(fft.IRFFT(in("rfft32"), fft.Options{N: np.Int(4096)}))},
	} {
		refcheck.Check(t, "arrayfft_"+tc.name+"_oracle", tc.y)
		refcheck.Metadata(t, "arrayfft_"+tc.name, tc.y)
		got, _ := tc.y.Complex128s()
		want, _ := in(tc.name).Complex128s()
		maxMag, maxErr := 0., 0.
		for i, z := range want {
			maxMag = max(maxMag, cmplx.Abs(z))
			maxErr = max(maxErr, cmplx.Abs(got[i]-z))
		}
		bound := 4 * 0x1p-23 * math.Log2(4096) * maxMag
		if !(maxErr <= bound) {
			t.Fatalf("%s: %g exceeds %g", tc.name, maxErr, bound)
		}
		t.Logf("%s: max NumPy single-precision error %g (bound %g)", tc.name, maxErr, bound)
	}
}
