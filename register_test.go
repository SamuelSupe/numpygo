package np_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/fft"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
)

func TestRegisterComplex64(t *testing.T) {
	m := refcheck.Require(t)
	lookup := refcheck.RequireValue[*np.UFunc](t)
	a, b := refcheck.Input(t, "register_complex_a"), refcheck.Input(t, "register_complex_b")
	mask := refcheck.Input(t, "register_complex_mask")
	for _, name := range []string{"multiply", "divide"} {
		t.Run(name, func(t *testing.T) {
			m := refcheck.Require(t)
			u := lookup(np.LookupUFunc(name))
			check := func(v *np.Array) {
				refcheck.Check(t, "register_"+name, v)
				refcheck.Check(t, "register_"+name+"_wide", v)
			}
			check(m(u.Call(a, b)))
			base := m(np.Full(np.Complex64, -7, 259))
			dst := m(base.Slice(np.Range(1, 258)))
			if m(u.CallWithOptions(np.UFuncOptions{Out: dst}, a, b)) != dst {
				t.Fatal("out identity")
			}
			check(dst)
			values, e := base.Complex128s()
			if e != nil {
				t.Fatal(e)
			}
			if values[0] != -7 || values[258] != -7 {
				t.Fatal("out guard")
			}
			x, y := m(a.Copy()), m(b.Copy())
			check(m(u.CallWithOptions(np.UFuncOptions{Out: x}, x, b)))
			check(m(u.CallWithOptions(np.UFuncOptions{Out: y}, a, y)))
			base = m(np.Full(np.Complex64, -7, 259))
			dst = m(base.Slice(np.Range(1, 258)))
			m(u.CallWithOptions(np.UFuncOptions{Out: dst, Where: mask}, a, b))
			refcheck.Check(t, "register_"+name+"_masked", base)
		})
	}
	// Preserve input arrays across all operations.
	refcheck.CheckExact(t, "register_complex_a", m(a.Copy()))
	refcheck.CheckExact(t, "register_complex_b", m(b.Copy()))
}

func TestRegisterOddFFT(t *testing.T) {
	for _, d := range []np.DType{np.Float32, np.Float64} {
		t.Run(d.String(), func(t *testing.T) {
			m := refcheck.Require(t)
			tag := d.String()
			a, c := refcheck.Input(t, "register_real_input_"+tag), refcheck.Input(t, "register_spectral_input_"+tag)
			cd := c.DType()
			for _, tc := range []struct {
				n    int
				norm fft.Norm
			}{{75, fft.Forward}, {81, fft.Backward}, {125, fft.Ortho}, {243, fft.Backward}, {625, fft.Forward}, {729, fft.Ortho}, {1125, fft.Backward}, {2025, fft.Ortho}, {3125, fft.Forward}} {
				t.Run(fmt.Sprintf("%d/%s", tc.n, tc.norm), func(t *testing.T) {
					m := refcheck.Require(t)
					n := tc.n
					key := fmt.Sprintf("%s_%d_%s", tag, n, tc.norm)
					hbase := m(np.Full(cd, -9, 3*(n/2+1)+2))
					ho := m(m(hbase.Slice(np.Range(1, 1+3*(n/2+1)))).Reshape(3, n/2+1))
					rbase := m(np.Full(d, -7, 3*n+2))
					ro := m(m(rbase.Slice(np.Range(1, 1+3*n))).Reshape(3, n))
					o := fft.Options{N: np.Int(n), Norm: tc.norm, Out: ho}
					if m(fft.RFFT(a, o)) != ho {
						t.Fatal("rfft out identity")
					}
					refcheck.Check(t, "register_rfft_"+key, ho)
					o.Out = ro
					if m(fft.IRFFT(c, o)) != ro {
						t.Fatal("irfft out identity")
					}
					refcheck.Check(t, "register_irfft_"+key, ro)
					refcheck.Check(t, "register_roundtrip_"+key, m(fft.IRFFT(ho, o)))
					v, _ := hbase.Complex128s()
					w, _ := rbase.Complex128s()
					if v[0] != -9 || v[len(v)-1] != -9 || w[0] != -7 || w[len(w)-1] != -7 {
						t.Fatal("FFT out guard")
					}
				})
			}
			refcheck.Check(t, "register_rfft_reverse_"+tag, m(fft.RFFT(m(a.Slice(np.Slice{}, np.Reverse())), fft.Options{N: np.Int(729), Norm: fft.Ortho})))
			refcheck.Check(t, "register_irfft_reverse_"+tag, m(fft.IRFFT(m(c.Slice(np.Slice{}, np.Reverse())), fft.Options{N: np.Int(729), Norm: fft.Forward})))
			ax := fft.Options{N: np.Int(125), Axis: np.Int(0), Norm: fft.Ortho}
			refcheck.Check(t, "register_rfft_axis0_"+tag, m(fft.RFFT(m(a.Slice(np.Slice{}, np.Range(0, 7))), ax)))
			refcheck.Check(t, "register_irfft_axis0_"+tag, m(fft.IRFFT(m(c.Slice(np.Slice{}, np.Range(0, 7))), ax)))
			signal := refcheck.Input(t, "register_signal_"+tag)
			spec := m(fft.RFFT(signal))
			refcheck.Check(t, "register_signal_fft_"+tag, spec)
			refcheck.Check(t, "register_signal_roundtrip_"+tag, m(fft.IRFFT(spec, fft.Options{N: np.Int(729)})))
			wide := refcheck.Input(t, "register_wide_spectrum_"+tag)
			wideResult := m(fft.IRFFT(wide, fft.Options{N: np.Int(729)}))
			refcheck.Check(t, "register_wide_irfft_"+tag, wideResult)
			// NumPy's native single-precision transform has cancellation
			// error proportional to the whole signal, not the tiny bin.
			// Also enforce the independent, pointwise wide-rounding oracle.
			nativeKey := "register_wide_native_irfft_" + tag
			refcheck.Metadata(t, nativeKey, wideResult)
			want, e := refcheck.Input(t, nativeKey).Complex128s()
			if e != nil {
				t.Fatal(e)
			}
			got, e := wideResult.Complex128s()
			if e != nil {
				t.Fatal(e)
			}
			var peak, errMax float64
			for i, z := range want {
				peak = math.Max(peak, math.Abs(real(z)))
				errMax = math.Max(errMax, math.Abs(real(got[i]-z)))
			}
			tol := 1e-11
			if d == np.Float32 {
				tol = 2e-6
			}
			if errMax > tol*peak {
				t.Fatalf("native NumPy wide signal: error %g at peak %g", errMax, peak)
			}
		})
	}
}

func TestRegisterMixedFFT(t *testing.T) {
	for _, d := range []np.DType{np.Float32, np.Float64} {
		t.Run(d.String(), func(t *testing.T) {
			m := refcheck.Require(t)
			tag := d.String()
			a, c := refcheck.Input(t, "register_real_input_"+tag), refcheck.Input(t, "register_spectral_input_"+tag)
			cx := refcheck.Input(t, "register_mixed_complex_input_"+tag)
			for _, n := range []int{150, 1000, 1500, 1536, 3125} {
				t.Run(fmt.Sprint(n), func(t *testing.T) {
					key := fmt.Sprintf("%s_%d", tag, n)
					o := fft.Options{N: np.Int(n), Norm: fft.Ortho}
					refcheck.Check(t, "register_mixed_fft_"+key, m(fft.FFT(cx, o)))
					if n%2 == 0 {
						refcheck.Check(t, "register_mixed_rfft_"+key, m(fft.RFFT(a, o)))
						o.Norm = fft.Forward
						refcheck.Check(t, "register_mixed_irfft_"+key, m(fft.IRFFT(c, o)))
					}
				})
			}
		})
	}
}
