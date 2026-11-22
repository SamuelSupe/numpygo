package np_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/fft"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
)

func TestVectorExp(t *testing.T) {
	for _, d := range []np.DType{np.Float32, np.Float64} {
		t.Run(d.String(), func(t *testing.T) {
			m := refcheck.Require(t)
			u := refcheck.RequireValue[*np.UFunc](t)(np.LookupUFunc("exp"))
			check := func(key string, a *np.Array) {
				refcheck.Metadata(t, key, a)
				w := refcheck.Input(t, key)
				got, e := a.Complex128s()
				if e != nil {
					t.Fatal(e)
				}
				want, e := w.Complex128s()
				if e != nil {
					t.Fatal(e)
				}
				rel, absolute := 5e-14, 2*math.SmallestNonzeroFloat64
				if d == np.Float32 {
					rel, absolute = 1.5e-6, 2*math.SmallestNonzeroFloat32
				}
				for i, value := range want {
					x, y := real(value), real(got[i])
					if math.IsNaN(x) && math.IsNaN(y) || x == y {
						continue
					}
					if math.IsInf(x, 0) || math.IsInf(y, 0) || math.IsNaN(x) || math.IsNaN(y) || math.Abs(y-x) > absolute+rel*math.Abs(x) {
						t.Errorf("%s [%d] got %.17g want %.17g", key, i, y, x)
					}
				}
			}
			for _, pair := range [][2]string{{"exp_input_", "exp_result_"}, {"exp_edge_input_", "exp_edge_result_"}} {
				a := refcheck.Input(t, "vector_"+pair[0]+d.String())
				key := "vector_" + pair[1] + d.String()
				check(key, m(np.Exp(a)))
				var out *np.Array
				base := m(np.Full(d, -7, a.Size()+2))
				out = m(base.Slice(np.Range(1, a.Size()+1)))
				if m(u.CallWithOptions(np.UFuncOptions{Out: out}, a)) != out {
					t.Fatal("out identity")
				}
				check(key, out)
				values, e := base.Complex128s()
				if e != nil {
					t.Fatal(e)
				}
				if values[0] != -7 || values[len(values)-1] != -7 {
					t.Fatal("out guard changed")
				}
				alias := m(a.Copy())
				check(key, m(u.CallWithOptions(np.UFuncOptions{Out: alias}, alias)))
			}
		})
	}
}

func TestVectorBinary(t *testing.T) {
	for _, d := range []np.DType{np.Float32, np.Float64, np.Complex64, np.Complex128} {
		t.Run(d.String(), func(t *testing.T) {
			lookup := refcheck.RequireValue[*np.UFunc](t)
			prefix := "extrema"
			names := []string{"minimum", "maximum"}
			if d.IsComplex() {
				prefix = "complex"
				names = []string{"add", "subtract", "multiply", "divide"}
			}
			a, b := refcheck.Input(t, "vector_"+prefix+"_a_"+d.String()), refcheck.Input(t, "vector_"+prefix+"_b_"+d.String())
			mask := refcheck.Input(t, "vector_mask_"+d.String())
			check := refcheck.Check
			if !d.IsComplex() {
				check = func(t testing.TB, key string, out *np.Array) {
					refcheck.Metadata(t, key, out)
					got, e := out.Complex128s()
					if e != nil {
						t.Fatal(e)
					}
					want, e := refcheck.Input(t, key).Complex128s()
					if e != nil {
						t.Fatal(e)
					}
					x, _ := a.Complex128s()
					y, _ := b.Complex128s()
					mv, _ := np.Values[bool](mask)
					off := (len(got) - len(x)) / 2
					for i, v := range got {
						w := real(want[i])
						j := i - off
						// NumPy's ARM64 FMIN/FMAX and amd64 min/max
						// differ for equal opposite-signed zeros. Keep
						// the library's documented right-operand policy.
						if j >= 0 && j < len(x) && (off == 0 || mv[j]) && x[j] == 0 && y[j] == 0 && w == 0 {
							w = real(y[j])
						}
						if !(math.Float64bits(real(v)) == math.Float64bits(w) || math.IsNaN(real(v)) && math.IsNaN(w)) {
							t.Errorf("%s[%d] got %v want %g", key, i, v, w)
						}
					}
				}
			}
			for _, name := range names {
				t.Run(name, func(t *testing.T) {
					m := refcheck.Require(t)
					u := lookup(np.LookupUFunc(name))
					key := "vector_" + name + "_" + d.String()
					check(t, key, m(u.Call(a, b)))
					base := m(np.Full(d, -7, a.Size()+2))
					dst := m(base.Slice(np.Range(1, a.Size()+1)))
					if m(u.CallWithOptions(np.UFuncOptions{Out: dst}, a, b)) != dst {
						t.Fatal("out identity")
					}
					check(t, key, dst)
					v, e := base.Complex128s()
					if e != nil {
						t.Fatal(e)
					}
					if v[0] != -7 || v[len(v)-1] != -7 {
						t.Fatal("out guard changed")
					}
					alias := m(a.Copy())
					check(t, key, m(u.CallWithOptions(np.UFuncOptions{Out: alias}, alias, b)))
					alias = m(b.Copy())
					check(t, key, m(u.CallWithOptions(np.UFuncOptions{Out: alias}, a, alias)))
					base = m(np.Full(d, -7, a.Size()+2))
					dst = m(base.Slice(np.Range(1, a.Size()+1)))
					m(u.CallWithOptions(np.UFuncOptions{Out: dst, Where: mask}, a, b))
					check(t, "vector_"+name+"_masked_"+d.String(), base)
				})
			}
		})
	}
}

func TestVectorFFT(t *testing.T) {
	for _, d := range []np.DType{np.Float32, np.Float64} {
		t.Run(d.String(), func(t *testing.T) {
			m := refcheck.Require(t)
			a, c := refcheck.Input(t, "vector_fft_real_input_"+d.String()), refcheck.Input(t, "vector_fft_complex_input_"+d.String())
			cd := c.DType()
			for _, tc := range []struct {
				n    int
				norm fft.Norm
			}{{90, fft.Ortho}, {150, fft.Backward}, {729, fft.Forward}, {1000, fft.Backward}, {1500, fft.Ortho}, {1536, fft.Forward}} {
				t.Run(fmt.Sprintf("%d/%s", tc.n, tc.norm), func(t *testing.T) {
					m := refcheck.Require(t)
					n := tc.n
					tag := fmt.Sprintf("%s_%d_%s", d, n, tc.norm)
					co, ho, ro := m(np.Zeros(cd, 3, n)), m(np.Zeros(cd, 3, n/2+1)), m(np.Zeros(d, 3, n))
					o := fft.Options{N: np.Int(n), Norm: tc.norm, Out: ho}
					if m(fft.RFFT(a, o)) != ho {
						t.Fatal("rfft out identity")
					}
					refcheck.Check(t, "vector_rfft_"+tag, ho)
					o.Out = ro
					if m(fft.IRFFT(refcheck.Input(t, "vector_rfft_"+tag), o)) != ro {
						t.Fatal("irfft out identity")
					}
					refcheck.Check(t, "vector_irfft_"+tag, ro)
					o.Out = co
					refcheck.Check(t, "vector_fft_"+tag, m(fft.FFT(c, o)))
					refcheck.Check(t, "vector_ifft_"+tag, m(fft.IFFT(c, o)))
				})
			}
			refcheck.Check(t, "vector_fft_real_"+d.String(), m(fft.FFT(a, fft.Options{N: np.Int(150)})))
			refcheck.Check(t, "vector_fft_axis0_"+d.String(), m(fft.FFT(m(c.Slice(np.Slice{}, np.Range(0, 7))), fft.Options{N: np.Int(150), Axis: np.Int(0), Norm: fft.Ortho})))
			refcheck.Check(t, "vector_rfft_reverse_"+d.String(), m(fft.RFFT(m(a.Slice(np.Slice{}, np.Reverse())), fft.Options{N: np.Int(150), Norm: fft.Forward})))
			alias := m(m(c.Slice(np.Slice{}, np.Range(0, 150))).Copy())
			if m(fft.FFT(alias, fft.Options{Out: alias, Norm: fft.Ortho})) != alias {
				t.Fatal("alias identity")
			}
			refcheck.Check(t, "vector_fft_alias_"+d.String(), alias)
		})
	}
}
