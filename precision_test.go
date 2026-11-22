package np_test

import (
	"fmt"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/fft"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
)

func TestPrecisionReductions(t *testing.T) {
	for _, d := range []np.DType{np.Float32, np.Float64} {
		t.Run(d.String(), func(t *testing.T) {
			m := refcheck.Require(t)
			a := refcheck.Input(t, "precision_reduce_"+d.String())
			for _, tag := range []string{"packed", "reverse"} {
				x := a
				if tag == "reverse" {
					x = m(x.Slice(np.Slice{}, np.Reverse()))
				}
				for _, ax := range []int{-1, 0, 1} {
					var axes []int
					key := "None"
					if ax >= 0 {
						axes, key = []int{ax}, fmt.Sprint(ax)
					}
					suffix := "_" + tag + "_" + key + "_" + d.String()
					o := np.SumOptions{Axes: axes}
					refcheck.CheckExact(t, "precision_sum"+suffix, m(np.Sum(x, o)))
					refcheck.Check(t, "precision_mean"+suffix, m(np.Mean(x, o)))
					if tag == "packed" {
						v := np.VarOptions{Axes: axes, DDOF: .5}
						refcheck.Check(t, "precision_var"+suffix, m(np.Var(x, v)))
						refcheck.Check(t, "precision_std"+suffix, m(np.Std(x, v)))
					}
				}
			}
			edge := refcheck.Input(t, "precision_reduce_edge_"+d.String())
			o := np.SumOptions{Axes: []int{1}}
			refcheck.CheckExact(t, "precision_sum_edge_"+d.String(), m(np.Sum(edge, o)))
			refcheck.Check(t, "precision_mean_edge_"+d.String(), m(np.Mean(edge, o)))
			for _, tc := range []struct {
				name string
				f    func(*np.Array, ...np.VarOptions) (*np.Array, error)
			}{{"var", np.Var}, {"std", np.Std}} {
				refcheck.Check(t, "precision_"+tc.name+"_edge_"+d.String(), m(tc.f(edge, np.VarOptions{Axes: []int{1}})))
				refcheck.Check(t, "precision_"+tc.name+"_zero_den_"+d.String(), m(tc.f(edge, np.VarOptions{Axes: []int{1}, DDOF: 137})))
				refcheck.Check(t, "precision_"+tc.name+"_empty_"+d.String(), m(tc.f(m(np.Zeros(d, 2, 0)), np.VarOptions{Axes: []int{1}})))
			}
		})
	}
}

func TestPrecisionUFuncOut(t *testing.T) {
	for _, d := range []np.DType{np.Float32, np.Float64, np.Complex64, np.Complex128} {
		t.Run(d.String(), func(t *testing.T) {
			m := refcheck.Require(t)
			lookup := refcheck.RequireValue[*np.UFunc](t)
			tag := d.String()
			x := refcheck.Input(t, "precision_unary_"+tag)
			y := refcheck.Input(t, "precision_binary_"+tag)
			mask := refcheck.Input(t, "precision_mask_"+tag)
			names := []string{"positive", "negative", "conjugate", "square", "reciprocal"}
			binary := []string{"minimum", "maximum", "fmin", "fmax"}
			if d.IsComplex() {
				binary = []string{"add", "subtract", "multiply", "divide"}
			} else {
				names = append(names, "absolute", "fabs", "sqrt", "exp", "log", "sin", "cos", "floor", "ceil", "trunc", "rint")
			}
			test := func(name string, arity int) {
				t.Run(name, func(t *testing.T) {
					m := refcheck.Require(t)
					u := lookup(np.LookupUFunc(name))
					base := m(np.Full(d, -7, 139))
					dst := m(base.Slice(np.Range(1, 138)))
					args := []*np.Array{x}
					if arity == 2 {
						args = append(args, y)
					}
					if m(u.CallWithOptions(np.UFuncOptions{Out: dst, Where: mask}, args...)) != dst {
						t.Fatal("output identity changed")
					}
					refcheck.Check(t, "precision_"+name+"_masked_"+tag, base)
					alias := m(x.Copy())
					args[0] = alias
					if m(u.CallWithOptions(np.UFuncOptions{Out: alias}, args...)) != alias {
						t.Fatal("alias identity changed")
					}
					refcheck.Check(t, "precision_"+name+"_alias_"+tag, alias)
				})
			}
			for _, name := range names {
				test(name, 1)
			}
			for _, name := range binary {
				test(name, 2)
			}
			if !d.IsComplex() {
				shift := refcheck.Input(t, "precision_shift_input_"+tag)
				u := lookup(np.LookupUFunc("square"))
				m(u.CallWithOptions(np.UFuncOptions{Out: m(shift.Slice(np.Range(1, 138)))}, m(shift.Slice(np.Range(0, 137)))))
				refcheck.CheckExact(t, "precision_shift_output_"+tag, shift)
			}
		})
	}
}

func TestPrecisionFFT(t *testing.T) {
	m := refcheck.Require(t)
	a := refcheck.Input(t, "precision_fft_real")
	c := refcheck.Input(t, "precision_fft_complex")
	for _, n := range []int{1024, 1000, 509, 2048} {
		for _, norm := range []fft.Norm{fft.Backward, fft.Ortho, fft.Forward} {
			t.Run(fmt.Sprintf("%d/%s", n, norm), func(t *testing.T) {
				m := refcheck.Require(t)
				tag := fmt.Sprintf("%d_%s", n, norm)
				co := m(np.Zeros(np.Complex64, 3, n))
				ho := m(np.Zeros(np.Complex64, 3, n/2+1))
				ro := m(np.Zeros(np.Float32, 3, n))
				o := fft.Options{N: np.Int(n), Norm: norm, Out: ho}
				if m(fft.RFFT(a, o)) != ho {
					t.Fatal("rfft out identity")
				}
				refcheck.Check(t, "precision_rfft_"+tag, ho)
				o.Out = ro
				if m(fft.IRFFT(refcheck.Input(t, "precision_rfft_"+tag), o)) != ro {
					t.Fatal("irfft out identity")
				}
				refcheck.Check(t, "precision_irfft_"+tag, ro)
				o.Out = co
				if m(fft.FFT(c, o)) != co {
					t.Fatal("fft out identity")
				}
				refcheck.Check(t, "precision_fft_"+tag, co)
				if m(fft.IFFT(c, o)) != co {
					t.Fatal("ifft out identity")
				}
				refcheck.Check(t, "precision_ifft_"+tag, co)
			})
		}
	}
	refcheck.Check(t, "precision_fft_realcomplex", m(fft.FFT(a, fft.Options{N: np.Int(1024)})))
	refcheck.Check(t, "precision_fft_axis0", m(fft.FFT(c, fft.Options{N: np.Int(64), Axis: np.Int(0), Norm: fft.Ortho})))
	refcheck.Check(t, "precision_rfft_reverse", m(fft.RFFT(m(a.Slice(np.Slice{}, np.Reverse())), fft.Options{N: np.Int(1024)})))
	u, v := refcheck.Input(t, "precision_fft2_real_input"), refcheck.Input(t, "precision_fft2_complex_input")
	refcheck.CheckExact(t, "precision_fft2_real", m(fft.FFT(u)))
	refcheck.CheckExact(t, "precision_rfft2", m(fft.RFFT(u)))
	refcheck.CheckExact(t, "precision_fft2_complex", m(fft.FFT(v)))
	refcheck.CheckExact(t, "precision_ifft2", m(fft.IFFT(v, fft.Options{Norm: fft.Forward})))
	refcheck.CheckExact(t, "precision_irfft2", m(fft.IRFFT(v, fft.Options{Norm: fft.Forward})))
	alias := m(m(c.Slice(np.Slice{}, np.Range(0, 1024))).Copy())
	if m(fft.FFT(alias, fft.Options{Norm: fft.Ortho, Out: alias})) != alias {
		t.Fatal("complex64 alias identity changed")
	}
	refcheck.Check(t, "precision_fft_alias", alias)
}
