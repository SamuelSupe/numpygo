package poly_test

import (
	"iter"
	"math"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"github.com/SamuelSupe/numpygo/poly"
)

func TestSeriesClassesNumPy(t *testing.T) {
	bases := []poly.Basis{poly.Power, poly.Chebyshev, poly.Legendre, poly.Hermite, poly.HermiteE, poly.Laguerre}
	domain, window := [2]float64{1, 5}, [2]float64{-1, 2}
	for _, b := range bases {
		t.Run(b.String(), func(t *testing.T) {
			m := refcheck.Require(t)
			sm := refcheck.RequireValue[*poly.Series](t)
			bm := refcheck.RequireValue[bool](t)
			tm := refcheck.RequireValue[string](t)
			tag := "series_" + b.String() + "_"
			opt := poly.SeriesOptions{Domain: &domain, Window: &window, Symbol: "t"}
			check := func(n string, a *np.Array) {
				t.Helper()
				refcheck.Check(t, tag+n, a)
			}
			text := func(n, s string) {
				t.Helper()
				refcheck.Text(t, tag+n, s)
			}
			interval := func(n string, v [2]float64) {
				t.Helper()
				want := refcheck.Input(t, tag+n)
				// The Go interval representation is explicitly [2]float64,
				// even for a domain inferred from lower-precision samples.
				if want.DType() != np.Float64 {
					want = m(want.Astype(np.Float64))
					actual := m(np.FromSlice(v[:], 2))
					if !bm(np.ArrayEqual(want, actual, false)) {
						t.Errorf("%s: interval values %v do not match NumPy", tag+n, v)
					}
					return
				}
				check(n, m(np.FromSlice(v[:], 2)))
			}
			checkSeries := func(n string, v *poly.Series, kind poly.Basis) {
				t.Helper()
				if v.Basis != kind {
					t.Fatalf("%s: want basis %s, got %s", n, kind, v.Basis)
				}
				check(n+"_coef", v.CoefficientView())
				interval(n+"_domain", v.Domain())
				interval(n+"_window", v.Window())
				text(n+"_symbol", v.Symbol())
			}
			yes := func(v bool) int64 {
				if v {
					return 1
				}
				return 0
			}
			ok := func(e error) {
				t.Helper()
				if e != nil {
					t.Fatal(e)
				}
			}
			coeff := refcheck.Input(t, "series_input_coeff")
			other := refcheck.Input(t, "series_input_other")
			x := refcheck.Input(t, "series_input_x")
			p := sm(poly.NewSeries(b, coeff, opt))
			q := sm(poly.NewSeries(b, other, opt))
			check("coef", p.CoefficientView())
			interval("domain", p.Domain())
			interval("window", p.Window())
			interval("default_domain", b.DefaultDomain())
			interval("default_window", b.DefaultWindow())
			check("metadata", m(np.FromSlice([]int64{int64(p.Len()), int64(p.Degree()), int64(b.MaxPower())}, 3)))
			text("symbol", p.Symbol())
			text("basis_name", b.BasisName())
			values := []float64{}
			for v := range refcheck.RequireValue[iter.Seq[any]](t)(p.Iter()) {
				values = append(values, v.(float64))
			}
			check("iter", m(np.FromSlice(values, len(values))))
			off, scale := p.MapParameters()
			check("mapparms", m(np.FromSlice([]float64{off, scale}, 2)))
			check("call", m(p.Eval(x)))
			text("repr", tm(p.Repr()))
			text("ascii", tm(p.Format("ascii")))
			text("unicode", tm(p.Format("unicode")))
			text("str", p.String())
			ok(coeff.SetValue(100., 0))
			check("constructor_copy", p.CoefficientView())
			cp := sm(p.Copy())
			checkSeries("copy", cp, b)
			check("copy_equal", m(np.FromSlice([]int64{yes(bm(p.Equal(cp))), yes(p == cp), yes(p.Positive() == p)}, 3)))
			ok(cp.CoefficientView().SetValue(-9., 0))
			check("copy_mutated", cp.CoefficientView())
			check("copy_original", p.CoefficientView())
			checkSeries("coef_view_mutation", cp, b)
			snapshot := m(p.Coefficients())
			ok(snapshot.SetValue(44., 0))
			check("copy_original", p.CoefficientView())
			for n, v := range map[string]*poly.Series{
				"cutdeg": sm(p.CutDegree(2)), "cutdeg_large": sm(p.CutDegree(100)),
				"truncate": sm(p.Truncate(2)), "truncate_large": sm(p.Truncate(100)),
				"trim":     sm(p.Trim(0)),
				"trim_tol": sm(sm(poly.NewSeries(b, m(np.FromSlice([]float64{1, 2, .001, 0}, 4)), poly.SeriesOptions{Symbol: "t"})).Trim(.01)),
				"trim_all": sm(sm(poly.NewSeries(b, m(np.FromSlice([]float64{.01, -.01}, 2)), poly.SeriesOptions{Symbol: "t"})).Trim(.02)),
			} {
				checkSeries(n, v, b)
			}
			diffBasis := poly.Power
			if b == poly.Power {
				diffBasis = poly.Chebyshev
			}
			checks := []*poly.Series{
				sm(p.Copy()), q,
				sm(poly.NewSeries(b, p.CoefficientView(), poly.SeriesOptions{Domain: &[2]float64{-1, 1}, Window: &window, Symbol: "t"})),
				sm(poly.NewSeries(b, p.CoefficientView(), poly.SeriesOptions{Domain: &domain, Window: &[2]float64{0, 1}, Symbol: "t"})),
				sm(poly.NewSeries(b, p.CoefficientView(), poly.SeriesOptions{Domain: &domain, Window: &window, Symbol: "s"})),
				sm(poly.NewSeries(diffBasis, p.CoefficientView(), opt)),
			}
			cmp := []int64{}
			for _, v := range checks {
				eq := bm(p.Equal(v))
				cmp = append(cmp, yes(bm(p.HasSameCoefficients(v))), yes(p.HasSameDomain(v)), yes(p.HasSameWindow(v)), yes(p.HasSameType(v)), yes(eq), yes(!eq))
			}
			check("comparisons", m(np.FromSlice(cmp, 6, 6)))
			checkSeries("basis", sm(b.SeriesBasis(3, opt)), b)
			checkSeries("basis_default", sm(b.SeriesBasis(0, poly.SeriesOptions{})), b)
			checkSeries("identity", sm(b.Identity(opt)), b)
			checkSeries("identity_default", sm(b.Identity(poly.SeriesOptions{})), b)
			check("identity_eval", m(sm(b.Identity(opt)).Eval(x)))
			two := np.Scalar(2)
			for n, v := range map[string]*poly.Series{
				"add": sm(p.Add(q)), "sub": sm(p.Sub(q)), "mul": sm(p.Mul(q)),
				"neg": sm(p.Negative()), "pos": p.Positive(),
				"pow": sm(p.Pow(3)), "pow_zero": sm(p.Pow(0)),
				"add_coef": sm(p.Add(m(np.FromSlice([]float64{2, 3}, 2)))),
				"radd":     sm(p.Add(two)), "rsub": sm(p.RSub(two)), "rmul": sm(p.Mul(two)),
				"true_div": sm(p.DivideScalar(two)),
				"pow_17":   sm(sm(poly.NewSeries(b, two)).Pow(17)),
			} {
				checkSeries(n, v, b)
			}
			quo, rem, e := p.DivMod(q, false)
			ok(e)
			checkSeries("floor_div", quo, b)
			checkSeries("mod", rem, b)
			checkSeries("divmod_q", quo, b)
			checkSeries("divmod_r", rem, b)
			quo, rem, e = p.DivMod(m(np.FromSlice([]float64{2, 3, 4, 5, 6, 7}, 6)), true)
			ok(e)
			checkSeries("rdivmod_q", quo, b)
			checkSeries("rdivmod_r", rem, b)
			checkSeries("rfloordiv", quo, b)
			checkSeries("rmod", rem, b)
			checkSeries("deriv", sm(p.Derivative(2)), b)
			checkSeries("integ", sm(p.Integral(2, []complex128{2, -1}, 2)), b)
			checkSeries("integ_default", sm(p.Integrate(1, []complex128{.5}, nil)), b)
			for _, tc := range []struct {
				name string
				n    int
				d    *[2]float64
			}{
				{"default", 5, nil}, {"reverse", 6, &[2]float64{5, 1}}, {"empty", 0, nil},
			} {
				a, y, e := p.Linspace(tc.n, tc.d)
				ok(e)
				check("linspace_"+tc.name+"_x", a)
				check("linspace_"+tc.name+"_y", y)
			}
			checkSeries("convert_default", sm(p.Convert(b, poly.SeriesOptions{})), b)
			checkSeries("convert_same", sm(p.Convert(b, opt)), b)
			for _, target := range bases {
				co := poly.SeriesOptions{Domain: &[2]float64{-2, 2}, Window: &[2]float64{.5, 1.5}, Symbol: "u"}
				v := sm(poly.NewSeries(target, m(np.FromSlice([]float64{.4, -.3, .2}, 3)), co))
				composed := sm(p.Compose(v))
				checkSeries("compose_"+target.String(), composed, target)
				check("compose_"+target.String()+"_eval", m(composed.Eval(x)))
				to := poly.SeriesOptions{Domain: &[2]float64{-2, 2}, Window: &[2]float64{-1, 1}}
				converted := sm(p.Convert(target, to))
				checkSeries("convert_"+target.String(), converted, target)
				check("convert_"+target.String()+"_eval", m(converted.Eval(x)))
				checkSeries("cast_"+target.String(), sm(target.Cast(p, to)), target)
			}
			roots := refcheck.Input(t, "series_input_roots")
			r := sm(b.SeriesFromRoots(roots, opt, false))
			checkSeries("fromroots", r, b)
			checkRoots(t, tag+"roots", m(r.Roots()))
			checkSeries("fromroots_default", sm(b.SeriesFromRoots(roots, poly.SeriesOptions{}, false)), b)
			checkSeries("fromroots_infer", sm(b.SeriesFromRoots(roots, poly.SeriesOptions{}, true)), b)
			checkSeries("fromroots_complex", sm(b.SeriesFromRoots(m(np.FromSlice([]complex128{2 + 1i, 3 - .5i}, 2)), opt, false)), b)
			fitX := refcheck.Input(t, "series_input_fit_x")
			fitY := refcheck.Input(t, "series_input_fit_y")
			weights := refcheck.Input(t, "series_input_weights")
			for _, tc := range []struct {
				name string
				opt  poly.SeriesFitOptions
			}{
				{"fit", poly.SeriesFitOptions{SeriesOptions: opt, FitOptions: poly.FitOptions{Weights: weights, Degrees: []int{0, 2, 3}}}},
				{"fit_infer", poly.SeriesFitOptions{SeriesOptions: poly.SeriesOptions{Symbol: "t"}, FitOptions: poly.FitOptions{Degrees: []int{0, 2, 3}}}},
				{"fit_default", poly.SeriesFitOptions{SeriesOptions: poly.SeriesOptions{Symbol: "t"}, FitOptions: poly.FitOptions{Degrees: []int{0, 2, 3}}, UseDefaultDomain: true}},
			} {
				v, result, e := b.FitSeries(fitX, fitY, 3, tc.opt)
				ok(e)
				checkSeries(tc.name, v, b)
				check(tc.name+"_residual", result.Residuals)
				check(tc.name+"_rank", np.ScalarOf(result.Rank))
				check(tc.name+"_singular", result.SingularValues)
				check(tc.name+"_rcond", np.Scalar(result.Rcond))
				check(tc.name+"_eval", m(v.Eval(fitX)))
			}
			fit, _, e := b.FitSeries(m(np.FromSlice([]float64{2, 2, 2, 2}, 4)), m(np.FromSlice([]float64{1, 2, 3, 4}, 4)), 0, poly.SeriesFitOptions{})
			ok(e)
			checkSeries("fit_constant_domain", fit, b)
			for _, d := range []np.DType{np.Int32, np.Float32} {
				xv := m(m(np.ArangeInt64(1, 6, 1)).Astype(d))
				yv := m(np.FromSlice([]float64{3, 5, 7, 9, 11}, 5))
				v, _, e := b.FitSeries(xv, yv, 1, poly.SeriesFitOptions{})
				ok(e)
				checkSeries("fit_"+d.String(), v, b)
				checkSeries("fromroots_"+d.String(), sm(b.SeriesFromRoots(m(xv.Slice(np.Range(0, 2))), poly.SeriesOptions{}, true)), b)
			}
			f32 := sm(poly.NewSeries(b, m(np.FromSlice([]float32{1, -.5, 0}, 3)), poly.SeriesOptions{Domain: &domain, Window: &window, Symbol: "θ2"}))
			checkSeries("float32", f32, b)
			checkSeries("float32_neg", sm(f32.Negative()), b)
			checkSeries("float32_pow", sm(f32.Pow(2)), b)
			checkSeries("float32_convert", sm(f32.Convert(b, poly.SeriesOptions{})), b)
			text("float32_repr", tm(f32.Repr()))
			text("float32_format", tm(f32.Format("unicode")))
			c := sm(poly.NewSeries(b, m(np.FromSlice([]complex128{1 + .5i, -.3 + 2i, .5}, 3)), opt))
			checkSeries("complex_mul", sm(c.Mul(c)), b)
			checkSeries("complex_convert", sm(c.Convert(b, poly.SeriesOptions{})), b)
			check("complex_eval", m(c.Eval(m(np.Add(x, np.ScalarOf(1i))))))
			for _, d := range []np.DType{np.Int8, np.Uint8, np.Int16, np.Uint16, np.Int32, np.Uint32, np.Int64, np.Uint64, np.Float16, np.Float32, np.Float64, np.Complex64, np.Complex128} {
				n := "dtype_" + d.String()
				var a *np.Array
				if d.IsComplex() {
					a = m(m(np.FromSlice([]complex128{1, 2 + 1i, 0}, 3)).Astype(d))
				} else {
					a = m(m(np.FromSlice([]int64{1, 2, 0}, 3)).Astype(d))
				}
				v := sm(poly.NewSeries(b, a))
				checkSeries(n, v, b)
				check(n+"_neg", sm(v.Negative()).CoefficientView())
				check(n+"_trim", sm(v.Trim(0)).CoefficientView())
				check(n+"_pow", sm(v.Pow(2)).CoefficientView())
				check(n+"_eval", m(v.Eval(m(np.FromSlice([]float32{.25, .5}, 2)))))
			}
			_, e = poly.NewSeries(b, m(np.FromSlice([]bool{true, false}, 2)))
			refcheck.Error(t, tag+"bad_bool", "ValueError", e, np.ErrValue)
			_, e = poly.NewSeries(b, p.CoefficientView(), poly.SeriesOptions{Symbol: "a+b"})
			refcheck.Error(t, tag+"bad_symbol", "ValueError", e, np.ErrValue)
			for _, tc := range []struct {
				name string
				q    *poly.Series
				kind string
				err  error
			}{
				{"bad_type", checks[5], "TypeError", np.ErrDType},
				{"bad_domain", checks[2], "TypeError", np.ErrDType},
				{"bad_window", checks[3], "TypeError", np.ErrDType},
				{"different_symbol", checks[4], "ValueError", np.ErrValue},
			} {
				_, e = p.Add(tc.q)
				refcheck.Error(t, tag+tc.name, tc.kind, e, tc.err)
			}
			_, e = b.SeriesBasis(-1, opt)
			refcheck.Error(t, tag+"bad_basis", "ValueError", e, np.ErrValue)
			_, e = p.CutDegree(-1)
			refcheck.Error(t, tag+"bad_cutdeg", "ValueError", e, np.ErrValue)
			_, e = p.Truncate(0)
			refcheck.Error(t, tag+"bad_truncate", "ValueError", e, np.ErrValue)
			_, e = p.Trim(-.25)
			refcheck.Error(t, tag+"bad_trim", "ValueError", e, np.ErrValue)
			_, e = p.Pow(101)
			refcheck.Error(t, tag+"bad_power", "ValueError", e, np.ErrValue)
			_, e = p.Pow(-1)
			refcheck.Error(t, tag+"bad_power_negative", "ValueError", e, np.ErrValue)
			_, _, e = p.DivMod(sm(poly.NewSeries(b, np.Scalar(0), opt)), false)
			refcheck.Error(t, tag+"bad_division", "ZeroDivisionError", e, np.ErrValue)
			_, e = p.DivideScalar(q.CoefficientView())
			refcheck.Error(t, tag+"bad_true_div", "TypeError", e, np.ErrDType)
			_, e = p.Format("unknown")
			refcheck.Error(t, tag+"bad_format", "ValueError", e, np.ErrValue)
			_, e = poly.NewSeries(b, m(np.Zeros(np.Float64, 0)))
			refcheck.Error(t, tag+"bad_empty", "ValueError", e, np.ErrShape)
		})
	}
}

func TestChebyshevClassInterpolateNumPy(t *testing.T) {
	m := refcheck.Require(t)
	s := refcheck.RequireValue[*poly.Series](t)(poly.InterpolateChebyshev(func(x float64) complex128 { return complex(math.Exp(x), 0) }, 6, &[2]float64{1, 3}))
	refcheck.Check(t, "series_chebyshev_interpolate_coef", s.CoefficientView())
	d := s.Domain()
	refcheck.Check(t, "series_chebyshev_interpolate_domain", m(np.FromSlice(d[:], 2)))
	refcheck.Check(t, "series_chebyshev_interpolate_eval", m(s.Eval(m(np.FromSlice([]float64{1, 2, 3}, 3)))))
}
