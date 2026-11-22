package poly_test

import (
	"math"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"github.com/SamuelSupe/numpygo/poly"
)

func TestPolynomialBases(t *testing.T) {
	must := refcheck.Require(t)
	c, d := refcheck.Input(t, "poly_input_c"), refcheck.Input(t, "poly_input_d")
	x, y, z := refcheck.Input(t, "poly_input_x"), refcheck.Input(t, "poly_input_y"), refcheck.Input(t, "poly_input_z")
	coefND, coef3 := refcheck.Input(t, "poly_input_coeff_nd"), refcheck.Input(t, "poly_input_coeff_3")
	gx, gy, gz := refcheck.Input(t, "poly_input_gx"), refcheck.Input(t, "poly_input_gy"), refcheck.Input(t, "poly_input_gz")
	fitX, weights := refcheck.Input(t, "poly_input_fit_x"), refcheck.Input(t, "poly_input_weights")
	roots := refcheck.Input(t, "poly_input_roots")
	for _, b := range []poly.Basis{poly.Power, poly.Chebyshev, poly.Legendre, poly.Hermite, poly.HermiteE, poly.Laguerre} {
		t.Run(b.String(), func(t *testing.T) {
			must := refcheck.Require(t)
			prefix := "poly_" + b.String() + "_"
			check := func(name string, a *np.Array) { t.Helper(); refcheck.Check(t, prefix+name, a) }
			check("add", must(b.Add(c, d)))
			check("sub", must(b.Sub(c, d)))
			check("mul", must(b.Mul(c, d)))
			q, r, err := b.Div(c, d)
			if err != nil {
				t.Fatal(err)
			}
			check("div_q", q)
			check("div_r", r)
			check("mulx", must(b.MulX(c)))
			check("pow", must(b.Pow(c, 3)))
			check("trim", must(b.Trim(must(np.FromFloat64([]float64{1, 2, 1e-8, 0}, 4)), 1e-7)))
			check("line", must(b.Line(2, 3)))
			rc := must(b.FromRoots(roots))
			check("fromroots", rc)
			checkRoots(t, prefix+"roots", must(b.Roots(rc)))
			complexRoots := must(np.FromComplex128([]complex128{1 + 2i, -1 + .5i}, 2))
			check("complex_roots", must(b.Roots(must(b.FromRoots(complexRoots)))))
			check("companion", must(b.Companion(c)))
			check("der", must(b.Derivative(c, 2, 1.25)))
			check("int", must(b.Integral(c, 2, []complex128{2, -1}, .3, .7)))
			trailing := must(np.FromFloat64([]float64{1, 2, 0}, 3))
			check("int_zeros", must(b.Integral(trailing, 1, nil, 0, 1)))
			check("der_zeros", must(b.Derivative(trailing, 1, 1)))
			check("val", must(b.Eval(c, x)))
			cc := must(np.Stack([]*np.Array{c, must(np.Multiply(c, np.Scalar(2)))}, 1))
			check("val_series", must(b.Eval(cc, x)))
			check("val_complex", must(b.Eval(must(c.Astype(np.Complex128)), must(np.FromComplex128([]complex128{1 + 2i, -.3 + .5i}, 2)))))
			check("val_float32", must(b.Eval(must(c.Astype(np.Float32)), must(x.Astype(np.Float32)))))
			check("vander", must(b.Vander(x, 3)))
			check("val2d", must(b.EvalND(coefND, false, x, y)))
			check("val3d", must(b.EvalND(coef3, false, x, y, z)))
			check("valnd", must(b.EvalND(coef3, false, x, y, z)))
			check("grid2d", must(b.EvalND(coefND, true, gx, gy)))
			check("grid3d", must(b.EvalND(coef3, true, gx, gy, gz)))
			check("vander2d", must(b.VanderND([]int{2, 1}, x, y)))
			check("vander3d", must(b.VanderND([]int{1, 2, 1}, x, y, z)))
			fitY := refcheck.Input(t, prefix+"fit_y")
			fit, err := b.Fit(fitX, fitY, 3, poly.FitOptions{Weights: weights})
			if err != nil {
				t.Fatal(err)
			}
			check("fit", fit.Coefficients)
			check("fit_residual", fit.Residuals)
			check("fit_rank", np.ScalarOf(fit.Rank))
			check("fit_singular", fit.SingularValues)
			check("fit_rcond", np.Scalar(fit.Rcond))
			fit, err = b.Fit(fitX, fitY, 3, poly.FitOptions{Degrees: []int{0, 2, 3}})
			if err != nil {
				t.Fatal(err)
			}
			check("fit_selected", fit.Coefficients)
			opt := poly.SeriesOptions{Domain: &[2]float64{1, 5}, Window: &[2]float64{-1, 2}}
			s, err := poly.NewSeries(b, c, opt)
			if err != nil {
				t.Fatal(err)
			}
			sx := must(np.FromFloat64([]float64{1, 2, 3, 4, 5}, 5))
			check("series_eval", must(s.Eval(sx)))
			der, err := s.Derivative(2)
			if err != nil {
				t.Fatal(err)
			}
			check("series_derivative", must(der.Eval(sx)))
			integral, err := s.Integral(2, []complex128{2, -1}, 2)
			if err != nil {
				t.Fatal(err)
			}
			check("series_integral", must(integral.Eval(sx)))
			s, err = poly.NewSeries(b, rc, opt)
			if err != nil {
				t.Fatal(err)
			}
			checkRoots(t, prefix+"series_roots", must(s.Roots()))
			if b == poly.Power {
				check("valfromroots", must(poly.ValFromRoots(roots, x)))
			} else {
				check("to_power", must(b.Convert(c, poly.Power)))
				check("from_power", must(poly.Power.Convert(c, b)))
				nodes, w, err := b.Gauss(5)
				if err != nil {
					t.Fatal(err)
				}
				check("gauss_nodes", nodes)
				check("gauss_weights", w)
				check("weight", must(b.Weight(gx)))
			}
		})
	}
	refcheck.Check(t, "poly_chebyshev_pts1", must(poly.ChebPts1(7)))
	refcheck.Check(t, "poly_chebyshev_pts2", must(poly.ChebPts2(7)))
	refcheck.Check(t, "poly_chebyshev_interpolate", must(poly.ChebInterpolate(func(x float64) complex128 { return complex(math.Exp(x), 0) }, 5)))
	_, _, err := poly.Power.Div(c, np.Scalar(0))
	refcheck.Error(t, "poly_bad_division", "ZeroDivisionError", err, np.ErrValue)
	_, err = poly.Power.Derivative(c, -1, 1)
	refcheck.Error(t, "poly_bad_derivative", "ValueError", err, np.ErrValue)
	_, err = poly.Power.Pow(c, -1)
	refcheck.Error(t, "poly_bad_power", "ValueError", err, np.ErrValue)
	_, err = poly.Chebyshev.Vander(x, -1)
	refcheck.Error(t, "poly_bad_degree", "ValueError", err, np.ErrValue)
}

func checkRoots(t *testing.T, name string, got *np.Array) {
	t.Helper()
	want := refcheck.Input(t, name)
	if want.DType() != got.DType() {
		// Different eigensolvers may report an all-real spectrum in a real or complex array.
		for _, a := range []*np.Array{want, got} {
			values, _ := a.Complex128s()
			for _, z := range values {
				if math.Abs(imag(z)) > 1e-11 {
					t.Fatalf("%s: incompatible root dtype %s vs %s", name, got.DType(), want.DType())
				}
			}
		}
		t.Logf("%s: real-root dtype differs (%s vs %s)", name, got.DType(), want.DType())
		got = refcheck.Require(t)(got.Astype(want.DType()))
	}
	refcheck.Check(t, name, got)
}

func TestDecompPolynomialRootDType(t *testing.T) {
	m := refcheck.Require(t)
	for _, d := range []string{"float32", "float64", "int64", "complex64", "complex128"} {
		for _, kind := range []string{"real", "complex"} {
			tag := "decomp_poly_roots_" + kind + "_" + d
			a := refcheck.Input(t, tag+"_input")
			refcheck.Check(t, tag+"_legacy", m(poly.LegacyRoots(a)))
			refcheck.Check(t, tag+"_power", m(poly.Power.Roots(m(a.Slice(np.Reverse())))))
		}
	}
}

func TestLegacyPolynomial(t *testing.T) {
	must := refcheck.Require(t)
	c, d := refcheck.Input(t, "poly_input_c"), refcheck.Input(t, "poly_input_d")
	x := refcheck.Input(t, "poly_input_x")
	for name, fn := range map[string]func(*np.Array, *np.Array) (*np.Array, error){"add": poly.LegacyAdd, "sub": poly.LegacySub, "mul": poly.LegacyMul} {
		refcheck.Check(t, "poly_legacy_"+name, must(fn(c, d)))
	}
	q, r, err := poly.LegacyDiv(c, d)
	if err != nil {
		t.Fatal(err)
	}
	refcheck.Check(t, "poly_legacy_div_q", q)
	refcheck.Check(t, "poly_legacy_div_r", r)
	refcheck.Check(t, "poly_legacy_der", must(poly.LegacyDerivative(c, 2)))
	refcheck.Check(t, "poly_legacy_int", must(poly.LegacyIntegral(c, 2, []complex128{2, -1})))
	refcheck.Check(t, "poly_legacy_val", must(poly.LegacyEval(c, x)))
	refcheck.Check(t, "poly_legacy_roots", must(poly.LegacyRoots(c)))
	refcheck.Check(t, "poly_legacy_poly", must(poly.LegacyFromRoots(refcheck.Input(t, "poly_input_roots"))))
	fitX := refcheck.Input(t, "poly_input_fit_x")
	fit, err := poly.LegacyFit(fitX, must(poly.LegacyEval(c, fitX)), 3, poly.FitOptions{Weights: refcheck.Input(t, "poly_input_weights")})
	if err != nil {
		t.Fatal(err)
	}
	refcheck.Check(t, "poly_legacy_fit", fit.Coefficients)
	series, err := poly.NewLegacySeries(c)
	if err != nil {
		t.Fatal(err)
	}
	refcheck.Check(t, "poly_legacy_series", must(series.Eval(x)))
}
