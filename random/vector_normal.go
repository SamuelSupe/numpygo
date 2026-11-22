package random

import (
	"fmt"
	"slices"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/linalg"
)

var ErrCovariance = fmt.Errorf("%w: covariance is not symmetric positive-semidefinite", np.ErrValue)

// MultivariateNormalOptions controls factorization and covariance validation.
// The default method is svd; check_valid defaults to warn and tolerance to
// 1e-8. Warn optionally receives the diagnostic for invalid covariance.
// Legacy RandomState supports only the SVD method.
type MultivariateNormalOptions struct {
	Method, CheckValid string
	Tol                *float64
	Warn               func(error)
}

func (g *Generator) MultivariateNormal(mean []float64, cov *np.Array, method string, shape ...int) (*np.Array, error) {
	return g.MultivariateNormalWithOptions(mean, cov, MultivariateNormalOptions{Method: method}, shape...)
}

func (g *Generator) MultivariateNormalWithOptions(mean []float64, cov *np.Array, opt MultivariateNormalOptions, shape ...int) (*np.Array, error) {
	n := len(mean)
	if cov.DType() == 0 || cov.DType().IsComplex() || !slices.Equal(cov.Shape(), []int{n, n}) {
		return nil, np.ErrShape
	}
	method := opt.Method
	if method == "" {
		method = "svd"
	}
	if method != "svd" && method != "eigh" && method != "cholesky" || g.legacy && method != "svd" {
		return nil, np.ErrValue
	}
	check := opt.CheckValid
	if check == "" {
		check = "warn"
	}
	tol := 1e-8
	if opt.Tol != nil {
		tol = *opt.Tol
	}
	outShape := append(slices.Clone(shape), n)
	count, err := np.ElementCount(np.Float64, shape...)
	if err != nil {
		return nil, err
	}
	if _, err = np.ElementCount(np.Float64, outShape...); err != nil {
		return nil, err
	}
	c, err := cov.Astype(np.Float64)
	if err != nil {
		return nil, err
	}
	// NumPy draws the normal samples before factorization and checking
	// covariance. Errors after this point must advance the same RNG stream.
	z, err := g.StandardNormal(count, n)
	if err != nil {
		return nil, err
	}
	var basis, s, factor *np.Array
	valid := true
	switch method {
	case "cholesky":
		factor, err = linalg.Cholesky(c)
	case "eigh":
		s, basis, err = linalg.Eigh(c)
		if err == nil && check != "ignore" {
			ev, e := s.Float64s()
			if e != nil {
				return nil, e
			}
			for _, v := range ev {
				if v < -tol {
					valid = false
					break
				}
			}
		}
		if err == nil {
			s, err = np.Absolute(s)
		}
	case "svd":
		var vh *np.Array
		basis, s, vh, err = linalg.SVD(c)
		if err == nil && (g.legacy || check != "ignore") {
			vt, e := vh.Transpose()
			if e != nil {
				return nil, e
			}
			if g.legacy {
				basis = vt
			}
			if check != "ignore" {
				vs, e := np.Multiply(vt, s)
				if e != nil {
					return nil, e
				}
				reconstructed, e := linalg.MatMul(vs, vh)
				if e != nil {
					return nil, e
				}
				valid, e = np.AllClose(reconstructed, c, tol, tol, false)
				if e != nil {
					return nil, e
				}
			}
		}
	}
	if err != nil {
		return nil, err
	}
	if method != "cholesky" {
		switch check {
		case "ignore":
		case "raise":
			if !valid {
				return nil, ErrCovariance
			}
		case "warn":
			if !valid && opt.Warn != nil {
				opt.Warn(ErrCovariance)
			}
		default:
			return nil, np.ErrValue
		}
		s, err = np.Sqrt(s)
		if err != nil {
			return nil, err
		}
		factor, err = np.Multiply(basis, s)
		if err != nil {
			return nil, err
		}
	}
	ft, err := factor.Transpose()
	if err != nil {
		return nil, err
	}
	out, err := linalg.MatMul(z, ft)
	if err != nil {
		return nil, err
	}
	m, err := np.FromFloat64(mean, n)
	if err != nil {
		return nil, err
	}
	out, err = np.Add(out, m)
	if err != nil {
		return nil, err
	}
	return out.Reshape(outShape...)
}
