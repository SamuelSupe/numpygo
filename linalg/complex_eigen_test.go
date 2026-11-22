package linalg

import (
	"math"
	"math/cmplx"
	"slices"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
)

func TestComplexGeneralEigenNumPy(t *testing.T) {
	for _, name := range []string{"jordan", "repeated", "conjugate", "nonnormal", "random2", "random3", "random4", "random6", "random8", "random12", "random16", "random24", "random32", "random48", "graded24", "repeated16", "small", "large", "batch32"} {
		t.Run(name, func(t *testing.T) {
			prefix := "unified_eigen_" + name + "_"
			a := refcheck.Input(t, prefix+"input")
			e, v, err := Eig(a)
			if err != nil {
				t.Fatal(err)
			}
			refcheck.Metadata(t, prefix+"values", e)
			if v.DType() != a.DType() || !slices.Equal(v.Shape(), a.Shape()) {
				t.Fatal("wrong eigenvector metadata")
			}
			only := refcheck.Require(t)(Eigvals(a))
			refcheck.Metadata(t, prefix+"only", only)
			read := func(a *np.Array) []complex128 { return refcheck.RequireValue[[]complex128](t)(a.Complex128s()) }
			av, ev, vv, ov := read(a), read(e), read(v), read(only)
			want, wantOnly := read(refcheck.Input(t, prefix+"values")), read(refcheck.Input(t, prefix+"only"))
			n := a.Shape()[a.NDim()-1]
			tol := 2e-10
			if a.DType() == np.Complex64 {
				tol = 3e-6
			}
			for b := 0; b < a.Size()/(n*n); b++ {
				x := av[b*n*n : (b+1)*n*n]
				scale := 0.
				for _, z := range x {
					scale = max(scale, cmplx.Abs(z))
				}
				if scale == 0 {
					scale = 1
				}
				compare := func(got, want []complex128) {
					t.Helper()
					used := make([]bool, n)
					for _, z := range got {
						idx := -1
						dist := math.Inf(1)
						for j, w := range want {
							if !used[j] {
								d := cmplx.Abs(z/complex(scale, 0) - w/complex(scale, 0))
								if d < dist {
									dist = d
									idx = j
								}
							}
						}
						if idx < 0 || dist > tol {
							t.Fatalf("eigenvalue %v versus NumPy %v, scaled error %g", z, want, dist)
						}
						used[idx] = true
					}
				}
				compare(ev[b*n:(b+1)*n], want[b*n:(b+1)*n])
				compare(ov[b*n:(b+1)*n], wantOnly[b*n:(b+1)*n])
				for j := 0; j < n; j++ {
					norm, res := 0., 0.
					lambda := ev[b*n+j] / complex(scale, 0)
					for i := 0; i < n; i++ {
						z := vv[b*n*n+i*n+j]
						norm = math.Hypot(norm, cmplx.Abs(z))
						var sum complex128
						for k := 0; k < n; k++ {
							sum += (x[i*n+k] / complex(scale, 0)) * vv[b*n*n+k*n+j]
						}
						res = math.Hypot(res, cmplx.Abs(sum-lambda*z))
					}
					if math.IsNaN(res) || math.Abs(norm-1) > tol || res > float64(n)*tol {
						t.Fatalf("column %d: norm %g, scaled eigenvector residual %g", j, norm, res)
					}
				}
			}
		})
	}
	a := refcheck.Require(t)(np.FromSlice([]complex128{complex(math.NaN(), 0)}, 1, 1))
	_, _, e := Eig(a)
	refcheck.Error(t, "unified_eigen_nan", "LinAlgError", e, ErrConvergence)
}
