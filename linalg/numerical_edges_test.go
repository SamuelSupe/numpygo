package linalg_test

import (
	"math"
	"math/cmplx"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"github.com/SamuelSupe/numpygo/linalg"
)

// Relative checks here must not use a fixed absolute floor: a missing
// singular value of 1e-200 or a vanished small pseudoinverse can otherwise
// satisfy the normal differential tolerance.
func checkScaled(t *testing.T, key string, got *np.Array) {
	t.Helper()
	refcheck.Metadata(t, key, got)
	x, _ := got.Complex128s()
	y, _ := refcheck.Input(t, key).Complex128s()
	for i, w := range y {
		if w == 0 {
			if cmplx.Abs(x[i]) > 1e-12 || math.IsNaN(cmplx.Abs(x[i])) {
				t.Errorf("%s[%d]: got %v, want zero", key, i, x[i])
			}
		} else if r := cmplx.Abs(x[i]-w) / cmplx.Abs(w); math.IsNaN(r) || r > 2e-9 {
			t.Errorf("%s[%d]: relative error %g, got %v want %v", key, i, r, x[i], w)
		}
	}
}
func TestComplexSVDGradedAndScaled(t *testing.T) {
	for _, key := range []string{"graded", "triangular", "singular", "small", "large", "wide"} {
		t.Run(key, func(t *testing.T) {
			m := refcheck.Require(t)
			tag := "adv_linalg_" + key
			a := refcheck.Input(t, tag+"_input")
			u, s, vh, e := linalg.SVD(a, linalg.SVDOptions{Reduced: true})
			if e != nil {
				t.Fatal(e)
			}
			checkScaled(t, tag+"_s", s)
			pinv := m(linalg.Pinv(a, 0))
			if key == "triangular" || key == "graded" {
				// With no discarded singular values the mathematical
				// pseudoinverse equals the ordinary inverse. Compare against
				// both NumPy representations; NumPy SVD rounding can leave
				// a tiny artifact at a structurally zero inverse element.
				checkScaled(t, tag+"_inv", m(linalg.Inv(a)))
				checkScaled(t, tag+"_inv", pinv)
				v, _ := pinv.Complex128s()
				want, _ := refcheck.Input(t, tag+"_pinv").Complex128s()
				for i, w := range want {
					if key == "triangular" && i == 2 {
						if cmplx.Abs(v[i]-w) > 1e-170 {
							t.Errorf("graded inverse structural zero: got %v, NumPy %v", v[i], w)
						}
					} else if w != 0 && cmplx.Abs(v[i]-w)/cmplx.Abs(w) > 2e-9 {
						t.Errorf("pseudoinverse[%d] got %v want %v", i, v[i], w)
					}
				}
			} else {
				checkScaled(t, tag+"_pinv", pinv)
			}
			p := m(linalg.MatMul(m(np.Multiply(u, s)), vh))
			x, _ := p.Complex128s()
			y, _ := a.Complex128s()
			rows, cols := a.Shape()[0], a.Shape()[1]
			for j := 0; j < cols; j++ {
				var scale float64
				for i := 0; i < rows; i++ {
					scale = max(scale, cmplx.Abs(y[i*cols+j]))
				}
				for i := 0; i < rows; i++ {
					delta := cmplx.Abs(x[i*cols+j] - y[i*cols+j])
					if delta > scale*1e-10 || math.IsNaN(delta) {
						t.Errorf("column %d row %d relative reconstruction failed: got %v want %v", j, i, x[i*cols+j], y[i*cols+j])
					}
				}
			}
			uh := m(m(np.Conjugate(u)).Transpose())
			v := m(m(np.Conjugate(vh)).Transpose())
			for _, product := range []*np.Array{m(linalg.MatMul(uh, u)), m(linalg.MatMul(vh, v))} {
				values, _ := product.Complex128s()
				k := product.Shape()[0]
				for i, z := range values {
					want := complex128(0)
					if i/k == i%k {
						want = 1
					}
					if cmplx.Abs(z-want) > 1e-10 || math.IsNaN(cmplx.Abs(z)) {
						t.Errorf("nonorthogonal factor: %v want %v", z, want)
					}
				}
			}
			h, tau, e := linalg.QRRaw(a)
			if e != nil {
				t.Fatal(e)
			}
			checkScaled(t, tag+"_raw_h", h)
			checkScaled(t, tag+"_raw_tau", tau)
		})
	}
}

func TestNormIntermediatePrecision(t *testing.T) {
	for _, d := range []np.DType{np.Float16, np.Float32, np.Float64, np.Complex64, np.Complex128} {
		for _, key := range []string{"tiny", "large"} {
			name := "adv_norm_" + d.String() + "_" + key
			t.Run(name, func(t *testing.T) {
				m := refcheck.Require(t)
				a := refcheck.Input(t, name+"_input")
				refcheck.Check(t, name+"_flat", m(linalg.Norm(a)))
				refcheck.Check(t, name+"_rows", m(linalg.Norm(a, linalg.NormOptions{Axes: []int{1}})))
				refcheck.Check(t, name+"_p3", m(linalg.Norm(a, linalg.NormOptions{Axes: []int{1}, Ord: "3"})))
				refcheck.Check(t, name+"_fro", m(linalg.Norm(a, linalg.NormOptions{Axes: []int{0, 1}, Ord: "fro"})))
				refcheck.Check(t, name+"_vector", m(linalg.VectorNorm(a, "", []int{0, 1}, true)))
				refcheck.Check(t, name+"_matrix", m(linalg.MatrixNorm(a, "fro", true)))
			})
		}
	}
}
