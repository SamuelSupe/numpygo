package linalg_test

import (
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"github.com/SamuelSupe/numpygo/linalg"
)

func verifyNativeQR(t *testing.T, a, q, r *np.Array) {
	t.Helper()
	m := refcheck.Require(t)
	got := m(linalg.MatMul(q, r))
	x, _ := a.Complex128s()
	g, _ := got.Complex128s()
	for i, v := range x {
		assertComplex(t, g[i], v, q.DType())
	}
	qh := m(m(np.Conjugate(q)).Transpose())
	qq := m(linalg.MatMul(qh, q))
	vals, _ := qq.Complex128s()
	n := q.Shape()[1]
	for i, z := range vals {
		var w complex128
		if i/n == i%n {
			w = 1
		}
		assertComplex(t, z, w, q.DType())
	}
}
func TestDecompQR(t *testing.T) {
	m := refcheck.Require(t)
	in := func(name string) *np.Array { return refcheck.Input(t, "decomp_"+name) }
	for _, d := range []string{"float32", "float64", "complex64", "complex128"} {
		t.Run(d, func(t *testing.T) {
			m := refcheck.Require(t)
			a := refcheck.Input(t, "decomp_qr_input_"+d)
			for _, tc := range []struct {
				name string
				a    *np.Array
			}{
				{"tall", m(a.Slice(np.Reverse()))},
				{"wide", m(m(a.Transpose()).Slice(np.Slice{}, np.Reverse()))},
			} {
				for _, mode := range []string{"reduced", "complete"} {
					q, r, e := linalg.QR(tc.a, linalg.QROptions{Complete: mode == "complete"})
					if e != nil {
						t.Fatal(e)
					}
					tag := "decomp_qr_" + tc.name + "_" + mode + "_" + d
					refcheck.Metadata(t, tag+"_q", q)
					refcheck.Check(t, tag+"_r", r)
					if mode == "reduced" {
						refcheck.Check(t, tag+"_q", q)
					}
					verifyNativeQR(t, tc.a, q, r)
				}
				q, r, e := linalg.QR(tc.a, linalg.QROptions{ROnly: true})
				if e != nil {
					t.Fatal(e)
				}
				if q != nil {
					t.Fatal("R-only QR unexpectedly returned Q")
				}
				refcheck.Check(t, "decomp_qr_"+tc.name+"_r_"+d, r)
			}
			h, tau, e := linalg.QRRaw(a)
			if e != nil {
				t.Fatal(e)
			}
			refcheck.Check(t, "decomp_qr_raw_"+d+"_h", h)
			refcheck.Check(t, "decomp_qr_raw_"+d+"_tau", tau)
			refcheck.CheckExact(t, "decomp_qr_input_"+d, a)
		})
	}
	for _, tc := range []struct {
		name  string
		shape []int
	}{{"empty_rows", []int{0, 3}}, {"empty_cols", []int{5, 0}}} {
		for _, d := range []np.DType{np.Float64, np.Complex128} {
			q, r, e := linalg.QR(m(np.Zeros(d, tc.shape...)), linalg.QROptions{Complete: true})
			if e != nil {
				t.Fatal(e)
			}
			tag := "decomp_qr_" + tc.name + "_" + d.String()
			refcheck.CheckExact(t, tag+"_q", q)
			refcheck.CheckExact(t, tag+"_r", r)
		}
	}
	a := in("qr_rankdef_input")
	q, r, e := linalg.QR(a)
	if e != nil {
		t.Fatal(e)
	}
	refcheck.Metadata(t, "decomp_qr_rankdef_q", q)
	refcheck.Metadata(t, "decomp_qr_rankdef_r", r)
	verifyNativeQR(t, a, q, r)
}

func TestDecompSVD(t *testing.T) {
	for _, d := range []string{"complex64", "complex128"} {
		t.Run(d, func(t *testing.T) {
			m := refcheck.Require(t)
			a := refcheck.Input(t, "decomp_svd_input_"+d)
			for _, tc := range []struct {
				name string
				a    *np.Array
			}{
				{"tall", m(a.Slice(np.Reverse()))},
				{"wide", m(m(a.Transpose()).Slice(np.Slice{}, np.Reverse()))},
			} {
				for _, mode := range []string{"reduced", "full"} {
					u, s, h, e := linalg.SVD(tc.a, linalg.SVDOptions{Reduced: mode == "reduced"})
					if e != nil {
						t.Fatal(e)
					}
					key := "decomp_svd_" + tc.name + "_" + mode + "_" + d
					refcheck.Metadata(t, key+"_u", u)
					refcheck.Metadata(t, key+"_vh", h)
					refcheck.Check(t, key+"_s", s)
					verifyComplexSVD(t, tc.a, u, s, h)
				}
				u, s, h, e := linalg.SVD(tc.a, linalg.SVDOptions{ValuesOnly: true})
				if e != nil {
					t.Fatal(e)
				}
				if u != nil || h != nil {
					t.Fatal("values-only SVD returned vectors")
				}
				refcheck.Check(t, "decomp_svd_"+tc.name+"_values_"+d, s)
			}
			refcheck.CheckExact(t, "decomp_svd_input_"+d, a)
		})
	}
	a := refcheck.Input(t, "decomp_svd_zero_input")
	u, s, h, e := linalg.SVD(a)
	if e != nil {
		t.Fatal(e)
	}
	refcheck.Metadata(t, "decomp_svd_zero_u", u)
	refcheck.Metadata(t, "decomp_svd_zero_vh", h)
	refcheck.Check(t, "decomp_svd_zero_s", s)
	verifyComplexSVD(t, a, u, s, h)
	_, s, _, e = linalg.SVD(a, linalg.SVDOptions{ValuesOnly: true})
	if e != nil {
		t.Fatal(e)
	}
	refcheck.Check(t, "decomp_svd_zero_values", s)
	_, s, _, e = linalg.SVD(refcheck.Input(t, "decomp_svd_graded_input"), linalg.SVDOptions{ValuesOnly: true})
	if e != nil {
		t.Fatal(e)
	}
	checkScaled(t, "decomp_svd_graded_values", s)
}

func TestDecompEigen(t *testing.T) {
	m := refcheck.Require(t)
	in := func(name string) *np.Array { return refcheck.Input(t, "decomp_"+name) }
	for _, d := range []string{"complex64", "complex128"} {
		t.Run(d, func(t *testing.T) {
			m := refcheck.Require(t)
			a := refcheck.Input(t, "decomp_eig_input_"+d)
			v := m(a.Transpose())
			e, q, err := linalg.Eig(v)
			if err != nil {
				t.Fatal(err)
			}
			matchEigenvalues(t, "decomp_eig_values_"+d, e)
			refcheck.Metadata(t, "decomp_eig_vectors_"+d, q)
			verifyEigen(t, v, e, q, false)
			matchEigenvalues(t, "decomp_eig_only_"+d, m(linalg.Eigvals(v)))
			refcheck.CheckExact(t, "decomp_eig_input_"+d, a)
			a = refcheck.Input(t, "decomp_hermitian_input_"+d)
			e, q, err = linalg.Eigh(a)
			if err != nil {
				t.Fatal(err)
			}
			refcheck.Check(t, "decomp_eigh_values_"+d, e)
			refcheck.Metadata(t, "decomp_eigh_vectors_"+d, q)
			verifyEigen(t, a, e, q, true)
			refcheck.Check(t, "decomp_eigvalsh_"+d, m(linalg.Eigvalsh(a)))
			u, s, h, err := linalg.SVD(a, linalg.SVDOptions{Hermitian: true, ValuesOnly: true})
			if err != nil {
				t.Fatal(err)
			}
			if u != nil || h != nil {
				t.Fatal("values-only Hermitian SVD returned vectors")
			}
			refcheck.Check(t, "decomp_svd_hermitian_values_"+d, s)
			a = refcheck.Input(t, "decomp_hermitian_upper_input_"+d)
			refcheck.Check(t, "decomp_hermitian_upper_values_"+d, m(linalg.Eigvalsh(a, true)))
		})
	}
	matchEigenvalues(t, "decomp_real_eig_only", m(linalg.Eigvals(in("real_eig_input"))))
	matchEigenvalues(t, "decomp_real_sym_eig_only", m(linalg.Eigvals(in("real_sym_input"))))
	refcheck.Check(t, "decomp_real_sym_eigvalsh", m(linalg.Eigvalsh(in("real_sym_input"))))
	for _, d := range []string{"float32", "float64", "int64"} {
		a := in("real_eig_dtype_input_" + d)
		e, v, err := linalg.Eig(a)
		if err != nil {
			t.Fatal(err)
		}
		matchEigenvalues(t, "decomp_real_eig_dtype_values_"+d, e)
		refcheck.Metadata(t, "decomp_real_eig_dtype_vectors_"+d, v)
		verifyEigen(t, a, e, v, false)
		matchEigenvalues(t, "decomp_real_eig_dtype_only_"+d, m(linalg.Eigvals(a)))
		refcheck.Check(t, "decomp_real_eig_empty_"+d, m(linalg.Eigvals(m(np.Zeros(a.DType(), 0, 0)))))
	}
	_, e := linalg.Eigvals(in("nonfinite_input"))
	refcheck.Error(t, "decomp_eig_nonfinite", "LinAlgError", e, linalg.ErrConvergence)
}
