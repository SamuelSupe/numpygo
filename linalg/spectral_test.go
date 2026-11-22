package linalg_test

import (
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"github.com/SamuelSupe/numpygo/linalg"
)

func TestSpectralHermitian(t *testing.T) {
	m := refcheck.Require(t)
	in := func(k string) *np.Array { return refcheck.Input(t, "spectral_"+k) }
	for _, d := range []string{"complex64", "complex128"} {
		t.Run(d, func(t *testing.T) {
			m := refcheck.Require(t)
			a := in("herm_" + d)
			x := m(a.Slice(np.Slice{}, np.Reverse(), np.Reverse()))
			w, v, e := linalg.Eigh(x)
			if e != nil {
				t.Fatal(e)
			}
			refcheck.Check(t, "spectral_herm_w_"+d, w)
			refcheck.Metadata(t, "spectral_herm_v_"+d, v)
			verifyEigen(t, x, w, v, true)
			refcheck.Check(t, "spectral_herm_vals_"+d, m(linalg.Eigvalsh(x)))
			u, s, vh, e := linalg.SVD(x, linalg.SVDOptions{Hermitian: true})
			if e != nil {
				t.Fatal(e)
			}
			refcheck.Metadata(t, "spectral_herm_svd_u_"+d, u)
			refcheck.Check(t, "spectral_herm_svd_s_"+d, s)
			refcheck.Metadata(t, "spectral_herm_svd_vh_"+d, vh)
			verifyComplexSVD(t, x, u, s, vh)
			refcheck.CheckExact(t, "spectral_herm_"+d, a)
			bad := in("upper_" + d)
			w, v, e = linalg.Eigh(bad, true)
			if e != nil {
				t.Fatal(e)
			}
			refcheck.Check(t, "spectral_upper_w_"+d, w)
			refcheck.Metadata(t, "spectral_upper_v_"+d, v)
			verifyEigen(t, in("upper_clean_"+d), w, v, true)
			refcheck.Check(t, "spectral_upper_vals_"+d, m(linalg.Eigvalsh(bad, true)))
			refcheck.CheckExact(t, "spectral_upper_"+d, bad)
		})
	}
	for _, scale := range []string{"small", "large", "dynamic"} {
		a := in("herm_" + scale)
		checkScaled(t, "spectral_herm_"+scale+"_vals", m(linalg.Eigvalsh(a)))
		w, _, e := linalg.Eigh(a)
		if e != nil {
			t.Fatal(e)
		}
		checkScaled(t, "spectral_herm_"+scale+"_vals", w)
	}
	for _, tag := range []string{"zero", "diag"} {
		a := in("herm_" + tag)
		w, v, e := linalg.Eigh(a)
		if e != nil {
			t.Fatal(e)
		}
		refcheck.CheckExact(t, "spectral_herm_"+tag+"_vals", w)
		refcheck.CheckExact(t, "spectral_herm_"+tag+"_vecs", v)
		refcheck.CheckExact(t, "spectral_herm_"+tag+"_vals", m(linalg.Eigvalsh(a)))
	}
}

func TestSpectralSVD(t *testing.T) {
	m := refcheck.Require(t)
	in := func(k string) *np.Array { return refcheck.Input(t, "spectral_"+k) }
	for _, d := range []string{"complex64", "complex128"} {
		t.Run(d, func(t *testing.T) {
			m := refcheck.Require(t)
			a := in("svd_" + d)
			for _, tc := range []struct {
				name string
				a    *np.Array
			}{
				{"tall", m(a.Slice(np.Reverse()))},
				{"wide", m(m(a.Transpose()).Slice(np.Slice{}, np.Reverse()))},
			} {
				k := "spectral_svd_" + tc.name + "_" + d
				u, s, vh, e := linalg.SVD(tc.a, linalg.SVDOptions{Reduced: true})
				if e != nil {
					t.Fatal(e)
				}
				refcheck.Metadata(t, k+"_u", u)
				refcheck.Metadata(t, k+"_vh", vh)
				refcheck.Check(t, k+"_s", s)
				verifyComplexSVD(t, tc.a, u, s, vh)
				refcheck.Check(t, k+"_vals", m(linalg.SVDVals(tc.a)))
			}
			refcheck.CheckExact(t, "spectral_svd_"+d, a)
		})
	}
	for _, tag := range []string{"small", "large", "graded"} {
		a := in("svd_" + tag)
		checkScaled(t, "spectral_svd_"+tag+"_vals", m(linalg.SVDVals(a)))
		_, s, _, e := linalg.SVD(a, linalg.SVDOptions{Reduced: true})
		if e != nil {
			t.Fatal(e)
		}
		checkScaled(t, "spectral_svd_"+tag+"_vals", s)
	}
}
