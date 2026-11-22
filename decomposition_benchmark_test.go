package np_test

import (
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/fft"
	"github.com/SamuelSupe/numpygo/linalg"
)

func BenchmarkDecompositionKernels(b *testing.B) {
	must := func(a *np.Array, e error) *np.Array {
		if e != nil {
			b.Fatal(e)
		}
		return a
	}
	run := func(name string, fn func() (*np.Array, error)) {
		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				benchmarkResult = must(fn())
			}
		})
	}
	matrix := func(m, n int) (*np.Array, *np.Array) {
		state := uint64(123)
		next := func() float64 {
			state = state*6364136223846793005 + 1442695040888963407
			return float64(state>>11)*0x1p-53 - .5
		}
		re, co := make([]float64, m*n), make([]complex128, m*n)
		for i := range re {
			re[i] = next()
			co[i] = complex(re[i], next())
		}
		return must(np.WrapFloat64(re, m, n)), must(np.WrapComplex128(co, m, n))
	}
	realQR, complexQR := matrix(512, 64)
	qr := func(a *np.Array, opt linalg.QROptions) (*np.Array, error) {
		q, r, e := linalg.QR(a, opt)
		if opt.ROnly {
			return r, e
		}
		return q, e
	}
	run("qr_real_tall", func() (*np.Array, error) { return qr(realQR, linalg.QROptions{}) })
	run("qr_real_r", func() (*np.Array, error) { return qr(realQR, linalg.QROptions{ROnly: true}) })
	run("qr_complex_tall", func() (*np.Array, error) { return qr(complexQR, linalg.QROptions{}) })
	run("qr_complex_r", func() (*np.Array, error) { return qr(complexQR, linalg.QROptions{ROnly: true}) })
	_, svdA := matrix(96, 64)
	svd := func(a *np.Array, opt linalg.SVDOptions) (*np.Array, error) {
		_, s, _, e := linalg.SVD(a, opt)
		return s, e
	}
	run("svd_complex_reduced", func() (*np.Array, error) { return svd(svdA, linalg.SVDOptions{Reduced: true}) })
	run("svd_complex_values", func() (*np.Array, error) { return svd(svdA, linalg.SVDOptions{ValuesOnly: true}) })
	run("svd_complex_full", func() (*np.Array, error) { return svd(svdA, linalg.SVDOptions{}) })
	realEig, complexEig := matrix(96, 96)
	run("eig_complex", func() (*np.Array, error) { e, _, err := linalg.Eig(complexEig); return e, err })
	run("eigvals_complex", func() (*np.Array, error) { return linalg.Eigvals(complexEig) })
	run("eigvals_real", func() (*np.Array, error) { return linalg.Eigvals(realEig) })
	cv, _ := np.Values[complex128](complexEig)
	rv, _ := np.Values[float64](realEig)
	for i := 0; i < 96; i++ {
		cv[i*96+i] = complex(real(cv[i*96+i]), 0)
		for j := 0; j < i; j++ {
			cv[j*96+i] = complex(real(cv[i*96+j]), -imag(cv[i*96+j]))
			rv[j*96+i] = rv[i*96+j]
		}
	}
	herm := must(np.WrapComplex128(cv, 96, 96))
	sym := must(np.WrapFloat64(rv, 96, 96))
	run("eigh_complex", func() (*np.Array, error) { e, _, err := linalg.Eigh(herm); return e, err })
	run("eigvalsh_complex", func() (*np.Array, error) { return linalg.Eigvalsh(herm) })
	run("eigvalsh_real", func() (*np.Array, error) { return linalg.Eigvalsh(sym) })
	run("svd_hermitian_values", func() (*np.Array, error) { return svd(herm, linalg.SVDOptions{Hermitian: true, ValuesOnly: true}) })
	v := make([]float64, 1<<20)
	for i := range v {
		v[i] = float64(i%137-68) / 17
	}
	a := must(np.WrapFloat64(v, 1024, 1024))
	t := must(a.Transpose())
	run("sum_all_1m", func() (*np.Array, error) { return np.Sum(a) })
	run("sum_axis0_1m", func() (*np.Array, error) { return np.Sum(a, np.SumOptions{Axes: []int{0}}) })
	run("sum_transposed_axis1_1m", func() (*np.Array, error) { return np.Sum(t, np.SumOptions{Axes: []int{1}}) })
	run("mean_axis1_1m", func() (*np.Array, error) { return np.Mean(a, np.SumOptions{Axes: []int{1}}) })
	prime := must(np.WrapFloat64(v[:4*4093], 4, 4093))
	run("fft_prime4093", func() (*np.Array, error) { return fft.FFT(prime) })
	run("rfft_prime4093", func() (*np.Array, error) { return fft.RFFT(prime) })
}
