package fft_test

import (
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/fft"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"testing"
)

func TestFFTNumericDTypes(t *testing.T) {
	must := refcheck.Require(t)
	x := must(np.FromFloat64([]float64{1, 2, -3, 4, 5}, 5))
	for _, d := range []np.DType{np.Int16, np.Float16, np.Float32} {
		a := must(x.Astype(d))
		refcheck.Check(t, "fft_dtype_"+d.String(), must(fft.FFT(a)))
		r := must(fft.RFFT(a))
		refcheck.Check(t, "rfft_dtype_"+d.String(), r)
		refcheck.Check(t, "irfft_dtype_"+d.String(), must(fft.IRFFT(r, fft.Options{N: np.Int(5)})))
	}
	refcheck.Check(t, "fft_dtype_complex64", must(fft.FFT(must(np.FromSlice([]complex64{1 + 2i, 3 - 1i, 2}, 3)))))
}
