package fft_test

import (
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/fft"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
)

func TestSignalPipeline(t *testing.T) {
	must := refcheck.Require(t)
	signals := refcheck.Input(t, "fft_input")
	view := must(must(signals.Transpose()).Slice(np.Reverse()))
	for _, norm := range []fft.Norm{fft.Backward, fft.Forward, fft.Ortho} {
		t.Run(string(norm), func(t *testing.T) {
			must := refcheck.Require(t)
			options := fft.Options{Axis: np.Int(0), Norm: norm}
			padded := options
			padded.N = np.Int(7)
			refcheck.Check(t, "fft_"+string(norm), must(fft.FFT(view, padded)))
			spectrum := must(fft.RFFT(view, options))
			refcheck.Check(t, "rfft_"+string(norm), spectrum)
			odd := options
			odd.N = np.Int(5)
			refcheck.Check(t, "irfft_"+string(norm), must(fft.IRFFT(spectrum, odd)))
			refcheck.Check(t, "ifft_"+string(norm), must(fft.IFFT(must(fft.FFT(view, options)), options)))
		})
	}
	refcheck.Check(t, "fft_default", must(fft.FFT(signals)))
	refcheck.Check(t, "fft_input", signals)
}

func TestFourierLengthsAndDTypes(t *testing.T) {
	must := refcheck.Require(t)
	z := must(np.FromComplex128([]complex128{1 + 2i, 3 - 4i, 5 + 1i, -2 + 1i, 9 + 2i}, 5))
	refcheck.Check(t, "fft_cropped_complex", must(fft.FFT(z, fft.Options{N: np.Int(4)})))
	refcheck.Check(t, "ifft_complex", must(fft.IFFT(z)))
	signals := refcheck.Input(t, "fft_input")
	even := must(fft.RFFT(signals, fft.Options{N: np.Int(6), Axis: np.Int(-1)}))
	refcheck.Check(t, "rfft_even", even)
	refcheck.Check(t, "irfft_even_default", must(fft.IRFFT(even)))
	refcheck.Check(t, "irfft_real_spectrum", must(fft.IRFFT(must(np.FromFloat64([]float64{1, 2, 3}, 3)), fft.Options{N: np.Int(5)})))
	one := must(np.FromComplex128([]complex128{3 + 2i}, 1))
	refcheck.Check(t, "irfft_one", must(fft.IRFFT(one, fft.Options{N: np.Int(1)})))
	refcheck.Check(t, "fft_one", must(fft.FFT(one)))
	refcheck.Check(t, "rfft_one", must(fft.RFFT(must(np.FromFloat64([]float64{3}, 1)))))
	refcheck.Check(t, "fft_empty_batch", must(fft.FFT(must(np.Zeros(np.Float64, 0, 4)))))
	refcheck.Check(t, "fft_empty_padded", must(fft.FFT(must(np.Zeros(np.Float64, 2, 0)), fft.Options{N: np.Int(4)})))
	prime := make([]float64, 13)
	for i := range prime {
		prime[i] = float64(i)
	}
	refcheck.Check(t, "rfft_prime", must(fft.RFFT(must(np.WrapFloat64(prime, 13)))))
	refcheck.Check(t, "fft_middle_axis", must(fft.FFT(refcheck.Input(t, "array_base"), fft.Options{Axis: np.Int(1), N: np.Int(5)})))
	refcheck.Check(t, "irfft_padded_spectrum", must(fft.IRFFT(must(np.FromComplex128([]complex128{1 + 2i, 3 - 1i}, 2)), fft.Options{N: np.Int(7)})))
	refcheck.Check(t, "irfft_cropped_spectrum", must(fft.IRFFT(must(np.FromComplex128([]complex128{1 + 2i, 3 - 1i, 4 + 5i}, 3)), fft.Options{N: np.Int(2)})))
}

func TestFourierErrors(t *testing.T) {
	must := refcheck.Require(t)
	signals := refcheck.Input(t, "fft_input")
	_, err := fft.FFT(signals, fft.Options{Axis: np.Int(2)})
	refcheck.Error(t, "fft_axis", "IndexError", err, np.ErrAxis)
	_, err = fft.FFT(np.Scalar(1))
	refcheck.Error(t, "fft_rank0", "IndexError", err, np.ErrAxis)
	_, err = fft.FFT(signals, fft.Options{N: np.Int(0)})
	refcheck.Error(t, "fft_zero_length", "ValueError", err, np.ErrShape)
	_, err = fft.FFT(must(np.Zeros(np.Float64, 0)))
	refcheck.Error(t, "fft_empty_length", "ValueError", err, np.ErrShape)
	_, err = fft.FFT(signals, fft.Options{Norm: "unknown"})
	refcheck.Error(t, "fft_bad_norm", "ValueError", err, fft.ErrNorm)
	point := must(np.FromComplex128([]complex128{1 + 2i}, 1))
	_, err = fft.RFFT(point)
	refcheck.Error(t, "rfft_complex", "TypeError", err, np.ErrDType)
	_, err = fft.IRFFT(point)
	refcheck.Error(t, "irfft_bad_default", "ValueError", err, np.ErrShape)
}
