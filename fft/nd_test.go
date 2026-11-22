package fft_test

import (
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/fft"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"testing"
)

func TestMultiAxisAndHermitianFFT(t *testing.T) {
	must := refcheck.Require(t)
	a := refcheck.Input(t, "fft_nd_input")
	f := must(fft.FFTN(a))
	refcheck.Check(t, "fft_nd", f)
	refcheck.Check(t, "ifft_nd", must(fft.IFFTN(f)))
	refcheck.Check(t, "fft_2", must(fft.FFT2(a, fft.NDOptions{Shape: []int{4, 7}, Norm: fft.Ortho})))
	f2 := must(fft.FFT2(a, fft.NDOptions{Norm: fft.Ortho}))
	refcheck.Check(t, "ifft_2", must(fft.IFFT2(f2, fft.NDOptions{Norm: fft.Ortho})))
	r := must(fft.RFFTN(a))
	refcheck.Check(t, "rfft_nd", r)
	refcheck.Check(t, "irfft_nd", must(fft.IRFFTN(r, fft.NDOptions{Shape: []int{2, 3, 5}, Axes: []int{0, 1, 2}})))
	r2 := must(fft.RFFT2(a, fft.NDOptions{Axes: []int{0, 2}, Norm: fft.Forward}))
	refcheck.Check(t, "rfft_2", r2)
	refcheck.Check(t, "irfft_2", must(fft.IRFFT2(r2, fft.NDOptions{Shape: []int{2, 5}, Axes: []int{0, 2}, Norm: fft.Forward})))
	refcheck.Check(t, "fft_repeated_axes", must(fft.FFTN(a, fft.NDOptions{Shape: []int{4, 7}, Axes: []int{1, 1}})))
	h := must(np.FromComplex128([]complex128{1, 2 + 3i, 4 - 2i}, 3))
	power := must(fft.HFFT(h, fft.Options{N: np.Int(5)}))
	refcheck.Check(t, "fft_hfft", power)
	refcheck.Check(t, "fft_ihfft", must(fft.IHFFT(power, fft.Options{N: np.Int(5)})))
	refcheck.Check(t, "fft_fftfreq_odd", must(fft.FFTFreq(5, .25)))
	refcheck.Check(t, "fft_fftfreq_even", must(fft.FFTFreq(6, .25)))
	refcheck.Check(t, "fft_rfftfreq", must(fft.RFFTFreq(5, .25)))
	refcheck.Check(t, "fft_shift", must(fft.FFTShift(a, 1, 2)))
	refcheck.Check(t, "fft_ishift", must(fft.IFFTShift(a, 1, 2)))
}
