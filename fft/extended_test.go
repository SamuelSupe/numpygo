package fft

import (
	"fmt"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/extended"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
)

func TestExtendedFFTNumPy(t *testing.T) {
	m := refcheck.Require(t)
	a := refcheck.ExtendedInput(t, "exfft_real_input", np.Float128)
	z := refcheck.ExtendedInput(t, "exfft_complex_input", np.Complex256)
	r := m(a.Slice(np.Reverse(), np.Slice{}, np.Reverse()))
	c := m(z.Slice(np.Reverse(), np.Slice{}, np.Reverse()))
	for _, tc := range []struct {
		name string
		fn   func(*np.Array, ...Options) (*np.Array, error)
		src  *np.Array
		d    np.DType
	}{{"fft", FFT, c, np.Complex256}, {"ifft", IFFT, c, np.Complex256}, {"rfft", RFFT, r, np.Complex256},
		{"irfft", IRFFT, c, np.Float128}, {"hfft", HFFT, c, np.Float128}, {"ihfft", IHFFT, r, np.Complex256}} {
		for _, norm := range []Norm{Backward, Forward, Ortho} {
			for _, nn := range [][2]int{{-1, 7}, {-1, 8}, {-1, 17}, {1, 5}, {0, 1}, {0, 4}} {
				axis, n := nn[0], nn[1]
				name := fmt.Sprintf("exfft_%s_%s_%d_%d", tc.name, norm, axis+2, n)
				t.Run(name, func(t *testing.T) {
					m := refcheck.Require(t)
					v := m(tc.fn(tc.src, Options{N: &n, Axis: &axis, Norm: norm}))
					refcheck.CheckExtendedFFT(t, name, v, tc.d, 96)
				})
			}
		}
	}
	for _, tc := range []struct {
		name string
		fn   func(*np.Array, ...NDOptions) (*np.Array, error)
		src  *np.Array
		d    np.DType
	}{{"fftn", FFTN, z, np.Complex256}, {"ifftn", IFFTN, z, np.Complex256}, {"rfftn", RFFTN, a, np.Complex256}, {"irfftn", IRFFTN, z, np.Float128},
		{"fft2", FFT2, z, np.Complex256}, {"ifft2", IFFT2, z, np.Complex256}, {"rfft2", RFFT2, a, np.Complex256}, {"irfft2", IRFFT2, z, np.Float128}} {
		t.Run(tc.name, func(t *testing.T) {
			v := m(tc.fn(m(tc.src.Transpose(2, 0, 1)), NDOptions{Shape: []int{5, 8}, Axes: []int{0, 2}, Norm: Ortho}))
			refcheck.CheckExtendedFFT(t, "exfft_"+tc.name, v, tc.d, 128)
		})
	}
	for _, tc := range []struct {
		name string
		n    int
	}{{"small", 31}, {"large", 31}, {"prime", 127}} {
		t.Run(tc.name, func(t *testing.T) {
			a := refcheck.ExtendedInput(t, "exfft_"+tc.name+"_input", np.Float128)
			v := m(FFT(a, Options{N: &tc.n}))
			refcheck.CheckExtendedFFT(t, "exfft_"+tc.name, v, np.Complex256, 256)
		})
	}
	precision := refcheck.ExtendedInput(t, "exfft_precision_input", np.Float128)
	// The DC component is 2^-63; a binary64 FFT would produce zero.
	refcheck.CheckExtended(t, "exfft_precision", m(FFT(precision)), np.Complex256, 0)
	out := m(m(np.Zeros(np.Complex256, 2, 3, 14)).Slice(np.Slice{}, np.Slice{}, np.Strided(0, 14, 2)))
	v := m(FFT(z, Options{Out: out, N: np.Int(7)}))
	if v != out {
		t.Fatal("out alias lost")
	}
	refcheck.CheckExtendedFFT(t, "exfft_out", v, np.Complex256, 96)
	source := m(np.FromSlice([]float32{1.25, 2.5, -3, 4, 5}, 5))
	for _, tc := range []struct {
		name  string
		fn    func(*np.Array, ...Options) (*np.Array, error)
		dtype np.DType
		size  int
	}{
		{"fft", FFT, np.Complex256, 7}, {"ifft", IFFT, np.Complex256, 7}, {"rfft", RFFT, np.Complex256, 4}, {"irfft", IRFFT, np.Float128, 7},
	} {
		dest := m(np.Zeros(tc.dtype, tc.size))
		// Out does not itself select the extended loop for native inputs.
		refcheck.CheckExtendedFFT(t, "exfft_native_out_"+tc.name, m(tc.fn(source, Options{Out: dest, N: np.Int(7)})), tc.dtype, 65536)
	}
	_, e := RFFT(z)
	refcheck.Error(t, "exfft_bad_real", "TypeError", e, np.ErrDType)
	_, e = FFT(a, Options{N: np.Int(0)})
	refcheck.Error(t, "exfft_bad_length", "ValueError", e, np.ErrShape)
	refcheck.CheckExtended(t, "exfft_shift", m(FFTShift(z, 0, 2)), np.Complex256, 0)
	refcheck.CheckExtended(t, "exfft_ishift", m(IFFTShift(z, 0, 2)), np.Complex256, 0)
	for _, tc := range []struct {
		name string
		fn   func(int, np.ExtendedFloat) (*np.Array, error)
	}{{"fftfreq", FFTFreqExtended}, {"rfftfreq", RFFTFreqExtended}} {
		for _, n := range []int{1, 6, 7} {
			for i, d := range []string{"1.0000000000000000001", "-1e4000", "1e-4000", "0", "inf"} {
				name := fmt.Sprintf("exfft_%s_%d_%d", tc.name, n, i)
				t.Run(name, func(t *testing.T) {
					refcheck.CheckExtended(t, name, refcheck.Require(t)(tc.fn(n, extended.MustParse(d))), np.Float128, 1)
				})
			}
		}
	}
}
