package np_test

import (
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/fft"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
)

func TestSpectralSIMD(t *testing.T) {
	m := refcheck.Require(t)
	for _, d := range []string{"float32", "float64"} {
		a, b := refcheck.Input(t, "spectral_simd_a_"+d), refcheck.Input(t, "spectral_simd_b_"+d)
		x, y := m(a.Slice(np.Range(1, 70))), m(b.Slice(np.Range(2, 71)))
		for _, op := range []struct {
			name string
			fn   func(*np.Array, *np.Array) (*np.Array, error)
		}{
			{"add", np.Add}, {"sub", np.Subtract}, {"mul", np.Multiply}, {"div", np.Divide},
		} {
			refcheck.CheckExact(t, "spectral_simd_"+op.name+"_"+d, m(op.fn(x, y)))
		}
		refcheck.CheckExact(t, "spectral_simd_a_"+d, a)
		refcheck.CheckExact(t, "spectral_simd_b_"+d, b)
	}
}

func TestSpectralFFT(t *testing.T) {
	m := refcheck.Require(t)
	a := refcheck.Input(t, "spectral_fft_input")
	x := m(a.Slice(np.Slice{}, np.Reverse()))
	refcheck.Check(t, "spectral_fft", m(fft.FFT(x, fft.Options{Norm: fft.Ortho})))
	refcheck.Check(t, "spectral_ifft", m(fft.IFFT(x, fft.Options{Norm: fft.Forward})))
	axis0 := m(m(a.Transpose()).Copy())
	refcheck.Check(t, "spectral_fft_axis0", m(fft.FFT(axis0, fft.Options{Axis: np.Int(0)})))
	prime := m(a.Slice(np.Slice{}, np.Range(0, 509)))
	refcheck.Check(t, "spectral_prime_fft", m(fft.FFT(prime)))
	refcheck.Check(t, "spectral_prime_ifft", m(fft.IFFT(prime)))
	refcheck.CheckExact(t, "spectral_fft_input", a)
	refcheck.CheckExact(t, "spectral_fft_input", m(axis0.Transpose()))
	b := refcheck.Input(t, "spectral_rfft_input")
	z := m(fft.RFFT(b, fft.Options{N: np.Int(509), Norm: fft.Ortho}))
	refcheck.Check(t, "spectral_rfft", z)
	refcheck.Check(t, "spectral_irfft", m(fft.IRFFT(refcheck.Input(t, "spectral_rfft"), fft.Options{N: np.Int(509), Norm: fft.Ortho})))
}
