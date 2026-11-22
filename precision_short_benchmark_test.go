package np_test

import (
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/fft"
)

func BenchmarkPrecisionShortFFT(b *testing.B) {
	const lanes = 65536
	x, y := make([]float32, lanes*2), make([]complex64, lanes*2)
	for i := range x {
		x[i] = float32(i%137-68) / 17
		y[i] = complex(x[i], float32(i%97-48)/31)
	}
	must := func(a *np.Array, e error) *np.Array {
		if e != nil {
			b.Fatal(e)
		}
		return a
	}
	a, c := must(np.WrapSlice(x, lanes, 2)), must(np.WrapSlice(y, lanes, 2))
	dst := must(np.Zeros(np.Complex64, lanes, 2))
	realDst := must(np.Zeros(np.Float32, lanes, 2))
	for _, tc := range []struct {
		name string
		f    func() (*np.Array, error)
	}{
		{"rfft2_out", func() (*np.Array, error) { return fft.RFFT(a, fft.Options{Out: dst}) }},
		{"fft2_out", func() (*np.Array, error) { return fft.FFT(c, fft.Options{Out: dst}) }},
		{"irfft2_out", func() (*np.Array, error) { return fft.IRFFT(c, fft.Options{Out: realDst, Norm: fft.Forward}) }},
	} {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				benchmarkResult = must(tc.f())
			}
		})
	}
}
