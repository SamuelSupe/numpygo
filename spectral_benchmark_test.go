package np_test

import (
	"fmt"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/fft"
	"github.com/SamuelSupe/numpygo/linalg"
)

func BenchmarkSpectralKernels(b *testing.B) {
	must := func(a *np.Array, e error) *np.Array {
		if e != nil {
			b.Fatal(e)
		}
		return a
	}
	run := func(name string, f func() (*np.Array, error)) {
		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				benchmarkResult = must(f())
			}
		})
	}
	matrix := func(m, n int, hermitian bool) *np.Array {
		state := uint64(123)
		next := func() float64 {
			state = state*6364136223846793005 + 1442695040888963407
			return float64(state>>11)*0x1p-53 - .5
		}
		v := make([]complex128, m*n)
		for i := range v {
			v[i] = complex(next(), next())
		}
		if hermitian {
			for i := 0; i < n; i++ {
				v[i*n+i] = complex(real(v[i*n+i]), 0)
				for j := 0; j < i; j++ {
					z := v[i*n+j]
					v[j*n+i] = complex(real(z), -imag(z))
				}
			}
		}
		return must(np.WrapComplex128(v, m, n))
	}
	for _, n := range []int{32, 96, 192} {
		a := matrix(n, n, true)
		run(fmt.Sprintf("eigh_%d", n), func() (*np.Array, error) { s, _, e := linalg.Eigh(a); return s, e })
		run(fmt.Sprintf("eigvalsh_%d", n), func() (*np.Array, error) { return linalg.Eigvalsh(a) })
	}
	for _, sh := range [][2]int{{32, 32}, {96, 64}, {256, 48}} {
		a := matrix(sh[0], sh[1], false)
		run(fmt.Sprintf("svd_%dx%d", sh[0], sh[1]), func() (*np.Array, error) { _, s, _, e := linalg.SVD(a, linalg.SVDOptions{Reduced: true}); return s, e })
		run(fmt.Sprintf("svdvals_%dx%d", sh[0], sh[1]), func() (*np.Array, error) {
			_, s, _, e := linalg.SVD(a, linalg.SVDOptions{ValuesOnly: true})
			return s, e
		})
	}
	v := make([]float64, 128*1024)
	z := make([]complex128, len(v))
	for i := range v {
		v[i] = float64(i%137-68) / 17
		z[i] = complex(v[i], float64(i%97-48)/31)
	}
	a := must(np.WrapFloat64(v, 128, 1024))
	c := must(np.WrapComplex128(z, 128, 1024))
	half := must(c.Slice(np.Slice{}, np.Range(0, 513)))
	run("fft_128x1024", func() (*np.Array, error) { return fft.FFT(c) })
	run("ifft_128x1024", func() (*np.Array, error) { return fft.IFFT(c) })
	run("rfft_128x1024", func() (*np.Array, error) { return fft.RFFT(a) })
	run("irfft_128x1024", func() (*np.Array, error) { return fft.IRFFT(half, fft.Options{N: np.Int(1024)}) })
	prime := must(np.WrapFloat64(v[:4*4093], 4, 4093))
	run("fft_4x4093", func() (*np.Array, error) { return fft.FFT(prime) })
	run("rfft_4x4093", func() (*np.Array, error) { return fft.RFFT(prime) })
	long := must(np.WrapComplex128(z[:65536], 65536))
	run("fft_65536", func() (*np.Array, error) { return fft.FFT(long) })
	for _, dtype := range []np.DType{np.Float32, np.Float64} {
		x := must(a.Astype(dtype))
		y := must(np.Add(x, must(np.Full(dtype, 5, 128, 1024))))
		for _, op := range []struct {
			name string
			f    func(*np.Array, *np.Array) (*np.Array, error)
		}{{"add", np.Add}, {"sub", np.Subtract}, {"mul", np.Multiply}, {"div", np.Divide}} {
			run(fmt.Sprintf("%s_%s_128k", op.name, dtype), func() (*np.Array, error) { return op.f(x, y) })
		}
	}
}
