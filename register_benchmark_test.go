package np_test

import (
	"fmt"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/fft"
)

func BenchmarkRegisterKernels(b *testing.B) {
	must := func(a *np.Array, e error) *np.Array {
		if e != nil {
			b.Fatal(e)
		}
		return a
	}
	lookup := func(name string) *np.UFunc {
		u, e := np.LookupUFunc(name)
		if e != nil {
			b.Fatal(e)
		}
		return u
	}
	run := func(name string, f func() (*np.Array, error)) {
		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				benchmarkResult = must(f())
			}
		})
	}
	for _, n := range []int{32, 257, 128 * 1024} {
		x, y := make([]complex64, n), make([]complex64, n)
		for i := range x {
			x[i] = complex(float32(float64(i%137-68)/17), float32(float64(i%97-48)/31))
			y[i] = complex(float32(float64(i%149-74)/23+.51), float32(float64(i%101-50)/27+.31))
		}
		a, c := must(np.WrapSlice(x, n)), must(np.WrapSlice(y, n))
		out := must(np.Zeros(np.Complex64, n))
		for _, name := range []string{"multiply", "divide"} {
			u := lookup(name)
			run(fmt.Sprintf("%s_c64_%d_out", name, n), func() (*np.Array, error) { return u.CallWithOptions(np.UFuncOptions{Out: out}, a, c) })
			run(fmt.Sprintf("%s_c64_%d_alloc", name, n), func() (*np.Array, error) { return u.Call(a, c) })
		}
	}
	fftCases := []struct {
		n, rows       int
		real, complex bool
		both          bool
	}{
		{81, 128, true, false, true}, {125, 128, true, false, true}, {729, 128, true, true, true},
		{1125, 128, true, false, true}, {2025, 64, true, false, true}, {3125, 32, true, true, true},
		{2025, 1, true, false, false}, {3125, 1, true, true, false},
		{150, 128, true, true, false}, {1000, 128, true, true, true}, {1536, 128, true, true, true},
		{1500, 128, true, true, false}, {1536, 1, false, true, true},
	}
	for _, tc := range fftCases {
		n, rows := tc.n, tc.rows
		ax, cx := make([]float64, rows*n), make([]complex128, rows*n)
		for i := range ax {
			ax[i] = float64(i%137-68) / 17
			cx[i] = complex(ax[i], float64(i%97-48)/31)
		}
		aa := must(np.WrapFloat64(ax, rows, n))
		cc := must(np.WrapComplex128(cx, rows, n))
		dtypes := []np.DType{np.Float32}
		if tc.both {
			dtypes = append(dtypes, np.Float64)
		}
		for _, d := range dtypes {
			cd := np.Complex64
			if d == np.Float64 {
				cd = np.Complex128
			}
			a, c := aa, cc
			if d == np.Float32 {
				a, c = must(a.Astype(d)), must(c.Astype(cd))
			}
			tag := fmt.Sprintf("%s_%dx%d", d, rows, n)
			if tc.complex {
				co := must(np.Zeros(cd, rows, n))
				run("fft_"+tag, func() (*np.Array, error) { return fft.FFT(c, fft.Options{Out: co}) })
			}
			if tc.real {
				ho := must(np.Zeros(cd, rows, n/2+1))
				ro := must(np.Zeros(d, rows, n))
				half := must(must(c.Slice(np.Slice{}, np.Range(0, n/2+1))).Copy())
				run("rfft_"+tag, func() (*np.Array, error) { return fft.RFFT(a, fft.Options{Out: ho}) })
				run("irfft_"+tag, func() (*np.Array, error) { return fft.IRFFT(half, fft.Options{N: np.Int(n), Out: ro}) })
			}
		}
	}
}
