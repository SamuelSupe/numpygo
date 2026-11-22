package np_test

import (
	"fmt"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/fft"
)

func BenchmarkVectorKernels(b *testing.B) {
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
	const size = 128 * 1024
	x, y := make([]float64, size), make([]float64, size)
	c, z := make([]complex128, size), make([]complex128, size)
	for i := range x {
		x[i] = float64(i%137-68) / 17
		y[i] = float64(i%149-74) / 23
		c[i] = complex(x[i], float64(i%97-48)/31)
		z[i] = complex(y[i]+.51, float64(i%101-50)/27+.31)
	}
	a64, b64 := must(np.WrapFloat64(x, size)), must(np.WrapFloat64(y, size))
	c128, d128 := must(np.WrapComplex128(c, size)), must(np.WrapComplex128(z, size))
	for _, d := range []np.DType{np.Float32, np.Float64, np.Complex64, np.Complex128} {
		a, q := a64, b64
		names := []string{"exp", "minimum", "maximum"}
		if d.IsComplex() {
			a, q = c128, d128
			names = []string{"add", "subtract", "multiply", "divide"}
		}
		if a.DType() != d {
			a, q = must(a.Astype(d)), must(q.Astype(d))
		}
		out := must(np.Zeros(d, size))
		for _, name := range names {
			u := lookup(name)
			args := []*np.Array{a}
			if name != "exp" {
				args = append(args, q)
			}
			run(name+"_"+d.String()+"_out", func() (*np.Array, error) { return u.CallWithOptions(np.UFuncOptions{Out: out}, args...) })
			run(name+"_"+d.String()+"_alloc", func() (*np.Array, error) { return u.Call(args...) })
		}
	}
	for _, n := range []int{1000, 1536, 150, 1500, 729, 4093, 1024} {
		rows := 128
		if n == 4093 {
			rows = 4
		}
		ax, cx := make([]float64, rows*n), make([]complex128, rows*n)
		for i := range ax {
			ax[i] = float64(i%137-68) / 17
			cx[i] = complex(ax[i], float64(i%97-48)/31)
		}
		aa := must(np.WrapFloat64(ax, rows, n))
		cc := must(np.WrapComplex128(cx, rows, n))
		for _, d := range []np.DType{np.Float32, np.Float64} {
			cd := np.Complex64
			if d == np.Float64 {
				cd = np.Complex128
			}
			a, c := aa, cc
			if d == np.Float32 {
				a, c = must(a.Astype(d)), must(c.Astype(cd))
			}
			tag := fmt.Sprintf("%s_%d", d, n)
			co := must(np.Zeros(cd, rows, n))
			ho := must(np.Zeros(cd, rows, n/2+1))
			ro := must(np.Zeros(d, rows, n))
			half := must(must(c.Slice(np.Slice{}, np.Range(0, n/2+1))).Copy())
			run("fft_"+tag, func() (*np.Array, error) { return fft.FFT(c, fft.Options{Out: co}) })
			run("rfft_"+tag, func() (*np.Array, error) { return fft.RFFT(a, fft.Options{Out: ho}) })
			run("irfft_"+tag, func() (*np.Array, error) { return fft.IRFFT(half, fft.Options{N: np.Int(n), Out: ro}) })
		}
	}
}
