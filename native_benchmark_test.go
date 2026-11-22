package np_test

import (
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/fft"
	"github.com/SamuelSupe/numpygo/linalg"
	"github.com/SamuelSupe/numpygo/random"
)

func BenchmarkNativeKernels(b *testing.B) {
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
	const rows, cols = 128, 1024
	v := make([]float64, rows*cols)
	z := make([]complex128, len(v))
	for i := range v {
		v[i] = float64(i%137-68) / 17
		z[i] = complex(v[i], float64(i%97-48)/31)
	}
	a := must(np.WrapFloat64(v, rows, cols))
	c := must(np.WrapComplex128(z, rows, cols))
	at := must(a.Transpose())
	half := must(c.Slice(np.Slice{}, np.Range(0, cols/2+1)))
	run("rfft_1024", func() (*np.Array, error) { return fft.RFFT(a) })
	run("rfft_transposed_1024", func() (*np.Array, error) { return fft.RFFT(at, fft.Options{Axis: np.Int(0)}) })
	run("irfft_1024", func() (*np.Array, error) { return fft.IRFFT(half, fft.Options{N: np.Int(cols)}) })
	run("fft_1024", func() (*np.Array, error) { return fft.FFT(c) })
	run("ifft_1024", func() (*np.Array, error) { return fft.IFFT(c) })
	run("rfft_512", func() (*np.Array, error) { return fft.RFFT(a, fft.Options{N: np.Int(512)}) })
	big := make([]float64, 1024*1024)
	for i := range big {
		big[i] = float64(i%763) / 13
	}
	bigA := must(np.WrapFloat64(big, 1024, 1024))
	bigT := must(bigA.Transpose())
	dst := must(np.EmptyLike(bigT, np.Float64, "K"))
	src := must(np.Full(np.Float64, 1.25, 1024, 1024))
	run("copy_transposed_1m", func() (*np.Array, error) { return bigT.Astype(np.Float64) })
	run("assign_transposed_1m", func() (*np.Array, error) { return dst, dst.Assign(nil, src) })
	run("assign_same_order_1m", func() (*np.Array, error) { return dst, dst.Assign(nil, bigT) })
	run("add_transposed_1m", func() (*np.Array, error) { return np.Add(bigT, np.Scalar(.5)) })
	g := random.NewPCG64(123)
	run("pcg64_int64_small_range", func() (*np.Array, error) { return g.Integers(-1000000, 1000000, 1<<18) })
	run("pcg64_int64_large_range", func() (*np.Array, error) { return g.Integers(-1<<55, 1<<55, 1<<18) })
	run("pcg64_double", func() (*np.Array, error) { return g.Random(1 << 18) })
	run("pcg64_raw", func() (*np.Array, error) { return g.RandomRaw(1 << 18) })
	dx := random.NewPCG64DXSM(123)
	run("pcg64dxsm_int64", func() (*np.Array, error) { return dx.Integers(-1000000, 1000000, 1<<18) })
	p := random.NewPhilox(123)
	run("philox_raw", func() (*np.Array, error) { return p.RandomRaw(1 << 18) })
	n := 192
	matrix := make([]complex128, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			r := float64((min(i, j)*101+max(i, j)*79)%31-15) / 31
			if i == j {
				r += float64(n)
			}
			matrix[i*n+j] = complex(r, float64(i-j)/float64(n*n))
		}
	}
	mat := must(np.WrapComplex128(matrix, n, n))
	rhs := must(np.Full(np.Complex128, complex(1, .25), n, 16))
	run("complex_solve_192x16", func() (*np.Array, error) { return linalg.Solve(mat, rhs) })
	run("complex_cholesky_192", func() (*np.Array, error) { return linalg.Cholesky(mat) })
	run("complex_matmul_192", func() (*np.Array, error) { return linalg.MatMul(mat, mat) })
	mat32 := must(mat.Astype(np.Complex64))
	run("complex64_matmul_192", func() (*np.Array, error) { return linalg.MatMul(mat32, mat32) })
	// Include the crossover cases, where a plan or real embedding costs
	// more relative to the amount of arithmetic.
	short := must(np.WrapFloat64(v[:64], 64))
	one := must(np.WrapFloat64(v[:1024], 1024))
	long := must(np.WrapFloat64(v[:65536], 65536))
	longc := must(np.WrapComplex128(z[:65536], 65536))
	run("rfft_single_64", func() (*np.Array, error) { return fft.RFFT(short) })
	run("rfft_single_1024", func() (*np.Array, error) { return fft.RFFT(one) })
	run("rfft_single_65536", func() (*np.Array, error) { return fft.RFFT(long) })
	run("fft_single_65536", func() (*np.Array, error) { return fft.FFT(longc) })
	small := must(mat.Slice(np.Range(0, 64), np.Range(0, 64)))
	small32 := must(small.Astype(np.Complex64))
	run("complex_matmul_64", func() (*np.Array, error) { return linalg.MatMul(small, small) })
	run("complex64_matmul_64", func() (*np.Array, error) { return linalg.MatMul(small32, small32) })
}
