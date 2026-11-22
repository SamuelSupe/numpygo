package np_test

import (
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/fft"
	"github.com/SamuelSupe/numpygo/linalg"
	"github.com/SamuelSupe/numpygo/random"
)

func BenchmarkHotPaths(b *testing.B) {
	must := func(a *np.Array, err error) *np.Array {
		if err != nil {
			b.Fatal(err)
		}
		return a
	}
	run := func(name string, fn func() (*np.Array, error)) {
		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				benchmarkResult = must(fn())
			}
		})
	}
	n := 64
	matrix := make([]float64, n*n)
	cmat := make([]complex128, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			v := float64((min(i, j)*101+max(i, j)*79)%31-15) / 31
			if i == j {
				v += float64(n)
			}
			matrix[i*n+j] = v
			cmat[i*n+j] = complex(v, float64(i-j)/float64(n*n))
		}
	}
	a := must(np.WrapFloat64(matrix, n, n))
	c := must(np.WrapComplex128(cmat, n, n))
	rhs := must(np.Full(np.Float64, 1., 64, n, 1))
	crhs := must(np.Full(np.Complex128, complex(1., .5), 64, n, 1))
	run("solve_broadcast_f64", func() (*np.Array, error) { return linalg.Solve(a, rhs) })
	run("solve_broadcast_c128", func() (*np.Array, error) { return linalg.Solve(c, crhs) })
	n = 192
	symmetric := make([]float64, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			v := float64((min(i, j)*101+max(i, j)*79)%31-15) / 31
			if i == j {
				v += float64(n)
			}
			symmetric[i*n+j] = v
		}
	}
	mat := must(np.WrapFloat64(symmetric, n, n))
	run("cholesky_f64", func() (*np.Array, error) { return linalg.Cholesky(mat) })
	run("det_f64", func() (*np.Array, error) { return linalg.Det(mat) })
	signal := make([]float64, 128*1024)
	for i := range signal {
		signal[i] = float64(i%137-68) / 17
	}
	x := must(np.WrapFloat64(signal, 128, 1024))
	xt := must(x.Transpose())
	run("rfft_dense", func() (*np.Array, error) { return fft.RFFT(x) })
	run("rfft_strided", func() (*np.Array, error) { return fft.RFFT(xt, fft.Options{Axis: np.Int(0)}) })
	prime := must(np.WrapFloat64(signal[:4093*4], 4, 4093))
	run("fft_prime4093", func() (*np.Array, error) { return fft.FFT(prime) })
	run("rfft_prime4093", func() (*np.Array, error) { return fft.RFFT(prime) })
	other := must(np.Full(np.Float64, 1.25, 128, 1))
	run("add_transposed", func() (*np.Array, error) { return np.Add(xt, np.Scalar(.5)) })
	run("mul_broadcast", func() (*np.Array, error) { return np.Multiply(x, other) })
	dst := must(np.EmptyLike(xt, np.Float64, "K"))
	src := must(np.Full(np.Float64, 1.25, 1024, 128))
	run("assign_strided", func() (*np.Array, error) { return dst, dst.Assign(nil, src) })
	for _, tc := range []struct {
		name string
		g    *random.Generator
	}{
		{"pcg64", random.NewPCG64(123)}, {"mt19937", random.NewMT19937(123)},
		{"philox", random.NewPhilox(123)}, {"sfc64", random.NewSFC64(123)},
	} {
		run("raw_"+tc.name, func() (*np.Array, error) { return tc.g.RandomRaw(1 << 18) })
	}
	g := random.NewPCG64(123)
	run("integers_i8", func() (*np.Array, error) { return g.IntegersDType(-50, 50, np.Int8, false, 1<<18) })
	run("integers_i64", func() (*np.Array, error) { return g.IntegersDType(-1000000, 1000000, np.Int64, false, 1<<18) })
	mu := must(np.WrapFloat64(signal[:1024], 1, 1024))
	sigma := must(np.Full(np.Float64, 1., 128, 1))
	run("normal_broadcast", func() (*np.Array, error) { return g.Sample("normal", nil, mu, sigma) })
	cv := make([]complex128, 4*4093)
	for i := range cv {
		cv[i] = complex(signal[i], signal[i+17])
	}
	spectrum := must(np.WrapComplex128(cv, 4, 4093))
	half := must(spectrum.Slice(np.Slice{}, np.Range(0, 2047)))
	run("ifft_prime4093", func() (*np.Array, error) { return fft.IFFT(spectrum) })
	run("irfft_prime4093", func() (*np.Array, error) { return fft.IRFFT(half, fft.Options{N: np.Int(4093)}) })
	dx := random.NewPCG64DXSM(123)
	run("raw_pcg64dxsm", func() (*np.Array, error) { return dx.RandomRaw(1 << 18) })
	legacy := random.NewRandomState(123)
	run("legacy_i16", func() (*np.Array, error) { return legacy.Randint(-3000, 3000, np.Int16, 1<<18) })
	lo := make([]int16, 128)
	for i := range lo {
		lo[i] = int16(-i - 10)
	}
	lows := must(np.WrapSlice(lo, 128, 1))
	hi := np.ScalarOf(int16(100))
	run("integers_array", func() (*np.Array, error) {
		return g.IntegersArray(lows, hi, random.IntegerOptions{Size: []int{128, 1024}, DType: np.Int16})
	})
	run("astype_strided", func() (*np.Array, error) { return xt.Astype(np.Float64) })
}
