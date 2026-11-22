package np_test

import (
	"fmt"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/fft"
)

func BenchmarkArrayFFTKernels(b *testing.B) {
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
	v := make([]float64, 128*1024)
	c := make([]complex128, len(v))
	for i := range v {
		v[i] = float64(i%137-68) / 17
		c[i] = complex(v[i], float64(i%97-48)/31)
	}
	a := must(np.WrapFloat64(v, 128, 1024))
	z := must(np.WrapComplex128(c, 128, 1024))
	a32 := must(a.Astype(np.Float32))
	z32 := must(z.Astype(np.Complex64))
	for _, tc := range []struct {
		name string
		a, z *np.Array
	}{{"64", a, z}, {"32", a32, z32}} {
		run("fft_batch_"+tc.name, func() (*np.Array, error) { return fft.FFT(tc.z) })
		run("ifft_batch_"+tc.name, func() (*np.Array, error) { return fft.IFFT(tc.z) })
		run("rfft_batch_"+tc.name, func() (*np.Array, error) { return fft.RFFT(tc.a) })
		half := must(tc.z.Slice(np.Slice{}, np.Range(0, 513)))
		run("irfft_batch_"+tc.name, func() (*np.Array, error) { return fft.IRFFT(half, fft.Options{N: np.Int(1024)}) })
	}
	for _, n := range []int{4096, 65536} {
		x := must(np.WrapFloat64(v[:n], n))
		run(fmt.Sprintf("rfft_single_%d", n), func() (*np.Array, error) { return fft.RFFT(x) })
		s := must(np.WrapComplex128(c[:n/2+1], n/2+1))
		run(fmt.Sprintf("irfft_single_%d", n), func() (*np.Array, error) { return fft.IRFFT(s, fft.Options{N: np.Int(n)}) })
	}
	long := must(np.WrapComplex128(c[:65536], 65536))
	run("fft_single_65536", func() (*np.Array, error) { return fft.FFT(long) })
	prime := must(np.WrapFloat64(v[:4093], 4093))
	run("rfft_prime_single_4093", func() (*np.Array, error) { return fft.RFFT(prime) })
	batchPrime := must(np.WrapFloat64(v[:4*4093], 4, 4093))
	run("fft_prime_4x4093", func() (*np.Array, error) { return fft.FFT(batchPrime) })
	run("rfft_prime_4x4093", func() (*np.Array, error) { return fft.RFFT(batchPrime) })
	at := must(a32.Transpose())
	for _, tc := range []struct {
		name string
		a    *np.Array
		d    np.DType
	}{
		{"f32_f64", a32, np.Float64}, {"f64_f32", a, np.Float32}, {"c64_c128", z32, np.Complex128}, {"c128_c64", z, np.Complex64}, {"f32_c128", a32, np.Complex128}, {"transpose_f32_f64", at, np.Float64},
	} {
		run("cast_"+tc.name, func() (*np.Array, error) { return tc.a.Astype(tc.d) })
	}
	for _, tc := range []struct {
		name      string
		x, scalar *np.Array
	}{{"64", a, np.ScalarOf(float64(1.25))}, {"32", a32, np.ScalarOf(float32(1.25))}} {
		run("scalar_add_"+tc.name, func() (*np.Array, error) { return np.Add(tc.x, tc.scalar) })
		run("scalar_div_"+tc.name, func() (*np.Array, error) { return np.Divide(tc.scalar, tc.x) })
		row := must(tc.x.Slice(np.AtIndex(0)))
		run("row_add_"+tc.name, func() (*np.Array, error) { return np.Add(tc.x, row) })
		run("dense_mul_"+tc.name, func() (*np.Array, error) { return np.Multiply(tc.x, tc.x) })
	}
}
