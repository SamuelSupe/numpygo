package np_test

import (
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/fft"
)

func BenchmarkDirectKernels(b *testing.B) {
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
	const rows, n = 128, 1024
	v := make([]float64, rows*n)
	z := make([]complex128, len(v))
	m := make([]bool, len(v))
	alt := make([]bool, len(v))
	for i := range v {
		v[i] = float64(i%137-68) / 17
		z[i] = complex(v[i], float64(i%97-48)/31)
		m[i] = i%256 < 128
		alt[i] = i%2 == 0
	}
	a := must(np.WrapFloat64(v, rows, n))
	c := must(np.WrapComplex128(z, rows, n))
	a32 := must(a.Astype(np.Float32))
	c32 := must(c.Astype(np.Complex64))
	mask := must(np.WrapSlice(m, rows, n))
	alternating := must(np.WrapSlice(alt, rows, n))
	for _, tc := range []struct {
		name string
		a    *np.Array
		d    np.DType
	}{
		{"f32_f64", a32, np.Float64}, {"f64_f32", a, np.Float32}, {"c64_c128", c32, np.Complex128}, {"c128_c64", c, np.Complex64},
		{"reverse_rows_f32_f64", must(a32.Slice(np.Reverse(), np.Slice{})), np.Float64},
	} {
		run("cast_"+tc.name, func() (*np.Array, error) { return tc.a.Astype(tc.d) })
	}
	lookup := func(name string) *np.UFunc {
		u, e := np.LookupUFunc(name)
		if e != nil {
			b.Fatal(e)
		}
		return u
	}
	add, mul, div := lookup("add"), lookup("multiply"), lookup("divide")
	for _, tc := range []struct {
		name   string
		a      *np.Array
		scalar *np.Array
	}{{"64", a, np.ScalarOf(float64(1.25))}, {"32", a32, np.ScalarOf(float32(1.25))}} {
		x := tc.a
		d := x.DType()
		out := must(np.Zeros(d, rows, n))
		alias := must(x.Copy())
		row := must(x.Slice(np.AtIndex(0)))
		one := must(np.Full(d, 1))
		opt := np.UFuncOptions{Out: out}
		run("out_add_"+tc.name, func() (*np.Array, error) { return add.CallWithOptions(opt, x, x) })
		run("out_scalar_div_"+tc.name, func() (*np.Array, error) { return div.CallWithOptions(opt, tc.scalar, x) })
		run("out_row_add_"+tc.name, func() (*np.Array, error) { return add.CallWithOptions(opt, x, row) })
		run("out_mask_blocks_"+tc.name, func() (*np.Array, error) { return add.CallWithOptions(np.UFuncOptions{Out: out, Where: mask}, x, x) })
		run("out_mask_alternating_"+tc.name, func() (*np.Array, error) {
			return add.CallWithOptions(np.UFuncOptions{Out: out, Where: alternating}, x, x)
		})
		run("out_alias_mul_"+tc.name, func() (*np.Array, error) { return mul.CallWithOptions(np.UFuncOptions{Out: alias}, alias, one) })
	}
	half := must(must(c.Slice(np.Slice{}, np.Range(0, n/2+1))).Copy())
	co := must(np.Zeros(np.Complex128, rows, n))
	ro := must(np.Zeros(np.Float64, rows, n))
	ho := must(np.Zeros(np.Complex128, rows, n/2+1))
	run("fft_alloc", func() (*np.Array, error) { return fft.FFT(c) })
	run("rfft_alloc", func() (*np.Array, error) { return fft.RFFT(a) })
	run("irfft_alloc", func() (*np.Array, error) { return fft.IRFFT(half, fft.Options{N: np.Int(n)}) })
	run("fft_out", func() (*np.Array, error) { return fft.FFT(c, fft.Options{Out: co}) })
	run("ifft_out", func() (*np.Array, error) { return fft.IFFT(c, fft.Options{Out: co}) })
	cAlias := must(c.Copy())
	run("fft_alias", func() (*np.Array, error) { return fft.FFT(cAlias, fft.Options{Norm: fft.Ortho, Out: cAlias}) })
	run("rfft_out", func() (*np.Array, error) { return fft.RFFT(a, fft.Options{Out: ho}) })
	run("irfft_out", func() (*np.Array, error) { return fft.IRFFT(half, fft.Options{N: np.Int(n), Out: ro}) })
	run("fft32_alloc", func() (*np.Array, error) { return fft.FFT(c32) })
	run("rfft32_alloc", func() (*np.Array, error) { return fft.RFFT(a32) })
	co32 := must(np.Zeros(np.Complex64, rows, n))
	ho32 := must(np.Zeros(np.Complex64, rows, n/2+1))
	ro32 := must(np.Zeros(np.Float32, rows, n))
	half32 := must(half.Astype(np.Complex64))
	run("fft32_out", func() (*np.Array, error) { return fft.FFT(c32, fft.Options{Out: co32}) })
	run("rfft32_out", func() (*np.Array, error) { return fft.RFFT(a32, fft.Options{Out: ho32}) })
	run("irfft32_out", func() (*np.Array, error) { return fft.IRFFT(half32, fft.Options{N: np.Int(n), Out: ro32}) })
	small := must(np.WrapFloat64(v[:4*4093], 4, 4093))
	primeOut := must(np.Zeros(np.Complex128, 4, 2047))
	run("prime_rfft_alloc", func() (*np.Array, error) { return fft.RFFT(small) })
	run("prime_rfft_out", func() (*np.Array, error) { return fft.RFFT(small, fft.Options{Out: primeOut}) })
	mixedOut := must(np.Zeros(np.Complex128, rows, 501))
	run("fftpack_rfft_out", func() (*np.Array, error) { return fft.RFFT(a, fft.Options{N: np.Int(1000), Out: mixedOut}) })
}
