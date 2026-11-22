package np_test

import (
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/fft"
)

func BenchmarkPrecisionKernels(b *testing.B) {
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
	v := make([]float64, 1024*1024)
	for i := range v {
		v[i] = float64(i%137-68) / 17
	}
	a64 := must(np.WrapFloat64(v, 1024, 1024))
	a32 := must(a64.Astype(np.Float32))
	for _, a := range []*np.Array{a64, a32} {
		tag := a.DType().String()
		for _, ax := range []struct {
			tag  string
			axes []int
		}{{"all", nil}, {"axis0", []int{0}}, {"axis1", []int{1}}} {
			o := np.SumOptions{Axes: ax.axes}
			w := np.VarOptions{Axes: ax.axes}
			run("sum_"+tag+"_"+ax.tag, func() (*np.Array, error) { return np.Sum(a, o) })
			run("mean_"+tag+"_"+ax.tag, func() (*np.Array, error) { return np.Mean(a, o) })
			run("var_"+tag+"_"+ax.tag, func() (*np.Array, error) { return np.Var(a, w) })
			run("std_"+tag+"_"+ax.tag, func() (*np.Array, error) { return np.Std(a, w) })
		}
	}
	a := must(np.WrapFloat64(v[:128*1024], 128, 1024))
	x := must(a.Astype(np.Float32))
	z := make([]complex64, x.Size())
	mask := make([]bool, x.Size())
	for i := range z {
		z[i] = complex(float32(v[i]), float32(float64(i%97-48)/31))
		mask[i] = i%256 < 128
	}
	c := must(np.WrapSlice(z, 128, 1024))
	half := must(must(c.Slice(np.Slice{}, np.Range(0, 513))).Copy())
	co := must(np.Zeros(np.Complex64, 128, 1024))
	ho := must(np.Zeros(np.Complex64, 128, 513))
	ro := must(np.Zeros(np.Float32, 128, 1024))
	run("fft32_alloc", func() (*np.Array, error) { return fft.FFT(c) })
	run("fft32_out", func() (*np.Array, error) { return fft.FFT(c, fft.Options{Out: co}) })
	run("ifft32_out", func() (*np.Array, error) { return fft.IFFT(c, fft.Options{Out: co}) })
	run("rfft32_alloc", func() (*np.Array, error) { return fft.RFFT(x) })
	run("rfft32_out", func() (*np.Array, error) { return fft.RFFT(x, fft.Options{Out: ho}) })
	run("irfft32_out", func() (*np.Array, error) { return fft.IRFFT(half, fft.Options{N: np.Int(1024), Out: ro}) })
	mixed := must(np.Zeros(np.Complex64, 128, 501))
	run("fftpack_rfft32_out", func() (*np.Array, error) { return fft.RFFT(x, fft.Options{N: np.Int(1000), Out: mixed}) })
	prime := must(must(np.WrapFloat64(v[:4*4093], 4, 4093)).Astype(np.Float32))
	primeOut := must(np.Zeros(np.Complex64, 4, 2047))
	run("prime_rfft32_out", func() (*np.Array, error) { return fft.RFFT(prime, fft.Options{Out: primeOut}) })
	m := must(np.WrapSlice(mask, 128, 1024))
	lookup := func(name string) *np.UFunc {
		u, e := np.LookupUFunc(name)
		if e != nil {
			b.Fatal(e)
		}
		return u
	}
	for _, q := range []*np.Array{a, x} {
		d := q.DType()
		tag := d.String()
		out := must(np.Zeros(d, 128, 1024))
		for _, name := range []string{"negative", "absolute", "square", "reciprocal", "sqrt", "exp", "minimum"} {
			u := lookup(name)
			args := []*np.Array{q}
			if name == "minimum" {
				args = append(args, q)
			}
			run("out_"+name+"_"+tag, func() (*np.Array, error) { return u.CallWithOptions(np.UFuncOptions{Out: out}, args...) })
		}
		square := lookup("square")
		run("out_mask_square_"+tag, func() (*np.Array, error) { return square.CallWithOptions(np.UFuncOptions{Out: out, Where: m}, q) })
	}
	for _, name := range []string{"conjugate", "square", "multiply", "divide"} {
		u := lookup(name)
		args := []*np.Array{c}
		if name == "multiply" || name == "divide" {
			args = append(args, c)
		}
		run("out_"+name+"_complex64", func() (*np.Array, error) { return u.CallWithOptions(np.UFuncOptions{Out: co}, args...) })
	}
}
