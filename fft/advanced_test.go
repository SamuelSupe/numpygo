package fft_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/fft"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
)

var oneFFT = map[string]func(*np.Array, ...fft.Options) (*np.Array, error){
	"fft": fft.FFT, "ifft": fft.IFFT, "rfft": fft.RFFT, "irfft": fft.IRFFT, "hfft": fft.HFFT, "ihfft": fft.IHFFT,
}
var multiFFT = map[string]func(*np.Array, ...fft.NDOptions) (*np.Array, error){
	"fftn": fft.FFTN, "ifftn": fft.IFFTN, "rfftn": fft.RFFTN, "irfftn": fft.IRFFTN,
	"fft2": fft.FFT2, "ifft2": fft.IFFT2, "rfft2": fft.RFFT2, "irfft2": fft.IRFFT2,
}
var fftTypes = []np.DType{np.Bool, np.Int8, np.Int16, np.Int32, np.Int64, np.Uint8, np.Uint16, np.Uint32, np.Uint64, np.Float16, np.Float32, np.Float64, np.Complex64, np.Complex128}
var fftNorms = []fft.Norm{fft.Backward, fft.Forward, fft.Ortho}

func TestAdvancedFFTDTypeGrid(t *testing.T) {
	for _, d := range fftTypes {
		m := refcheck.Require(t)
		a := refcheck.Input(t, "adv_fft_input_"+d.String())
		a = m(m(a.Transpose(1, 2, 0)).Slice(np.Reverse()))
		for name, fn := range oneFFT {
			for _, norm := range fftNorms {
				for _, axis := range []int{1, -1} {
					for _, n := range []int{0, 1, 5} {
						label := "None"
						var length *int
						if n != 0 {
							label = fmt.Sprint(n)
							length = np.Int(n)
						}
						key := fmt.Sprintf("adv_fft_%s_%s_%s_%d_%s", name, d, norm, axis, label)
						t.Run(key, func(t *testing.T) {
							v, e := fn(a, fft.Options{Axis: np.Int(axis), N: length, Norm: norm})
							refcheck.Case(t, key, v, e)
							if e == nil && n == 5 {
								refcheck.CheckLayout(t, key, v)
							}
						})
					}
				}
			}
		}
	}
}

func TestAdvancedFFTNDGrid(t *testing.T) {
	for _, d := range []np.DType{np.Float32, np.Float64, np.Complex64} {
		a := refcheck.Input(t, "adv_fft_multi_input_"+d.String())
		for name, fn := range multiFFT {
			if d.IsComplex() && (name == "rfft2" || name == "rfftn") {
				continue
			}
			for ak, axes := range map[string][]int{"repeat": {1, 1}, "outer": {0, -1}, "reverse": {2, 0}} {
				for sk, shape := range map[string][]int{"default": nil, "full": {-1, -1}, "resize": {5, 7}} {
					for _, norm := range fftNorms {
						key := fmt.Sprintf("adv_fft_%s_%s_%s_%s_%s", name, d, ak, sk, norm)
						t.Run(key, func(t *testing.T) {
							refcheck.Check(t, key, refcheck.Require(t)(fn(a, fft.NDOptions{Shape: shape, Axes: axes, Norm: norm})))
						})
					}
				}
			}
		}
	}
}
func TestFFTOutputBuffers(t *testing.T) {
	a := refcheck.Input(t, "adv_fft_out_input")
	for name, fn := range oneFFT {
		for _, norm := range fftNorms {
			for _, n := range []int{4, 7} {
				types := []np.DType{np.Complex64, np.Complex128}
				if name == "irfft" || name == "hfft" {
					types = []np.DType{np.Float16, np.Float32, np.Float64, np.Complex128}
				}
				for _, d := range types {
					key := fmt.Sprintf("adv_fft_out_%s_%s_%d_%s", name, norm, n, d)
					t.Run(key, func(t *testing.T) {
						m := refcheck.Require(t)
						sh := refcheck.Input(t, key+"_backing").Shape()
						backing := m(np.Full(d, -99, sh...))
						out := m(backing.Slice(np.Slice{}, np.Slice{Step: np.Int(-2)}))
						got := m(fn(a, fft.Options{N: np.Int(n), Norm: norm, Out: out}))
						if got != out {
							t.Fatal("out identity not preserved")
						}
						refcheck.CheckLayout(t, key, got)
						refcheck.Check(t, key+"_backing", backing)
					})
				}
			}
		}
		for _, d := range []np.DType{np.Float16, np.Float32, np.Complex64} {
			if d == np.Complex64 && (name == "rfft" || name == "ihfft") {
				continue
			}
			key := fmt.Sprintf("adv_fft_wide_%s_%s", name, d)
			t.Run(key, func(t *testing.T) {
				m := refcheck.Require(t)
				input := refcheck.Input(t, "adv_fft_wide_input_"+d.String())
				want := refcheck.Input(t, key)
				out := m(np.Zeros(want.DType(), want.Shape()...))
				got := m(fn(input, fft.Options{N: np.Int(7), Norm: fft.Ortho, Out: out}))
				if d == np.Complex64 || d == np.Float32 && (name == "rfft" || name == "ihfft") {
					refcheck.Metadata(t, key, got)
					x, _ := got.Complex128s()
					y, _ := want.Complex128s()
					for i, z := range x {
						// A wider out does not promote the selected single-precision loop.
						if z != complex128(complex64(z)) || !refcheck.CloseDType(real(z), real(y[i]), np.Complex64) || !refcheck.CloseDType(imag(z), imag(y[i]), np.Complex64) {
							t.Errorf("%s [%d] got %v want %v (single-precision loop)", key, i, z, y[i])
						}
					}
				} else {
					refcheck.Check(t, key, got)
				}
			})
		}
	}
	m := refcheck.Require(t)
	a = m(np.Arange(0, 12, 1, np.Float64))
	a = m(a.Reshape(3, 4))
	for name, fn := range multiFFT {
		t.Run(name, func(t *testing.T) {
			key := "adv_fft_ndout_" + name
			m := refcheck.Require(t)
			want := refcheck.Input(t, key)
			out := m(np.EmptyLike(want, want.DType(), "F"))
			s := []int{3, 7}
			if name == "irfft2" || name == "irfftn" {
				s[0] = 5
			}
			r := m(fn(a, fft.NDOptions{Shape: s, Axes: []int{0, 1}, Out: out}))
			if r != out {
				t.Fatal("nd out identity")
			}
			refcheck.CheckLayout(t, key, r)
		})
	}
	x := m(np.Arange(0, 4, 1, np.Float64))
	x = m(x.Reshape(1, 4))
	refcheck.Check(t, "adv_fft_broadcast", m(fft.FFT(x, fft.Options{Out: m(np.Zeros(np.Complex128, 3, 4))})))
	alias := refcheck.Input(t, "adv_fft_alias_input")
	refcheck.Check(t, "adv_fft_alias", m(fft.FFT(alias, fft.Options{Out: alias})))
	for _, c := range []struct {
		name, class string
		out         *np.Array
		want        error
	}{
		{"shape", "ValueError", m(np.Zeros(np.Complex128, 1, 3)), np.ErrShape},
		{"rank", "ValueError", m(np.Zeros(np.Complex128, 1, 1, 4)), np.ErrShape},
		{"bad_dtype", "UFuncTypeError", m(np.Zeros(np.Float64, 1, 4)), np.ErrDType},
		{"readonly", "ValueError", m(np.ScalarOf(complex128(1+1i)).BroadcastTo(1, 4)), np.ErrReadOnly},
	} {
		_, err := fft.FFT(x, fft.Options{Out: c.out})
		refcheck.Error(t, "adv_fft_out_"+c.name, c.class, err, c.want)
	}
	_, err := fft.FFTN(a, fft.NDOptions{Shape: []int{5, 7}, Axes: []int{0, 1}, Out: m(np.Zeros(np.Complex128, 5, 7))})
	refcheck.Error(t, "adv_fft_ndout_resize", "ValueError", err, np.ErrShape)
	_, err = fft.RFFTN(x, fft.NDOptions{Axes: []int{}})
	refcheck.Error(t, "adv_fft_empty_realaxes", "IndexError", err, np.ErrAxis)
	_, err = fft.IRFFTN(x, fft.NDOptions{Axes: []int{}})
	refcheck.Error(t, "adv_fft_inverse_empty_realaxes", "IndexError", err, np.ErrAxis)
}
func TestFFTFrequencyEdge(t *testing.T) {
	for _, n := range []int{1, 2, 5, 6} {
		for key, d := range map[string]float64{"normal": .25, "negative": -.5, "nan": math.NaN(), "inf": math.Inf(1)} {
			t.Run(fmt.Sprintf("%d_%s", n, key), func(t *testing.T) {
				m := refcheck.Require(t)
				refcheck.CheckExact(t, fmt.Sprintf("adv_fft_freq_%d_%s", n, key), m(fft.FFTFreq(n, d)))
				refcheck.CheckExact(t, fmt.Sprintf("adv_fft_rfreq_%d_%s", n, key), m(fft.RFFTFreq(n, d)))
			})
		}
	}
}
