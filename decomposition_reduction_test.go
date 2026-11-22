package np_test

import (
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/fft"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
)

func TestDecompNativeReductions(t *testing.T) {
	for _, d := range []string{"float32", "float64"} {
		t.Run(d, func(t *testing.T) {
			m := refcheck.Require(t)
			a := refcheck.Input(t, "decomp_reduce_input_"+d)
			for _, tc := range []struct {
				name string
				v    *np.Array
				axes []int
			}{
				{"all", a, nil}, {"axis0", a, []int{0}}, {"axis1", a, []int{1}},
				{"transpose", m(a.Transpose()), []int{1}},
				{"reverse", m(a.Slice(np.Reverse(), np.Reverse())), []int{1}},
				{"broadcast", m(m(a.Slice(np.Range(0, 1), np.Slice{})).BroadcastTo(37, 41)), []int{0}},
				{"nonlinear", m(a.Slice(np.Strided(0, 37, 2), np.Strided(0, 41, 2))), nil},
			} {
				refcheck.CheckExact(t, "decomp_sum_"+tc.name+"_"+d, m(np.Sum(tc.v, np.SumOptions{Axes: tc.axes})))
				refcheck.Check(t, "decomp_mean_"+tc.name+"_"+d, m(np.Mean(tc.v, np.SumOptions{Axes: tc.axes})))
			}
			refcheck.CheckExact(t, "decomp_sum_noaxes_"+d, m(np.Sum(a, np.SumOptions{Axes: []int{}})))
			empty := m(np.Zeros(a.DType(), 3, 0))
			opt := np.SumOptions{Axes: []int{1}}
			refcheck.CheckExact(t, "decomp_sum_empty_"+d, m(np.Sum(empty, opt)))
			refcheck.Check(t, "decomp_mean_empty_"+d, m(np.Mean(empty, opt)))
			special := refcheck.Input(t, "decomp_reduce_special_input_"+d)
			refcheck.CheckExact(t, "decomp_sum_special_"+d, m(np.Sum(special, opt)))
			refcheck.Check(t, "decomp_mean_special_"+d, m(np.Mean(special, opt)))
			refcheck.CheckExact(t, "decomp_reduce_input_"+d, a)
		})
	}
}

func TestDecompBatchedPrimeFFT(t *testing.T) {
	m := refcheck.Require(t)
	a := refcheck.Input(t, "decomp_fft_prime_input")
	refcheck.Check(t, "decomp_fft_prime", m(fft.FFT(a)))
	refcheck.Check(t, "decomp_rfft_prime", m(fft.RFFT(a)))
	z := refcheck.Input(t, "decomp_rfft_prime")
	refcheck.Check(t, "decomp_irfft_prime", m(fft.IRFFT(z, fft.Options{N: np.Int(4093)})))
	c := refcheck.Input(t, "decomp_ifft_prime_input")
	refcheck.Check(t, "decomp_ifft_prime", m(fft.IFFT(c)))
}
