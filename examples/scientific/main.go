// Demonstrates batch linear algebra and a real multi-channel Fourier round trip.
package main

import (
	"fmt"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/fft"
	"github.com/SamuelSupe/numpygo/linalg"
)

func must(a *np.Array, err error) *np.Array {
	if err != nil {
		panic(err)
	}
	return a
}

func printArray(label string, a *np.Array) {
	values, err := a.Complex128s()
	if err != nil {
		panic(err)
	}
	fmt.Printf("%s shape=%v dtype=%s values=%v\n", label, a.Shape(), a.DType(), values)
}

func main() {
	a := must(np.FromFloat64([]float64{3, 1, 2, 4, 4, 2, 1, 3}, 2, 2, 2))
	a = must(a.Transpose(0, 2, 1))
	b := must(np.FromFloat64([]float64{7, 11}, 2))
	printArray("solve", must(linalg.Solve(a, b)))
	_, s, _, err := linalg.SVD(a, linalg.SVDOptions{Reduced: true})
	if err != nil {
		panic(err)
	}
	printArray("singular values", s)

	signals := must(np.FromFloat64([]float64{1, -2, 3, .5, 4, 2, 3, -1, 1, 6}, 2, 5))
	signals = must(must(signals.Transpose()).Slice(np.Reverse()))
	spectrum := must(fft.RFFT(signals, fft.Options{Axis: np.Int(0), Norm: fft.Ortho}))
	printArray("spectrum", spectrum)
	restored := must(fft.IRFFT(spectrum, fft.Options{Axis: np.Int(0), N: np.Int(5), Norm: fft.Ortho}))
	printArray("restored", restored)
}
