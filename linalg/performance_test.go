package linalg_test

import (
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/linalg"
)

var matmulResult *np.Array

func BenchmarkMatrixProducts(b *testing.B) {
	for _, d := range []np.DType{np.Float32, np.Float64, np.Complex64, np.Complex128, np.Int64} {
		a, e := np.Full(d, 1.25, 128, 128)
		if e != nil {
			b.Fatal(e)
		}
		x, e := np.Full(d, 2.5, 128, 128)
		if e != nil {
			b.Fatal(e)
		}
		if d.IsComplex() {
			a, e = np.Full(d, complex(1.25, .5), 128, 128)
			if e != nil {
				b.Fatal(e)
			}
		}
		b.Run(d.String(), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				matmulResult, e = linalg.MatMul(a, x)
				if e != nil {
					b.Fatal(e)
				}
			}
		})
	}
}
