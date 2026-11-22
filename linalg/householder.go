package linalg

import (
	"math/cmplx"

	"gonum.org/v1/gonum/blas/cblas128"
)

func contiguousVector(x []complex128) cblas128.Vector {
	return cblas128.Vector{N: len(x), Inc: 1, Data: x}
}

// applyHouseholderLeft applies A -= tau*v*(vᴴ*A) on the selected trailing
// rectangular block. Rows are streamed contiguously, retaining the order
// of each dot product and sharing one scratch vector across reflectors.
func applyHouseholderLeft(a []complex128, stride, row, col int, v []complex128, tau complex128, scratch []complex128, fast bool) {
	w := scratch[:stride-col]
	clear(w)
	axpy := fast && len(w) >= 32
	for i, z := range v {
		x := a[(row+i)*stride+col : (row+i+1)*stride]
		c := cmplx.Conj(z)
		if axpy {
			cblas128.Axpy(c, contiguousVector(x), contiguousVector(w))
		} else {
			for j, y := range x {
				w[j] += c * y
			}
		}
	}
	for j := range w {
		w[j] *= tau
	}
	for i, z := range v {
		x := a[(row+i)*stride+col : (row+i+1)*stride]
		if axpy {
			cblas128.Axpy(-z, contiguousVector(w), contiguousVector(x))
		} else {
			for j, y := range w {
				x[j] -= z * y
			}
		}
	}
}
