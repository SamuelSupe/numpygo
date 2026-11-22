package numeric

import (
	"math"
	"testing"
)

func TestComplexVectorRounding(t *testing.T) {
	values := []float64{0, math.Copysign(0, -1), 1, -2, 1e200, 1e-200, math.Inf(1), math.Inf(-1), math.NaN()}
	x, y := make([]complex128, 257), make([]complex128, 257)
	for i := range x {
		x[i] = complex(values[i%len(values)], values[(i/len(values))%len(values)])
		y[i] = complex(values[(i*7+1)%len(values)], values[(i*3+i/len(values))%len(values)])
	}
	same := func(a, b float64) bool {
		return math.Float64bits(a) == math.Float64bits(b) || math.IsNaN(a) && math.IsNaN(b)
	}
	for op := uint8(0); op < 4; op++ {
		dst := make([]complex128, len(x))
		if !ComplexBinary128(dst, x, y, op) {
			continue
		}
		for i, a := range x {
			b := y[i]
			var want complex128
			switch op {
			case 0:
				want = a + b
			case 1:
				want = a - b
			case 2:
				want = a * b
			case 3:
				want = ComplexDivide(a, b)
			}
			if !same(real(dst[i]), real(want)) || !same(imag(dst[i]), imag(want)) {
				t.Errorf("op %d index %d %v %v: got %v want %v", op, i, a, b, dst[i], want)
			}
		}
	}
}

func TestComplex64RegisterRounding(t *testing.T) {
	values := []float32{0, float32(math.Copysign(0, -1)), 1, -2, math.MaxFloat32, math.SmallestNonzeroFloat32, 1e-20, 1e20, float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN())}
	same := func(a, b float32) bool {
		return math.Float32bits(a) == math.Float32bits(b) || math.IsNaN(float64(a)) && math.IsNaN(float64(b))
	}
	for _, n := range []int{16, 17, 31, 32, 257} {
		x, y := make([]complex64, n), make([]complex64, n)
		for i := range x {
			x[i] = complex(values[i%len(values)], values[(i/len(values))%len(values)])
			y[i] = complex(values[(i*7+1)%len(values)], values[(i*3+i/len(values))%len(values)])
		}
		for _, op := range []uint8{2, 3} {
			want := make([]complex64, n)
			for i, a := range x {
				b := complex128(y[i])
				if op == 2 {
					want[i] = complex64(complex128(a) * b)
				} else {
					want[i] = complex64(ComplexDivide(complex128(a), b))
				}
			}
			for alias := 0; alias < 3; alias++ {
				a, b := append([]complex64(nil), x...), append([]complex64(nil), y...)
				storage := make([]complex64, n+2)
				storage[0], storage[n+1] = 17, -23
				dst := storage[1 : n+1]
				if alias == 1 {
					dst = a
				} else if alias == 2 {
					dst = b
				}
				if !ComplexBinary64(dst, a, b, op) {
					if advancedVectorsAvailable() {
						t.Fatal("missing SIMD")
					}
					continue
				}
				for i, z := range dst {
					if !same(real(z), real(want[i])) || !same(imag(z), imag(want[i])) {
						t.Fatalf("length %d op %d alias %d index %d: got %v want %v", n, op, alias, i, z, want[i])
					}
				}
				if storage[0] != 17 || storage[n+1] != -23 {
					t.Fatal("out guard overwritten")
				}
			}
		}
	}
}
