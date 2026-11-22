package numeric

var mixedConstants = [...]float64{.5, .86602540378443864676, .3090169943749474241, -.8090169943749474241, .9510565162951535721, .5877852522924731292}

// Butterfly applies radix-2 complex butterflies to non-overlapping halves.
// The caller has already validated the transform's numerical range.
func Butterfly(left, right, roots []complex128) {
	if len(left) != len(right) || len(left) != len(roots) {
		panic("FFT butterfly lane length mismatch")
	}
	if len(left) >= 32 && len(left)%4 == 0 && butterflySIMD(left, right, roots) {
		return
	}
	for i, w := range roots {
		u, v := left[i], right[i]*w
		left[i], right[i] = u+v, u-v
	}
}

// Radix4 fuses two radix-2 stages over four contiguous, bit-reversed
// subtransforms. The three streams are w², w and w³, respectively.
func Radix4(x, w1, w2, w3 []complex128, sign float64) {
	h := len(w1)
	if len(x) != 4*h || len(w2) != h || len(w3) != h {
		panic("FFT radix-4 lane length mismatch")
	}
	if h >= 16 && h%2 == 0 && radix4SIMD(x, w1, w2, w3, sign) {
		return
	}
	a, b, c, d := x[:h], x[h:2*h], x[2*h:3*h], x[3*h:]
	for i, u := range a {
		v, p, q := b[i]*w1[i], c[i]*w2[i], d[i]*w3[i]
		s, t := u+v, u-v
		r, z := p+q, p-q
		y := complex(-sign*imag(z), sign*real(z))
		a[i], b[i], c[i], d[i] = s+r, t+y, s-r, t-y
	}
}

// Radix3/5 combine natural-order subtransforms in a mixed-radix plan.
// Twiddles are laid out as r-1 contiguous streams; symmetric pairs
// avoid computing a general small DFT for every output.
func Radix3(x, w1, w2 []complex128, sign float64) {
	h := len(w1)
	a, b, c := x[:h], x[h:2*h], x[2*h:]
	k := complex(0, sign*0.86602540378443864676)
	start := 0
	if h >= 8 && radix3SIMD(x, w1, w2, sign) {
		start = h &^ 1
	}
	for j := start; j < h; j++ {
		u := a[j]
		v, w := b[j]*w1[j], c[j]*w2[j]
		s, d := v+w, v-w
		t := u - s*.5
		q := d * k
		a[j], b[j], c[j] = u+s, t+q, t-q
	}
}
func Radix5(x, roots []complex128, sign float64) {
	h := len(roots) / 4
	a, b, c, d, e := x[:h], x[h:2*h], x[2*h:3*h], x[3*h:4*h], x[4*h:]
	w1, w2, w3, w4 := roots[:h], roots[h:2*h], roots[2*h:3*h], roots[3*h:]
	const c1 = 0.3090169943749474241
	const c2 = -0.8090169943749474241
	const s1 = 0.9510565162951535721
	const s2 = 0.5877852522924731292
	is1, is2 := complex(0, sign*s1), complex(0, sign*s2)
	start := 0
	if h >= 8 && radix5SIMD(x, roots, sign) {
		start = h &^ 1
	}
	for j := start; j < h; j++ {
		u := a[j]
		v, w, y, z := b[j]*w1[j], c[j]*w2[j], d[j]*w3[j], e[j]*w4[j]
		p, q := v+z, w+y
		r, t := v-z, w-y
		f, g := u+c1*p+c2*q, u+c2*p+c1*q
		m, n := r*is1+t*is2, r*is2-t*is1
		a[j], b[j], c[j], d[j], e[j] = u+p+q, f+m, g+n, g-n, f-m
	}
}

// Binary64 and Binary32 process whole vectors and use the same scalar
// arithmetic for the remainder. Operation codes are add/sub/mul/div.
// They return false on platforms without a SIMD implementation.
func Binary64(dst, x, y []float64, op uint8) bool {
	n := len(dst) &^ 3
	if n < 32 || !binary64SIMD(dst[:n], x[:n], y[:n], op) {
		return false
	}
	for i := n; i < len(dst); i++ {
		switch op {
		case 0:
			dst[i] = x[i] + y[i]
		case 1:
			dst[i] = x[i] - y[i]
		case 2:
			dst[i] = x[i] * y[i]
		case 3:
			dst[i] = x[i] / y[i]
		}
	}
	return true
}
func Binary32(dst, x, y []float32, op uint8) bool {
	n := len(dst) &^ 7
	if n < 32 || !binary32SIMD(dst[:n], x[:n], y[:n], op) {
		return false
	}
	for i := n; i < len(dst); i++ {
		switch op {
		case 0:
			dst[i] = x[i] + y[i]
		case 1:
			dst[i] = x[i] - y[i]
		case 2:
			dst[i] = x[i] * y[i]
		case 3:
			dst[i] = x[i] / y[i]
		}
	}
	return true
}

func Lane64(dst, x, y []float64, as, bs int, op uint8) bool {
	if as == 1 && bs == 1 {
		return Binary64(dst, x, y, op)
	}
	if len(dst) < 32 || !((as == 0 && bs == 1) || (as == 1 && bs == 0)) {
		return false
	}
	var scalar float64
	if as == 0 {
		scalar = x[0]
		x = y
		op |= 4
	} else {
		scalar = y[0]
	}
	n := len(dst) &^ 3
	if !broadcast64SIMD(dst[:n], x[:n], scalar, op) {
		return false
	}
	broadcastTail(dst[n:], x[n:], scalar, op)
	return true
}
func Lane32(dst, x, y []float32, as, bs int, op uint8) bool {
	if as == 1 && bs == 1 {
		return Binary32(dst, x, y, op)
	}
	if len(dst) < 32 || !((as == 0 && bs == 1) || (as == 1 && bs == 0)) {
		return false
	}
	var scalar float32
	if as == 0 {
		scalar = x[0]
		x = y
		op |= 4
	} else {
		scalar = y[0]
	}
	n := len(dst) &^ 7
	if !broadcast32SIMD(dst[:n], x[:n], scalar, op) {
		return false
	}
	broadcastTail(dst[n:], x[n:], scalar, op)
	return true
}

func broadcastTail[T float32 | float64](dst, x []T, scalar T, op uint8) {
	for i, v := range x[:len(dst)] {
		a, b := v, scalar
		if op&4 != 0 {
			a, b = b, a
		}
		switch op & 3 {
		case 0:
			dst[i] = a + b
		case 1:
			dst[i] = a - b
		case 2:
			dst[i] = a * b
		case 3:
			dst[i] = a / b
		}
	}
}
