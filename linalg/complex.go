package linalg

import (
	"fmt"
	"math"
	"math/cmplx"
	"slices"
	"sort"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/numeric"

	"gonum.org/v1/gonum/blas/cblas128"
)

func linearDType(types ...np.DType) (np.DType, error) {
	double, complexType := false, false
	for _, d := range types {
		if !d.IsNativeNumeric() || d == np.Float16 {
			return 0, np.ErrDType
		}
		if d == np.Float64 || d == np.Complex128 || d.IsInteger() || d == np.Bool {
			double = true
		}
		if d.IsComplex() {
			complexType = true
		}
	}
	if complexType {
		if double {
			return np.Complex128, nil
		}
		return np.Complex64, nil
	}
	if double {
		return np.Float64, nil
	}
	return np.Float32, nil
}

type arithmeticType interface {
	int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | float32 | complex64 | complex128
}

func matMulNumeric[T arithmeticType](a, b *np.Array, shape, batch, abatch, bbatch []int, m, k, n, size int) (*np.Array, error) {
	av, e := np.Values[T](a)
	if e != nil {
		return nil, e
	}
	bv, e := np.Values[T](b)
	if e != nil {
		return nil, e
	}
	out := make([]T, size)
	// Keep the k order for modular integer arithmetic while traversing
	// the RHS and output rows contiguously.
	for q := 0; q < size/(m*n); q++ {
		ai, bi := matrixOffset(q, batch, abatch, m*k), matrixOffset(q, batch, bbatch, k*n)
		for i := 0; i < m; i++ {
			row := out[q*m*n+i*n : q*m*n+(i+1)*n]
			for h := 0; h < k; h++ {
				x := av[ai+i*k+h]
				rhs := bv[bi+h*n : bi+(h+1)*n]
				for j, y := range rhs {
					row[j] += x * y
				}
			}
		}
	}
	return np.WrapSlice(out, shape...)
}
func matMulBLAS[T numeric.Floating](a, b *np.Array, shape, batch, abatch, bbatch []int, m, k, n, size int) (*np.Array, error) {
	av, e := np.Values[T](a)
	if e != nil {
		return nil, e
	}
	bv, e := np.Values[T](b)
	if e != nil {
		return nil, e
	}
	out := make([]T, size)
	for q := 0; q < size/(m*n); q++ {
		ai, bi := matrixOffset(q, batch, abatch, m*k), matrixOffset(q, batch, bbatch, k*n)
		numeric.Gemm(m, k, n, av[ai:ai+m*k], bv[bi:bi+k*n], out[q*m*n:(q+1)*m*n])
	}
	return np.WrapSlice(out, shape...)
}
func matMul32(a, b *np.Array, shape, batch, abatch, bbatch []int, m, k, n, size int) (*np.Array, error) {
	return matMulBLAS[float32](a, b, shape, batch, abatch, bbatch, m, k, n, size)
}
func matMulTyped[T np.Number](a, b *np.Array, shape, batch, abatch, bbatch []int, m, k, n, size int, accumulate func(T, T, T) T) (*np.Array, error) {
	av, err := np.Values[T](a)
	if err != nil {
		return nil, err
	}
	bv, err := np.Values[T](b)
	if err != nil {
		return nil, err
	}
	out := make([]T, size)
	for q := 0; q < size/(m*n); q++ {
		ai, bi := matrixOffset(q, batch, abatch, m*k), matrixOffset(q, batch, bbatch, k*n)
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				var sum T
				for h := 0; h < k; h++ {
					sum = accumulate(sum, av[ai+i*k+h], bv[bi+h*n+j])
				}
				out[q*m*n+i*n+j] = sum
			}
		}
	}
	return np.WrapSlice(out, shape...)
}
func matMulOther(a, b *np.Array, d np.DType, shape, batch, abatch, bbatch []int, m, k, n, size int) (*np.Array, error) {
	switch d {
	case np.Float128:
		return matMulTyped(a, b, shape, batch, abatch, bbatch, m, k, n, size, func(s, x, y np.ExtendedFloat) np.ExtendedFloat { return s.Add(x.Mul(y)) })
	case np.Complex256:
		return matMulTyped(a, b, shape, batch, abatch, bbatch, m, k, n, size, func(s, x, y np.ExtendedComplex) np.ExtendedComplex { return s.Add(x.Mul(y)) })
	case np.Bool:
		return matMulTyped(a, b, shape, batch, abatch, bbatch, m, k, n, size, func(sum, x, y bool) bool { return sum || x && y })
	case np.Float16:
		av, err := a.Astype(np.Float32)
		if err != nil {
			return nil, err
		}
		bv, err := b.Astype(np.Float32)
		if err != nil {
			return nil, err
		}
		out, err := matMul32(av, bv, shape, batch, abatch, bbatch, m, k, n, size)
		if err != nil {
			return nil, err
		}
		return out.Astype(np.Float16)
	case np.Int8:
		return matMulNumeric[int8](a, b, shape, batch, abatch, bbatch, m, k, n, size)
	case np.Int16:
		return matMulNumeric[int16](a, b, shape, batch, abatch, bbatch, m, k, n, size)
	case np.Int32:
		return matMulNumeric[int32](a, b, shape, batch, abatch, bbatch, m, k, n, size)
	case np.Int64:
		return matMulNumeric[int64](a, b, shape, batch, abatch, bbatch, m, k, n, size)
	case np.Uint8:
		return matMulNumeric[uint8](a, b, shape, batch, abatch, bbatch, m, k, n, size)
	case np.Uint16:
		return matMulNumeric[uint16](a, b, shape, batch, abatch, bbatch, m, k, n, size)
	case np.Uint32:
		return matMulNumeric[uint32](a, b, shape, batch, abatch, bbatch, m, k, n, size)
	case np.Uint64:
		return matMulNumeric[uint64](a, b, shape, batch, abatch, bbatch, m, k, n, size)
	case np.Float32:
		return matMul32(a, b, shape, batch, abatch, bbatch, m, k, n, size)
	case np.Complex64:
		return matMulBLAS[complex64](a, b, shape, batch, abatch, bbatch, m, k, n, size)
	case np.Complex128:
		return matMulBLAS[complex128](a, b, shape, batch, abatch, bbatch, m, k, n, size)
	}
	return nil, np.ErrDType
}

func asComplexArray(v []complex128, d np.DType, shape []int) (*np.Array, error) {
	out, err := np.WrapComplex128(v, shape...)
	if err != nil || d == np.Complex128 {
		return out, err
	}
	return out.Astype(d)
}

func complexPivot(z complex128) float64 { return math.Abs(real(z)) + math.Abs(imag(z)) }

// factorComplex produces in-place LU and row pivots using partial pivoting.
func factorComplex(a []complex128, n int) ([]int, int, bool) {
	pivots := make([]int, n)
	parity := 1
	for j := 0; j < n; j++ {
		p := j
		best := complexPivot(a[j*n+j])
		for i := j + 1; i < n; i++ {
			if v := complexPivot(a[i*n+j]); v > best {
				p = i
				best = v
			}
		}
		pivots[j] = p
		if a[p*n+j] == 0 {
			return pivots, parity, false
		}
		if p != j {
			parity = -parity
			x, y := a[p*n:(p+1)*n], a[j*n:(j+1)*n]
			for k := range x {
				x[k], y[k] = y[k], x[k]
			}
		}
		pivot := a[j*n+j]
		rhs := a[j*n+j+1 : (j+1)*n]
		fast := len(rhs) >= 32
		if fast {
			for _, v := range rhs {
				if v-v != 0 {
					fast = false
					break
				}
			}
		}
		x := cblas128.Vector{N: len(rhs), Inc: 1, Data: rhs}
		for i := j + 1; i < n; i++ {
			row := a[i*n+j : (i+1)*n]
			mult := row[0] / pivot
			row[0] = mult
			tail := row[1:]
			if fast && mult != 0 && mult-mult == 0 {
				cblas128.Axpy(-mult, x, cblas128.Vector{N: len(tail), Inc: 1, Data: tail})
				continue
			}
			for k, v := range rhs {
				tail[k] -= mult * v
			}
		}
	}
	return pivots, parity, true
}
func solveLU(a, b []complex128, pivots []int, n, k int) {
	for j, p := range pivots {
		if j != p {
			for h := 0; h < k; h++ {
				b[j*k+h], b[p*k+h] = b[p*k+h], b[j*k+h]
			}
		}
	}
	for i := 0; i < n; i++ {
		row := b[i*k : (i+1)*k]
		for j := 0; j < i; j++ {
			mult := a[i*n+j]
			for h, v := range b[j*k : (j+1)*k] {
				row[h] -= mult * v
			}
		}
	}
	for i := n - 1; i >= 0; i-- {
		row := b[i*k : (i+1)*k]
		for j := i + 1; j < n; j++ {
			mult := a[i*n+j]
			for h, v := range b[j*k : (j+1)*k] {
				row[h] -= mult * v
			}
		}
		for h := range row {
			row[h] /= a[i*n+i]
		}
	}
}
func solveComplex(a, b *np.Array, d np.DType, shape, batch, abatch, bbatch []int, n, k, size, batches int) (*np.Array, error) {
	av, _ := a.Complex128s()
	bv, _ := b.Complex128s()
	out := make([]complex128, size)
	lu := make([]complex128, n*n)
	last := -1
	var pivots []int
	for i := 0; i < batches; i++ {
		ai := matrixOffset(i, batch, abatch, n*n)
		if ai != last {
			copy(lu, av[ai:ai+n*n])
			var ok bool
			pivots, _, ok = factorComplex(lu, n)
			if !ok {
				return nil, fmt.Errorf("%w: batch matrix %d", ErrSingular, i)
			}
			last = ai
		}
		if k == 0 {
			continue
		}
		bi := matrixOffset(i, batch, bbatch, n*k)
		rhs := out[i*n*k : (i+1)*n*k]
		copy(rhs, bv[bi:bi+n*k])
		solveLU(lu, rhs, pivots, n, k)
	}
	return asComplexArray(out, d, shape)
}

func adjoint(a []complex128, m, n int) []complex128 {
	out := make([]complex128, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			out[j*m+i] = cmplx.Conj(a[i*n+j])
		}
	}
	return out
}
func identity(n int) []complex128 {
	a := make([]complex128, n*n)
	for i := 0; i < n; i++ {
		a[i*n+i] = 1
	}
	return a
}
func normColumn(a []complex128, m, n, j int) float64 {
	var norm float64
	for i := 0; i < m; i++ {
		norm = math.Hypot(norm, cmplx.Abs(a[i*n+j]))
	}
	return norm
}

func complexSVD(a *np.Array, d np.DType, options []SVDOptions) (*np.Array, *np.Array, *np.Array, error) {
	if a.NDim() < 2 || len(options) > 1 {
		return nil, nil, nil, np.ErrShape
	}
	var opt SVDOptions
	if len(options) == 1 {
		opt = options[0]
	}
	shape := a.Shape()
	batch := shape[:len(shape)-2]
	m, n := shape[len(shape)-2], shape[len(shape)-1]
	k := min(m, n)
	uc, vr := m, n
	if opt.Reduced {
		uc, vr = k, k
	}
	ss := append(slices.Clone(batch), k)
	us := append(slices.Clone(batch), m, uc)
	vs := append(slices.Clone(batch), vr, n)
	batches, err := np.ElementCount(np.Complex128, batch...)
	if err != nil {
		return nil, nil, nil, err
	}
	sd := np.Float64
	if d == np.Complex64 {
		sd = np.Float32
	}
	sn, err := np.ElementCount(sd, ss...)
	if err != nil {
		return nil, nil, nil, err
	}
	sv := make([]float64, sn)
	var uv, vv []complex128
	if !opt.ValuesOnly {
		un, err := np.ElementCount(d, us...)
		if err != nil {
			return nil, nil, nil, err
		}
		vn, err := np.ElementCount(d, vs...)
		if err != nil {
			return nil, nil, nil, err
		}
		uv, vv = make([]complex128, un), make([]complex128, vn)
	}
	values, _ := a.Complex128s()
	for q := 0; q < batches; q++ {
		x := values[q*m*n : (q+1)*m*n]
		full := !opt.Reduced && !opt.ValuesOnly
		var ul, vl []complex128
		var s []float64
		if m >= n {
			ul, s, vl, err = complexTallSVD(x, m, n, full, !opt.ValuesOnly)
		} else {
			var ut, vt []complex128
			ut, s, vt, err = complexTallSVD(adjoint(x, m, n), n, m, full, !opt.ValuesOnly)
			if err == nil && !opt.ValuesOnly {
				ul = adjoint(vt, m, m)
				cols := m
				if full {
					cols = n
				}
				vl = adjoint(ut, n, cols)
			}
		}
		if err != nil {
			return nil, nil, nil, fmt.Errorf("%w: batch matrix %d", err, q)
		}
		copy(sv[q*k:(q+1)*k], s)
		if !opt.ValuesOnly {
			copy(uv[q*m*uc:(q+1)*m*uc], ul)
			copy(vv[q*vr*n:(q+1)*vr*n], vl)
		}
	}
	s, err := np.WrapFloat64(sv, ss...)
	if err != nil {
		return nil, nil, nil, err
	}
	if sd == np.Float32 {
		s, err = s.Astype(sd)
		if err != nil {
			return nil, nil, nil, err
		}
	}
	if opt.ValuesOnly {
		return nil, s, nil, nil
	}
	u, err := asComplexArray(uv, d, us)
	if err != nil {
		return nil, nil, nil, err
	}
	vh, err := asComplexArray(vv, d, vs)
	return u, s, vh, err
}

// jacobiSVD orthogonalizes columns of a scaled tall complex matrix using
// unitary Jacobi rotations. It avoids squaring the condition number via AᴴA.
func jacobiSVD(input []complex128, m, n int, full, vectors bool) ([]complex128, []float64, []complex128, error) {
	cols := n
	if full {
		cols = m
	}
	if n == 0 {
		if full {
			return identity(m), nil, nil, nil
		}
		return nil, nil, nil, nil
	}
	scale := 0.0
	for _, z := range input {
		if math.IsInf(real(z), 0) || math.IsInf(imag(z), 0) || math.IsNaN(real(z)) || math.IsNaN(imag(z)) {
			return nil, nil, nil, ErrConvergence
		}
		scale = max(scale, math.Abs(real(z)), math.Abs(imag(z)))
	}
	// Jacobi traverses pairs of columns many times. Store both matrices
	// column-major during the iteration; only transpose the final output.
	b := make([]complex128, len(input))
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			z := input[i*n+j]
			if scale > 0 {
				z = complex(real(z)/scale, imag(z)/scale)
			}
			b[j*m+i] = z
		}
	}
	var v []complex128
	if vectors {
		v = identity(n)
	}
	const eps = 2.220446049250313e-16
	converged := false
	for sweep := 0; sweep < 100; sweep++ {
		rotated := false
		for p := 0; p < n; p++ {
			for q := p + 1; q < n; q++ {
				bp, bq := b[p*m:(p+1)*m], b[q*m:(q+1)*m]
				var alpha, beta float64
				var gamma complex128
				for i, x := range bp {
					y := bq[i]
					alpha += real(x)*real(x) + imag(x)*imag(x)
					beta += real(y)*real(y) + imag(y)*imag(y)
					gamma += cmplx.Conj(x) * y
				}
				var delta, twoG float64
				g := cmplx.Abs(gamma)
				if min(alpha, beta) > 1e-280 {
					if g <= 4*eps*math.Sqrt(alpha)*math.Sqrt(beta) {
						continue
					}
					delta, twoG = beta-alpha, 2*g
				} else {
					// Squared norms can underflow for representable
					// singular values. Recover their normalized correlation
					// before forming a rotation with a safe relative scale.
					an, bn := vectorNorm(bp), vectorNorm(bq)
					if an == 0 || bn == 0 {
						continue
					}
					gamma = 0
					for i, x := range bp {
						y := bq[i]
						x = complex(real(x)/an, imag(x)/an)
						y = complex(real(y)/bn, imag(y)/bn)
						gamma += cmplx.Conj(x) * y
					}
					g = cmplx.Abs(gamma)
					if g <= 4*eps {
						continue
					}
					smallScale := max(an, bn)
					ap, bp := an/smallScale, bn/smallScale
					delta, twoG = bp*bp-ap*ap, 2*g*ap*bp
				}
				phase := gamma / complex(g, 0)
				if twoG == 0 {
					continue
				}
				t := math.Copysign(twoG/(math.Abs(delta)+math.Hypot(delta, twoG)), delta)
				if t == 0 {
					continue
				}
				c := 1 / math.Hypot(1, t)
				s := c * t
				w := scaleComplex(phase, s)
				wc := cmplx.Conj(w)
				jacobiRotate(bp, bq, c, w, wc)
				if vectors {
					jacobiRotate(v[p*n:(p+1)*n], v[q*n:(q+1)*n], c, w, wc)
				}
				rotated = true
			}
		}
		if !rotated {
			converged = true
			break
		}
	}
	if !converged {
		return nil, nil, nil, ErrConvergence
	}
	norms := make([]float64, n)
	order := make([]int, n)
	for i := range norms {
		norms[i] = vectorNorm(b[i*m : (i+1)*m])
		order[i] = i
	}
	sort.SliceStable(order, func(i, j int) bool { return norms[order[i]] > norms[order[j]] })
	s := make([]float64, n)
	for i, k := range order {
		s[i] = norms[k] * scale
	}
	if !vectors {
		return nil, s, nil, nil
	}
	vh := make([]complex128, n*n)
	nonzero := true
	for j, k := range order {
		for i, z := range v[k*n : (k+1)*n] {
			vh[j*n+i] = cmplx.Conj(z)
		}
		nonzero = nonzero && norms[k] > 0
	}
	if cols == n && nonzero {
		// A full-rank reduced decomposition needs no basis completion.
		// Pack the sorted singular vectors straight from contiguous B
		// columns into C-order U, avoiding another full rectangular copy.
		u := make([]complex128, m*n)
		for j, k := range order {
			norm := norms[k]
			for i, z := range b[k*m : (k+1)*m] {
				u[i*n+j] = complex(real(z)/norm, imag(z)/norm)
			}
		}
		return u, s, vh, nil
	}
	uc := make([]complex128, m*cols)
	work := make([]complex128, m)
	nextBasis := 0
	for j := 0; j < cols; j++ {
		if j < n {
			k := order[j]
			norm := norms[k]
			if norm > 0 {
				col := uc[j*m : (j+1)*m]
				copy(col, b[k*m:(k+1)*m])
				normalizeVector(col, norm)
				continue
			}
		}
		// Complete the orthonormal basis for zero singular values or full U.
		var ok bool
		nextBasis, ok = completeColumn(uc, m, j, nextBasis, work)
		if !ok {
			return nil, nil, nil, ErrConvergence
		}
	}
	// Full square U transposes in place; a reduced rectangular U needs
	// one final C-order packing pass.
	if m == cols {
		for i := 0; i < m; i++ {
			for j := 0; j < i; j++ {
				uc[i*m+j], uc[j*m+i] = uc[j*m+i], uc[i*m+j]
			}
		}
		return uc, s, vh, nil
	}
	u := make([]complex128, m*cols)
	for j := 0; j < cols; j++ {
		for i, z := range uc[j*m : (j+1)*m] {
			u[i*cols+j] = z
		}
	}
	return u, s, vh, nil
}

func jacobiRotate(x, y []complex128, c float64, w, wc complex128) {
	for i, a := range x {
		b := y[i]
		x[i] = scaleComplex(a, c) - wc*b
		y[i] = w*a + scaleComplex(b, c)
	}
}
func completeColumn(u []complex128, rows, j, start int, vec []complex128) (int, bool) {
	// A coordinate already represented by previous basis columns remains
	// represented as the basis grows. Continue after the last accepted
	// coordinate, with a wraparound fallback for numerical rank edges.
	for attempt := 0; attempt < rows; attempt++ {
		k := start + attempt
		if k >= rows {
			k -= rows
		}
		clear(vec)
		vec[k] = 1
		for pass := 0; pass < 2; pass++ {
			for p := 0; p < j; p++ {
				col := u[p*rows : (p+1)*rows]
				if rows >= 32 {
					dot := cblas128.Dotc(contiguousVector(col), contiguousVector(vec))
					cblas128.Axpy(-dot, contiguousVector(col), contiguousVector(vec))
					continue
				}
				var dot complex128
				for i := 0; i < rows; i++ {
					dot += cmplx.Conj(col[i]) * vec[i]
				}
				for i := 0; i < rows; i++ {
					vec[i] -= dot * col[i]
				}
			}
		}
		norm := vectorNorm(vec)
		if norm > 1e-8 {
			normalizeVector(vec, norm)
			copy(u[j*rows:(j+1)*rows], vec)
			return k + 1, true
		}
	}
	return start, false
}
