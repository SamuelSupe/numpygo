package linalg

import (
	"math"
	"math/cmplx"
	"slices"
	"sort"

	"github.com/SamuelSupe/numpygo"
	"gonum.org/v1/gonum/lapack/lapack64"
)

func hermitianSVD(a *np.Array, opt SVDOptions) (u, s, vh *np.Array, err error) {
	l, v, e := eigh(a, !opt.ValuesOnly)
	if e != nil {
		return nil, nil, nil, e
	}
	shape := a.Shape()
	n := shape[len(shape)-1]
	ls, _ := l.Float64s()
	if l.DType() != np.Float64 {
		w, e := l.Astype(np.Float64)
		if e != nil {
			return nil, nil, nil, e
		}
		ls, _ = w.Float64s()
	}
	sv := make([]float64, len(ls))
	var uv, vv, ev []complex128
	if !opt.ValuesOnly {
		ev, _ = v.Complex128s()
		uv = make([]complex128, len(ev))
		vv = make([]complex128, len(ev))
	}
	order := make([]int, n)
	for off := 0; n > 0 && off < len(ls); off += n {
		for j := range order {
			order[j] = j
		}
		sort.SliceStable(order, func(i, j int) bool { return math.Abs(ls[off+order[i]]) > math.Abs(ls[off+order[j]]) })
		mo := off * n
		for j, k := range order {
			sv[off+j] = math.Abs(ls[off+k])
			if opt.ValuesOnly {
				continue
			}
			sign := complex(math.Copysign(1, ls[off+k]), 0)
			for i := 0; i < n; i++ {
				z := ev[mo+i*n+k]
				uv[mo+i*n+j] = z
				vv[mo+j*n+i] = sign * cmplx.Conj(z)
			}
		}
	}
	s, e = np.WrapFloat64(sv, l.Shape()...)
	if e != nil {
		return nil, nil, nil, e
	}
	if l.DType() != np.Float64 {
		s, e = s.Astype(l.DType())
		if e != nil {
			return nil, nil, nil, e
		}
	}
	if opt.ValuesOnly {
		return nil, s, nil, nil
	}
	u, e = asComplexArray(uv, v.DType(), shape)
	if e != nil {
		return nil, nil, nil, e
	}
	vh, e = asComplexArray(vv, v.DType(), shape)
	return u, s, vh, e
}

// QRRaw returns the LAPACK packed Householder reflectors in transposed
// storage (batch...,N,M) and tau (batch...,min(M,N)). NumPy 2.5
// returns the decomposition's result precision, including float32/complex64.
func QRRaw(a *np.Array) (h, tau *np.Array, err error) {
	d, e := linearDType(a.DType())
	if e != nil {
		return nil, nil, e
	}
	if a.NDim() < 2 {
		return nil, nil, np.ErrShape
	}
	shape := a.Shape()
	batch := shape[:len(shape)-2]
	m, n := shape[len(shape)-2], shape[len(shape)-1]
	k := min(m, n)
	count, e := np.ElementCount(d, batch...)
	if e != nil {
		return nil, nil, e
	}
	hs := append(slices.Clone(batch), n, m)
	ts := append(slices.Clone(batch), k)
	hsize, e := np.ElementCount(d, hs...)
	if e != nil {
		return nil, nil, e
	}
	tsize, e := np.ElementCount(d, ts...)
	if e != nil {
		return nil, nil, e
	}
	av, e := a.Complex128s()
	if e != nil {
		return nil, nil, e
	}
	hv := make([]complex128, hsize)
	tv := make([]complex128, tsize)
	if k > 0 && count > 0 {
		// Query the LAPACK workspace once, then reuse it for all real batches.
		var packed, rtau, work []float64
		if !d.IsComplex() {
			packed = make([]float64, m*n)
			rtau = make([]float64, k)
			work = make([]float64, 1)
			lapack64.Geqrf(matrix(m, n, packed), rtau, work, -1)
			work = make([]float64, max(int(work[0]), max(m, n)))
		}
		for q := 0; q < count; q++ {
			src := av[q*m*n : (q+1)*m*n]
			t := tv[q*k : (q+1)*k]
			if d.IsComplex() {
				packedComplexQR(src, m, n, t)
			} else {
				for i, z := range src {
					packed[i] = real(z)
				}
				lapack64.Geqrf(matrix(m, n, packed), rtau, work, len(work))
				for i, z := range packed {
					src[i] = complex(z, 0)
				}
				for i, z := range rtau {
					t[i] = complex(z, 0)
				}
			}
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					hv[q*m*n+j*m+i] = src[i*n+j]
				}
			}
		}
	}
	h, e = asComplexArray(hv, d, hs)
	if e != nil {
		return nil, nil, e
	}
	tau, e = asComplexArray(tv, d, ts)
	return h, tau, e
}

func packedComplexQR(a []complex128, m, n int, tau []complex128) {
	clear(tau)
	fast := m >= 32 && n >= 32
	if fast {
		limit := math.MaxFloat64 / (32 * float64(m) * float64(n))
		for _, z := range a {
			if !(math.Abs(real(z)) <= limit && math.Abs(imag(z)) <= limit) {
				fast = false
				break
			}
		}
	}
	var v, work []complex128
	if fast {
		v, work = make([]complex128, m), make([]complex128, n)
	}
	for j := 0; j < min(m, n); j++ {
		alpha := a[j*n+j]
		var norm float64
		for i := j + 1; i < m; i++ {
			norm = math.Hypot(norm, cmplx.Abs(a[i*n+j]))
		}
		if norm == 0 && imag(alpha) == 0 {
			continue
		}
		beta := -math.Copysign(math.Hypot(cmplx.Abs(alpha), norm), real(alpha))
		ar := complex(real(alpha)/beta, imag(alpha)/beta)
		t := 1 - ar
		tau[j] = t
		for i := j + 1; i < m; i++ {
			z := a[i*n+j]
			a[i*n+j] = complex(real(z)/beta, imag(z)/beta) / (ar - 1)
		}
		a[j*n+j] = complex(beta, 0)
		if fast {
			u := v[:m-j]
			u[0] = 1
			for i := j + 1; i < m; i++ {
				u[i-j] = a[i*n+j]
			}
			applyHouseholderLeft(a, n, j, j+1, u, cmplx.Conj(t), work, true)
			continue
		}
		for col := j + 1; col < n; col++ {
			dot := a[j*n+col]
			for i := j + 1; i < m; i++ {
				dot += cmplx.Conj(a[i*n+j]) * a[i*n+col]
			}
			dot *= cmplx.Conj(t)
			a[j*n+col] -= dot
			for i := j + 1; i < m; i++ {
				a[i*n+col] -= a[i*n+j] * dot
			}
		}
	}
}
