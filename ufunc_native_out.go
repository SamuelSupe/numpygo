package np

// Policy-free numeric kernels can write directly after callElementwise
// has validated shapes and casting.
func nativeFloatOp(u *UFunc) (op, bool) {
	if u.NIn != 2 || u.binary == nil || u.nativeOp == 0 {
		return 0, false
	}
	return op(u.nativeOp - 1), true
}

func nativeFloatOpName(name string) (op, bool) {
	switch name {
	case "add":
		return addOp, true
	case "subtract":
		return subOp, true
	case "multiply":
		return mulOp, true
	case "divide":
		return divOp, true
	}
	return 0, false
}

func constantView(a *Array) bool {
	for i, n := range a.shape {
		if n > 1 && a.strides[i] != 0 {
			return false
		}
	}
	return true
}

func exactPackedAlias(out, a *Array) bool {
	return a.IsCContiguous() && a.size == out.size && a.DataIdentity() == out.DataIdentity()
}

// tryNativeFloatOut accepts already broadcast inputs. Exact elementwise
// aliasing is safe; shifted or broadcast aliases require the old snapshot.
func tryNativeFloatOut(u *UFunc, d DType, out, where *Array, args []*Array) bool {
	o, ok := nativeFloatOp(u)
	if !ok || d != Float32 && d != Float64 {
		return false
	}
	mask, skip, ok := prepareNativeOut(d, out, where, args)
	if !ok || skip {
		return ok
	}
	a, b := args[0], args[1]
	if d == Float64 {
		nativeFloatTo(out.data.real[out.offset:out.offset+out.size], a.data.real, b.data.real, a, b, mask, o)
	} else {
		z := out.data.raw.([]float32)
		nativeFloatTo(z[out.offset:out.offset+out.size], a.data.raw.([]float32), b.data.raw.([]float32), a, b, mask, o)
	}
	return true
}

// Shared eligibility check: a native same-dtype destination is safe from
// read-after-write hazards only for disjoint arrays or exact packed aliases.
func prepareNativeOut(d DType, out, where *Array, args []*Array) (mask []bool, skip, ok bool) {
	if out.DType() != d || !nativeNumericStorage(out) || !out.IsCContiguous() {
		return nil, false, false
	}
	for _, x := range args {
		if x.DType() != d || !nativeNumericStorage(x) || MayShareMemory(out, x) && !exactPackedAlias(out, x) {
			return nil, false, false
		}
	}
	if out.size == 0 {
		return nil, true, true
	}
	if where != nil {
		m, e := where.BroadcastTo(out.shape...)
		if e != nil || !nativeNumericStorage(m) || MayShareMemory(out, m) {
			return nil, false, false
		}
		raw := m.data.raw.([]bool)
		if constantView(m) {
			if !raw[m.offset] {
				return nil, true, true
			}
		} else if m.IsCContiguous() {
			mask = raw[m.offset : m.offset+m.size]
		} else {
			return nil, false, false
		}
	}
	return mask, false, true
}

func nativeFloatTo[T float32 | float64](dst, x, y []T, a, b *Array, mask []bool, o op) {
	as, bs := 1, 1
	ac, bc := a.IsCContiguous(), b.IsCContiguous()
	if !ac && constantView(a) {
		as = 0
		ac = true
	}
	if !bc && constantView(b) {
		bs = 0
		bc = true
	}
	if ac && bc {
		if mask == nil {
			nx, ny := 1+(len(dst)-1)*as, 1+(len(dst)-1)*bs
			denseLoop(dst, x[a.offset:a.offset+nx], y[b.offset:b.offset+ny], as, bs, o)
		} else {
			maskedRealLane(dst, x, y, mask, a.offset, b.offset, as, bs, o)
		}
		return
	}
	if mask == nil {
		stridedRealLoop(dst, x, y, a, b, a.shape, o)
		return
	}
	ax := a.NDim() - 1
	width := a.shape[ax]
	for row := 0; row < len(dst)/width; row++ {
		ai := a.offset + flatOffset(row, a.shape[:ax], a.strides[:ax])
		bi := b.offset + flatOffset(row, b.shape[:ax], b.strides[:ax])
		maskedRealLane(dst[row*width:(row+1)*width], x, y, mask[row*width:(row+1)*width], ai, bi, a.strides[ax], b.strides[ax], o)
	}
}

// Evaluate only true runs; long contiguous runs reuse the SIMD lane
// implementation and false positions are neither evaluated nor written.
func maskedRealLane[T float32 | float64](dst, x, y []T, mask []bool, ai, bi, as, bs int, o op) {
	for i := 0; i < len(mask); {
		if !mask[i] {
			i++
			continue
		}
		j := i + 1
		for j < len(mask) && mask[j] {
			j++
		}
		if j == i+1 {
			// Alternating masks otherwise pay for a lane dispatch per
			// selected value. An isolated true value is a scalar operation.
			a, b := x[ai+i*as], y[bi+i*bs]
			switch o {
			case addOp:
				dst[i] = a + b
			case subOp:
				dst[i] = a - b
			case mulOp:
				dst[i] = a * b
			case divOp:
				dst[i] = a / b
			}
			i = j
			continue
		}
		realLane(dst[i:j], x, y, ai+i*as, bi+i*bs, as, bs, o)
		i = j
	}
}
