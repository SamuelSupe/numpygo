package np

func (it *NDIterator) bufferCapacity() int {
	if it.opt.BufferSize > 0 {
		return it.opt.BufferSize
	}
	return 8192
}
func (it *NDIterator) ensureBuffer() error {
	if len(it.buffers) > 0 && it.pos >= it.bufferStart && it.pos < it.bufferEnd {
		return nil
	}
	if e := it.flushBuffer(); e != nil {
		return e
	}
	n := min(it.bufferCapacity(), it.end-it.pos)
	buffers := make([]*Array, len(it.views))
	for i, v := range it.views {
		var b *Array
		var e error
		if it.dtypes[i] == v.DType() {
			b, e = v.zerosLike([]int{n})
		} else {
			b, e = Zeros(it.dtypes[i], n)
		}
		if e != nil {
			return e
		}
		if it.access[i] != "writeonly" {
			for j := 0; j < n; j++ {
				off := v.offset + flatOffset(it.pos+j, v.shape, v.strides)
				x := v.elementAt(off)
				if b.DType() != v.DType() {
					x = castElement(x, v.DType(), b.DType())
				}
				b.setElement(j, x)
			}
		}
		buffers[i] = b
	}
	it.buffers = buffers
	it.bufferStart, it.bufferEnd = it.pos, it.pos+n
	return nil
}
func (it *NDIterator) flushBuffer() error {
	if len(it.buffers) == 0 {
		return nil
	}
	for i, b := range it.buffers {
		if it.access[i] == "readonly" {
			continue
		}
		v := it.views[i]
		if !v.Writable() {
			return ErrReadOnly
		}
		for j := 0; j < b.Size(); j++ {
			if len(it.opt.WriteMasked) > 0 && it.opt.WriteMasked[i] {
				mask := it.buffers[*it.opt.MaskOperand]
				if mask.elementAt(j).i == 0 {
					continue
				}
			}
			off := v.offset + flatOffset(it.bufferStart+j, v.shape, v.strides)
			x := b.elementAt(j)
			if b.DType() != v.DType() {
				x = castElement(x, b.DType(), v.DType())
			}
			v.setElement(off, x)
		}
	}
	it.buffers = nil
	it.bufferStart, it.bufferEnd = 0, 0
	return nil
}
