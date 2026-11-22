package np

// copyMatrix copies native elements to a non-overlapping contiguous matrix.
// Tiling makes C↔Fortran copies visit nearby source and destination cache
// lines instead of traversing the entire source column for each output row.
func copyMatrix[T Number](dst, src []T, rows, cols int, do, so int, ds, ss [2]int) {
	if ds[1] != 1 && ds[0] == 1 {
		rows, cols = cols, rows
		ds[0], ds[1] = ds[1], ds[0]
		ss[0], ss[1] = ss[1], ss[0]
	}
	if ss[1] == 1 && ds[1] == 1 {
		for i := 0; i < rows; i++ {
			d, s := do+i*ds[0], so+i*ss[0]
			copy(dst[d:d+cols], src[s:s+cols])
		}
		return
	}
	const tile = 32
	for i := 0; i < rows; i += tile {
		for j := 0; j < cols; j += tile {
			end := min(j+tile, cols)
			for r := i; r < min(i+tile, rows); r++ {
				d, s := do+r*ds[0]+j*ds[1], so+r*ss[0]+j*ss[1]
				for c := j; c < end; c++ {
					dst[d] = src[s]
					d += ds[1]
					s += ss[1]
				}
			}
		}
	}
}

func typedMatrixCopy[T Number](dst, src []T, d, s *Array) {
	copyMatrix(dst, src, d.shape[0], d.shape[1], d.offset, s.offset,
		[2]int{d.strides[0], d.strides[1]}, [2]int{s.strides[0], s.strides[1]})
}

// copyNative requires matching shape/dtype and independent storage (or a
// prior copy). A possibly overlapping destination is visited in C order.
func copyNative(dst, src *Array) {
	if dst.size == 0 {
		return
	}
	dc, sc := dst.IsCContiguous(), src.IsCContiguous()
	df, sf := dst.IsFContiguous(), src.IsFContiguous()
	width := dst.ItemSize()
	if dc && sc || df && sf {
		d, s := rawStorageBytes(dst.data), rawStorageBytes(src.data)
		copy(d[dst.offset*width:(dst.offset+dst.size)*width], s[src.offset*width:(src.offset+src.size)*width])
		return
	}
	if dst.NDim() == 2 && (dc || df) {
		switch d := storageSlice(dst.data).(type) {
		case []float64:
			typedMatrixCopy(d, storageSlice(src.data).([]float64), dst, src)
		case []float32:
			typedMatrixCopy(d, storageSlice(src.data).([]float32), dst, src)
		case []complex128:
			typedMatrixCopy(d, storageSlice(src.data).([]complex128), dst, src)
		case []complex64:
			typedMatrixCopy(d, storageSlice(src.data).([]complex64), dst, src)
		case []bool:
			typedMatrixCopy(d, storageSlice(src.data).([]bool), dst, src)
		case []int8:
			typedMatrixCopy(d, storageSlice(src.data).([]int8), dst, src)
		case []int16:
			typedMatrixCopy(d, storageSlice(src.data).([]int16), dst, src)
		case []int32:
			typedMatrixCopy(d, storageSlice(src.data).([]int32), dst, src)
		case []int64:
			typedMatrixCopy(d, storageSlice(src.data).([]int64), dst, src)
		case []uint8:
			typedMatrixCopy(d, storageSlice(src.data).([]uint8), dst, src)
		case []uint16:
			typedMatrixCopy(d, storageSlice(src.data).([]uint16), dst, src)
		case []uint32:
			typedMatrixCopy(d, storageSlice(src.data).([]uint32), dst, src)
		case []uint64:
			typedMatrixCopy(d, storageSlice(src.data).([]uint64), dst, src)
		case []Half:
			typedMatrixCopy(d, storageSlice(src.data).([]Half), dst, src)
		}
		return
	}
	d, s := rawStorageBytes(dst.data), rawStorageBytes(src.data)
	if dc {
		src.walk(func(i, off int) { j := dst.offset + i; copy(d[j*width:(j+1)*width], s[off*width:(off+1)*width]) })
	} else {
		dst.walk(func(i, off int) {
			j := src.offset + i
			if !sc {
				j = src.offset + flatOffset(i, src.shape, src.strides)
			}
			copy(d[off*width:(off+1)*width], s[j*width:(j+1)*width])
		})
	}
}

func nativeNumericStorage(a *Array) bool {
	return a.DType().IsNativeNumeric() && a.data.part == nil && a.data.buffer == nil
}
