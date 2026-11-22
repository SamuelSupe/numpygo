package np

import (
	"github.com/SamuelSupe/numpygo/internal/x87"
	"math"
)

type PadOptions struct {
	Mode                      string
	ConstantValues, EndValues [][2]any
	StatLength                [][2]int
	ReflectType               string
	// Callback receives each padded one-dimensional writable lane, axis
	// and padding widths. It implements NumPy's callable padding contract.
	Callback func(*Array, [2]int, int) error
}

func PadWithOptions(a *Array, widths [][2]int, opt PadOptions) (*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	rank := a.NDim()
	if len(widths) == 1 && rank > 1 {
		w := widths[0]
		widths = make([][2]int, rank)
		for i := range widths {
			widths[i] = w
		}
	}
	if len(widths) != rank {
		return nil, ErrShape
	}
	for _, n := range []int{len(opt.ConstantValues), len(opt.EndValues), len(opt.StatLength)} {
		if n != 0 && n != 1 && n != rank {
			return nil, ErrShape
		}
	}
	mode := opt.Mode
	if mode == "" {
		mode = "constant"
	}
	if opt.ReflectType != "" && opt.ReflectType != "even" && opt.ReflectType != "odd" {
		return nil, ErrValue
	}
	switch mode {
	case "constant", "edge", "wrap", "reflect", "symmetric", "linear_ramp", "minimum", "maximum", "mean", "median", "empty":
	default:
		if opt.Callback == nil {
			return nil, ErrValue
		}
	}
	shape := a.Shape()
	for i, w := range widths {
		if w[0] < 0 || w[1] < 0 || w[0] > maxInt-shape[i] || w[1] > maxInt-shape[i]-w[0] {
			return nil, ErrShape
		}
		shape[i] += w[0] + w[1]
	}
	out, e := a.zerosLike(shape)
	if e != nil {
		return nil, e
	}
	var zero element
	if mode == "constant" && (a.DType() == RecordDType || a.DType().IsFlexible() || a.DType() == StringDType) {
		v, e := a.zerosLike(nil)
		if e != nil {
			return nil, e
		}
		zero = v.elementAt(0)
	}
	spec := make([]Index, rank)
	for i, w := range widths {
		spec[i] = Range(w[0], w[0]+a.shape[i])
	}
	if e = out.Assign(spec, a); e != nil {
		return nil, e
	}
	pair := func(v [][2]any, axis, side int) any {
		if len(v) == 0 {
			return nil
		}
		if len(v) == 1 {
			return v[0][side]
		}
		return v[axis][side]
	}
	for axis, w := range widths {
		if w[0]+w[1] == 0 {
			continue
		}
		n := a.shape[axis]
		// Later axes still contain uninitialized padding; extend corners
		// only after the corresponding earlier axes have been filled.
		region := make([]Index, rank)
		for j := range region {
			region[j] = Slice{}
			if j > axis {
				region[j] = spec[j]
			}
		}
		v, e := out.Slice(region...)
		if e != nil {
			return nil, e
		}
		if opt.Callback != nil {
			v = out
		}
		r, _, e := reductionAny(v, []SumOptions{{Axes: []int{axis}}})
		if e != nil {
			return nil, e
		}
		length := v.shape[axis]
		if length == 0 || v.size == 0 {
			continue
		}
		if n == 0 && mode != "constant" && mode != "empty" && opt.Callback == nil {
			return nil, ErrValue
		}
		step := v.strides[axis]
		for i := 0; i < v.size/length; i++ {
			base := r.base(v, i)
			lane := &Array{data: v.data, shape: []int{length}, strides: []int{step}, size: length, offset: base, base: v.viewBase()}
			if opt.Callback != nil {
				if e := opt.Callback(lane, w, axis); e != nil {
					return nil, e
				}
				continue
			}
			if mode == "empty" {
				continue
			}
			for side := 0; side < 2; side++ {
				if w[side] == 0 {
					continue
				}
				var fill element
				switch mode {
				case "constant":
					value := pair(opt.ConstantValues, axis, side)
					if value != nil {
						fill, e = v.convertValue(value)
						if e != nil {
							return nil, e
						}
					} else if v.DType() == RecordDType || v.DType().IsFlexible() || v.DType() == StringDType {
						fill = zero
					}
				case "minimum", "maximum", "mean", "median":
					limit := n
					if len(opt.StatLength) > 0 {
						k := axis
						if len(opt.StatLength) == 1 {
							k = 0
						}
						limit = min(limit, opt.StatLength[k][side])
					}
					if limit <= 0 {
						return nil, ErrValue
					}
					b := w[0]
					if side == 1 {
						b += n - limit
					}
					s, e := lane.Slice(Range(b, b+limit))
					if e != nil {
						return nil, e
					}
					var stat *Array
					switch mode {
					case "minimum":
						stat, e = Min(s)
					case "maximum":
						stat, e = Max(s)
					case "mean":
						stat, e = Mean(s)
					case "median":
						stat, e = Median(s)
					}
					if e != nil {
						return nil, e
					}
					x := stat.elementAt(stat.offset)
					if v.DType().IsInteger() && stat.DType().IsFloat() {
						x.z = complex(math.RoundToEven(real(x.z)), 0)
					}
					fill = castElement(x, stat.DType(), v.DType())
				}
				for j := 0; j < w[side]; j++ {
					target := j
					if side == 1 {
						target = w[0] + n + j
					}
					z := fill
					x := target - w[0]
					switch mode {
					case "edge", "wrap", "reflect", "symmetric":
						p := x
						switch mode {
						case "edge":
							p = max(0, min(n-1, x))
						case "wrap":
							p = x % n
							if p < 0 {
								p += n
							}
						case "reflect", "symmetric":
							if n == 1 {
								p = 0
								break
							}
							m := n - 1
							if mode == "symmetric" {
								m = n
							}
							if m > maxInt/2 {
								return nil, ErrShape
							}
							period := 2 * m
							p = x % period
							if p < 0 {
								p += period
							}
							if p >= n {
								p = period - p
								if mode == "symmetric" {
									p--
								}
							}
						}
						z = lane.elementAt(base + (w[0]+p)*step)
						if (mode == "reflect" || mode == "symmetric") && opt.ReflectType == "odd" && n > 1 {
							// Each reflection adds twice the difference between
							// opposite edge values, also for multiple periods.
							m := n - 1
							if mode == "symmetric" {
								m = n
							}
							period := 2 * m
							q := x / period
							rem := x % period
							if rem < 0 {
								rem += period
								q--
							}
							l, r := lane.elementAt(base+w[0]*step), lane.elementAt(base+(w[0]+n-1)*step)
							d := v.DType()
							if d == Bool || !d.IsNativeNumeric() && !d.IsExtended() {
								return nil, ErrDType
							}
							delta, _ := arithmeticScalar(r, l, d, subOp)
							shift := multiplyScalar(delta, castElement(element{i: uint64(int64(2 * q))}, Int64, d), d)
							if rem >= m && (mode == "symmetric" || rem >= n) {
								z, _ = arithmeticScalar(addScalar(r, r, d), z, d, subOp)
							}
							z = addScalar(z, shift, d)
						}
					case "linear_ramp":
						value := pair(opt.EndValues, axis, side)
						if v.DType().IsExtended() {
							d := v.DType()
							var end element
							if value != nil {
								end, e = v.convertValue(value)
								if e != nil {
									return nil, e
								}
							}
							idx, numerator, denominator := w[0], j, w[0]
							if side == 1 {
								idx += n - 1
								numerator = w[1] - 1 - j
								denominator = w[1]
							}
							ratio := x87.FromInt64(int64(numerator)).Div(x87.FromInt64(int64(denominator)))
							f := castElement(element{extra: ratio}, Float128, d)
							delta, _ := arithmeticScalar(lane.elementAt(base+idx*step), end, d, subOp)
							z = addScalar(end, multiplyScalar(delta, f, d), d)
							lane.setElement(base+target*step, z)
							continue
						}
						end := complex128(0)
						if value != nil {
							d := dtypeOf(value)
							if d == 0 {
								return nil, ErrDType
							}
							end = elementFrom(value).number(d)
						}
						idx := w[0]
						frac := float64(j) / float64(w[0])
						if side == 1 {
							idx += n - 1
							frac = float64(w[1]-1-j) / float64(w[1])
						}
						edge := lane.elementAt(base + idx*step).number(v.DType())
						valueZ := end + complex(frac, 0)*(edge-end)
						if v.DType().IsInteger() {
							valueZ = complex(math.Floor(real(valueZ)), 0)
						}
						z = castElement(element{z: valueZ}, Complex128, v.DType())
					}
					lane.setElement(base+target*step, z)
				}
			}
		}
	}
	return out, nil
}
