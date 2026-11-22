package fft

import (
	"slices"

	"github.com/SamuelSupe/numpygo"
)

// NDOptions selects transform lengths and axes. Nil Axes with nil Shape uses
// all axes for FFTN, or the final two for FFT2. A length -1 uses the input axis.
type NDOptions struct {
	Shape, Axes []int
	Norm        Norm
	Out         *np.Array
}

func ndArguments(a *np.Array, options []NDOptions, two, inverseReal bool) ([]int, []int, Norm, error) {
	if a.DType() == 0 {
		return nil, nil, "", np.ErrDType
	}
	if len(options) > 1 {
		return nil, nil, "", np.ErrShape
	}
	var opt NDOptions
	if len(options) == 1 {
		opt = options[0]
	}
	axes := slices.Clone(opt.Axes)
	if opt.Axes == nil {
		count := a.NDim()
		if two {
			count = 2
		} else if opt.Shape != nil {
			count = len(opt.Shape)
		}
		if count > a.NDim() {
			return nil, nil, "", np.ErrAxis
		}
		axes = make([]int, count)
		for i := range axes {
			axes[i] = a.NDim() - count + i
		}
	}
	lengths := slices.Clone(opt.Shape)
	if opt.Shape == nil {
		lengths = make([]int, len(axes))
	}
	if len(axes) != len(lengths) {
		return nil, nil, "", np.ErrShape
	}
	shape := a.Shape()
	for i, axis := range axes {
		ax, err := np.NormalizeAxis(axis, a.NDim())
		if err != nil {
			return nil, nil, "", err
		}
		axes[i] = ax
		if opt.Shape == nil || lengths[i] == -1 {
			lengths[i] = shape[ax]
		}
		if inverseReal && opt.Shape == nil && i == len(axes)-1 {
			lengths[i] = 2 * (shape[ax] - 1)
		}
		if lengths[i] <= 0 {
			return nil, nil, "", np.ErrShape
		}
	}
	return axes, lengths, opt.Norm, nil
}

func FFTN(a *np.Array, o ...NDOptions) (*np.Array, error)   { return nd(a, o, false, false, false) }
func IFFTN(a *np.Array, o ...NDOptions) (*np.Array, error)  { return nd(a, o, false, true, false) }
func FFT2(a *np.Array, o ...NDOptions) (*np.Array, error)   { return nd(a, o, true, false, false) }
func IFFT2(a *np.Array, o ...NDOptions) (*np.Array, error)  { return nd(a, o, true, true, false) }
func RFFTN(a *np.Array, o ...NDOptions) (*np.Array, error)  { return nd(a, o, false, false, true) }
func IRFFTN(a *np.Array, o ...NDOptions) (*np.Array, error) { return nd(a, o, false, true, true) }
func RFFT2(a *np.Array, o ...NDOptions) (*np.Array, error)  { return nd(a, o, true, false, true) }
func IRFFT2(a *np.Array, o ...NDOptions) (*np.Array, error) { return nd(a, o, true, true, true) }

func nd(a *np.Array, options []NDOptions, two, inverse, realTransform bool) (*np.Array, error) {
	axes, lengths, norm, err := ndArguments(a, options, two, inverse && realTransform)
	if err != nil {
		return nil, err
	}
	if realTransform && len(axes) == 0 {
		return nil, np.ErrAxis
	}
	var out *np.Array
	if len(options) == 1 {
		out = options[0].Out
	}
	at := func(i int) Options {
		opt := Options{Axis: np.Int(axes[i]), N: np.Int(lengths[i]), Norm: norm, Out: out}
		if inverse && realTransform && i < len(axes)-1 {
			opt.Out = nil
		}
		return opt
	}
	if realTransform && inverse {
		for i := 0; i < len(axes)-1; i++ {
			a, err = IFFT(a, at(i))
			if err != nil {
				return nil, err
			}
		}
		return IRFFT(a, at(len(axes)-1))
	}
	last := len(axes) - 1
	if realTransform {
		a, err = RFFT(a, at(last))
		if err != nil {
			return nil, err
		}
		last--
		for i := last; i >= 0; i-- {
			a, err = FFT(a, at(i))
			if err != nil {
				return nil, err
			}
		}
		return a, nil
	}
	for i := last; i >= 0; i-- {
		if inverse {
			a, err = IFFT(a, at(i))
		} else {
			a, err = FFT(a, at(i))
		}
		if err != nil {
			return nil, err
		}
	}
	return a, nil
}
func swapNorm(n Norm) (Norm, error) {
	switch n {
	case "", Backward:
		return Forward, nil
	case Forward:
		return Backward, nil
	case Ortho:
		return Ortho, nil
	}
	return "", ErrNorm
}
func HFFT(a *np.Array, options ...Options) (*np.Array, error) {
	if len(options) > 1 {
		return nil, np.ErrShape
	}
	var opt Options
	if len(options) == 1 {
		opt = options[0]
	}
	n, err := swapNorm(opt.Norm)
	if err != nil {
		return nil, err
	}
	opt.Norm = n
	conj, err := np.Conjugate(a)
	if err != nil {
		return nil, err
	}
	if !a.IsCContiguous() {
		c, e := np.EmptyLike(a, conj.DType(), "K")
		if e != nil {
			return nil, e
		}
		if e = c.Assign(nil, conj); e != nil {
			return nil, e
		}
		conj = c
	}
	return IRFFT(conj, opt)
}
func IHFFT(a *np.Array, options ...Options) (*np.Array, error) {
	if len(options) > 1 {
		return nil, np.ErrShape
	}
	var opt Options
	if len(options) == 1 {
		opt = options[0]
	}
	n, err := swapNorm(opt.Norm)
	if err != nil {
		return nil, err
	}
	opt.Norm = n
	out, err := RFFT(a, opt)
	if err != nil {
		return nil, err
	}
	result, err := np.Conjugate(out)
	if err != nil {
		return nil, err
	}
	if err = out.Assign(nil, result); err != nil {
		return nil, err
	}
	return out, nil
}
func FFTFreq(n int, d float64) (*np.Array, error) {
	if n <= 0 || d == 0 {
		return nil, np.ErrValue
	}
	if _, err := np.ElementCount(np.Float64, n); err != nil {
		return nil, err
	}
	v := 1 / (float64(n) * d)
	out := make([]float64, n)
	for i := range out {
		j := i
		if i >= (n+1)/2 {
			j = i - n
		}
		out[i] = float64(j) * v
	}
	return np.WrapSlice(out, n)
}
func RFFTFreq(n int, d float64) (*np.Array, error) {
	if n <= 0 || d == 0 {
		return nil, np.ErrValue
	}
	size := n/2 + 1
	if _, err := np.ElementCount(np.Float64, size); err != nil {
		return nil, err
	}
	v := 1 / (float64(n) * d)
	out := make([]float64, size)
	for i := range out {
		out[i] = float64(i) * v
	}
	return np.WrapSlice(out, size)
}
func FFTShift(a *np.Array, axes ...int) (*np.Array, error)  { return shift(a, false, axes) }
func IFFTShift(a *np.Array, axes ...int) (*np.Array, error) { return shift(a, true, axes) }
func shift(a *np.Array, inverse bool, axes []int) (*np.Array, error) {
	if a.DType() == 0 {
		return nil, np.ErrDType
	}
	if axes == nil {
		axes = make([]int, a.NDim())
		for i := range axes {
			axes[i] = i
		}
	}
	shape := a.Shape()
	for _, axis := range axes {
		ax, err := np.NormalizeAxis(axis, len(shape))
		if err != nil {
			return nil, err
		}
		n := shape[ax] / 2
		if inverse {
			n = -n
		}
		a, err = np.Roll(a, n, &ax)
		if err != nil {
			return nil, err
		}
	}
	return a, nil
}
