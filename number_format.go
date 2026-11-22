package np

import (
	"math/big"
	"strconv"
	"strings"
)

func BaseRepr(n int64, base, padding int) (string, error) {
	if base < 2 || base > 36 || padding < 0 {
		return "", ErrValue
	}
	sign := ""
	var magnitude uint64
	if n < 0 {
		sign = "-"
		magnitude = 0 - uint64(n)
	} else {
		magnitude = uint64(n)
	}
	if magnitude == 0 && padding > 0 {
		return strings.Repeat("0", padding), nil
	}
	return sign + strings.Repeat("0", padding) + strings.ToUpper(strconv.FormatUint(magnitude, base)), nil
}
func BinaryRepr(n int64, width *int) (string, error) {
	if width == nil {
		return strconv.FormatInt(n, 2), nil
	}
	w := *width
	if w < 0 {
		return "", ErrValue
	}
	if n >= 0 {
		s := strconv.FormatInt(n, 2)
		// NumPy permits zero with zero width for its minimal representation.
		if n == 0 && w == 0 {
			return "0", nil
		}
		if len(s) > w {
			return "", ErrValue
		}
		return strings.Repeat("0", w-len(s)) + s, nil
	}
	if w == 0 {
		return "", ErrValue
	}
	v := big.NewInt(n)
	minimum := new(big.Int).Neg(new(big.Int).Lsh(big.NewInt(1), uint(w-1)))
	if v.Cmp(minimum) < 0 {
		return "", ErrValue
	}
	v.Add(v, new(big.Int).Lsh(big.NewInt(1), uint(w)))
	s := v.Text(2)
	return strings.Repeat("0", w-len(s)) + s, nil
}

type FloatFormatOptions struct {
	Precision *int // nil uses shortest round-trip representation
	Fixed     bool // when true, precision is exact rather than a maximum
	Sign      bool
	Trim      string
}

func formatFloat(value float64, form byte, opt FloatFormatOptions) (string, error) {
	trim := opt.Trim
	if trim == "" {
		trim = "k"
	}
	switch trim {
	case "k", ".", "0", "-":
	default:
		return "", ErrValue
	}
	precision := -1
	if opt.Precision != nil {
		precision = *opt.Precision
		if precision < 0 {
			return "", ErrValue
		}
	}
	short := strconv.FormatFloat(value, form, -1, 64)
	if short == "+Inf" {
		return "inf", nil
	}
	if short == "-Inf" {
		return "-inf", nil
	}
	if short == "NaN" {
		return "nan", nil
	}
	if precision >= 0 {
		mantissa := short
		if form == 'e' {
			mantissa = strings.SplitN(short, "e", 2)[0]
		}
		fraction := 0
		if p := strings.IndexByte(mantissa, '.'); p >= 0 {
			fraction = len(mantissa) - p - 1
		}
		if opt.Fixed || fraction > precision {
			short = strconv.FormatFloat(value, form, precision, 64)
		}
	}
	mantissa, exp := short, ""
	if form == 'e' {
		i := strings.IndexByte(short, 'e')
		mantissa, exp = short[:i], short[i:]
	}
	if !strings.ContainsRune(mantissa, '.') {
		mantissa += "."
	}
	switch trim {
	case ".", "0", "-":
		mantissa = strings.TrimRight(mantissa, "0")
		if trim == "0" && strings.HasSuffix(mantissa, ".") {
			mantissa += "0"
		}
		if trim == "-" {
			mantissa = strings.TrimSuffix(mantissa, ".")
		}
	}
	if opt.Sign && !strings.HasPrefix(mantissa, "-") {
		mantissa = "+" + mantissa
	}
	return mantissa + exp, nil
}
func FormatFloatPositional(value float64, opt FloatFormatOptions) (string, error) {
	return formatFloat(value, 'f', opt)
}
func FormatFloatScientific(value float64, opt FloatFormatOptions) (string, error) {
	return formatFloat(value, 'e', opt)
}
