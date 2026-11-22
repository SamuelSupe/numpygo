package np

import (
	"bufio"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/SamuelSupe/numpygo/internal/x87"
)

func parseNumber(s string, d DType) (any, error) {
	s = strings.TrimSpace(s)
	if d == Bool {
		switch strings.ToLower(s) {
		case "true":
			return true, nil
		case "false":
			return false, nil
		}
		v, e := strconv.ParseFloat(s, 64)
		return v != 0, e
	}
	if d.IsSigned() {
		return strconv.ParseInt(s, 10, d.ItemSize()*8)
	}
	if d.IsUnsigned() {
		return strconv.ParseUint(s, 10, d.ItemSize()*8)
	}
	if d == Float128 {
		return x87.Parse(s)
	}
	if d.IsFloat() {
		return strconv.ParseFloat(s, 64)
	}
	if d.IsComplex() {
		s = strings.ReplaceAll(strings.ReplaceAll(s, "J", "i"), "j", "i")
		if strings.HasPrefix(s, "(") && strings.HasSuffix(s, ")") {
			s = s[1 : len(s)-1]
		}
		switch s {
		case "i", "+i":
			s = "1i"
		case "-i":
			s = "-1i"
		}
		return strconv.ParseComplex(s, 128)
	}
	return nil, ErrDType
}

// ParseExtendedComplex is the lossless Go parser for independently
// extended real and imaginary literals. NumPy's complex text conversions
// use binary64 components; this is an explicit higher-precision alternative.
func ParseExtendedComplex(s string) (x87.Complex, error) {
	s = strings.TrimSpace(s)
	if strings.HasPrefix(s, "(") && strings.HasSuffix(s, ")") {
		s = strings.TrimSpace(s[1 : len(s)-1])
	}
	if !strings.HasSuffix(s, "i") && !strings.HasSuffix(s, "j") && !strings.HasSuffix(s, "I") && !strings.HasSuffix(s, "J") {
		f, e := x87.Parse(s)
		return x87.Complex{Real: f}, e
	}
	s = s[:len(s)-1]
	cut := -1
	for i := 1; i < len(s); i++ {
		if (s[i] == '+' || s[i] == '-') && s[i-1] != 'e' && s[i-1] != 'E' {
			cut = i
		}
	}
	real, imag := "0", s
	if cut >= 0 {
		real, imag = s[:cut], s[cut:]
	}
	switch imag {
	case "", "+":
		imag = "1"
	case "-":
		imag = "-1"
	}
	return x87.ParseComplex(real, imag)
}
func textFields(line, delimiter string) []string {
	if delimiter == "" || strings.TrimSpace(delimiter) == "" {
		return strings.Fields(line)
	}
	return strings.Split(line, delimiter)
}

// FromString parses an explicitly separated numeric sequence.
func FromString(text string, d DType, separator string, count int) (*Array, error) {
	if d.ItemSize() == 0 {
		return nil, ErrDType
	}
	if d == Complex256 {
		return nil, ErrValue
	}
	if count < -1 || separator == "" {
		return nil, ErrValue
	}
	fields := textFields(strings.TrimSpace(text), separator)
	if len(fields) == 1 && strings.TrimSpace(fields[0]) == "" {
		fields = nil
	}
	if count >= 0 {
		if len(fields) < count {
			return nil, ErrValue
		}
		fields = fields[:count]
	}
	out, e := Zeros(d, len(fields))
	if e != nil {
		return nil, e
	}
	for i, s := range fields {
		v, e := parseNumber(s, d)
		if e != nil {
			return nil, fmt.Errorf("%w: invalid numeric token %q", ErrValue, s)
		}
		if e = out.SetValue(v, i); e != nil {
			return nil, e
		}
	}
	return out, nil
}

type TextOptions struct {
	DType             DType
	Delimiter         string
	Comment           string // empty uses NumPy's default "#"
	SkipRows, MaxRows int
	UseCols           []int
	Unpack            bool
	NDMin             int
	MissingValues     []string
	FillingValue      any
	QuoteChar         rune
	Converter         func(string) (any, error)
	Converters        map[int]func(string) (any, error)
}

// splitTextLine strips comments only outside quoted fields and supports
// doubled quote escapes. Whitespace delimiters collapse consecutive spaces.
func splitTextLine(line, delimiter, comment string, quote rune) ([]string, error) {
	if quote == 0 {
		if i := strings.Index(line, comment); comment != "" && i >= 0 {
			line = line[:i]
		}
		if strings.TrimSpace(line) == "" {
			return nil, nil
		}
		return textFields(strings.TrimSpace(line), delimiter), nil
	}
	var fields []string
	var field strings.Builder
	q := string(quote)
	whitespace := delimiter == "" || strings.TrimSpace(delimiter) == ""
	inQuote, started := false, false
	for i := 0; i < len(line); {
		if strings.HasPrefix(line[i:], q) {
			if inQuote && strings.HasPrefix(line[i+len(q):], q) {
				field.WriteString(q)
				i += 2 * len(q)
				continue
			}
			inQuote = !inQuote
			started = true
			i += len(q)
			continue
		}
		if !inQuote && comment != "" && strings.HasPrefix(line[i:], comment) {
			break
		}
		ch, sz := utf8.DecodeRuneInString(line[i:])
		if !inQuote && (whitespace && unicode.IsSpace(ch) || !whitespace && strings.HasPrefix(line[i:], delimiter)) {
			if started || !whitespace {
				fields = append(fields, field.String())
			}
			field.Reset()
			started = false
			if whitespace {
				i += sz
			} else {
				i += len(delimiter)
			}
			continue
		}
		field.WriteString(line[i : i+sz])
		started = true
		i += sz
	}
	if inQuote {
		return nil, ErrValue
	}
	if started || !whitespace && len(fields) > 0 {
		fields = append(fields, field.String())
	}
	return fields, nil
}

func loadText(r io.Reader, opt TextOptions, allowMissing bool) (*Array, error) {
	d := opt.DType
	if d == 0 {
		d = Float64
	}
	if d.ItemSize() == 0 {
		return nil, ErrDType
	}
	if opt.SkipRows < 0 || opt.MaxRows < 0 || opt.NDMin < 0 || opt.NDMin > 2 {
		return nil, ErrValue
	}
	comment := opt.Comment
	if comment == "" {
		comment = "#"
	}
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 4096), 64*1024*1024)
	rows, cols, lineNum := 0, -1, 0
	var values []any
	for scanner.Scan() {
		lineNum++
		if lineNum <= opt.SkipRows {
			continue
		}
		line := scanner.Text()
		fields, e := splitTextLine(line, opt.Delimiter, comment, opt.QuoteChar)
		if e != nil {
			return nil, fmt.Errorf("%w: line %d", e, lineNum)
		}
		if len(fields) == 0 {
			continue
		}
		columnIndices := make([]int, len(fields))
		for i := range columnIndices {
			columnIndices[i] = i
		}
		if opt.UseCols != nil {
			selected := make([]string, len(opt.UseCols))
			columnIndices = make([]int, len(opt.UseCols))
			for i, col := range opt.UseCols {
				if col < 0 {
					col += len(fields)
				}
				if col < 0 || col >= len(fields) {
					return nil, ErrIndex
				}
				selected[i] = fields[col]
				columnIndices[i] = col
			}
			fields = selected
		}
		if cols < 0 {
			cols = len(fields)
		}
		if len(fields) != cols {
			return nil, ErrShape
		}
		for col, s := range fields {
			s = strings.TrimSpace(s)
			missing := s == ""
			for _, v := range opt.MissingValues {
				missing = missing || s == v
			}
			var v any
			var e error
			if allowMissing && missing {
				v = opt.FillingValue
				if v == nil {
					switch {
					case d.IsFloat():
						v = math.NaN()
					case d.IsComplex():
						v = complex(math.NaN(), 0)
					case d == Bool:
						v = false
					default:
						v = int64(-1)
					}
				}
			} else {
				converter := opt.Converters[columnIndices[col]]
				if converter == nil {
					converter = opt.Converter
				}
				if converter != nil {
					v, e = converter(s)
				} else {
					v, e = parseNumber(s, d)
				}
				if e != nil {
					return nil, fmt.Errorf("%w: line %d token %q", ErrValue, lineNum, s)
				}
			}
			values = append(values, v)
		}
		rows++
		if opt.MaxRows > 0 && rows == opt.MaxRows {
			break
		}
	}
	if e := scanner.Err(); e != nil {
		return nil, e
	}
	if cols < 0 {
		cols = 0
		if opt.UseCols != nil {
			cols = len(opt.UseCols)
		}
	}
	out, e := Zeros(d, rows, cols)
	if e != nil {
		return nil, e
	}
	for i, v := range values {
		from := dtypeOf(v)
		if from == 0 {
			return nil, ErrDType
		}
		out.setElement(i, castElement(elementFrom(v), from, d))
	}
	if opt.Unpack {
		out, e = out.Transpose()
		if e != nil {
			return nil, e
		}
	}
	if opt.NDMin < 2 {
		if out.size == 0 {
			return out.Reshape(0)
		}
		out, e = Squeeze(out)
		if e != nil {
			return nil, e
		}
		if opt.NDMin == 1 && out.NDim() == 0 {
			return out.Reshape(1)
		}
	}
	return out, nil
}
func LoadText(r io.Reader, opt TextOptions) (*Array, error)    { return loadText(r, opt, false) }
func GenFromText(r io.Reader, opt TextOptions) (*Array, error) { return loadText(r, opt, true) }

type SaveTextOptions struct {
	Format, Delimiter, Newline string
	Header, Footer, Comments   string
}

func textNumber(value any, format string) (string, error) {
	d := dtypeOf(value)
	if d == 0 {
		return "", ErrDType
	}
	if format == "" {
		format = "%.18e"
	}
	e := elementFrom(value)
	if !strings.HasPrefix(format, "%") || len(format) < 2 {
		return "", ErrValue
	}
	// Format a single numeric conversion, matching printf's numeric coercion.
	verb := format[len(format)-1]
	if d.IsComplex() {
		if !strings.ContainsRune("eEfFgG", rune(verb)) {
			return "", ErrValue
		}
		r := fmt.Sprintf(format, real(e.z))
		i := fmt.Sprintf(format, imag(e.z))
		if strings.HasPrefix(i, "-") {
			return " (" + r + i + "j)", nil
		}
		return " (" + r + "+" + i + "j)", nil
	}
	if strings.ContainsRune("diuxXo", rune(verb)) {
		if verb == 'i' || verb == 'u' {
			format = format[:len(format)-1] + "d"
		}
		if d.IsUnsigned() {
			return fmt.Sprintf(format, e.i), nil
		}
		if d.IsSigned() || d == Bool {
			return fmt.Sprintf(format, int64(e.i)), nil
		}
		return fmt.Sprintf(format, int64(real(e.z))), nil
	}
	if strings.ContainsRune("eEfFgG", rune(verb)) {
		x := real(e.number(d))
		if math.IsNaN(x) {
			return "nan", nil
		}
		if math.IsInf(x, 1) {
			return "inf", nil
		}
		if math.IsInf(x, -1) {
			return "-inf", nil
		}
		return fmt.Sprintf(format, x), nil
	}
	return "", ErrValue
}
func SaveText(w io.Writer, a *Array, opt SaveTextOptions) error {
	if a.DType() == 0 || a.NDim() < 1 || a.NDim() > 2 {
		return ErrShape
	}
	delimiter := opt.Delimiter
	if delimiter == "" {
		delimiter = " "
	}
	newline := opt.Newline
	if newline == "" {
		newline = "\n"
	}
	comments := opt.Comments
	if comments == "" {
		comments = "# "
	}
	bw := bufio.NewWriter(w)
	writeHeader := func(s string) error {
		if s == "" {
			return nil
		}
		for _, line := range strings.Split(s, "\n") {
			if _, e := bw.WriteString(comments + line + newline); e != nil {
				return e
			}
		}
		return nil
	}
	if e := writeHeader(opt.Header); e != nil {
		return e
	}
	rows, cols := a.shape[0], 1
	if a.NDim() == 2 {
		cols = a.shape[1]
	}
	for i := 0; i < rows; i++ {
		line := make([]string, cols)
		for j := 0; j < cols; j++ {
			var v any
			var e error
			if a.NDim() == 1 {
				v, e = a.AtValue(i)
			} else {
				v, e = a.AtValue(i, j)
			}
			if e != nil {
				return e
			}
			line[j], e = textNumber(v, opt.Format)
			if e != nil {
				return e
			}
		}
		if _, e := bw.WriteString(strings.Join(line, delimiter) + newline); e != nil {
			return e
		}
	}
	if e := writeHeader(opt.Footer); e != nil {
		return e
	}
	return bw.Flush()
}
