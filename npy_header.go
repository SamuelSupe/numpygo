package np

import (
	"encoding/binary"
	"fmt"
	"io"
	"strconv"
	"strings"
)

// NPYHeader is the parsed, restricted data-literal metadata of an NPY file.
// Descriptor is a scalar string or a sequence of field tuples (each []any),
// including nested descriptors and integer shape tuples. No Python code is run.
type NPYHeader struct {
	Descriptor   any
	FortranOrder bool
	Shape        []int
}

type NPYVersion struct{ Major, Minor int }

func NPYMagic(version NPYVersion) ([]byte, error) {
	if version.Major < 0 || version.Major > 255 || version.Minor < 0 || version.Minor > 255 {
		return nil, ErrValue
	}
	return []byte(npyMagic + string([]byte{byte(version.Major), byte(version.Minor)})), nil
}
func ReadNPYMagic(r io.Reader) (NPYVersion, error) {
	var prefix [8]byte
	if _, e := io.ReadFull(r, prefix[:]); e != nil {
		return NPYVersion{}, e
	}
	if string(prefix[:6]) != npyMagic {
		return NPYVersion{}, fmt.Errorf("%w: invalid .npy signature", ErrValue)
	}
	return NPYVersion{int(prefix[6]), int(prefix[7])}, nil
}

func ReadNPYHeader(r io.Reader) (NPYHeader, error) {
	version, e := ReadNPYMagic(r)
	if e != nil {
		return NPYHeader{}, e
	}
	return ReadNPYHeaderVersion(r, version, 1<<20)
}

// ReadNPYHeaderVersion reads the length and restricted literal after the
// magic bytes. An explicit header size limit is enforced before allocation.
func ReadNPYHeaderVersion(r io.Reader, version NPYVersion, maxHeaderSize int) (NPYHeader, error) {
	var h NPYHeader
	if version.Minor != 0 {
		return h, ErrValue
	}
	var n uint32
	switch version.Major {
	case 1:
		var v uint16
		if e := binary.Read(r, binary.LittleEndian, &v); e != nil {
			return h, e
		}
		n = uint32(v)
	case 2, 3:
		if e := binary.Read(r, binary.LittleEndian, &n); e != nil {
			return h, e
		}
	default:
		return h, ErrValue
	}
	if maxHeaderSize < 0 || n == 0 || uint64(n) > uint64(maxHeaderSize) {
		return h, fmt.Errorf("%w: invalid .npy header length", ErrShape)
	}
	b := make([]byte, n)
	if _, e := io.ReadFull(r, b); e != nil {
		return h, e
	}
	if b[len(b)-1] != '\n' {
		return h, ErrValue
	}
	text := string(b)
	if version.Major != 3 {
		// Versions 1 and 2 use latin-1, version 3 uses UTF-8.
		runes := make([]rune, len(b))
		for i, v := range b {
			runes[i] = rune(v)
		}
		text = string(runes)
	}
	p := npyLiteral{text: text}
	if !p.accept('{') {
		return h, ErrValue
	}
	seen := map[string]bool{}
	for !p.accept('}') {
		key, e := p.value(0)
		if e != nil {
			return h, e
		}
		name, ok := key.(string)
		if !ok || seen[name] || !p.accept(':') {
			return h, ErrValue
		}
		v, e := p.value(0)
		if e != nil {
			return h, e
		}
		seen[name] = true
		switch name {
		case "descr":
			switch v.(type) {
			case string, []any:
				h.Descriptor = v
			default:
				return h, ErrDType
			}
		case "fortran_order":
			f, ok := v.(bool)
			if !ok {
				return h, ErrValue
			}
			h.FortranOrder = f
		case "shape":
			list, ok := v.([]any)
			if !ok {
				return h, ErrShape
			}
			h.Shape = make([]int, len(list))
			for i, n := range list {
				q, ok := n.(int)
				if !ok || q < 0 {
					return h, ErrShape
				}
				h.Shape[i] = q
			}
		default:
			return h, ErrValue
		}
		if p.accept('}') {
			break
		}
		if !p.accept(',') {
			return h, ErrValue
		}
	}
	p.space()
	if p.pos != len(p.text) || !seen["descr"] || !seen["fortran_order"] || !seen["shape"] {
		return h, ErrValue
	}
	return h, nil
}

// ParseNPYDescriptorLiteral decodes only data-literal descriptor syntax.
func ParseNPYDescriptorLiteral(text string) (any, error) {
	p := npyLiteral{text: text}
	v, e := p.value(0)
	if e != nil {
		return nil, e
	}
	p.space()
	if p.pos != len(p.text) {
		return nil, ErrValue
	}
	return v, nil
}

type npyLiteral struct {
	text string
	pos  int
}

func (p *npyLiteral) space() {
	for p.pos < len(p.text) && strings.ContainsRune(" \r\n\t", rune(p.text[p.pos])) {
		p.pos++
	}
}
func (p *npyLiteral) accept(c byte) bool {
	p.space()
	if p.pos < len(p.text) && p.text[p.pos] == c {
		p.pos++
		return true
	}
	return false
}
func (p *npyLiteral) value(depth int) (any, error) {
	if depth > 64 {
		return nil, ErrValue
	}
	p.space()
	if p.pos == len(p.text) {
		return nil, ErrValue
	}
	c := p.text[p.pos]
	switch c {
	case '\'', '"':
		p.pos++
		var out strings.Builder
		for p.pos < len(p.text) {
			if p.text[p.pos] == c {
				p.pos++
				return out.String(), nil
			}
			v, _, tail, e := strconv.UnquoteChar(p.text[p.pos:], c)
			if e != nil {
				return nil, ErrValue
			}
			out.WriteRune(v)
			p.pos = len(p.text) - len(tail)
		}
		return nil, ErrValue
	case '[', '(':
		p.pos++
		end := byte(']')
		if c == '(' {
			end = ')'
		}
		list := []any{}
		for !p.accept(end) {
			v, e := p.value(depth + 1)
			if e != nil {
				return nil, e
			}
			list = append(list, v)
			if p.accept(end) {
				return list, nil
			}
			if !p.accept(',') {
				return nil, ErrValue
			}
		}
		return list, nil
	case 'T', 'F':
		if strings.HasPrefix(p.text[p.pos:], "True") {
			p.pos += 4
			return true, nil
		}
		if strings.HasPrefix(p.text[p.pos:], "False") {
			p.pos += 5
			return false, nil
		}
		return nil, ErrValue
	default:
		start := p.pos
		if c == '-' {
			p.pos++
		}
		for p.pos < len(p.text) && p.text[p.pos] >= '0' && p.text[p.pos] <= '9' {
			p.pos++
		}
		if p.pos == start {
			return nil, ErrValue
		}
		v, e := strconv.Atoi(p.text[start:p.pos])
		if e != nil {
			return nil, ErrShape
		}
		return v, nil
	}
}
