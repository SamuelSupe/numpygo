package lib

import (
	"errors"
	"strings"

	"github.com/SamuelSupe/numpygo"
)

var ErrDocConflict = errors.New("object already has a different docstring")

// DocTarget holds runtime documentation for an explicitly registered Go
// symbol. It does not mutate the compiler's source documentation or ABI.
type DocTarget struct {
	Documentation           *string
	ReadOnly, SourceDefined bool
	Members                 map[string]*DocTarget
}

type DocRegistry map[string]map[string]*DocTarget

type DocEntry struct {
	Attribute, Text string
}

type DocOptions struct {
	NoWarn bool
	Warn   func(string)
}

// AddDocString installs documentation once. Repeating the same text is
// idempotent; conflicting documentation is never overwritten.
func AddDocString(target *DocTarget, doc string) error {
	if target == nil {
		return np.ErrDType
	}
	if target.Documentation != nil {
		if *target.Documentation == doc {
			return nil
		}
		return ErrDocConflict
	}
	if target.ReadOnly {
		return np.ErrDType
	}
	target.Documentation = &doc
	return nil
}

// AddNewDoc resolves a registered module/symbol, then documents either that
// symbol or one or more attributes. Like NumPy, failures attaching text are
// ignored, but a missing target/attribute is reported. Caller synchronization
// is required for concurrent mutation of the same registry.
func AddNewDoc(registry DocRegistry, module, name string, doc any, opt DocOptions) error {
	target := registry[module][name]
	if target == nil {
		return np.ErrIndex
	}
	attach := func(obj *DocTarget, label, text string) {
		if !opt.NoWarn && opt.Warn != nil && obj.SourceDefined {
			opt.Warn(label)
		}
		_ = AddDocString(obj, cleanDoc(text))
	}
	entry := func(e DocEntry) error {
		attr := target.Members[e.Attribute]
		if attr == nil {
			return np.ErrIndex
		}
		attach(attr, module+"."+name+"."+e.Attribute, e.Text)
		return nil
	}
	switch v := doc.(type) {
	case string:
		attach(target, module+"."+name, v)
	case DocEntry:
		return entry(v)
	case []DocEntry:
		for _, e := range v {
			if err := entry(e); err != nil {
				return err
			}
		}
	}
	return nil
}

func cleanDoc(doc string) string {
	// Python inspect.cleandoc expands tabs before measuring indentation.
	var expanded strings.Builder
	col := 0
	for _, r := range doc {
		switch r {
		case '\t':
			n := 8 - col%8
			expanded.WriteString(strings.Repeat(" ", n))
			col += n
		case '\n', '\r':
			expanded.WriteRune(r)
			col = 0
		default:
			expanded.WriteRune(r)
			col++
		}
	}
	lines := strings.Split(expanded.String(), "\n")
	margin := -1
	for _, line := range lines[1:] {
		stripped := strings.TrimLeft(line, " ")
		if strings.TrimSpace(stripped) != "" {
			n := len(line) - len(stripped)
			if margin < 0 || n < margin {
				margin = n
			}
		}
	}
	lines[0] = strings.TrimLeft(lines[0], " ")
	if margin >= 0 {
		for i := 1; i < len(lines); i++ {
			lines[i] = lines[i][min(margin, len(lines[i])):]
		}
	}
	for len(lines) > 0 && lines[0] == "" {
		lines = lines[1:]
	}
	for len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	return strings.Join(lines, "\n")
}
