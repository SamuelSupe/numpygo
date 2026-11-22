package rec

import (
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/dtypes"
	"github.com/SamuelSupe/numpygo/records"
	"unicode/utf8"
)

func scalarDescriptor(v any) (dtypes.Descriptor, error) {
	var name string
	opt := dtypes.Options{}
	switch x := v.(type) {
	case bool:
		name = "BoolDType"
	case int8:
		name = "Int8DType"
	case int16:
		name = "Int16DType"
	case int32:
		name = "Int32DType"
	case int, int64:
		name = "Int64DType"
	case uint8:
		name = "UInt8DType"
	case uint16:
		name = "UInt16DType"
	case uint32:
		name = "UInt32DType"
	case uint, uint64:
		name = "UInt64DType"
	case np.Half:
		name = "Float16DType"
	case float32:
		name = "Float32DType"
	case float64:
		name = "Float64DType"
	case np.ExtendedFloat:
		name = "LongDoubleDType"
	case complex64:
		name = "Complex64DType"
	case complex128:
		name = "Complex128DType"
	case np.ExtendedComplex:
		name = "CLongDoubleDType"
	case string:
		name = "StrDType"
		opt.Width = max(1, utf8.RuneCountInString(x))
	case np.TextScalar:
		name = "StrDType"
		opt.Width = max(1, utf8.RuneCountInString(string(x)))
	case np.BytesScalar:
		name = "BytesDType"
		opt.Width = max(1, len(x))
	case np.VoidScalar:
		name = "VoidDType"
		opt.Width = len(x)
	case np.DateTime:
		name = "DateTime64DType"
		opt.Unit = x.Unit.String()
	case np.TimeDelta:
		name = "TimeDelta64DType"
		opt.Unit = x.Unit.String()
	case *records.Record:
		if x == nil {
			return dtypes.Descriptor{}, np.ErrDType
		}
		name = "VoidDType"
		opt.Schema = x.Schema()
	default:
		return dtypes.Descriptor{}, np.ErrDType
	}
	return dtypes.New(name, opt)
}
func FromRecords(rows [][]any, opt Options) (*records.Array, error) {
	s, e := explicitSchema(opt)
	if e != nil {
		return nil, e
	}
	if s == nil {
		if len(rows) == 0 || len(rows[0]) == 0 {
			return nil, np.ErrShape
		}
		ds := make([]dtypes.Descriptor, len(rows[0]))
		for i, row := range rows {
			if len(row) != len(ds) {
				return nil, np.ErrShape
			}
			for j, v := range row {
				d, e := scalarDescriptor(v)
				if e != nil {
					return nil, e
				}
				if i == 0 {
					ds[j] = d
				} else {
					ds[j], e = dtypes.Promote(ds[j], d)
					if e != nil {
						return nil, e
					}
				}
			}
		}
		types := make([]records.Type, len(ds))
		for i, d := range ds {
			types[i], e = d.RecordType()
			if e != nil {
				return nil, e
			}
		}
		s, e = schemaFromTypes(types, opt)
		if e != nil {
			return nil, e
		}
	}
	shape := opt.Shape
	if shape == nil {
		shape = []int{len(rows)}
	}
	return records.FromRows(s, rows, shape...)
}
