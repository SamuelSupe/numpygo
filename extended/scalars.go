// Package extended exposes the x87 scalar kernels shared with numpygo.Array.
package extended

import "github.com/SamuelSupe/numpygo/internal/x87"

type Float = x87.Float
type Complex = x87.Complex
type FloatInfo = x87.FloatInfo

func Zero(negative bool) Float                    { return x87.Zero(negative) }
func Inf(negative bool) Float                     { return x87.Inf(negative) }
func NaN() Float                                  { return x87.NaN() }
func FromRaw(b []byte) (Float, error)             { return x87.FromRaw(b) }
func FromFloat64(v float64) Float                 { return x87.FromFloat64(v) }
func FromInt64(v int64) Float                     { return x87.FromInt64(v) }
func FromUint64(v uint64) Float                   { return x87.FromUint64(v) }
func Parse(s string) (Float, error)               { return x87.Parse(s) }
func MustParse(s string) Float                    { return x87.MustParse(s) }
func Info() FloatInfo                             { return x87.Info() }
func FromComplex128(v complex128) Complex         { return x87.FromComplex128(v) }
func ParseComplex(re, im string) (Complex, error) { return x87.ParseComplex(re, im) }
