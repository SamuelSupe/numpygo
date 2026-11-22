# Dtypes and data interchange

## Supported dtype families

| NumPy dtype | Go representation |
|:--|:--|
| bool | bool |
| int8/16/32/64 | int8/int16/int32/int64 |
| uint8/16/32/64 | uint8/uint16/uint32/uint64 |
| float16 | np.Half |
| float32/64 | float32/float64 |
| complex64/128 | complex64/complex128 |
| longdouble, float128 (Linux/x86 representation) | np.Float128 / np.ExtendedFloat / extended.Float |
| clongdouble, complex256 (Linux/x86 representation) | np.Complex256 / np.ExtendedComplex / extended.Complex |
| datetime64, timedelta64 | np.DateTime64 / np.TimeDelta64, explicit units |
| S, U, V | Fixed-width bytes, Unicode and opaque values using np.FlexibleDType |
| StringDType | Variable-length UTF-8 strings using Go-managed slots |
| Structured/record | np.RecordDType with records.Schema |
| object_ | np.Object with Go any values |

These share the ndarray representation. Numeric values and descriptors preserve exact storage where supported. A descriptor carries extra metadata such as temporal units, string width, record names, subarray shapes and byte order. The `dtypes`, `records`, `datetime`, `extended` and `objects` packages supply constructors and typed adapters.

Extended precision implements the x87 80-bit value format (64-bit significand and 15-bit exponent), stored in NumPy's 16-byte Linux/x86 long-double layout. It is not IEEE binary128. Arithmetic and transcendental operations use Go high-precision algorithms; the implementation does not depend on the host C long-double ABI.

```go
x := extended.MustParse("1.0000000000000000001")
y := extended.MustParse("0.0000000000000000001")
a, err := np.FromSlice([]np.ExtendedFloat{x, y}, 2)
if err != nil {
    return err
}
sum, err := np.Sum(a)
if err != nil {
    return err
}
_ = sum
```

Structured schemas support packed/aligned fields, nested records, fixed subarrays, supported field titles, fixed-width strings and temporal fields. Field views share the parent storage, including unaligned strides and byte order. `np.PromoteRecordSchemas` and `Array.CastRecords` handle compatible schema promotion. The `rec` and `recfunctions` packages provide record construction and transformation utilities.

`np.FromObjects` uses Go-managed reference slots and shallow reference copying rather than Python objects. Object callbacks require explicit Go functions. `np.FromStrings` creates variable-length string arrays; the `strings` and `char` packages provide modern and legacy string operations. Pointer-managed object and variable-length string arrays cannot be represented as a raw fixed-width memory map.

## NumPy files

Use Go's standard I/O interfaces to read and write numeric arrays:

```go
var buf bytes.Buffer
if err := np.SaveNPY(&buf, a); err != nil {
    return err
}
restored, err := np.LoadNPY(&buf)
if err != nil {
    return err
}
_ = restored
```

`SaveNPZ` and `LoadNPZ` operate on named arrays using a ZIP container. Use `records.LoadNPY`, `records.LoadAnyNPY` and `records.LoadNPZ` for supported structured data. NPY v1/v2/v3 headers, C/F layouts, endianness, fixed-width strings, temporal units, nested schemas and subarray fields are supported within the documented scope.

The [npyformat package](https://pkg.go.dev/github.com/SamuelSupe/numpygo/npyformat) offers explicit header/version functions, layout-preserving reads, NPY memory maps and lazy NPZ archives:

- `ReadArray` preserves supported byte order and C/F layout. The earlier root numeric `LoadNPY` contract normalizes numeric data to native endian.
- `OpenNPZ` / `OpenNPZPath` read members on demand, validate CRCs and do not extract member paths to disk.
- `NewDataSource` supports explicitly requested HTTP(S) sources, local data, gzip and bzip2 reads with a caller-controlled cache.

NPY readers parse dtype literals rather than execute Python. Arbitrary object arrays requiring pickle, overlapping/unsupported structured field layouts and Python-specific metadata are outside the supported interchange contract.

## Raw and text data

`FromBuffer` supports caller-owned byte storage, explicit byte order, offset and count. `View`, `GetField` and `SetField` provide supported shared byte reinterpretations. `ToBytes` returns a copy; `DataBytes` exposes a writable C-contiguous array's byte span when allowed.

The text API includes `LoadText`, `SaveText`, `GenFromText`, delimiters, comments, selected quoting and converter callbacks. Structured text can be read via the records package. Not every Python converter, encoding, dtype inference or quoting combination is supported.

## Memory mapping

```go
mm, err := np.MemMap("data.bin", np.MemMapOptions{
    DType: np.Int32,
    Shape: []int{2, 3},
    Mode: "w+",
    Order: "F",
})
if err != nil {
    return err
}
defer mm.Close()
if err := mm.SetValue(int32(42), 0, 0); err != nil {
    return err
}
if err := mm.Flush(); err != nil {
    return err
}
```

The OS-backed implementation supports Linux/macOS builds, fixed-width numeric and supported flexible/temporal/structured layouts, byte offsets, endian options and C/F order. Modes are read-only (`r`), read/write (`r+`), create/truncate (`w+`) and copy-on-write (`c`). `npyformat.OpenMemMap` handles an NPY header and its actual payload offset.

**Lifetime matters:** close a mapped array explicitly, do not access its views after close, and do not truncate the backing file while it is mapped. Synchronize concurrent access and close operations. `Flush` persists shared writes; copy-on-write does not modify the backing file.
