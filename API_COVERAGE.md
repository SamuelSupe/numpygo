# NumPy 2.5.0 numerical API inventory

The inventory enumerates the pinned version's public callable exports, selected public class members and documented instance attributes. It covers the core, linear algebra, FFT, random, masked array and polynomial namespaces (including six Series classes and polyutils), strings, char, dtypes, rec, and selected public numerical utilities in numpy.lib (scimath/emath, array_utils, Arrayterator, NumpyVersion, npyio, introspect, mixins, recfunctions, stride_tricks and format). It also tracks public members and selected numerical protocols of ufunc, flatiter, broadcast, nditer, ndindex, ndenumerate, poly1d, memmap, MaskedArray/MaskedIterator, chararray, recarray/record, matrix, dtype and numeric scalar classes. Unchanged inherited ndarray members are not counted twice; shared Series exports are counted once under numpy.polynomial.<Class>. Shared descriptor and scalar members are grouped by base class. Private APIs, the Python extension C ABI and unlisted namespaces are excluded. Otherwise, aliases and separate public exports are counted independently.

The inventory includes 40 additional shared numpy.generic members, 5 numpy.void-specific members, 6 public numpy.ma.mrecords exports and 5 MaskedRecords-specific members. Shared inherited implementations are grouped under their base classes.

verified-subset means that the documented behavior has NumPy differential fixtures. verified-go-equivalent means that a Go contract for a Python/C ABI or platform-specific interface has observable-behavior evidence, without cross-runtime binary compatibility. pending indicates that no verified mapping exists. The CSV records each mapping, reference fixtures and scope; these counts do not represent exhaustive coverage of all arguments or behaviors.

- Public symbols/members with verified mappings: 2260 (behavior subsets: 2075; Go equivalents: 185)
- Pending public symbols/members in this inventory: 0
- Total tracked public symbols/members: 2260
- NumPy reference arrays: 31662; error records: 2732.

| Namespace | Verified subsets | Go equivalents | Pending | Total |
|---|---:|---:|---:|---:|
| numpy | 441 | 21 | 0 | 462 |
| numpy.broadcast | 10 | 0 | 0 | 10 |
| numpy.char | 50 | 3 | 0 | 53 |
| numpy.char.chararray | 67 | 0 | 0 | 67 |
| numpy.dtype | 29 | 1 | 0 | 30 |
| numpy.dtypes | 29 | 5 | 0 | 34 |
| numpy.dtypes.StringDType | 0 | 2 | 0 | 2 |
| numpy.emath | 9 | 0 | 0 | 9 |
| numpy.fft | 18 | 0 | 0 | 18 |
| numpy.flatiter | 10 | 0 | 0 | 10 |
| numpy.float16 | 2 | 0 | 0 | 2 |
| numpy.float32 | 2 | 0 | 0 | 2 |
| numpy.float64 | 4 | 0 | 0 | 4 |
| numpy.generic | 59 | 6 | 0 | 65 |
| numpy.int16 | 1 | 0 | 0 | 1 |
| numpy.int32 | 1 | 0 | 0 | 1 |
| numpy.int64 | 1 | 0 | 0 | 1 |
| numpy.int8 | 1 | 0 | 0 | 1 |
| numpy.integer | 3 | 0 | 0 | 3 |
| numpy.lib | 1 | 3 | 0 | 4 |
| numpy.lib.Arrayterator | 0 | 10 | 0 | 10 |
| numpy.lib.NumpyVersion | 8 | 6 | 0 | 14 |
| numpy.lib.array_utils | 2 | 1 | 0 | 3 |
| numpy.lib.format | 12 | 2 | 0 | 14 |
| numpy.lib.introspect | 0 | 1 | 0 | 1 |
| numpy.lib.mixins | 0 | 1 | 0 | 1 |
| numpy.lib.mixins.NDArrayOperatorsMixin | 0 | 51 | 0 | 51 |
| numpy.lib.npyio | 0 | 2 | 0 | 2 |
| numpy.lib.npyio.DataSource | 0 | 3 | 0 | 3 |
| numpy.lib.npyio.NpzFile | 0 | 16 | 0 | 16 |
| numpy.lib.recfunctions | 20 | 1 | 0 | 21 |
| numpy.lib.scimath | 9 | 0 | 0 | 9 |
| numpy.lib.stride_tricks | 1 | 1 | 0 | 2 |
| numpy.linalg | 32 | 0 | 0 | 32 |
| numpy.longdouble | 2 | 0 | 0 | 2 |
| numpy.longlong | 1 | 0 | 0 | 1 |
| numpy.ma | 217 | 2 | 0 | 219 |
| numpy.ma.MaskedArray | 82 | 5 | 0 | 87 |
| numpy.ma.core.MaskedIterator | 0 | 7 | 0 | 7 |
| numpy.ma.mrecords | 6 | 0 | 0 | 6 |
| numpy.ma.mrecords.MaskedRecords | 5 | 0 | 0 | 5 |
| numpy.matrix | 46 | 0 | 0 | 46 |
| numpy.memmap | 4 | 1 | 0 | 5 |
| numpy.ndarray | 69 | 3 | 0 | 72 |
| numpy.ndenumerate | 3 | 0 | 0 | 3 |
| numpy.ndindex | 2 | 0 | 0 | 2 |
| numpy.nditer | 27 | 4 | 0 | 31 |
| numpy.poly1d | 25 | 0 | 0 | 25 |
| numpy.polynomial | 7 | 0 | 0 | 7 |
| numpy.polynomial.Chebyshev | 51 | 0 | 0 | 51 |
| numpy.polynomial.Hermite | 50 | 0 | 0 | 50 |
| numpy.polynomial.HermiteE | 50 | 0 | 0 | 50 |
| numpy.polynomial.Laguerre | 50 | 0 | 0 | 50 |
| numpy.polynomial.Legendre | 50 | 0 | 0 | 50 |
| numpy.polynomial.Polynomial | 50 | 0 | 0 | 50 |
| numpy.polynomial.chebyshev | 31 | 0 | 0 | 31 |
| numpy.polynomial.hermite | 28 | 0 | 0 | 28 |
| numpy.polynomial.hermite_e | 28 | 0 | 0 | 28 |
| numpy.polynomial.laguerre | 28 | 0 | 0 | 28 |
| numpy.polynomial.legendre | 28 | 0 | 0 | 28 |
| numpy.polynomial.polynomial | 25 | 0 | 0 | 25 |
| numpy.polynomial.polyutils | 7 | 0 | 0 | 7 |
| numpy.random | 59 | 1 | 0 | 60 |
| numpy.random.BitGenerator | 5 | 3 | 0 | 8 |
| numpy.random.Generator | 45 | 0 | 0 | 45 |
| numpy.random.MT19937 | 6 | 3 | 0 | 9 |
| numpy.random.PCG64 | 7 | 3 | 0 | 10 |
| numpy.random.PCG64DXSM | 7 | 3 | 0 | 10 |
| numpy.random.Philox | 7 | 3 | 0 | 10 |
| numpy.random.RandomState | 49 | 0 | 0 | 49 |
| numpy.random.SFC64 | 5 | 3 | 0 | 8 |
| numpy.random.SeedSequence | 8 | 0 | 0 | 8 |
| numpy.rec | 6 | 3 | 0 | 9 |
| numpy.recarray | 3 | 3 | 0 | 6 |
| numpy.record | 3 | 2 | 0 | 5 |
| numpy.strings | 46 | 0 | 0 | 46 |
| numpy.ufunc | 15 | 0 | 0 | 15 |
| numpy.uint16 | 1 | 0 | 0 | 1 |
| numpy.uint32 | 1 | 0 | 0 | 1 |
| numpy.uint64 | 1 | 0 | 0 | 1 |
| numpy.uint8 | 1 | 0 | 0 | 1 |
| numpy.ulonglong | 1 | 0 | 0 | 1 |
| numpy.void | 5 | 0 | 0 | 5 |

To regenerate, run python3 tools/generate_reference.py followed by python3 tools/generate_inventory.py in an environment with NumPy 2.5.0 installed.

## Remaining entries

No entries in the defined inventory are pending. See the CSV and README for Go equivalents and untested arguments, dtypes, namespaces and runtime boundaries.
