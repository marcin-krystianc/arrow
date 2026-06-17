# Proposal: decode FIXED_LEN_BYTE_ARRAY values directly into a caller-owned buffer

Status: draft

Target: Apache Arrow C++ Parquet reader (`cpp/src/parquet`)

Audience: this document is written for developers who do not work on Arrow
internals. It explains the relevant Parquet and Arrow concepts before stating
the change.

## 1. Summary

Add one method to the Parquet C++ decoder layer:

```cpp
// in class FLBADecoder (cpp/src/parquet/encoding.h)
virtual int Decode(uint8_t* buffer, int max_values);
```

It decodes up to `max_values` fixed-length values straight into a contiguous,
caller-owned byte buffer, with no per-value pointer indirection and no Arrow
memory allocation. The caller owns `buffer`.

The change is limited to the decoder layer. It does not touch `ColumnReader` or
`RecordReader`. It supports columns without nulls only.

The motivating use case is reading FLOAT16 (half-precision float) columns into a
pre-allocated numeric buffer, for example a NumPy array, without an intermediate
Arrow allocation or a per-value copy.

## 2. Background for non-Arrow developers

### 2.1 Parquet physical types and FLOAT16

Parquet stores every column using one of a small set of *physical types*
(`BOOLEAN`, `INT32`, `INT64`, `FLOAT`, `DOUBLE`, `BYTE_ARRAY`,
`FIXED_LEN_BYTE_ARRAY`). Logical types are layered on top.

There is no 16-bit float physical type. A FLOAT16 column is stored as
`FIXED_LEN_BYTE_ARRAY` (FLBA) with a fixed length of 2 bytes per value, plus a
logical type annotation. So every FLOAT16 value is exactly 2 raw bytes on disk.

This matters: because the width is fixed and known (2 bytes), a buffer of N
FLOAT16 values is just `2 * N` contiguous bytes. There is no need to describe
each value separately.

### 2.2 How Arrow represents an FLBA value in memory

In the Parquet C++ code, a decoded FLBA value is this struct
(`cpp/src/parquet/types.h:682`):

```cpp
struct FixedLenByteArray {
  FixedLenByteArray() : ptr(NULLPTR) {}
  explicit FixedLenByteArray(const uint8_t* ptr) : ptr(ptr) {}
  const uint8_t* ptr;
};
using FLBA = FixedLenByteArray;
```

It holds only a pointer. The pointer points at the value's bytes, which live
somewhere else (usually inside the decoded data page buffer). The struct itself
carries no length, because the length is fixed for the whole column.

On a 64-bit machine this struct is 8 bytes. A FLOAT16 value is 2 bytes. So an
array of `FixedLenByteArray` uses 8 bytes of pointer to describe each 2 bytes of
data, and the actual data sits elsewhere.

### 2.3 The decoder layer

A Parquet column is split into data pages. Each page is encoded with one
*encoding*. For FLBA columns the encodings in practice are:

- `PLAIN`
- `RLE_DICTIONARY` (dictionary encoding)
- `DELTA_BYTE_ARRAY`
- `BYTE_STREAM_SPLIT`

For each encoding there is a *decoder* class. The decoder is given the raw page
bytes through `SetData`, and then produces decoded values on demand. The decoder
interface is templated on the physical type
(`cpp/src/parquet/encoding.h:267`):

```cpp
template <typename DType>
class TypedDecoder : virtual public Decoder {
 public:
  using T = typename DType::c_type;
  virtual int Decode(T* buffer, int max_values) = 0;
  virtual int DecodeSpaced(T* buffer, int num_values, int null_count,
                           const uint8_t* valid_bits, int64_t valid_bits_offset) = 0;
  // plus DecodeArrow(...) into an Arrow builder
};
```

For an FLBA column, `T` is `FixedLenByteArray`. So the standard
`Decode(T* buffer, int max_values)` fills an array of pointer structs, not raw
bytes.

A caller can construct a decoder directly through the public factory
(`cpp/src/parquet/encoding.h:462`):

```cpp
template <typename DType>
std::unique_ptr<typename EncodingTraits<DType>::Decoder> MakeTypedDecoder(
    Encoding::type encoding, const ColumnDescriptor* descr, MemoryPool* pool);
```

## 3. The problem

For a 2-byte FLOAT16 value, the standard decode path produces an 8-byte pointer
per value and leaves the data where it was. Concretely, the PLAIN decode for
FLBA does this (`cpp/src/parquet/decoder.cc:508`):

```cpp
template <>
inline int DecodePlain<FixedLenByteArray>(const uint8_t* data, int64_t data_size,
                                          int num_values, int type_length,
                                          FixedLenByteArray* out) {
  int64_t bytes_to_decode = static_cast<int64_t>(type_length) * num_values;
  if (bytes_to_decode > data_size || bytes_to_decode > INT_MAX) {
    ParquetException::EofException();
  }
  for (int i = 0; i < num_values; ++i) {
    out[i].ptr = data + i * static_cast<int64_t>(type_length);   // one pointer per value
  }
  return static_cast<int>(bytes_to_decode);
}
```

A consumer that wants a plain `2 * N`-byte buffer of FLOAT16 values must then
walk the `N` pointers and copy 2 bytes from each. That second pass is pure
overhead. The width is fixed, so the indirection adds nothing.

The goal is to let the decoder write the `2 * N` raw bytes directly into the
caller's buffer and skip the pointer array entirely.

## 4. Why the existing mechanisms do not solve it

Arrow already has code that produces contiguous raw FLBA bytes. It does so while
decoding into an Arrow builder, not into caller memory.

### 4.1 DecodeArrow into FixedSizeBinaryBuilder

Each FLBA decoder implements `DecodeArrow`, which fills a
`FixedSizeBinaryBuilder`. That builder stores values contiguously and packed, at
`byte_width` bytes each, with no per-value pointers. For PLAIN
(`cpp/src/parquet/decoder.cc:668`):

```cpp
inline int PlainDecoder<FLBAType>::DecodeArrow(
    int num_values, int null_count, const uint8_t* valid_bits, int64_t valid_bits_offset,
    typename EncodingTraits<FLBAType>::Accumulator* builder) {
  const int byte_width = this->type_length_;
  const int values_decoded = num_values - null_count;
  CheckPageLargeEnough(len_, byte_width, values_decoded);
  PARQUET_THROW_NOT_OK(builder->Reserve(num_values));
  uint8_t* decode_out = builder->GetMutableValue(builder->length() + null_count);
  memcpy(decode_out, data_, values_decoded * byte_width);
  ...
}
```

This proves two things:

1. Contiguous, packed, pointer-free output is achievable for FLBA.
2. The other three encodings already do the same. Their `DecodeArrow`
   implementations also write packed bytes into a `FixedSizeBinaryBuilder`
   (dictionary at `cpp/src/parquet/decoder.cc:1149`, BYTE_STREAM_SPLIT into the
   builder data buffer, and DELTA_BYTE_ARRAY through its internal decode).

So the per-value indirection is *already* removed on the `DecodeArrow` path. The
problem is the destination.

### 4.2 The destination is Arrow-owned, not caller-owned

`FixedSizeBinaryBuilder` always owns its output buffer and allocates it from an
Arrow `MemoryPool`. Its only constructor takes a memory pool:

```cpp
// cpp/src/arrow/array/builder_binary.h
explicit FixedSizeBinaryBuilder(const std::shared_ptr<DataType>& type,
                                MemoryPool* pool = default_memory_pool(),
                                int64_t alignment = kDefaultBufferAlignment);
```

Its internal `BufferBuilder` allocates a `ResizableBuffer` from that pool on the
first `Reserve`/`Resize` and grows it by reallocating. There is no public way to
hand the builder a fixed, caller-provided region of memory and have it write
there. Even the lower-level `BufferBuilder` resize path reallocates through the
pool, so wrapping caller memory as a non-resizable buffer does not work either.

Consequence: to land FLOAT16 values in a caller buffer (a NumPy array) using
`DecodeArrow`, you must let Arrow allocate, then copy out of the finished array
into the caller buffer. That is an extra allocation and an extra copy. The
caller cannot choose where the bytes go.

### 4.3 Summary of the gap

Two separate properties are wanted. Only one exists today.

| Property | Provided today? | By what |
|---|---|---|
| No per-value pointer indirection for FLBA | Yes | `DecodeArrow` into `FixedSizeBinaryBuilder`, all encodings |
| Decode into caller-owned memory | No | nothing at the decoder layer |

This proposal adds the second property, reusing the contiguous-output logic that
already backs the first.

## 5. Scope and constraints

These constraints were chosen deliberately to keep the change small and to avoid
a known design hazard (see section 9.1).

- Decoder layer only. No change to `ColumnReader` or `RecordReader`.
- Columns without nulls only. The new method has no null bitmap parameter and
  does not leave gaps. Callers that need nulls keep using the existing
  `FixedLenByteArray*` or `DecodeArrow` paths.
- Caller-owned memory. The method writes into a plain `uint8_t*`. No Arrow
  `Buffer`, no `MemoryPool`.
- FLBA physical type only. This is where the fixed-width property holds and
  where FLOAT16 lives.

## 6. Proposed API

Add a virtual method to `FLBADecoder` (`cpp/src/parquet/encoding.h:411`):

```cpp
class FLBADecoder : virtual public TypedDecoder<FLBAType> {
 public:
  using TypedDecoder<FLBAType>::Decode;
  using TypedDecoder<FLBAType>::DecodeSpaced;

  /// \brief Decode up to max_values values into a contiguous, densely packed
  /// byte buffer holding max_values * descr->type_length() bytes.
  ///
  /// Unlike Decode(FixedLenByteArray*, int), which writes one pointer per value,
  /// this writes the raw fixed-width values back to back, with no per-value
  /// pointers and no gaps. The caller owns buffer and must size it to at least
  /// max_values * type_length bytes.
  ///
  /// This method is for columns without nulls. The return value is the number
  /// of values decoded, bounded by the values remaining in the current page.
  ///
  /// The default implementation throws. Each FLBA encoding overrides it.
  ///
  /// \note API EXPERIMENTAL
  virtual int Decode(uint8_t* buffer, int max_values);
};
```

The `using TypedDecoder<FLBAType>::Decode;` line keeps the existing
`Decode(FixedLenByteArray*, int)` overload visible alongside the new one.

### 6.1 Precedent in the same file

This mirrors an existing pattern. `BooleanDecoder` already adds a packed-output
`Decode(uint8_t*, int)` overload for its physical type
(`cpp/src/parquet/encoding.h:395`):

```cpp
class BooleanDecoder : virtual public TypedDecoder<BooleanType> {
 public:
  using TypedDecoder<BooleanType>::Decode;
  /// \brief Decode and bit-pack values into a buffer
  virtual int Decode(uint8_t* buffer, int max_values) = 0;
};
```

The proposed `FLBADecoder::Decode(uint8_t*, int)` is the FLBA analogue: a
physical-type-specific overload that writes packed output. The only difference
is that the default throws (instead of being pure virtual), so encodings can opt
in one at a time.

## 7. Per-encoding implementation plan

There are four FLBA decoders. Each already produces packed bytes inside its
`DecodeArrow`. The new override reuses that logic with a caller pointer as the
destination and no null handling.

### 7.1 Default (base) implementation

In `cpp/src/parquet/decoder.cc`, define the base `FLBADecoder::Decode(uint8_t*,
int)` to throw, so any encoding that has not opted in fails clearly:

```cpp
int FLBADecoder::Decode(uint8_t* /*buffer*/, int /*max_values*/) {
  throw ParquetException(
      "Dense FIXED_LEN_BYTE_ARRAY decoding is not implemented for this encoding");
}
```

### 7.2 PLAIN (`PlainFLBADecoder`, `cpp/src/parquet/decoder.cc:854`)

PLAIN-encoded FLBA values are already contiguous in the page buffer. The
override is a single bounds check plus one `memcpy`:

```cpp
int Decode(uint8_t* buffer, int max_values) override {
  max_values = std::min(max_values, this->num_values_);
  const int64_t bytes_to_decode =
      static_cast<int64_t>(this->type_length_) * max_values;
  if (bytes_to_decode > this->len_ || bytes_to_decode > INT_MAX) {
    ParquetException::EofException();
  }
  if (bytes_to_decode > 0) {
    memcpy(buffer, this->data_, static_cast<size_t>(bytes_to_decode));
  }
  this->data_ += bytes_to_decode;
  this->len_ -= static_cast<int>(bytes_to_decode);
  this->num_values_ -= max_values;
  return max_values;
}
```

This is the same copy already used by `PlainDecoder<FLBAType>::DecodeArrow`
(`cpp/src/parquet/decoder.cc:679`), without the builder.

### 7.3 RLE_DICTIONARY (`DictDecoderImpl<FLBAType>`, `cpp/src/parquet/decoder.cc:864`)

The dictionary holds the distinct values as packed FLBA entries. Decoding reads
an index per value and copies that dictionary entry's `type_length` bytes. The
no-null override follows the existing `DecodeArrow`
(`cpp/src/parquet/decoder.cc:1149`), dropping the bitmap visit:

```cpp
int Decode(uint8_t* buffer, int max_values) override {
  max_values = std::min(max_values, this->num_values_);
  const auto* dict_values = dictionary_->data_as<FLBA>();
  const int type_length = this->type_length_;
  for (int i = 0; i < max_values; ++i) {
    int32_t index;
    if (ARROW_PREDICT_FALSE(!idx_decoder_.Get(&index))) {
      throw ParquetException("Dict decoding failed");
    }
    if (ARROW_PREDICT_FALSE(index < 0 || index >= dictionary_length_)) {
      throw ParquetException("Index out of bounds in dictionary decoding");
    }
    memcpy(buffer + static_cast<int64_t>(i) * type_length,
           dict_values[index].ptr, type_length);
  }
  this->num_values_ -= max_values;
  return max_values;
}
```

(The exact index bounds check should reuse the existing `IndexInBounds` helper
used by `DecodeArrow`.)

### 7.4 BYTE_STREAM_SPLIT (`ByteStreamSplitDecoder<FLBAType>`, `cpp/src/parquet/decoder.cc:2351`)

This decoder already has a protected `DecodeRaw(uint8_t*, int)` that writes
unsplit, contiguous bytes directly into a destination buffer. The current
`Decode(FixedLenByteArray*, int)` decodes into a scratch buffer and then builds
pointers (`cpp/src/parquet/decoder.cc:2360`). The dense override skips the
scratch buffer and the pointer building:

```cpp
int Decode(uint8_t* buffer, int max_values) override {
  return this->DecodeRaw(buffer, max_values);
}
```

This is the cleanest of the four and is strictly less work than the existing
pointer path.

### 7.5 DELTA_BYTE_ARRAY (`DeltaByteArrayFLBADecoder`, `cpp/src/parquet/decoder.cc:2214`)

This decoder reconstructs values through an internal `GetInternal` that produces
`ByteArray` (pointer plus length) values into a temporary vector. The existing
`Decode(FixedLenByteArray*, int)` already does this and copies out the pointers
(`cpp/src/parquet/decoder.cc:2221`). The dense override does the same decode, but
copies the bytes contiguously and validates the fixed length:

```cpp
int Decode(uint8_t* buffer, int max_values) override {
  std::vector<ByteArray> decoded(max_values);
  const int n = GetInternal(decoded.data(), max_values);
  const uint32_t type_length = static_cast<uint32_t>(this->type_length_);
  for (int i = 0; i < n; ++i) {
    if (ARROW_PREDICT_FALSE(decoded[i].len != type_length)) {
      throw ParquetException("Fixed length byte array length mismatch");
    }
    memcpy(buffer + static_cast<int64_t>(i) * type_length, decoded[i].ptr, type_length);
  }
  return n;
}
```

This one keeps the existing temporary `ByteArray` vector, because
`GetInternal` is defined in terms of `ByteArray`. The extra copy here is inherent
to how DELTA_BYTE_ARRAY reconstructs values; it is the same cost the existing
pointer path pays, minus the later per-pointer copy by the consumer.

### 7.6 Per-encoding cost summary

| Encoding | Override cost vs. existing pointer path |
|---|---|
| PLAIN | one `memcpy`, no pointer array |
| BYTE_STREAM_SPLIT | reuses `DecodeRaw`, drops scratch buffer and pointer building |
| RLE_DICTIONARY | per-value `memcpy` from dictionary, same as existing `DecodeArrow` minus bitmap |
| DELTA_BYTE_ARRAY | same internal decode, contiguous copy instead of pointer copy |

## 8. Semantics and error handling

- `buffer` must hold at least `max_values * descr->type_length()` bytes. The
  caller is responsible for sizing it.
- The method decodes at most the number of values remaining in the current data
  page. The return value is the number actually decoded. A caller reads a page
  to exhaustion by calling until the return value is 0 or the expected count is
  reached, the same pattern as the existing `Decode`.
- Nulls are not handled. The method assumes the column has no nulls. If the
  source data contains nulls, the caller must not use this method; behavior with
  null pages is out of scope and not supported.
- Bounds and corruption checks match the existing decode paths
  (`EofException` on short pages, index bounds checks for dictionary).
- The base implementation throws `ParquetException`, so an encoding that has not
  implemented the override fails with a clear message rather than silently.

## 9. Alternatives considered

### 9.1 A ReadBatchDense method on the column reader (rejected)

An earlier local experiment added a flat `ReadBatchDense(batch_size, def_levels,
rep_levels, uint8_t* values, values_read)` to `TypedColumnReader`. This is the
wrong layer.

Arrow already removed a method of exactly this shape. `ReadBatchSpaced` was a
`TypedColumnReader<DType>` method that read definition and repetition levels plus
a flat values buffer in one call. It was deprecated in 4.0.0 and removed in
commit `990cdf7dca8ba74d1c3b9a7b14898011a1142356`
("GH-44079: [C++][Parquet] Remove deprecated APIs") with the stated reason:

> Doesn't handle nesting correctly and unused outside of unit tests.

A flat values buffer combined with repetition levels cannot represent nested or
repeated data correctly. Re-introducing a method with the same shape would
re-introduce the same defect. Keeping the change at the decoder layer avoids this
entirely, because decoders never see repetition or definition levels. They only
turn encoded bytes into values.

### 9.2 Let RecordReader decode into a caller buffer (out of scope)

`RecordReader` handles levels and nesting correctly and produces a contiguous
values buffer. It could be extended to accept a caller-owned output buffer. This
is a larger change to a higher layer and is not needed for the FLOAT16,
no-nulls, non-nested use case. It is out of scope here but could be a separate
follow-up.

### 9.3 Decode into FixedSizeBinaryBuilder then copy out (status quo)

This works today but forces an Arrow allocation and an extra copy, and the caller
cannot choose the destination. See section 4.2. The proposal exists to avoid
exactly this.

## 10. How a caller uses it

A consumer that manages its own memory (for example, filling a NumPy array)
constructs the decoder for the page's encoding, points it at the page bytes, and
decodes into its own buffer:

```cpp
// descr: the ColumnDescriptor for the FLOAT16 column (type_length() == 2)
auto decoder = parquet::MakeTypedDecoder<parquet::FLBAType>(encoding, descr);
decoder->SetData(num_values, page_data, page_len);

auto* flba = dynamic_cast<parquet::FLBADecoder*>(decoder.get());
// user_buffer points at the caller's memory, sized num_values * 2 bytes
int decoded = flba->Decode(user_buffer, num_values);
```

`MakeTypedDecoder` is the existing public factory
(`cpp/src/parquet/encoding.h:462`). No new factory or reader API is required.

## 11. Backward compatibility

- Additive only. The new method has a default implementation, so existing
  decoders and existing callers are unaffected.
- The existing `Decode(FixedLenByteArray*, int)` and `DecodeArrow` paths are
  unchanged.
- Marked `API EXPERIMENTAL` so the signature can evolve before it is treated as
  stable.

## 12. Testing plan

- Round-trip test per encoding: write a FLOAT16 column with PLAIN,
  RLE_DICTIONARY, DELTA_BYTE_ARRAY, and BYTE_STREAM_SPLIT, then decode with the
  new method and compare the raw `2 * N` bytes against the source values.
- Cross-check against the existing path: decode the same page with
  `Decode(FixedLenByteArray*, int)`, gather the 2-byte values by following the
  pointers, and assert the bytes match the dense output.
- Boundary cases: `max_values` larger than the values left in the page (expect a
  short decode and the correct return count), empty page, and a page whose byte
  length is too small (expect `EofException`).
- Dictionary corruption: an out-of-range index must throw, matching the existing
  `DecodeArrow` behavior.
- Negative test: calling the new method on an encoding before its override
  exists must throw the base `ParquetException`.

## 13. Open questions

- Should the method assert or document the FLOAT16 logical type, or stay generic
  to any FLBA width? Staying generic costs nothing and also serves fixed-size
  decimals.
- Whether to add a matching capability check (for example, a virtual
  `bool supports_dense_decode()`), so callers can detect support without relying
  on a thrown exception. Not required for the four encodings above, since all
  will implement it.
