"""
_mini_parquet.py

A minimal, dependency-free Parquet reader (stdlib only: struct, zlib).

Why this exists: this sandbox's package index blocks every third-party
package that isn't already preinstalled (pyarrow, fastparquet, duckdb,
polars all return 403 from PyPI itself -- confirmed, not a network fluke),
so pandas.read_parquet has no engine available. The uploaded
`est_hourly.paruqet` file needs pyarrow or fastparquet to read via the
normal route, so this hand-rolled reader decodes the Thrift compact-protocol
footer and the column data pages directly.

Scope (enough for this specific file, not a general-purpose library):
  - Flat (non-nested) columns only
  - REQUIRED or OPTIONAL repetition (definition levels: RLE/bit-packed hybrid,
    max_definition_level 0 or 1)
  - Encodings: PLAIN, PLAIN_DICTIONARY / RLE_DICTIONARY
  - Codecs: UNCOMPRESSED, SNAPPY (pure-Python decompressor below), GZIP
  - Physical types: DOUBLE, INT32, INT64, BYTE_ARRAY (as used by this file)
"""
import struct
import zlib

# ---------- Thrift compact-protocol primitives ----------

def read_uvarint(buf, pos):
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return result, pos


def zigzag_decode(n):
    return (n >> 1) ^ -(n & 1)


def read_zigzag_varint(buf, pos):
    v, pos = read_uvarint(buf, pos)
    return zigzag_decode(v), pos


CT_STOP, CT_TRUE, CT_FALSE, CT_BYTE, CT_I16, CT_I32, CT_I64, CT_DOUBLE, \
    CT_BINARY, CT_LIST, CT_SET, CT_MAP, CT_STRUCT = range(13)


def read_value(buf, pos, ctype):
    if ctype in (CT_TRUE, CT_FALSE):
        return (ctype == CT_TRUE), pos
    if ctype == CT_BYTE:
        return struct.unpack_from("<b", buf, pos)[0], pos + 1
    if ctype in (CT_I16, CT_I32, CT_I64):
        return read_zigzag_varint(buf, pos)
    if ctype == CT_DOUBLE:
        return struct.unpack_from("<d", buf, pos)[0], pos + 8
    if ctype == CT_BINARY:
        length, pos = read_uvarint(buf, pos)
        return bytes(buf[pos:pos + length]), pos + length
    if ctype == CT_STRUCT:
        return read_struct(buf, pos)
    if ctype in (CT_LIST, CT_SET):
        header = buf[pos]; pos += 1
        size = header >> 4
        elem_type = header & 0x0F
        if size == 15:
            size, pos = read_uvarint(buf, pos)
        items = []
        for _ in range(size):
            v, pos = read_value(buf, pos, elem_type)
            items.append(v)
        return items, pos
    if ctype == CT_MAP:
        size, pos = read_uvarint(buf, pos)
        result = []
        if size > 0:
            types_byte = buf[pos]; pos += 1
            key_type, val_type = types_byte >> 4, types_byte & 0x0F
            for _ in range(size):
                k, pos = read_value(buf, pos, key_type)
                v, pos = read_value(buf, pos, val_type)
                result.append((k, v))
        return result, pos
    raise ValueError(f"Unsupported compact type {ctype} at pos {pos}")


def read_struct(buf, pos):
    """Returns {field_id: value} and advances pos past the terminating STOP."""
    fields = {}
    last_fid = 0
    while True:
        header = buf[pos]; pos += 1
        if header == 0:
            break
        delta = header >> 4
        ctype = header & 0x0F
        if delta == 0:
            fid, pos = read_zigzag_varint(buf, pos)
        else:
            fid = last_fid + delta
        last_fid = fid
        if ctype in (CT_TRUE, CT_FALSE):
            val, pos = read_value(buf, pos, ctype)
        else:
            val, pos = read_value(buf, pos, ctype)
        fields[fid] = val
    return fields, pos


# ---------- Pure-python raw Snappy block decompressor ----------

def snappy_decompress(data: bytes) -> bytes:
    length, pos = read_uvarint(data, 0)
    out = bytearray()
    n = len(data)
    while pos < n:
        tag = data[pos]; pos += 1
        kind = tag & 0x03
        if kind == 0:  # literal
            len_hi = tag >> 2
            if len_hi < 60:
                lit_len = len_hi + 1
            else:
                nbytes = len_hi - 59
                lit_len = int.from_bytes(data[pos:pos + nbytes], "little") + 1
                pos += nbytes
            out += data[pos:pos + lit_len]
            pos += lit_len
        elif kind == 1:  # copy, 1-byte offset
            copy_len = ((tag >> 2) & 0x07) + 4
            offset = ((tag >> 5) << 8) | data[pos]
            pos += 1
            _snappy_copy(out, offset, copy_len)
        elif kind == 2:  # copy, 2-byte offset
            copy_len = (tag >> 2) + 1
            offset = int.from_bytes(data[pos:pos + 2], "little")
            pos += 2
            _snappy_copy(out, offset, copy_len)
        else:  # kind == 3, 4-byte offset
            copy_len = (tag >> 2) + 1
            offset = int.from_bytes(data[pos:pos + 4], "little")
            pos += 4
            _snappy_copy(out, offset, copy_len)
    assert len(out) == length, f"snappy decompressed length mismatch: {len(out)} != {length}"
    return bytes(out)


def _snappy_copy(out: bytearray, offset: int, length: int):
    start = len(out) - offset
    for i in range(length):
        out.append(out[start + i])


# ---------- RLE / bit-packed hybrid (definition levels) ----------

def read_rle_bitpacked_hybrid(buf, pos, bit_width, num_values):
    values = []
    byte_width = (bit_width + 7) // 8
    while len(values) < num_values:
        header, pos = read_uvarint(buf, pos)
        if header & 1:  # bit-packed run
            num_groups = header >> 1
            count = num_groups * 8
            total_bits = count * bit_width
            nbytes = (total_bits + 7) // 8
            chunk = buf[pos:pos + nbytes]
            pos += nbytes
            bit_pos = 0
            for _ in range(count):
                v = 0
                for b in range(bit_width):
                    byte_idx = bit_pos // 8
                    bit_idx = bit_pos % 8
                    if byte_idx < len(chunk) and (chunk[byte_idx] >> bit_idx) & 1:
                        v |= (1 << b)
                    bit_pos += 1
                values.append(v)
        else:  # RLE run
            run_len = header >> 1
            raw = buf[pos:pos + byte_width]
            pos += byte_width
            val = int.from_bytes(raw, "little")
            values.extend([val] * run_len)
    return values[:num_values], pos


# ---------- Footer / metadata ----------

PARQUET_TYPE_NAMES = {0: "BOOLEAN", 1: "INT32", 2: "INT64", 3: "INT96",
                       4: "FLOAT", 5: "DOUBLE", 6: "BYTE_ARRAY", 7: "FIXED_LEN_BYTE_ARRAY"}
CODEC_NAMES = {0: "UNCOMPRESSED", 1: "SNAPPY", 2: "GZIP", 3: "LZO", 4: "BROTLI", 5: "LZ4", 6: "ZSTD"}
ENCODING_NAMES = {0: "PLAIN", 2: "PLAIN_DICTIONARY", 3: "RLE", 8: "RLE_DICTIONARY"}


def read_file_metadata(path):
    with open(path, "rb") as f:
        data = f.read()
    assert data[:4] == b"PAR1" and data[-4:] == b"PAR1", "not a parquet file"
    footer_len = struct.unpack("<I", data[-8:-4])[0]
    footer = data[-8 - footer_len:-8]
    meta, _ = read_struct(footer, 0)
    return data, meta


def get_schema_names(meta):
    """Returns column names in file order, skipping the root schema element."""
    schema_list = meta[2]  # list of SchemaElement structs, each {field_id: val}
    names = []
    for elem in schema_list[1:]:  # [0] is the root
        names.append(elem[4].decode("utf-8"))
    return names


def decompress_page(raw, codec_id):
    codec = CODEC_NAMES.get(codec_id, "UNKNOWN")
    if codec == "UNCOMPRESSED":
        return raw
    if codec == "SNAPPY":
        return snappy_decompress(raw)
    if codec == "GZIP":
        return zlib.decompress(raw, 16 + zlib.MAX_WBITS)
    raise ValueError(f"Unsupported codec {codec}")


def read_column(data, col_chunk_meta, num_rows_in_group):
    """col_chunk_meta: the ColumnMetaData struct (field 3 of a ColumnChunk)."""
    ptype = col_chunk_meta[1]
    codec_id = col_chunk_meta[4]
    data_page_offset = col_chunk_meta.get(9)
    dict_page_offset = col_chunk_meta.get(11)

    pos = dict_page_offset if dict_page_offset is not None else data_page_offset
    dictionary = None
    values = []
    def_levels_all = []

    while len(values) < num_rows_in_group and pos < len(data):
        page_header, pos = read_struct(data, pos)
        page_type = page_header[1]  # 0=DATA_PAGE,1=INDEX_PAGE,2=DICTIONARY_PAGE,3=DATA_PAGE_V2
        uncompressed_size = page_header[2]
        compressed_size = page_header[3]
        raw = data[pos:pos + compressed_size]
        pos += compressed_size
        page_bytes = decompress_page(raw, codec_id)
        assert len(page_bytes) == uncompressed_size

        if page_type == 2:  # DICTIONARY_PAGE
            dph = page_header[7]
            n_dict = dph[1]
            dictionary, _ = read_plain_values(page_bytes, 0, ptype, n_dict)
            continue

        if page_type == 0:  # DATA_PAGE (v1)
            dph = page_header[5]
            n_vals = dph[1]
            encoding = dph[2]
            def_level_encoding = dph.get(3)
            ppos = 0

            # Definition levels (only present if column is OPTIONAL, i.e. max_def_level==1)
            max_def_level = col_chunk_meta.get("_max_def_level", 0)
            if max_def_level > 0:
                level_section_len = struct.unpack_from("<i", page_bytes, ppos)[0]
                ppos += 4
                def_levels, _ = read_rle_bitpacked_hybrid(
                    page_bytes, ppos, bit_width=1, num_values=n_vals)
                ppos += level_section_len
            else:
                def_levels = [1] * n_vals

            def_levels_all.extend(def_levels)
            n_present = sum(1 for d in def_levels if d == 1)

            if encoding in (2, 8):  # (RLE_)PLAIN_DICTIONARY
                bit_width = page_bytes[ppos]; ppos += 1
                idxs, ppos = read_rle_bitpacked_hybrid(page_bytes, ppos, bit_width, n_present)
                vals = [dictionary[i] for i in idxs]
            else:  # PLAIN
                vals, ppos = read_plain_values(page_bytes, ppos, ptype, n_present)

            it = iter(vals)
            for d in def_levels:
                values.append(next(it) if d == 1 else None)

    return values[:num_rows_in_group]


def read_plain_values(buf, pos, ptype, n):
    out = []
    if ptype == 5:  # DOUBLE
        for _ in range(n):
            out.append(struct.unpack_from("<d", buf, pos)[0]); pos += 8
    elif ptype == 4:  # FLOAT
        for _ in range(n):
            out.append(struct.unpack_from("<f", buf, pos)[0]); pos += 4
    elif ptype == 1:  # INT32
        for _ in range(n):
            out.append(struct.unpack_from("<i", buf, pos)[0]); pos += 4
    elif ptype == 2:  # INT64
        for _ in range(n):
            out.append(struct.unpack_from("<q", buf, pos)[0]); pos += 8
    elif ptype == 6:  # BYTE_ARRAY
        for _ in range(n):
            ln = struct.unpack_from("<i", buf, pos)[0]; pos += 4
            out.append(bytes(buf[pos:pos + ln])); pos += ln
    else:
        raise ValueError(f"Unsupported physical type {ptype}")
    return out, pos


def load_parquet_columns(path, wanted_columns=None):
    """Returns dict[column_name] -> list of values (across all row groups)."""
    data, meta = read_file_metadata(path)
    names = get_schema_names(meta)
    schema_list = meta[2][1:]
    name_to_schema = dict(zip(names, schema_list))

    row_groups = meta[4]
    result = {name: [] for name in (wanted_columns or names)}

    for rg in row_groups:
        num_rows = rg[3]
        col_chunks = rg[1]
        for cc in col_chunks:
            col_meta = cc[3]
            path_in_schema = col_meta[3]
            col_name = path_in_schema[0].decode("utf-8") if isinstance(path_in_schema[0], bytes) else path_in_schema[0]
            if wanted_columns and col_name not in wanted_columns:
                continue
            schema_elem = name_to_schema[col_name]
            repetition_type = schema_elem.get(3, 0)  # 0=REQUIRED,1=OPTIONAL
            col_meta["_max_def_level"] = 1 if repetition_type == 1 else 0
            vals = read_column(data, col_meta, num_rows)
            result[col_name].extend(vals)

    return result
