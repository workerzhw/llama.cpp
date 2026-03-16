#!/usr/bin/env python3
"""Parse a llama.cpp KV cache sequence state binary file (.bin) and produce
a human-readable text report.

Usage examples:
  # Summary only (stats per layer, no values)
  python3 scripts/kv_cache_dump.py ke_seq_xxx.bin

  # Export layer 0 with all values
  python3 scripts/kv_cache_dump.py ke_seq_xxx.bin --layers 0

  # Export layers 0,1,31 with all values
  python3 scripts/kv_cache_dump.py ke_seq_xxx.bin --layers 0,1,31

  # Export all layers with all values
  python3 scripts/kv_cache_dump.py ke_seq_xxx.bin --layers all

  # Export layers 0-3 with all values, only cells 0-15
  python3 scripts/kv_cache_dump.py ke_seq_xxx.bin --layers 0-3 --cells 0-15

  # Only K or V
  python3 scripts/kv_cache_dump.py ke_seq_xxx.bin --layers 0 --only k
  python3 scripts/kv_cache_dump.py ke_seq_xxx.bin --layers 0 --only v

Binary format (LLAMA_STATE_SEQ_VERSION = 2):
  Header:  u32 magic, u32 version, u32 n_tokens, i32 tokens[n_tokens]
  KV Cache (per stream):
    u32 n_stream
    For each stream:
      u32 cell_count
      Meta: [i32 pos, u32 n_seq_id, i32 seq_ids[n_seq_id]] * cell_count
      Data: u32 v_trans, u32 n_layer
        K: [i32 type, u64 row_size, bytes[cell_count * row_size]] * n_layer
        V (contiguous): [i32 type, u64 row_size, bytes[cell_count * row_size]] * n_layer
        V (transposed): [i32 type, u32 el_size, u32 n_embd, bytes[n_embd * cell_count * el_size]] * n_layer
"""

import argparse
import struct
import sys
import os
import numpy as np
from pathlib import Path

LLAMA_STATE_SEQ_MAGIC = 0x67677371
LLAMA_STATE_SEQ_VERSION = 2

GGML_TYPES = {
    0:  ("f32",   4, 1),   1:  ("f16",   2, 1),   2:  ("q4_0", 18, 32),
    3:  ("q4_1", 20, 32),  6:  ("q5_0", 22, 32),  7:  ("q5_1", 24, 32),
    8:  ("q8_0", 34, 32),  9:  ("q8_1", 40, 32), 10:  ("q2_K", 64, 256),
    11: ("q3_K", 64, 256), 12: ("q4_K", 72, 256), 13: ("q5_K", 88, 256),
    14: ("q6_K",104, 256), 15: ("q8_K",108, 256), 30: ("bf16",  2, 1),
}


def type_name(t: int) -> str:
    return GGML_TYPES.get(t, ("unknown",))[0]


def read_u32(f) -> int:
    return struct.unpack("<I", f.read(4))[0]


def read_i32(f) -> int:
    return struct.unpack("<i", f.read(4))[0]


def read_u64(f) -> int:
    return struct.unpack("<Q", f.read(8))[0]


def decode_kv_data(raw: bytes, dtype_id: int, n_elements: int):
    t = GGML_TYPES.get(dtype_id)
    if t is None:
        return None
    name = t[0]
    if name == "f32":
        return np.frombuffer(raw, dtype=np.float32, count=n_elements)
    elif name == "f16":
        return np.frombuffer(raw, dtype=np.float16, count=n_elements)
    elif name == "bf16":
        u16 = np.frombuffer(raw, dtype=np.uint16, count=n_elements)
        f32 = np.zeros(n_elements, dtype=np.float32)
        f32.view(np.uint32)[:] = u16.astype(np.uint32) << 16
        return f32
    return None


def stats_str(arr) -> str:
    if arr is None or len(arr) == 0:
        return "(quantized, raw bytes)"
    a = arr.astype(np.float32) if arr.dtype != np.float32 else arr
    return (
        f"min={a.min():.6g}, max={a.max():.6g}, "
        f"mean={a.mean():.6g}, std={a.std():.6g}, "
        f"absmax={np.abs(a).max():.6g}"
    )


def format_matrix(arr, n_rows: int, n_cols: int, cell_start: int, cell_end: int,
                   prefix: str = "  │  │    ") -> str:
    """Format selected rows of a [n_rows x n_cols] array."""
    lines = []
    if arr is None:
        lines.append(f"{prefix}(quantized data, cannot decode)")
        return "\n".join(lines)

    mat = arr.reshape(n_rows, n_cols)
    cs = max(0, cell_start)
    ce = min(n_rows, cell_end)
    lines.append(f"{prefix}shape = [{n_rows}, {n_cols}], showing cells [{cs}..{ce - 1}]")
    for r in range(cs, ce):
        row_vals = ", ".join(f"{v:.6g}" for v in mat[r])
        lines.append(f"{prefix}cell[{r:4d}]: [{row_vals}]")
    return "\n".join(lines)


def format_tokens(tokens: list, per_line: int = 16) -> str:
    lines = []
    for i in range(0, len(tokens), per_line):
        chunk = tokens[i : i + per_line]
        lines.append(f"  [{i:6d}] " + " ".join(f"{t:6d}" for t in chunk))
    return "\n".join(lines)


def parse_layers_arg(s: str, n_layer: int) -> set:
    """Parse --layers argument: 'all', '0', '0,1,31', '0-3', '0-3,31'."""
    if s is None:
        return set()
    s = s.strip()
    if s.lower() == "all":
        return set(range(n_layer))
    result = set()
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            for i in range(int(a), int(b) + 1):
                if 0 <= i < n_layer:
                    result.add(i)
        else:
            i = int(part)
            if 0 <= i < n_layer:
                result.add(i)
    return result


def parse_cells_arg(s: str, cell_count: int) -> tuple:
    """Parse --cells argument: 'all', '0-15', '100-200'. Returns (start, end)."""
    if s is None or s.strip().lower() == "all":
        return (0, cell_count)
    s = s.strip()
    if "-" in s:
        a, b = s.split("-", 1)
        return (max(0, int(a)), min(cell_count, int(b) + 1))
    i = int(s)
    return (i, i + 1)


def parse_and_dump(bin_path: str, txt_path: str, layers_arg: str, cells_arg: str, only: str):
    file_size = os.path.getsize(bin_path)
    dump_values = layers_arg is not None
    show_k = only in (None, "k")
    show_v = only in (None, "v")

    with open(bin_path, "rb") as f:
        out = []

        def w(s=""):
            out.append(s)

        magic = read_u32(f)
        version = read_u32(f)
        if magic != LLAMA_STATE_SEQ_MAGIC:
            print(f"ERROR: bad magic 0x{magic:08x}", file=sys.stderr)
            sys.exit(1)
        if version != LLAMA_STATE_SEQ_VERSION:
            print(f"WARNING: version {version}, file=sys.stderr")

        n_tokens = read_u32(f)
        tokens = [read_i32(f) for _ in range(n_tokens)]

        w("=" * 80)
        w("  KV CACHE SEQUENCE STATE DUMP")
        w("=" * 80)
        w(f"  File         : {bin_path}")
        w(f"  Size         : {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MiB)")
        w(f"  Magic        : 0x{magic:08x} ('ggsq')")
        w(f"  Version      : {version}")
        w(f"  Tokens       : {n_tokens}")
        if dump_values:
            w(f"  Layers       : {layers_arg}")
            w(f"  Cells        : {cells_arg or 'all'}")
            w(f"  Only         : {only or 'K+V'}")
        w()
        w("─" * 80)
        w("  TOKENS")
        w("─" * 80)
        w(format_tokens(tokens))
        w()

        n_stream = read_u32(f)
        w("─" * 80)
        w(f"  KV CACHE  (n_stream = {n_stream})")
        w("─" * 80)
        w()

        for s in range(n_stream):
            cell_count = read_u32(f)
            w(f"  ┌─ Stream {s}")
            w(f"  │  cell_count = {cell_count}")

            if cell_count == 0:
                w(f"  └─")
                w()
                continue

            cell_start, cell_end = parse_cells_arg(cells_arg, cell_count)

            # Meta
            cells_meta = []
            for c in range(cell_count):
                pos = read_i32(f)
                n_seq_id = read_u32(f)
                seq_ids = [read_i32(f) for _ in range(n_seq_id)]
                cells_meta.append((pos, seq_ids))

            w(f"  │")
            w(f"  │  ┌─ Cell Metadata (pos range: [{cells_meta[0][0]}..{cells_meta[-1][0]}])")
            show_n = min(4, cell_count)
            for c in range(show_n):
                pos, sids = cells_meta[c]
                w(f"  │  │  cell[{c:4d}]: pos={pos:6d}, seq_ids={sids}")
            if cell_count > show_n:
                w(f"  │  │  ... ({cell_count - show_n} more cells)")
            w(f"  │  └─")

            # Data header
            v_trans = read_u32(f)
            n_layer = read_u32(f)

            layer_set = parse_layers_arg(layers_arg, n_layer) if dump_values else set()

            w(f"  │")
            w(f"  │  v_trans = {v_trans}, n_layer = {n_layer}")
            w(f"  │")

            # ── K ──
            w(f"  │  ┌─ K (Key)")
            total_k = 0
            for li in range(n_layer):
                k_type = read_i32(f)
                k_row = read_u64(f)
                k_bytes = cell_count * k_row
                k_raw = f.read(k_bytes)
                total_k += k_bytes

                tname = type_name(k_type)
                el_sz = GGML_TYPES.get(k_type, (None, 1))[1]
                n_cols = k_row // max(1, el_sz)
                n_elem = cell_count * n_cols
                arr = decode_kv_data(k_raw, k_type, n_elem)

                w(f"  │  │  layer[{li:2d}] K: {tname}, [{cell_count}×{n_cols}], {k_bytes:,} B")
                w(f"  │  │    {stats_str(arr)}")

                if li in layer_set and show_k:
                    w(format_matrix(arr, cell_count, n_cols, cell_start, cell_end))

            w(f"  │  │  K total: {total_k:,} B ({total_k / 1048576:.2f} MiB)")
            w(f"  │  └─")
            w(f"  │")

            # ── V ──
            w(f"  │  ┌─ V (Value)")
            total_v = 0

            if not v_trans:
                for li in range(n_layer):
                    v_type = read_i32(f)
                    v_row = read_u64(f)
                    v_bytes = cell_count * v_row
                    v_raw = f.read(v_bytes)
                    total_v += v_bytes

                    tname = type_name(v_type)
                    el_sz = GGML_TYPES.get(v_type, (None, 1))[1]
                    n_cols = v_row // max(1, el_sz)
                    n_elem = cell_count * n_cols
                    arr = decode_kv_data(v_raw, v_type, n_elem)

                    w(f"  │  │  layer[{li:2d}] V: {tname}, [{cell_count}×{n_cols}], {v_bytes:,} B")
                    w(f"  │  │    {stats_str(arr)}")

                    if li in layer_set and show_v:
                        w(format_matrix(arr, cell_count, n_cols, cell_start, cell_end))
            else:
                for li in range(n_layer):
                    v_type = read_i32(f)
                    v_el = read_u32(f)
                    n_embd = read_u32(f)
                    v_bytes = n_embd * cell_count * v_el
                    v_raw = f.read(v_bytes)
                    total_v += v_bytes

                    tname = type_name(v_type)
                    n_elem = n_embd * cell_count
                    arr = decode_kv_data(v_raw, v_type, n_elem)

                    w(f"  │  │  layer[{li:2d}] V: {tname}, [{cell_count}×{n_embd}] (transposed), {v_bytes:,} B")
                    w(f"  │  │    {stats_str(arr)}")

                    if li in layer_set and show_v:
                        if arr is not None:
                            arr_t = arr.reshape(n_embd, cell_count).T.copy()
                            w(format_matrix(arr_t.ravel(), cell_count, n_embd, cell_start, cell_end))
                        else:
                            w(format_matrix(arr, cell_count, n_embd, cell_start, cell_end))

            w(f"  │  │  V total: {total_v:,} B ({total_v / 1048576:.2f} MiB)")
            w(f"  │  └─")
            w(f"  │")
            w(f"  │  KV total: {total_k + total_v:,} B ({(total_k + total_v) / 1048576:.2f} MiB)")
            w(f"  └─")
            w()

        consumed = f.tell()
        w("─" * 80)
        w("  SUMMARY")
        w("─" * 80)
        w(f"  Parsed: {consumed:,} / {file_size:,} bytes")
        if consumed != file_size:
            w(f"  WARNING: {file_size - consumed} trailing bytes!")
        w("=" * 80)

    report = "\n".join(out) + "\n"
    with open(txt_path, "w") as fout:
        fout.write(report)
    print(report, end="")
    print(f"\nWritten to: {txt_path}")


def main():
    p = argparse.ArgumentParser(
        description="Parse llama.cpp KV cache binary (.bin) -> human-readable text.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Summary only (stats per layer, no actual values)
  python3 %(prog)s ke_seq_xxx.bin

  # Dump layer 0 values
  python3 %(prog)s ke_seq_xxx.bin --layers 0

  # Dump layers 0-3 and 31
  python3 %(prog)s ke_seq_xxx.bin --layers 0-3,31

  # Dump all layers
  python3 %(prog)s ke_seq_xxx.bin --layers all

  # Dump layer 0, only cells 0-15
  python3 %(prog)s ke_seq_xxx.bin --layers 0 --cells 0-15

  # Dump layer 0, only K (no V)
  python3 %(prog)s ke_seq_xxx.bin --layers 0 --only k
""",
    )
    p.add_argument("bin_file", help="KV cache .bin file")
    p.add_argument("-o", "--output", help="Output .txt path (default: <bin_file>.txt)")
    p.add_argument(
        "--layers",
        help="Layers to dump values for: 'all', '0', '0,1,31', '0-3', '0-3,31'. "
             "Omit to show stats only (no values).",
    )
    p.add_argument(
        "--cells",
        help="Cell range to dump: 'all', '0-15', '100-200'. Default: all.",
    )
    p.add_argument(
        "--only",
        choices=["k", "v"],
        help="Only dump K or V values (default: both).",
    )

    args = p.parse_args()
    txt_path = args.output or str(Path(args.bin_file).with_suffix(".txt"))
    parse_and_dump(args.bin_file, txt_path, args.layers, args.cells, args.only)


if __name__ == "__main__":
    main()
