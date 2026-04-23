#!/usr/bin/env python3
"""
Tokenize a directory of .txt files into a flat binary dataset.

Reads every non-empty .txt file, tokenises with the gpt2 tokenizer,
separates documents with EOS, and writes a single uint16 binary file
that train_lite.py can load directly.

Usage:
    python training/prepare_corpus.py \
        --input-dir /path/to/txt_files \
        --output-dir data/corpus

Outputs:
    data/corpus/corpus.bin   -- flat uint16 token array
    data/corpus/meta.json    -- token count, file count, skipped files
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from transformers import GPT2TokenizerFast


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir",  type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--tokenizer",  default="gpt2")
    p.add_argument("--min-tokens", type=int, default=32,
                   help="Skip documents with fewer than this many tokens")
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = GPT2TokenizerFast.from_pretrained(args.tokenizer)
    eos = tokenizer.eos_token_id  # 50256

    txt_files = sorted(args.input_dir.glob("*.txt"))
    print(f"Found {len(txt_files)} .txt files in {args.input_dir}")

    all_tokens: list[np.ndarray] = []
    total_tokens = 0
    skipped_empty = 0
    skipped_short = 0
    skipped_errors = 0
    processed = 0

    for i, path in enumerate(txt_files):
        if path.stat().st_size == 0:
            skipped_empty += 1
            continue

        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        except Exception as e:
            print(f"  [skip] {path.name}: read error — {e}")
            skipped_errors += 1
            continue

        if not text:
            skipped_empty += 1
            continue

        ids = tokenizer.encode(text, add_special_tokens=False)

        if len(ids) < args.min_tokens:
            skipped_short += 1
            continue

        # append document tokens + EOS separator
        doc = np.array(ids + [eos], dtype=np.uint16)
        all_tokens.append(doc)
        total_tokens += len(doc)
        processed += 1

        if (i + 1) % 100 == 0 or (i + 1) == len(txt_files):
            print(f"  {i+1}/{len(txt_files)}  processed={processed}  "
                  f"tokens={total_tokens:,}", flush=True)

    print(f"\nTokenisation complete:")
    print(f"  Processed  : {processed} files")
    print(f"  Skipped    : {skipped_empty} empty, "
          f"{skipped_short} too short (<{args.min_tokens} tok), "
          f"{skipped_errors} read errors")
    print(f"  Total tokens: {total_tokens:,}  "
          f"({total_tokens/1e6:.1f}M)")

    # write flat binary
    out_bin = args.output_dir / "corpus.bin"
    flat = np.concatenate(all_tokens)
    flat.tofile(out_bin)
    print(f"  Written    : {out_bin}  ({out_bin.stat().st_size/1e6:.1f} MB)")

    # write metadata
    meta = {
        "total_tokens": int(total_tokens),
        "files_processed": processed,
        "files_skipped_empty": skipped_empty,
        "files_skipped_short": skipped_short,
        "files_skipped_errors": skipped_errors,
        "tokenizer": args.tokenizer,
        "eos_token_id": int(eos),
        "dtype": "uint16",
        "input_dir": str(args.input_dir),
    }
    meta_path = args.output_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  Metadata   : {meta_path}")
    print(f"\nDone. Load with CorpusDataset('{out_bin}', seq_len=512)")


if __name__ == "__main__":
    main()
