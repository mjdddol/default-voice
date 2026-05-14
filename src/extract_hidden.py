#!/usr/bin/env python
"""
Hidden state extraction for E6: mechanistic supplement.

Re-encodes existing JSONL generations through the model with
output_hidden_states=True, saves last-token (or mean-pooled response)
hidden states from a specified layer to NPZ.

Used for the linear-probe analysis (src/probe.py) that strengthens the
persistence finding by showing the family / register signature is
encoded in hidden states, not just surface output.

Usage:
    python -m src.extract_hidden \
        --model_name qwen3_it \
        --input outputs/generations/en/empty_user/qwen3_it.jsonl \
        --output outputs/hidden/qwen3_it__empty_user.npz \
        --layer -1 --n_samples 200
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_models_cfg() -> dict:
    return yaml.safe_load(open(CONFIG_DIR / "models.yaml"))


def find_model_spec(name: str, cfg: dict) -> dict:
    tier_keys = (
        "cross_family",
        "trajectory_olmo3",
        "trajectory_ministral3",
        "scale_qwen3",
        "reasoning_pairs",
    )
    for tier in tier_keys:
        for entry in cfg.get(tier, []):
            if entry["name"] == name:
                return entry
    raise KeyError(f"model '{name}' not found in models.yaml")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--input", required=True, help="JSONL generations file")
    ap.add_argument("--output", required=True, help="NPZ destination")
    ap.add_argument("--layer", type=int, default=-1,
                    help="hidden-state layer index (-1 = final)")
    ap.add_argument("--n_samples", type=int, default=200)
    ap.add_argument("--max_tokens", type=int, default=512,
                    help="truncate input to at most this many tokens")
    ap.add_argument("--pool", choices=["last", "mean"], default="last",
                    help="how to reduce per-token states to a single vector")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"[skip] {out_path} already exists")
        return 0

    cfg = load_models_cfg()
    spec = find_model_spec(args.model_name, cfg)
    quant = spec.get("quantization")

    # Lazy imports to keep --help fast
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"[init] loading {spec['hf_id']} (quant={quant or 'none'})")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(spec["hf_id"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(torch_dtype=torch.bfloat16, trust_remote_code=True)
    if quant == "awq":
        # transformers loads AWQ via auto if installed; fall back to bf16 if not
        try:
            from awq import AutoAWQForCausalLM      # noqa: F401
        except ImportError:
            print(f"[warn] AWQ requested but `awq` not installed; "
                  f"falling back to bf16: this may OOM for >24GB models")
    model = AutoModelForCausalLM.from_pretrained(
        spec["hf_id"], device_map=args.device, **model_kwargs)
    model.eval()
    print(f"[init] loaded in {time.time() - t0:.1f}s")

    # Load JSONL up to n_samples
    rows = []
    with open(args.input) as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= args.n_samples:
                break
    print(f"[data] {len(rows)} samples")

    # Re-encode each (input + generation) and extract hidden states
    hidden_vecs = []
    sample_ids = []
    n_skipped = 0

    with torch.no_grad():
        for i, row in enumerate(rows):
            text = (row.get("input_text") or "") + row["text"]
            if not text.strip():
                n_skipped += 1
                continue
            enc = tokenizer(text, return_tensors="pt",
                            truncation=True, max_length=args.max_tokens).to(args.device)
            try:
                out = model(**enc, output_hidden_states=True)
            except RuntimeError as e:
                print(f"[err] sample {i}: {e}")
                n_skipped += 1
                continue
            # out.hidden_states is a tuple of (n_layers+1) tensors,
            # each [batch=1, seq_len, hidden_dim].
            h = out.hidden_states[args.layer][0]   # [seq_len, hidden_dim]
            if args.pool == "last":
                vec = h[-1]
            else:
                vec = h.mean(dim=0)
            hidden_vecs.append(vec.float().cpu().numpy())
            sample_ids.append(row.get("sample_id", i))
            if (i + 1) % 50 == 0:
                print(f"[progress] {i + 1}/{len(rows)}")

    H = np.stack(hidden_vecs)
    sids = np.asarray(sample_ids, dtype=np.int64)
    np.savez(out_path,
             hidden=H,
             sample_ids=sids,
             model_name=args.model_name,
             model_hf_id=spec["hf_id"],
             model_family=spec.get("family"),
             model_stage=spec.get("stage"),
             layer=args.layer,
             pool=args.pool,
             n_skipped=n_skipped)
    print(f"[done] wrote {len(H)} vectors of dim {H.shape[1]} to {out_path}")
    if n_skipped:
        print(f"[done] skipped {n_skipped} samples (errors / empty)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
