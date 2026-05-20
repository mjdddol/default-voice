#!/usr/bin/env python
"""
Raw-sample-weighted vs deduplicated-unique feature means on the
main-panel cells (7 instruct families x 3 prompt conditions, English).

The body of the paper reports feature means over the deduplicated set of
unique generations per (model, condition). This script contrasts that
estimand with raw-sample-weighted means computed on the *same generations
the body uses* (so the comparison stays self-consistent with Figure 1,
Appendix C, Appendix A, etc.).

Implementation:
  * Load per-sample features from outputs/all_features.parquet (the
    source of truth for the body).
  * For each (model, condition) cell, drop sample_id duplicates that
    exist from merged feature runs.
  * Look up the corresponding generation texts from
    outputs/generations/en/<condition>/<model>.jsonl so we can group
    rows by string (text dedup, not just sample_id dedup).
  * Apply the body's 30-word filter.
  * Compute raw_mean (every retained row counts once, so degenerate
    repetitions are weighted by their multiplicity) and dedup_mean
    (each unique string counts once, averaging feature values across
    rows that share that string).

Writes:
  outputs/raw_vs_dedup_body_full.csv   (per-cell, per-feature)
  outputs/raw_vs_dedup_main_panel.csv  (compatibility alias, same content)
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GEN_ROOT = ROOT / "outputs" / "generations" / "en"
FEATURES_PARQUET = ROOT / "outputs" / "all_features.parquet"
OUT_FULL = ROOT / "outputs" / "raw_vs_dedup_body_full.csv"
OUT_COMPAT = ROOT / "outputs" / "raw_vs_dedup_main_panel.csv"

MODELS = {
    "llama31_it":      "Llama 3.1",
    "qwen3_it":        "Qwen 3",
    "olmo3_it":        "OLMo 3",
    "granite41_8b_it": "Granite 4.1",
    "gemma4_it":       "Gemma 4",
    "aya_expanse":     "Aya Expanse",
    "m3_8_it":         "Ministral 3 8B",
}
CONDS = {
    "bos_only":                    "BOS-only",
    "chat_template_no_gen_prompt": "Chat-no-gen",
    "empty_user":                  "Empty-user",
}
FEATS = {
    "assistant_phrase_ratio": "Assist.-phrase",
    "refusal_density":        "Refusal-like",
    "first_person_density":   "Self-ref.",
    "hedging_density":        "Hedging",
    "mean_sentence_length":   "Sent. length",
}
MIN_WORDS = 30


def load_text_lookup() -> dict:
    """Map (model_key, condition_key, sample_id) -> generation text."""
    out = {}
    for ckey in CONDS:
        for mkey in MODELS:
            p = GEN_ROOT / ckey / f"{mkey}.jsonl"
            if not p.exists():
                continue
            with open(p) as f:
                for line in f:
                    d = json.loads(line)
                    out[(mkey, ckey, d["sample_id"])] = d["text"]
    return out


def main():
    df = pd.read_parquet(FEATURES_PARQUET)
    text_lookup = load_text_lookup()
    if not text_lookup:
        print(
            f"WARNING: no generation jsonl found under {GEN_ROOT}.\n"
            "  The pre-computed result is in "
            "outputs/raw_vs_dedup_body_full.csv (one row per "
            "(model, condition, feature)). Re-running this script "
            "requires the raw generations, which are sampled with "
            "src/sample.py and not bundled in the lightweight "
            "supplementary archive.")
        return
    print(f"loaded {len(text_lookup)} (model, cond, sample_id) texts; "
          f"{len(df)} feature rows in {FEATURES_PARQUET.name}")

    rows = []
    for mkey, mlabel in MODELS.items():
        for ckey, clabel in CONDS.items():
            sub = df[(df.model == mkey) &
                     (df.condition == ckey) &
                     (df.language == "en")]
            if not len(sub):
                continue
            # Drop sample_id duplicates that arise from merged feature runs.
            sub = sub.drop_duplicates(subset=["sample_id"]).copy()
            # Attach raw text by sample_id.
            sub["text"] = sub.sample_id.apply(
                lambda s, m=mkey, c=ckey: text_lookup.get((m, c, int(s))))
            sub = sub[sub.text.notna()]
            # Body's 30-word filter.
            sub = sub[sub.n_words >= MIN_WORDS]
            if not len(sub):
                continue
            for fkey, flabel in FEATS.items():
                if fkey not in sub.columns:
                    continue
                # Group rows by raw text — each unique string counted once.
                grp = sub.groupby("text")[fkey].mean()
                raw_mean = float(sub[fkey].mean())
                dedup_mean = float(grp.mean())
                rows.append({
                    "model":     mlabel,
                    "condition": clabel,
                    "feature":   flabel,
                    "raw_n":     int(len(sub)),
                    "unique_n":  int(grp.size),
                    "raw_mean":  raw_mean,
                    "dedup_mean": dedup_mean,
                    "diff":      raw_mean - dedup_mean,
                })
            print(f"  {mlabel:<16} {clabel:<14} "
                  f"raw_n={len(sub):>4}  unique_n={sub.text.nunique():>4}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_FULL, index=False)
    out.to_csv(OUT_COMPAT, index=False)
    print(f"\nwrote {OUT_FULL} ({len(out)} rows)")

    # Summary
    print("\n=== Per-feature |raw - dedup| summary (across all 20 cells) ===")
    for flabel in FEATS.values():
        s = out[out.feature == flabel]
        if not len(s):
            continue
        unit = "pp" if flabel != "Sent. length" else "words"
        mult = 100.0 if flabel != "Sent. length" else 1.0
        print(f"  {flabel:<16}  max|diff|={s['diff'].abs().max()*mult:.3f}{unit}  "
              f"median|diff|={s['diff'].abs().median()*mult:.4f}{unit}")

    # Headline cell-by-cell for assistant-phrase
    print("\n=== Assistant-phrase per (model, condition), raw / dedup % ===")
    for m in MODELS.values():
        print(f"\n{m}:")
        for c in CONDS.values():
            sub = out[(out.model == m) &
                      (out.condition == c) &
                      (out.feature == "Assist.-phrase")]
            if not len(sub):
                continue
            r = sub.iloc[0]
            print(f"  {c:<14} raw={r.raw_mean*100:6.3f}%  "
                  f"dedup={r.dedup_mean*100:6.3f}%  "
                  f"diff={(r.raw_mean-r.dedup_mean)*100:+6.3f}pp  "
                  f"(N raw={int(r.raw_n):>4} uniq={int(r.unique_n):>4})")


if __name__ == "__main__":
    main()
