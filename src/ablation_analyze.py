#!/usr/bin/env python
"""
Analysis + figure for the register-direction ablation experiment.

Reads:
    outputs/ablation/generations/<family>/<cond>.jsonl
Computes:
    feature pipeline (src.features) for each (family, condition)
Saves:
    outputs/ablation/features/<family>/<cond>.parquet
    outputs/ablation/summary.csv          (one row per (family, cond))
    paper/assets/fig_ablation.pdf         (4 register features x 4 conditions)

Conditions (fixed order):
    empty_baseline  empty_anti  empty_ablate  empty_anti_ablate

Focus features:
    assistant_phrase_ratio       headline register density
    refusal_density              spontaneous refusal-like phrasing
    hedging_density              softening / uncertainty
    first_person_density         self-reference
    mean_sentence_length         (paired check)
    n_words                      (paired check)
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({"font.family": "DejaVu Sans"})

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "ablation"
ASSETS = ROOT / "paper" / "assets"

FAMILIES = ["llama31_it", "qwen3_it", "olmo3_it", "aya_expanse", "granite41_8b_it"]
DISPLAY = {
    "llama31_it": "Llama 3.1",
    "qwen3_it": "Qwen 3",
    "olmo3_it": "OLMo 3",
    "aya_expanse": "Aya Expanse",
    "granite41_8b_it": "Granite 4.1",
}
CONDS = ["empty_baseline", "empty_anti", "empty_ablate", "empty_anti_ablate"]
COND_DISPLAY = {
    "empty_baseline": "Baseline",
    "empty_anti":     "Prompt-anti",
    "empty_ablate":   "Direction\nablation",
    "empty_anti_ablate": "Prompt-anti\n+ ablation",
}
FEATURES = [
    ("assistant_phrase_ratio", "Assistant phrase (\\%)",   100),
    ("first_person_density",   "Self-reference (\\%)",     100),
    ("mean_sentence_length",   "Sentence length (words)",  1),
]


# -------------------- feature extraction --------------------

def ensure_features():
    """Run src.features on each (family, condition) that lacks a parquet."""
    py = sys.executable
    for fam in FAMILIES:
        gen_dir = OUT / "generations" / fam
        feat_dir = OUT / "features" / fam
        feat_dir.mkdir(parents=True, exist_ok=True)
        for cond in CONDS:
            jsonl = gen_dir / f"{cond}.jsonl"
            parquet = feat_dir / f"{cond}.parquet"
            if not jsonl.exists():
                print(f"  missing {jsonl}, skip")
                continue
            if parquet.exists() and parquet.stat().st_size > 0:
                continue
            print(f"  extract: {fam} / {cond}")
            subprocess.run(
                [py, "-m", "src.features", "extract",
                 "--input", str(jsonl), "--output", str(parquet)],
                check=False,
                cwd=ROOT,
            )


# -------------------- summary --------------------

def build_summary() -> pd.DataFrame:
    rows = []
    for fam in FAMILIES:
        for cond in CONDS:
            p = OUT / "features" / fam / f"{cond}.parquet"
            if not p.exists():
                continue
            df = pd.read_parquet(p)
            row = {"family": fam, "condition": cond, "n": len(df)}
            for fc, _, _ in FEATURES:
                if fc in df.columns:
                    row[fc] = float(df[fc].mean())
                    row[fc + "_sd"] = float(df[fc].std())
                else:
                    row[fc] = np.nan
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "summary.csv", index=False)
    return out


# -------------------- figure --------------------

def fig_ablation(summary: pd.DataFrame):
    ASSETS.mkdir(parents=True, exist_ok=True)
    n_feat = len(FEATURES)
    # Wide horizontal layout for figure* (spanning two columns).
    fig, axes = plt.subplots(1, n_feat, figsize=(11.0, 2.55),
                             sharex=False)
    cmap = plt.get_cmap("viridis")
    fam_colors = {fam: cmap(0.10 + 0.78 * i / max(len(FAMILIES) - 1, 1))
                  for i, fam in enumerate(FAMILIES)}

    for ax_idx, (ax, (fc, label, mult)) in enumerate(zip(axes, FEATURES)):
        for fam in FAMILIES:
            vals = []
            for cond in CONDS:
                m = summary[(summary.family == fam) &
                            (summary.condition == cond)]
                vals.append(m[fc].iloc[0] * mult if len(m) else np.nan)
            x = np.arange(len(CONDS))
            ax.plot(x, vals, "-o",
                    color=fam_colors[fam], markersize=4.5, linewidth=1.6,
                    alpha=0.92, label=DISPLAY[fam])

        ax.set_xticks(np.arange(len(CONDS)))
        ax.set_xticklabels([COND_DISPLAY[c] for c in CONDS],
                           fontsize=8.5, rotation=0)
        ax.set_ylabel(label, fontsize=9.5)
        ax.tick_params(axis="y", labelsize=8.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.5, 1.06),
        ncol=len(FAMILIES), fontsize=9, frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = ASSETS / "fig_ablation.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# -------------------- entrypoint --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_features", action="store_true")
    args = ap.parse_args()
    if not args.skip_features:
        ensure_features()
    summary = build_summary()
    print("\n=== Per-(family, condition) feature summary ===")
    # pivot-style display
    for fc, label, _ in FEATURES:
        print(f"\n  {label}")
        for fam in FAMILIES:
            vals = []
            for cond in CONDS:
                m = summary[(summary.family == fam) &
                            (summary.condition == cond)]
                vals.append(m[fc].iloc[0] if len(m) else np.nan)
            print(f"    {DISPLAY[fam]:<14} " +
                  "  ".join(f"{v:8.4f}" for v in vals))

    fig_ablation(summary)


if __name__ == "__main__":
    main()
