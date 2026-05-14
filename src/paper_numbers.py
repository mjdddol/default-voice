#!/usr/bin/env python
"""
Compute the precise numbers cited in the paper.
Writes machine-readable JSON to analysis/paper_numbers.json
and a Markdown summary to analysis/paper_numbers.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(".")
OUT = PROJ / "analysis"
OUT.mkdir(exist_ok=True)


def fold(post: float, pre: float) -> float:
    pre = max(pre, 1e-6)
    return post / pre


def main() -> None:
    df = pd.read_parquet(PROJ / "outputs" / "all_features.parquet")
    results: dict = {}

    # ============================================================
    # F1: Format-gating gradient: cross-family, instruct models only
    # ============================================================
    inst = df[df["model_stage"] == "instruct"].copy()
    # Group by family x condition
    f1_table = inst.groupby(["model_family", "condition"])[[
        "assistant_phrase_ratio", "first_person_density",
        "hedging_density", "refusal_density", "mean_sentence_length",
        "n_words"
    ]].mean()
    # Pivot to (family, feature) x condition
    f1 = {}
    for fam in inst["model_family"].dropna().unique():
        f1[fam] = {}
        for cond in ["bos_only", "chat_template_no_gen_prompt",
                     "continue_cue", "empty_user"]:
            try:
                row = inst[(inst.model_family == fam) & (inst.condition == cond)]
                if len(row) > 50:
                    f1[fam][cond] = {
                        "assistant_phrase_ratio": float(row["assistant_phrase_ratio"].mean()),
                        "first_person_density":   float(row["first_person_density"].mean()),
                        "hedging_density":        float(row["hedging_density"].mean()),
                        "refusal_density":        float(row["refusal_density"].mean()),
                        "mean_sentence_length":   float(row["mean_sentence_length"].mean()),
                        "n_words":                float(row["n_words"].mean()),
                        "n_samples":              int(len(row)),
                    }
            except Exception:
                pass
    results["F1_cross_family"] = f1

    # F1 fold change empty_user / bos_only per family
    fold_table = {}
    for fam, conds in f1.items():
        if "empty_user" in conds and "bos_only" in conds:
            fold_table[fam] = {
                feat: fold(conds["empty_user"][feat], conds["bos_only"][feat])
                for feat in ["assistant_phrase_ratio", "first_person_density",
                              "hedging_density", "mean_sentence_length", "n_words"]
            }
    results["F1_fold_changes"] = fold_table

    # ============================================================
    # F2: OLMo 3 stage trajectory (empty_user condition)
    # ============================================================
    olmo3_order = ["olmo3_t_base", "olmo3_t_sft", "olmo3_t_dpo",
                   "olmo3_t_rlvr", "olmo3_t_rl_code", "olmo3_t_rl_math",
                   "olmo3_t_think_sft", "olmo3_t_think"]
    f2 = {}
    for cond in ["bos_only", "empty_user"]:
        f2[cond] = {}
        for m in olmo3_order:
            row = df[(df.model == m) & (df.condition == cond)]
            if len(row) > 50:
                f2[cond][m] = {
                    "assistant_phrase_ratio": float(row["assistant_phrase_ratio"].mean()),
                    "first_person_density":   float(row["first_person_density"].mean()),
                    "hedging_density":        float(row["hedging_density"].mean()),
                    "refusal_density":        float(row["refusal_density"].mean()),
                    "apology_density":        float(row["apology_density"].mean()),
                    "mean_sentence_length":   float(row["mean_sentence_length"].mean()),
                    "ttr":                    float(row["ttr"].mean()),
                    "markdown_density":       float(row["markdown_density"].mean()),
                    "n_words":                float(row["n_words"].mean()),
                    "n_samples":              int(len(row)),
                }
    results["F2_olmo3_trajectory"] = f2

    # Key OLMo 3 fold changes (base -> RLVR under empty_user)
    if ("empty_user" in f2 and
        "olmo3_t_base" in f2["empty_user"] and
        "olmo3_t_rlvr" in f2["empty_user"]):
        b = f2["empty_user"]["olmo3_t_base"]
        r = f2["empty_user"]["olmo3_t_rlvr"]
        s = f2["empty_user"].get("olmo3_t_sft", {})
        d = f2["empty_user"].get("olmo3_t_dpo", {})
        results["F2_key_folds_olmo3"] = {
            "assistant_phrase_base_to_rlvr": fold(r["assistant_phrase_ratio"],
                                                   b["assistant_phrase_ratio"]),
            "refusal_sft_to_dpo":           fold(d["refusal_density"],
                                                   s["refusal_density"]) if s and d else None,
            "n_words_base_to_rlvr":         fold(r["n_words"], b["n_words"]),
            "markdown_base_to_rlvr":        fold(r["markdown_density"],
                                                   b["markdown_density"]),
            "first_person_base_to_rlvr":    fold(r["first_person_density"],
                                                   b["first_person_density"]),
            "hedging_base_to_rlvr":         fold(r["hedging_density"],
                                                   b["hedging_density"]),
            "ttr_base_to_rlvr":             fold(r["ttr"], b["ttr"]),
        }

    # ============================================================
    # F2: Ministral 3 trajectory (Base / Instruct / Reasoning)
    # ============================================================
    ministral_models = ["m3_3_base", "m3_3_it", "m3_3_reason",
                        "m3_8_base", "m3_8_it", "m3_8_reason",
                        "m3_14_base", "m3_14_it", "m3_14_reason"]
    f2b = {}
    for cond in ["bos_only", "empty_user"]:
        f2b[cond] = {}
        for m in ministral_models:
            row = df[(df.model == m) & (df.condition == cond)]
            if len(row) > 30:
                f2b[cond][m] = {
                    "assistant_phrase_ratio": float(row["assistant_phrase_ratio"].mean()),
                    "first_person_density":   float(row["first_person_density"].mean()),
                    "hedging_density":        float(row["hedging_density"].mean()),
                    "refusal_density":        float(row["refusal_density"].mean()),
                    "mean_sentence_length":   float(row["mean_sentence_length"].mean()),
                    "n_words":                float(row["n_words"].mean()),
                    "n_samples":              int(len(row)),
                }
    results["F2b_ministral3_trajectory"] = f2b

    # Ministral 3 8B base -> it fold change (for replication claim)
    if ("empty_user" in f2b
        and "m3_8_base" in f2b["empty_user"]
        and "m3_8_it" in f2b["empty_user"]):
        b = f2b["empty_user"]["m3_8_base"]
        i = f2b["empty_user"]["m3_8_it"]
        results["F2b_key_folds_ministral3_8b"] = {
            "assistant_phrase_base_to_it": fold(i["assistant_phrase_ratio"],
                                                  b["assistant_phrase_ratio"]),
            "n_words_base_to_it":         fold(i["n_words"], b["n_words"]),
        }

    # ============================================================
    # F3: Persistence asymmetry (|Delta_pro| vs |Delta_anti|)
    # ============================================================
    pairs = [
        ("self_ref",  "first_person_density"),
        ("hedging",   "hedging_density"),
        ("assistant", "assistant_phrase_ratio"),
        ("verbosity", "mean_sentence_length"),
    ]
    f3 = {}
    for model_name in df["model"].unique():
        ctrl = df[(df.model == model_name) & (df.condition == "control_neutral")]
        if len(ctrl) == 0:
            continue
        f3[model_name] = {}
        for tag, feat in pairs:
            pro = df[(df.model == model_name) & (df.condition == f"pro_{tag}")]
            ant = df[(df.model == model_name) & (df.condition == f"anti_{tag}")]
            if len(pro) and len(ant):
                d_pro = pro[feat].mean() - ctrl[feat].mean()
                d_ant = ant[feat].mean() - ctrl[feat].mean()
                f3[model_name][tag] = {
                    "delta_pro":   float(d_pro),
                    "delta_anti":  float(d_ant),
                    "abs_pro":     float(abs(d_pro)),
                    "abs_anti":    float(abs(d_ant)),
                    "ratio_pro_anti": float(abs(d_pro) / max(abs(d_ant), 1e-6)),
                }
    results["F3_persistence"] = f3

    # Mean |Delta_pro| / |Delta_anti| ratio
    all_ratios = [v[tag]["ratio_pro_anti"]
                  for v in f3.values() for tag in v
                  if "ratio_pro_anti" in v[tag]]
    results["F3_mean_ratio_pro_over_anti"] = (
        float(np.mean(all_ratios)) if all_ratios else None
    )
    results["F3_median_ratio"] = (
        float(np.median(all_ratios)) if all_ratios else None
    )

    # ============================================================
    # F4: Hidden state probe (read from existing analysis JSON if present)
    # ============================================================
    probe_files = list((PROJ / "analysis").glob("probe_*.json"))
    f4 = {}
    for pf in probe_files:
        f4[pf.stem] = json.load(open(pf))
    results["F4_probe"] = f4

    # ============================================================
    # Write
    # ============================================================
    json.dump(results, open(OUT / "paper_numbers.json", "w"),
              indent=2, ensure_ascii=False)
    print(f"[done] wrote {OUT}/paper_numbers.json "
          f"({len(json.dumps(results)):,} chars)")

    # ============================================================
    # Markdown summary (for quick eyeball)
    # ============================================================
    lines = ["# Paper Numbers: auto-generated", ""]

    lines.append("## F1: Fold change empty_user / bos_only per family")
    lines.append("")
    lines.append("| Family | assistant_phrase | first_person | hedging | sentence_len | n_words |")
    lines.append("|---|---|---|---|---|---|")
    for fam, vals in fold_table.items():
        lines.append(f"| {fam} | {vals['assistant_phrase_ratio']:.1f}x | "
                     f"{vals['first_person_density']:.1f}x | "
                     f"{vals['hedging_density']:.1f}x | "
                     f"{vals['mean_sentence_length']:.2f}x | "
                     f"{vals['n_words']:.2f}x |")

    lines.append("")
    lines.append("## F2: OLMo 3 trajectory (key folds, base -> RLVR under empty_user)")
    lines.append("")
    if "F2_key_folds_olmo3" in results:
        for k, v in results["F2_key_folds_olmo3"].items():
            if v is not None:
                lines.append(f"- **{k}**: {v:.2f}x")

    lines.append("")
    lines.append("## F3: Persistence asymmetry (|Delta_pro| / |Delta_anti| mean ratio)")
    lines.append("")
    if results.get("F3_mean_ratio_pro_over_anti") is not None:
        lines.append(f"- mean: {results['F3_mean_ratio_pro_over_anti']:.2f}")
        lines.append(f"- median: {results['F3_median_ratio']:.2f}")

    lines.append("")
    lines.append("## F4: Hidden state probe")
    lines.append("")
    for k, v in f4.items():
        if isinstance(v, dict):
            if "cv_mean" in v:
                lines.append(f"- {k}: cv={v['cv_mean']:.3f} +/- {v.get('cv_std',0):.3f}, n={v.get('n_samples','?')}")
            elif "in_acc" in v:
                lines.append(f"- {k}: in={v['in_acc']:.3f}, transfer={v.get('transfer_acc',0):.3f}")

    with open(OUT / "paper_numbers.md", "w") as f:
        f.write("\n".join(lines))
    print(f"[done] wrote {OUT}/paper_numbers.md")


if __name__ == "__main__":
    main()
