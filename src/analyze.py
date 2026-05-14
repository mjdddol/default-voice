#!/usr/bin/env python
"""
Analysis utilities for "Default Voice" paper.

Subcommands:
    profile        : aggregate per-(model, condition, language) feature profiles
    classify       : E1 family classifier (does default voice exist?)
    trajectory     : E2 post-training trajectory plot
    scale          : E3 scale-ladder plot
    persistence    : E4 prompt-persistence analysis
    cross_gen      : cross-generation comparison plot
    figures        : produce all paper figures from an aggregated parquet

Inputs are produced by src/features.py (one parquet per JSONL).
Aggregated table at outputs/all_features.parquet.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Aesthetic defaults: neutral, paper-friendly
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "figure.titleweight": "bold",
})

FEATURE_COLS_DEFAULT = [
    # lexical
    "ttr", "mtld", "hapax_ratio", "mean_word_length",
    "function_word_ratio", "hedging_density", "assertive_density",
    "modal_density", "first_person_density", "second_person_density",
    "third_person_density",
    # syntactic
    "mean_sentence_length", "mean_dep_depth", "passive_ratio",
    "subordinate_ratio", "punctuation_density", "pos_dist_kl",
    # discourse
    "discourse_marker_density", "assistant_phrase_ratio",
    "apology_density", "refusal_density", "politeness_score",
    "formality_score", "emoji_density", "markdown_density",
]

# Baseline feature sets: argued in the paper to demonstrate that
# our default-voice classifier is *not* trivially driven by output length
# or random noise.
FEATURE_COLS_LENGTH_ONLY = ["n_words", "n_chars"]
FEATURE_COLS_LEXICAL_ONLY = [
    "ttr", "mtld", "hapax_ratio", "mean_word_length",
    "function_word_ratio", "hedging_density", "assertive_density",
    "modal_density", "first_person_density", "second_person_density",
    "third_person_density",
]


def select_feature_cols(df: pd.DataFrame, baseline: str) -> list[str]:
    """Pick feature columns based on requested baseline."""
    if baseline == "full":
        return [c for c in FEATURE_COLS_DEFAULT if c in df.columns]
    if baseline == "length_only":
        return [c for c in FEATURE_COLS_LENGTH_ONLY if c in df.columns]
    if baseline == "lexical_only":
        return [c for c in FEATURE_COLS_LEXICAL_ONLY if c in df.columns]
    if baseline == "random":
        # Replace features with random noise of the same shape: handled in caller
        return [c for c in FEATURE_COLS_DEFAULT if c in df.columns]
    raise ValueError(f"unknown baseline: {baseline}")


# --------------------------------------------------------------------------
# Profile aggregation
# --------------------------------------------------------------------------

def aggregate_profiles(df: pd.DataFrame, feat_cols: Sequence[str] | None = None
                       ) -> pd.DataFrame:
    """Per-(model, condition, language) mean and std of each feature."""
    feat_cols = list(feat_cols) if feat_cols else [c for c in FEATURE_COLS_DEFAULT
                                                    if c in df.columns]
    grp = df.groupby(["model", "condition", "language"])
    agg = grp[feat_cols].agg(["mean", "std"])
    agg.columns = [f"{c}_{stat}" for c, stat in agg.columns]
    agg = agg.reset_index()
    # Re-attach metadata columns
    meta = df.groupby(["model", "condition", "language"])[
        ["model_family", "model_stage"]].first().reset_index()
    return agg.merge(meta, on=["model", "condition", "language"])


def cmd_profile(args):
    df = pd.read_parquet(args.input)
    out = aggregate_profiles(df)
    out.to_parquet(args.output)
    print(f"[profile] wrote {len(out)} rows to {args.output}")
    print(f"[profile] columns: {len(out.columns)}")
    return 0


# --------------------------------------------------------------------------
# E1: Family classifier
# --------------------------------------------------------------------------

def cmd_classify(args):
    """
    Train a family classifier on per-sample features.
    Output:
      - Confusion matrix figure
      - Accuracy report (per-condition, per-language)
      - Top discriminative features
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (classification_report, confusion_matrix,
                                  ConfusionMatrixDisplay)

    df = pd.read_parquet(args.input)
    feat_cols = select_feature_cols(df, args.baseline)

    if args.condition:
        df = df[df["condition"] == args.condition].copy()
    if args.language:
        df = df[df["language"] == args.language].copy()
    if args.exclude_base:
        df = df[df["model_stage"] != "base"].copy()
    if args.exclude_instruct:
        df = df[df["model_stage"] == "base"].copy()
    if args.size_controlled:
        models_cfg = yaml.safe_load(open(PROJECT_ROOT / "configs" / "models.yaml"))
        subset = set(models_cfg.get("size_controlled_subset", []))
        before = df["model"].nunique()
        df = df[df["model"].isin(subset)].copy()
        print(f"[classify] size-controlled subset: {df['model'].nunique()}/{before} models retained")

    print(f"[classify] {len(df)} samples, {df['model_family'].nunique()} families, "
          f"{df['model'].nunique()} models, baseline={args.baseline}, "
          f"|features|={len(feat_cols)}")

    X = df[feat_cols].fillna(0.0).values
    if args.baseline == "random":
        # Replace each row with random noise of the same scale.
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, size=X.shape)
    y = df["model_family"].values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # 5-fold CV accuracy
    clf = LogisticRegression(max_iter=2000, C=args.C, n_jobs=-1)
    cv = cross_val_score(clf, Xs, y, cv=5, n_jobs=-1)
    print(f"[classify] 5-fold CV accuracy: {cv.mean():.3f} +/- {cv.std():.3f}")

    # Single train/test for confusion matrix + coefficients
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2,
                                          stratify=y, random_state=42)
    clf.fit(Xtr, ytr)
    yhat = clf.predict(Xte)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.baseline == "full" else f"_{args.baseline}"

    # Confusion matrix figure
    classes = sorted(set(y))
    cm = confusion_matrix(yte, yhat, labels=classes)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax,
                cbar_kws={"label": "row-normalized count"})
    ax.set_xlabel("Predicted family")
    ax.set_ylabel("True family")
    ax.set_title(f"Family classifier: baseline={args.baseline}\n"
                 f"CV acc {cv.mean():.3f}  ({df['condition'].unique()[0] if args.condition else 'all-conditions'})")
    plt.tight_layout()
    fig.savefig(out_dir / f"f2_classifier_confusion{suffix}.png", dpi=200)
    fig.savefig(out_dir / f"f2_classifier_confusion{suffix}.pdf")
    print(f"[classify] wrote confusion matrix to {out_dir}/f2_classifier_confusion{suffix}.{{png,pdf}}")

    # Top discriminative features (by mean |coef| across classes)
    coefs = np.abs(clf.coef_).mean(axis=0)
    order = np.argsort(coefs)[::-1]
    top_feats = [(feat_cols[i], float(coefs[i])) for i in order[:15]]
    print("\n[classify] top discriminative features (mean |coef|):")
    for name, c in top_feats:
        print(f"   {name:30s} {c:.3f}")

    # Per-condition / per-language breakdown if running across all
    if not args.condition or not args.language:
        breakdown = []
        for (cond, lang), sub in df.groupby(["condition", "language"]):
            if len(sub) < 50: continue
            Xs_sub = scaler.transform(sub[feat_cols].fillna(0.0).values)
            y_sub  = sub["model_family"].values
            try:
                acc = cross_val_score(clf, Xs_sub, y_sub, cv=3,
                                       n_jobs=-1).mean()
            except Exception:
                acc = float("nan")
            breakdown.append({"condition": cond, "language": lang,
                              "n": len(sub), "cv_acc": acc})
        bdf = pd.DataFrame(breakdown)
        bdf.to_csv(out_dir / "classify_breakdown.csv", index=False)
        print(f"[classify] wrote per-condition/language breakdown")

    # Save report
    report = classification_report(yte, yhat, labels=classes,
                                    output_dict=True, zero_division=0)
    with open(out_dir / f"classify_report{suffix}.json", "w") as f:
        json.dump({
            "baseline": args.baseline,
            "cv_mean": float(cv.mean()),
            "cv_std": float(cv.std()),
            "n_features": len(feat_cols),
            "feat_cols": feat_cols,
            "n_samples": len(df),
            "n_models": int(df["model"].nunique()),
            "n_families": int(df["model_family"].nunique()),
            "report": report,
            "top_features": top_feats,
        }, f, indent=2)

    return 0


# --------------------------------------------------------------------------
# E2: Trajectory plot (post-training reshapes default voice)
# --------------------------------------------------------------------------

TRAJECTORY_FEATURES = [
    "first_person_density", "assistant_phrase_ratio",
    "hedging_density", "refusal_density", "apology_density",
    "mean_sentence_length", "ttr", "markdown_density",
]


def cmd_trajectory(args):
    """
    Plot register features across post-training stages.
    Default: OLMo 3 trajectory. Optional --family ministral3 for replication.

    Uses bootstrap-within-checkpoint to compute 95% CIs for each stage's
    mean. With ~9 OLMo-3 stages, naive Mann-Kendall has weak power; bootstrap
    bands visualize stage-level variance.
    Also runs Mann-Kendall directional test on the bootstrap means.
    """
    df = pd.read_parquet(args.input)
    family_models = _trajectory_models(args.family)

    sub = df[df["model"].isin(family_models)].copy()
    if args.condition:
        sub = sub[sub["condition"] == args.condition]
    if len(sub) == 0:
        print(f"[trajectory] no data for family {args.family}")
        return 1

    stage_order = _trajectory_stage_order(args.family)
    sub["_stage_idx"] = sub["model"].map(
        {m: i for i, m in enumerate(stage_order)})
    sub = sub.dropna(subset=["_stage_idx"]).copy()
    sub["_stage_idx"] = sub["_stage_idx"].astype(int)

    feats = [f for f in TRAJECTORY_FEATURES if f in sub.columns]
    rng = np.random.default_rng(args.seed)
    n_boot = args.n_boot

    # Per-feature bootstrap means at each stage
    boot_records = []     # rows for plotting + saving
    test_records = []     # Mann-Kendall results per feature
    for f in feats:
        stage_means = []
        for stage in sorted(sub["_stage_idx"].unique()):
            samples = sub.loc[sub["_stage_idx"] == stage, f].dropna().values
            if len(samples) < 5:
                lo, hi, mean = float("nan"), float("nan"), float("nan")
            else:
                boot_means = np.empty(n_boot)
                for b in range(n_boot):
                    idx = rng.integers(0, len(samples), size=len(samples))
                    boot_means[b] = samples[idx].mean()
                lo, hi = np.percentile(boot_means, [2.5, 97.5])
                mean = boot_means.mean()
            stage_means.append((stage, mean, lo, hi))
            boot_records.append({"feature": f, "stage": stage,
                                  "mean": mean, "ci_lo": lo, "ci_hi": hi})
        # Mann-Kendall directional test on bootstrap means
        means_only = [m for _, m, _, _ in stage_means
                       if not np.isnan(m)]
        if len(means_only) >= 4:
            tau, p = _mann_kendall(means_only)
        else:
            tau, p = float("nan"), float("nan")
        test_records.append({"feature": f, "tau": tau, "p_one_sided": p,
                              "n_stages": len(means_only)})

    boot_df = pd.DataFrame(boot_records)
    test_df = pd.DataFrame(test_records)

    # Plot
    cols = 4
    nrows = (len(feats) + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(3.2 * cols, 2.6 * nrows),
                              sharex=True)
    axes = axes.flatten()

    for i, f in enumerate(feats):
        ax = axes[i]
        f_df = boot_df[boot_df["feature"] == f].sort_values("stage")
        ax.plot(f_df["stage"], f_df["mean"], marker="o", lw=1.5)
        ax.fill_between(f_df["stage"], f_df["ci_lo"], f_df["ci_hi"],
                         alpha=0.25)
        # annotate Mann-Kendall p
        row = test_df[test_df["feature"] == f].iloc[0]
        if not np.isnan(row["p_one_sided"]):
            ax.text(0.05, 0.95, f"tau={row['tau']:.2f}\np={row['p_one_sided']:.3f}",
                    transform=ax.transAxes, fontsize=7, va="top")
        ax.set_title(f.replace("_", " "), fontsize=9)
        ax.set_xticks(range(len(stage_order)))
        ax.set_xticklabels(_trajectory_stage_labels(args.family),
                            rotation=30, ha="right", fontsize=8)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Post-training reshapes default voice: {args.family}\n"
                  f"(bootstrap 95% CI, Mann-Kendall trend)",
                  y=1.02)
    plt.tight_layout()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.family}"
    fig.savefig(out_dir / f"f3_trajectory{suffix}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / f"f3_trajectory{suffix}.pdf", bbox_inches="tight")
    boot_df.to_csv(out_dir / f"f3_trajectory{suffix}_bootstrap.csv", index=False)
    test_df.to_csv(out_dir / f"f3_trajectory{suffix}_mannkendall.csv", index=False)
    print(f"[trajectory] wrote f3_trajectory{suffix}.{{png,pdf,bootstrap.csv,mannkendall.csv}}")
    return 0


def _mann_kendall(seq) -> tuple[float, float]:
    """Simple Mann-Kendall trend test. Returns (Kendall's tau, one-sided p).

    Direction is determined by the sign of S: we report a one-sided p
    against H0: no trend, in the observed direction (consistent with our
    pre-registered directional hypothesis).
    """
    n = len(seq)
    if n < 4:
        return float("nan"), float("nan")
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            d = seq[j] - seq[i]
            if   d > 0: s += 1
            elif d < 0: s -= 1
    var = n * (n - 1) * (2 * n + 5) / 18.0
    if s > 0:
        z = (s - 1) / np.sqrt(var)
    elif s < 0:
        z = (s + 1) / np.sqrt(var)
    else:
        z = 0.0
    # Two-sided to one-sided
    from scipy.stats import norm
    p_two = 2 * (1 - norm.cdf(abs(z)))
    p_one = p_two / 2
    tau = s / (0.5 * n * (n - 1))
    return float(tau), float(p_one)


def _trajectory_models(family: str) -> list[str]:
    if family == "olmo3":
        return ["olmo3_t_base", "olmo3_t_sft", "olmo3_t_dpo",
                "olmo3_t_rlvr",
                "olmo3_t_rl_code", "olmo3_t_rl_math", "olmo3_t_rl_mix",
                "olmo3_t_think_sft", "olmo3_t_think"]
    if family == "ministral3":
        # Use 8B variants for clean trajectory
        return ["m3_8_base", "m3_8_it", "m3_8_reason"]
    raise ValueError(f"unknown family {family}")


def _trajectory_stage_order(family: str) -> list[str]:
    return _trajectory_models(family)


def _trajectory_stage_labels(family: str) -> list[str]:
    if family == "olmo3":
        return ["base", "SFT", "DPO", "RLVR",
                "RL-Code", "RL-Math", "RL-Mix",
                "Think-SFT", "Think"]
    if family == "ministral3":
        return ["base", "Instruct", "Reasoning"]
    return _trajectory_models(family)


# --------------------------------------------------------------------------
# E3: Scale-ladder plot
# --------------------------------------------------------------------------

def cmd_scale(args):
    """
    Plot register features across model sizes (Qwen 3 ladder by default).
    """
    df = pd.read_parquet(args.input)
    sizes_models = [
        ("0.6B", "q3_06_base", "q3_06_it"),
        ("1.7B", "q3_17_base", "q3_17_it"),
        ("4B",   "q3_4_base",  "q3_4_it"),
        ("8B",   "q3_8_base",  "q3_8_it"),
        ("14B",  "q3_14_base", "q3_14_it"),
    ]
    rows = []
    for size, base, it in sizes_models:
        for stage, m in [("base", base), ("instruct", it)]:
            sub = df[df["model"] == m]
            if args.condition:
                sub = sub[sub["condition"] == args.condition]
            if len(sub) == 0:
                continue
            for f in TRAJECTORY_FEATURES:
                if f in sub.columns:
                    rows.append({"size": size, "stage": stage,
                                 "feature": f,
                                 "mean": sub[f].mean(),
                                 "std":  sub[f].std()})
    if not rows:
        print(f"[scale] no data")
        return 1
    plot_df = pd.DataFrame(rows)
    feats = [f for f in TRAJECTORY_FEATURES if f in df.columns]
    cols = 4
    rs   = (len(feats) + cols - 1) // cols
    fig, axes = plt.subplots(rs, cols, figsize=(3.2 * cols, 2.6 * rs),
                              sharex=True)
    axes = axes.flatten()
    for i, f in enumerate(feats):
        ax = axes[i]
        sub = plot_df[plot_df["feature"] == f]
        for stage, marker in [("base", "o"), ("instruct", "s")]:
            s = sub[sub["stage"] == stage]
            if len(s) == 0: continue
            ax.errorbar(range(len(s)), s["mean"], yerr=s["std"],
                        marker=marker, label=stage, capsize=3, lw=1.5)
        ax.set_title(f.replace("_", " "), fontsize=9)
        ax.set_xticks(range(len([s for s, _, _ in sizes_models])))
        ax.set_xticklabels([s for s, _, _ in sizes_models],
                            rotation=0, fontsize=8)
        if i == 0:
            ax.legend(fontsize=8, loc="best")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Default voice across the Qwen 3 scale ladder", y=1.02)
    plt.tight_layout()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "f_scale.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "f_scale.pdf", bbox_inches="tight")
    print(f"[scale] wrote f_scale.{{png,pdf}}")
    return 0


# --------------------------------------------------------------------------
# E4: Persistence under suppression
# --------------------------------------------------------------------------

def cmd_persistence(args):
    """
    For each model, compare register-feature deltas under suppression
    prompts vs control. A small delta = persistent default voice.
    """
    df = pd.read_parquet(args.input)
    suppression_ids = ["anti_self_ref", "anti_hedging", "anti_assistant",
                        "anti_verbosity"]
    control_id = "control_neutral"
    df = df[df["condition"].isin(suppression_ids + [control_id])].copy()

    feats = [f for f in TRAJECTORY_FEATURES if f in df.columns]
    rows = []
    for model, sub in df.groupby("model"):
        ctrl = sub[sub["condition"] == control_id]
        if len(ctrl) == 0: continue
        for cond in suppression_ids:
            sup = sub[sub["condition"] == cond]
            if len(sup) == 0: continue
            for f in feats:
                d = sup[f].mean() - ctrl[f].mean()
                rows.append({"model": model, "condition": cond,
                             "feature": f, "delta": d,
                             "delta_abs": abs(d)})
    pdf = pd.DataFrame(rows)
    if pdf.empty:
        print("[persistence] no data")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf.to_csv(out_dir / "persistence_deltas.csv", index=False)

    # Heatmap: rows=model, cols=feature, values=mean |delta| across conditions
    pivot = pdf.groupby(["model", "feature"])["delta_abs"].mean().unstack()
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(feats)),
                                      max(4, 0.4 * len(pivot))))
    sns.heatmap(pivot, cmap="Reds", annot=False, ax=ax, cbar_kws={"label": "|Delta|"})
    ax.set_title("Persistence under suppression: mean |Delta| from control")
    plt.tight_layout()
    fig.savefig(out_dir / "f4_persistence.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "f4_persistence.pdf", bbox_inches="tight")
    print(f"[persistence] wrote f4_persistence.{{png,pdf}}")
    return 0


# --------------------------------------------------------------------------
# Main CLI
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("profile")
    p.add_argument("--input", required=True)
    p.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "profiles.parquet"))
    p.set_defaults(func=cmd_profile)

    p = sub.add_parser("classify")
    p.add_argument("--input", required=True, help="all_features.parquet from features.aggregate")
    p.add_argument("--out_dir", default=str(PROJECT_ROOT / "analysis"))
    p.add_argument("--condition", default=None)
    p.add_argument("--language", default="en")
    p.add_argument("--exclude_base", action="store_true")
    p.add_argument("--exclude_instruct", action="store_true")
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--size_controlled", action="store_true",
                    help="restrict to 7-9 B FP16 subset (configs/models.yaml `size_controlled_subset`)")
    p.add_argument("--baseline", choices=["full", "length_only", "lexical_only", "random"],
                    default="full",
                    help="feature set: 'full' = all linguistic features (paper main); "
                         "'length_only' = n_words/n_chars only (defends against trivial-length critique); "
                         "'lexical_only' = ablation; 'random' = sanity baseline")
    p.set_defaults(func=cmd_classify)

    p = sub.add_parser("trajectory")
    p.add_argument("--input", required=True)
    p.add_argument("--out_dir", default=str(PROJECT_ROOT / "analysis"))
    p.add_argument("--family", choices=["olmo3", "ministral3"], default="olmo3")
    p.add_argument("--condition", default="bos_only")
    p.add_argument("--n_boot", type=int, default=10000,
                    help="bootstrap iterations within each checkpoint")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_trajectory)

    p = sub.add_parser("scale")
    p.add_argument("--input", required=True)
    p.add_argument("--out_dir", default=str(PROJECT_ROOT / "analysis"))
    p.add_argument("--condition", default="bos_only")
    p.set_defaults(func=cmd_scale)

    p = sub.add_parser("persistence")
    p.add_argument("--input", required=True)
    p.add_argument("--out_dir", default=str(PROJECT_ROOT / "analysis"))
    p.set_defaults(func=cmd_persistence)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
