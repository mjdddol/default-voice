#!/usr/bin/env python
"""
Publication-grade main-paper figures.

Design system:
 - Centralized PALETTE / FONTS / LINES.
 - Main figures show derived claims (effect sizes), not raw cross-tabs.
 - Raw cross-tabs and specialized analyses live in appendix figures.
 - Only outlier points get text labels. No bar-top annotations.
 - n/a markers replaced by caption footnotes.
 - Consistent spine/grid/marker treatment via apply_style().
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

PROJ = Path(".")
ASSETS = PROJ / "paper" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

# --- design tokens ----------------------------------------------------------

# colorblind-safe muted palette
PALETTE = {
    "primary":   "#3a6ea5",   # muted blue
    "secondary": "#c97c4a",   # muted orange
    "neutral":   "#9aa0a6",   # gray
    "muted":     "#b8bdc4",   # light gray
    "highlight": "#d44a3c",   # vermillion red
    "bos":       "#9aa0a6",   # gray (condition mapping)
    "no_gen":    "#d9a05b",   # warm tan
    "empty":     "#3a6ea5",   # blue
    "ink":       "#1a1a1a",
    "grid":      "#dadada",
}

FONTS = {
    "tick":    7.5,
    "label":   8.5,
    "title":   9.5,
    "legend":  7.5,
    "anno":    7.5,
}

plt.rcParams.update({
    "savefig.dpi":     300,
    "pdf.fonttype":    42,
    "ps.fonttype":     42,
    "font.family":     "serif",
    "font.serif":      ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset":"stix",
    "font.size":              FONTS["label"],
    "axes.titlesize":         FONTS["title"],
    "axes.titleweight":       "normal",
    "axes.labelsize":         FONTS["label"],
    "axes.linewidth":         0.6,
    "axes.spines.top":        False,
    "axes.spines.right":      False,
    "xtick.major.width":      0.5,
    "ytick.major.width":      0.5,
    "xtick.labelsize":        FONTS["tick"],
    "ytick.labelsize":        FONTS["tick"],
    "legend.fontsize":        FONTS["legend"],
    "legend.frameon":         False,
})


def apply_style(ax, ygrid=True, xgrid=False):
    """Uniform spine + grid treatment."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    if ygrid:
        ax.yaxis.grid(True, color=PALETTE["grid"], lw=0.5, alpha=0.7)
    if xgrid:
        ax.xaxis.grid(True, color=PALETTE["grid"], lw=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="both", length=2.5, pad=2)


# ============================================================
# data prep helpers
# ============================================================

FAMILY_SPECS = [
    ("Llama 3.1",   "llama",     None),
    ("Qwen 3",      "qwen3",     None),
    ("OLMo 3",      "olmo3",     None),
    ("Aya Expanse", "aya",       None),
    ("Gemma 4",     "gemma4",    None),
    ("Granite 4.1", "granite4",  None),
    ("Ministral 3", None,        "m3_8_it"),
]


def _per_family_table(df):
    conds = ["bos_only", "chat_template_no_gen_prompt", "empty_user"]
    M  = np.full((len(FAMILY_SPECS), len(conds)), np.nan)
    SE = np.full_like(M, np.nan)
    for i, (label, fam, mod) in enumerate(FAMILY_SPECS):
        for j, c in enumerate(conds):
            if fam is not None:
                sub = df[(df.model_family == fam) &
                         (df.condition == c) &
                         (df.model_stage == "instruct")]
            else:
                sub = df[(df.model == mod) & (df.condition == c)]
            if len(sub) > 50:
                v = sub["assistant_phrase_ratio"].values * 100
                M[i, j] = v.mean()
                SE[i, j] = v.std(ddof=1) / np.sqrt(len(v))
    return M, SE


# ============================================================
# FIG 2 (main): two effect-size dot plots
# ============================================================

def fig_format_gradient():
    df = pd.read_parquet(PROJ / "outputs" / "all_features.parquet")
    M, SE = _per_family_table(df)
    labels = [f[0] for f in FAMILY_SPECS]

    pp_empty = M[:, 2] - M[:, 0]       # EMPTY-USER minus BOS-ONLY
    pp_nogen = M[:, 1] - M[:, 0]       # CHAT-NO-GEN minus BOS-ONLY

    # sort families by panel-A effect, descending; Ministral kept at the
    # bottom even if it's negative-ish.
    order = np.argsort(pp_empty)       # ascending (so largest is at top in plot)
    labels_o   = [labels[i] for i in order]
    pp_empty_o = pp_empty[order]
    pp_nogen_o = pp_nogen[order]

    # error bars on the empty-BOS contrast (combine SEs in quadrature)
    SE_empty_BOS = np.sqrt(SE[:, 2] ** 2 + SE[:, 0] ** 2)[order]
    SE_nogen_BOS = np.sqrt(SE[:, 1] ** 2 + SE[:, 0] ** 2)[order]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9),
                              sharey=True,
                              gridspec_kw=dict(wspace=0.16, left=0.12,
                                               right=0.99, top=0.86,
                                               bottom=0.16))

    # marker shape per family: OLMo = diamond, Ministral = open circle,
    # others = filled circle. Shapes carry meaning beyond color so the
    # exception structure survives grayscale printing.
    def _marker_for(label, missing=False):
        if missing:
            return dict(marker="o", facecolor="white",
                        edgecolor=PALETTE["muted"], edgewidth=0.9)
        if label == "OLMo 3":
            return dict(marker="D", facecolor=PALETTE["secondary"],
                        edgecolor="black", edgewidth=0.4)
        if label == "Ministral 3":
            return dict(marker="o", facecolor="white",
                        edgecolor=PALETTE["highlight"], edgewidth=1.2)
        return dict(marker="o", facecolor=PALETTE["neutral"],
                    edgecolor="black", edgewidth=0.4)

    def _draw(ax, vals, ses, labels, panel_color_role):
        """panel_color_role is 'A' or 'B': affects line/edge color."""
        for yi, val, se, lbl in zip(y, vals, ses, labels):
            m = _marker_for(lbl, missing=np.isnan(val))
            if np.isnan(val):
                # don't draw open marker inside the plot for unavailable
                # entries; leave row blank, caption explains.
                continue
            line_col = (PALETTE["highlight"]
                        if lbl in ("OLMo 3", "Ministral 3")
                        else PALETTE["neutral"])
            ax.plot([0, val], [yi, yi], color=line_col, lw=1.3,
                    alpha=0.55, zorder=1)
            ax.errorbar(val, yi, xerr=se, fmt=m["marker"],
                        mfc=m["facecolor"], mec=m["edgecolor"],
                        ecolor=line_col, markersize=6.8,
                        markeredgewidth=m["edgewidth"],
                        elinewidth=1.0, capsize=0, zorder=3)

    y = np.arange(len(labels_o))

    # Panel A: OLMo is not special here (it's only the template-only
    # exception in Panel B), so use neutral markers for all activating
    # families and keep the Ministral outlier only.
    def _draw_panel_a(ax, vals, ses, labels):
        for yi, val, se, lbl in zip(y, vals, ses, labels):
            if np.isnan(val):
                continue
            if lbl == "Ministral 3":
                m = dict(marker="o", facecolor="white",
                         edgecolor=PALETTE["highlight"], edgewidth=1.2)
                line_col = PALETTE["highlight"]
            else:
                m = dict(marker="o", facecolor=PALETTE["neutral"],
                         edgecolor="black", edgewidth=0.4)
                line_col = PALETTE["neutral"]
            ax.plot([0, val], [yi, yi], color=line_col, lw=1.3,
                    alpha=0.55, zorder=1)
            ax.errorbar(val, yi, xerr=se, fmt=m["marker"],
                        mfc=m["facecolor"], mec=m["edgecolor"],
                        ecolor=line_col, markersize=6.8,
                        markeredgewidth=m["edgewidth"],
                        elinewidth=1.0, capsize=0, zorder=3)

    # ---------- Panel A: empty-user - BOS-only ----------
    axA = axes[0]
    _draw_panel_a(axA, pp_empty_o, SE_empty_BOS, labels_o)
    axA.axvline(0, color=PALETTE["ink"], lw=0.6, ls=(0, (4, 3)), alpha=0.4)
    axA.set_yticks(y)
    axA.set_yticklabels(labels_o, fontsize=FONTS["tick"])
    axA.set_ylim(-0.5, len(labels_o) - 0.5)
    axA.set_xlabel(r"$\Delta$ assistant-phrase density (pp)",
                   fontsize=FONTS["label"])
    axA.set_title(r"(a) Empty-user format effect (empty $-$ BOS)",
                  fontsize=FONTS["title"], pad=4)
    # Use same x-range on both panels so the two contrasts can be
    # compared directly.
    axA.set_xlim(-6, 50)
    axA.set_xticks([-5, 0, 10, 20, 30, 40, 50])
    apply_style(axA, ygrid=False, xgrid=True)

    # ---------- Panel B: chat-no-gen - BOS-only ----------
    axB = axes[1]
    _draw(axB, pp_nogen_o, SE_nogen_BOS, labels_o, "B")
    axB.axvline(0, color=PALETTE["ink"], lw=0.6, ls=(0, (4, 3)), alpha=0.4)
    axB.set_yticks(y)
    # y-tick labels suppressed: Panel A's labels are shared via sharey.
    axB.tick_params(axis="y", which="both", length=0, labelleft=False)
    axB.set_ylim(-0.5, len(labels_o) - 0.5)
    axB.set_xlabel(r"$\Delta$ assistant-phrase density (pp)",
                   fontsize=FONTS["label"])
    axB.set_title(r"(b) Template-only effect (no-gen $-$ BOS)",
                  fontsize=FONTS["title"], pad=4)
    # Match Panel A range so the two contrasts are visually comparable.
    axB.set_xlim(-6, 50)
    axB.set_xticks([-5, 0, 10, 20, 30, 40, 50])
    apply_style(axB, ygrid=False, xgrid=True)

    out = ASSETS / "fig_format_gradient.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)
    return M, SE


# ============================================================
# FIG 2b (appendix): full per-family x condition grouped bars
# ============================================================

def fig_per_family_density_appendix():
    df = pd.read_parquet(PROJ / "outputs" / "all_features.parquet")
    M, SE = _per_family_table(df)
    labels = [f[0] for f in FAMILY_SPECS]
    short  = ["Llama","Qwen","OLMo","Aya","Gemma","Granite","Ministral"]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    w = 0.27
    cond_labels = ["BOS", "No-gen", "Empty"]
    # Use same palette mapping as Figure 1: neutral / muted-orange / primary
    cond_cols   = [PALETTE["neutral"], PALETTE["secondary"],
                   PALETTE["primary"]]
    for j, (lbl, col) in enumerate(zip(cond_labels, cond_cols)):
        ax.bar(x + (j - 1) * w, M[:, j], width=w, yerr=SE[:, j],
                color=col, edgecolor="black", linewidth=0.4,
                error_kw=dict(lw=0.4, capthick=0.4), label=lbl)
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=FONTS["tick"])
    ax.set_ylabel("Assistant-phrase density (\\%)", fontsize=FONTS["label"])
    ax.legend(loc="upper left", ncol=3, fontsize=FONTS["legend"] - 0.5,
              handlelength=1.2, columnspacing=0.8, borderaxespad=0.3)
    ax.set_ylim(0, 52)
    apply_style(ax)
    out = ASSETS / "fig_per_family_density.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)


# ============================================================
# FIG 3 (main): OLMo 3 main-chain trajectory
# ============================================================

def fig_olmo3_trajectory():
    df = pd.read_parquet(PROJ / "outputs" / "all_features.parquet")
    main_stages = ["olmo3_t_base", "olmo3_t_sft",
                   "olmo3_t_dpo", "olmo3_t_rlvr"]
    main_labels = ["Base", "SFT", "DPO", "RLVR"]
    feats = [
        ("assistant_phrase_ratio", "Assistant-phrase (%)", 100),
        ("refusal_density",        "Refusal-like (%)",     100),
        ("n_words",                "Output length (words)",  1),
        ("markdown_density",       "Markdown (per 1k chars)", 1),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.6),
                              gridspec_kw=dict(wspace=0.45,
                                               top=0.86, bottom=0.18,
                                               left=0.07, right=0.99))
    for ax, (feat, label, mult) in zip(axes, feats):
        ys, errs = [], []
        for m in main_stages:
            sub = df[(df.model == m) & (df.condition == "empty_user")]
            v = sub[feat].values * mult if len(sub) > 30 else np.array([])
            ys.append(v.mean() if len(v) else np.nan)
            errs.append(v.std(ddof=1) / max(np.sqrt(len(v)), 1)
                        if len(v) else 0)
        ax.errorbar(range(len(main_stages)), ys, yerr=errs, fmt="-o",
                    color=PALETTE["primary"], lw=1.6, markersize=5,
                    capsize=2.5,
                    markeredgecolor="black", markeredgewidth=0.4)
        ax.set_title(label, fontsize=FONTS["title"], pad=4)
        ax.set_xticks(range(len(main_stages)))
        ax.set_xticklabels(main_labels, fontsize=FONTS["tick"])
        ax.tick_params(axis="x", length=0)
        ax.margins(y=0.22)
        # Density metrics cannot be negative: clip lower bound at 0 so
        # the visual baseline matches the quantity's support.
        if feat in ("refusal_density", "markdown_density",
                    "assistant_phrase_ratio"):
            ax.set_ylim(bottom=0)
        apply_style(ax)

    out = ASSETS / "fig_olmo3_trajectory.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)


# ============================================================
# FIG 3b (appendix): specialized RL variants (full-width)
# ============================================================

def fig_olmo3_specialized():
    df = pd.read_parquet(PROJ / "outputs" / "all_features.parquet")
    spec_stages = ["olmo3_t_rlvr", "olmo3_t_rl_code", "olmo3_t_rl_math",
                   "olmo3_t_think_sft", "olmo3_t_think"]
    spec_labels = ["RLVR", "Code", "Math", "Think-S", "Think"]
    feats = [
        ("assistant_phrase_ratio", "Assistant-phrase (%)", 100),
        ("refusal_density",        "Refusal-like (%)",     100),
        ("n_words",                "Output length (words)",  1),
        ("markdown_density",       "Markdown (per 1k chars)", 1),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.7),
                              gridspec_kw=dict(wspace=0.45,
                                               top=0.86, bottom=0.22,
                                               left=0.07, right=0.99))
    for ax, (feat, label, mult) in zip(axes, feats):
        ys, errs = [], []
        for m in spec_stages:
            sub = df[(df.model == m) & (df.condition == "empty_user")]
            v = sub[feat].values * mult if len(sub) > 30 else np.array([])
            ys.append(v.mean() if len(v) else np.nan)
            errs.append(v.std(ddof=1) / max(np.sqrt(len(v)), 1)
                        if len(v) else 0)
        # primary blue for RLVR baseline, neutral for the rest
        colors = [PALETTE["primary"]] + [PALETTE["neutral"]] * 4
        ax.bar(range(len(spec_stages)), ys, yerr=errs, capsize=2,
               color=colors, edgecolor="black", linewidth=0.4,
               error_kw=dict(lw=0.5, capthick=0.5))
        ax.set_title(label, fontsize=FONTS["title"], pad=4)
        ax.set_xticks(range(len(spec_stages)))
        ax.set_xticklabels(spec_labels, fontsize=FONTS["tick"],
                            rotation=35, ha="right")
        ax.tick_params(axis="x", length=0, pad=2)
        ax.margins(y=0.18)
        apply_style(ax)

    out = ASSETS / "fig_olmo3_specialized.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)


# ============================================================
# FIG 4 (main): desired-direction PRO vs ANTI shift, per feature
# ============================================================

PERSIST_PAIRS = [
    ("self_ref",  "first_person_density",  "Self-reference (%)",   100),
    ("hedging",   "hedging_density",       "Hedging (%)",          100),
    ("assistant", "assistant_phrase_ratio","Assistant-phrase (%)", 100),
    ("verbosity", "mean_sentence_length",  "Sentence length",        1),
]
PERSIST_MODELS = ["olmo3_t_rlvr", "llama31_it", "qwen3_it", "gemma4_it"]


def fig_persistence():
    """Paired slope plot: PRO vs ANTI desired-direction shift, per family.
    Each family is one line; mean is a bold black line + diamond. n=4
    families, so individual lines are the protagonists, not aggregated
    bars."""
    df = pd.read_parquet(PROJ / "outputs" / "all_features.parquet")
    target_models = [m for m in PERSIST_MODELS if (df.model == m).any()]
    family_short = {
        "olmo3_t_rlvr": "OLMo",
        "llama31_it":   "Llama",
        "qwen3_it":     "Qwen",
        "gemma4_it":    "Gemma",
    }

    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.5),
                              gridspec_kw=dict(wspace=0.40,
                                               top=0.84, bottom=0.20,
                                               left=0.09, right=0.99))
    for k, (ax, (tag, feat, ylabel, mult)) in enumerate(zip(axes, PERSIST_PAIRS)):
        d_pro_list, d_anti_list, fams = [], [], []
        for m in target_models:
            c = df[(df.model == m) & (df.condition == "control_neutral")][feat]
            p = df[(df.model == m) & (df.condition == f"pro_{tag}")][feat]
            a = df[(df.model == m) & (df.condition == f"anti_{tag}")][feat]
            if not (len(c) and len(p) and len(a)):
                continue
            cm, pm, am = c.mean() * mult, p.mean() * mult, a.mean() * mult
            d_pro_list.append(pm - cm)
            d_anti_list.append(cm - am)
            fams.append(family_short.get(m, m))
        d_pro  = np.array(d_pro_list)
        d_anti = np.array(d_anti_list)

        # Per-family slope lines (light gray, low alpha so the mean
        # line dominates the panel).
        for pv, av in zip(d_pro, d_anti):
            ax.plot([0, 1], [pv, av],
                    color=PALETTE["muted"], lw=0.9, alpha=0.38, zorder=1)
            ax.plot([0, 1], [pv, av], "o",
                    color="white", markersize=3.6,
                    markeredgecolor=PALETTE["muted"],
                    markeredgewidth=0.6, zorder=2)

        # Mean line + diamond (bold black, dominates the panel)
        mean_pro  = d_pro.mean()
        mean_anti = d_anti.mean()
        ax.plot([0, 1], [mean_pro, mean_anti],
                color=PALETTE["ink"], lw=2.2, zorder=3)
        ax.plot([0, 1], [mean_pro, mean_anti], "D",
                color=PALETTE["ink"], markersize=6.5,
                markeredgecolor="white", markeredgewidth=0.8, zorder=4)

        ax.axhline(0, color=PALETTE["muted"], lw=0.4,
                   ls=(0, (4, 3)), alpha=0.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["PRO", "ANTI"], fontsize=FONTS["tick"])
        ax.set_xlim(-0.25, 1.25)
        ax.set_title(ylabel, fontsize=FONTS["title"], pad=4)
        ax.tick_params(axis="x", length=0, pad=4)
        # y-pad
        ymax = max(d_pro.max(), d_anti.max(), 0)
        ymin = min(d_anti.min(), d_pro.min(), 0)
        rng_y = ymax - ymin
        pad   = rng_y * 0.18 if rng_y > 0 else 1
        ax.set_ylim(ymin - pad, ymax + pad)
        apply_style(ax)

    fig.supylabel("Desired-direction shift", fontsize=FONTS["label"],
                  x=0.005)

    out = ASSETS / "fig_persistence.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)


# ============================================================
# FIG 5 (main): 3x3 probe transfer matrix
# ============================================================

def _probe_matrix():
    """Return the 3x3 (M, conds) tuple, M[i,j] = train i, test j accuracy."""
    nums = json.load(open(PROJ / "analysis" / "paper_numbers.json"))
    probe = nums.get("F4_probe", {})
    conds = ["bos_only", "chat_template_no_gen_prompt", "empty_user"]
    M = np.full((3, 3), np.nan)
    for i, c in enumerate(conds):
        key = f"probe_family_{c}"
        if key in probe and "cv_mean" in probe[key]:
            M[i, i] = probe[key]["cv_mean"]
    for k, v in probe.items():
        if not k.startswith("probe_transfer_"):
            continue
        parts = k.replace("probe_transfer_", "").split("_to_")
        if len(parts) != 2: continue
        src, dst = parts
        if src in conds and dst in conds:
            i = conds.index(src); j = conds.index(dst)
            M[i, j] = v.get("transfer_acc", np.nan)
    return M, conds


def fig_probe():
    """Main paper Figure 4: gap-only dot plot of probe transfer
    asymmetry per condition pair. Each row is one unordered pair; the
    x value is the absolute difference between the two transfer
    directions (in percentage points). Per-direction accuracies appear
    as annotations to the right; the full 3x3 matrix lives in the
    appendix."""
    M, conds = _probe_matrix()
    short = {"bos_only": "BOS-only",
             "chat_template_no_gen_prompt": "Chat-no-gen",
             "empty_user": "Empty-user"}

    pairs = [
        ("empty_user",                  "bos_only"),
        ("empty_user",                  "chat_template_no_gen_prompt"),
        ("bos_only", "chat_template_no_gen_prompt"),
    ]
    rows = []
    for a, b in pairs:
        ia, ib = conds.index(a), conds.index(b)
        acc_a2b = M[ia, ib] * 100   # train a, test b
        acc_b2a = M[ib, ia] * 100   # train b, test a
        gap = abs(acc_a2b - acc_b2a)
        rows.append((short[a], short[b], acc_a2b, acc_b2a, gap))

    fig, ax = plt.subplots(figsize=(3.4, 1.7))
    y = np.arange(len(rows))

    for k, (a_lbl, b_lbl, a2b, b2a, gap) in enumerate(rows):
        col = PALETTE["highlight"] if gap >= 20 else PALETTE["neutral"]
        # thin stem from 0 to value
        ax.plot([0, gap], [k, k],
                color=col, lw=1.2, alpha=0.7, zorder=1)
        # filled endpoint
        ax.plot(gap, k, "o",
                color=col, markersize=7,
                markeredgecolor="black", markeredgewidth=0.5,
                zorder=3)
        ax.text(gap + 1.8, k, f"{gap:.0f} pp",
                va="center", ha="left",
                fontsize=FONTS["anno"], color=PALETTE["ink"])

    ax.axvline(0, color=PALETTE["ink"], lw=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r[0]} $\\leftrightarrow$ {r[1]}"
                         for r in rows], fontsize=FONTS["tick"])
    ax.invert_yaxis()
    ax.set_xlim(0, 60)
    ax.set_xlabel("Probe transfer asymmetry (pp)",
                  fontsize=FONTS["label"])
    apply_style(ax, ygrid=False, xgrid=True)
    ax.tick_params(axis="y", length=0, pad=4)
    # tighten y range so rows don't float
    ax.set_ylim(len(rows) - 0.5, -0.5)

    out = ASSETS / "fig_probe.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)


def fig_probe_matrix_appendix():
    """Appendix version: full 3x3 matrix as a quiet table-like display."""
    M, conds = _probe_matrix()
    short = {"bos_only": "BOS-only",
             "chat_template_no_gen_prompt": "Chat-no-gen",
             "empty_user": "Empty-user"}

    cmap = plt.get_cmap("Blues")
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    # Very light tinting; numbers carry the information.
    ax.imshow(M * 100, cmap=cmap, vmin=10, vmax=130, aspect="equal")
    for i in range(3):
        for j in range(3):
            v = M[i, j] * 100
            txt_col = PALETTE["ink"]
            w = "bold" if i == j else "normal"
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    fontsize=13, color=txt_col, weight=w)

    ax.set_xticks(range(3))
    ax.set_xticklabels([short[c] for c in conds],
                        fontsize=FONTS["tick"], rotation=18, ha="right")
    ax.set_yticks(range(3))
    ax.set_yticklabels([short[c] for c in conds], fontsize=FONTS["tick"])
    ax.set_xlabel("Test condition",  fontsize=FONTS["label"], labelpad=4)
    ax.set_ylabel("Train condition", fontsize=FONTS["label"], labelpad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0)
    plt.tight_layout()
    out = ASSETS / "fig_probe_matrix.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)


def fig_generalization():
    """Single-panel scale-ladder figure: Qwen 3 base vs instruct fold
    change across four model sizes."""
    df = pd.read_parquet(PROJ / "outputs" / "all_features.parquet")

    sizes = [0.6, 1.7, 4, 8]
    size_models_base    = ["q3_06_base", "q3_17_base", "q3_4_base", "q3_8_base"]
    size_models_instr   = ["q3_06_it",   "q3_17_it",   "q3_4_it",   "q3_8_it"]

    def _fold(m):
        bos = df[(df.model == m) & (df.condition == "bos_only")]
        emp = df[(df.model == m) & (df.condition == "empty_user")]
        if len(bos) < 30 or len(emp) < 30:
            return np.nan
        b = bos.assistant_phrase_ratio.mean()
        e = emp.assistant_phrase_ratio.mean()
        return e / max(b, 1e-5)

    base_folds  = [_fold(m) for m in size_models_base]
    instr_folds = [_fold(m) for m in size_models_instr]

    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    xs = np.arange(len(sizes))
    ax.plot(xs, instr_folds, "-o", color=PALETTE["primary"],
            lw=1.6, markersize=6,
            markeredgecolor="black", markeredgewidth=0.4,
            label="Instruct")
    ax.plot(xs, base_folds, "-o", color=PALETTE["muted"],
            lw=1.3, markersize=5,
            markeredgecolor="black", markeredgewidth=0.3,
            label="Base")
    ax.axhline(1, color=PALETTE["ink"], lw=0.5, ls=(0, (3, 3)), alpha=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{s} B" for s in sizes], fontsize=FONTS["tick"])
    ax.set_xlabel("Model size", fontsize=FONTS["label"], labelpad=3)
    ax.set_ylabel("Fold change\n(empty-user / BOS-only)",
                  fontsize=FONTS["label"])
    ax.legend(loc="upper left", fontsize=FONTS["legend"],
              handlelength=1.2, borderaxespad=0.3)
    ax.set_ylim(0, max(max(instr_folds), 14) + 1.5)
    apply_style(ax)

    plt.tight_layout()
    out = ASSETS / "fig_generalization.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating figures...")
    M, SE = fig_format_gradient()
    print("\nFig 2 numeric summary (mean assistant-phrase density %):")
    print(f"{'Family':<12} {'BOS':>7} {'no_gen':>7} {'empty':>7} {'empty-BOS':>10} {'nogen-BOS':>10}")
    for i, (label, _, _) in enumerate(FAMILY_SPECS):
        b = f"{M[i,0]:7.2f}" if not np.isnan(M[i,0]) else "    nan"
        n = f"{M[i,1]:7.2f}" if not np.isnan(M[i,1]) else "    nan"
        e = f"{M[i,2]:7.2f}" if not np.isnan(M[i,2]) else "    nan"
        eb = M[i,2]-M[i,0] if not np.isnan(M[i,2]) else float("nan")
        nb = M[i,1]-M[i,0] if not np.isnan(M[i,1]) else float("nan")
        ebs = f"{eb:+10.2f}" if not np.isnan(eb) else "       nan"
        nbs = f"{nb:+10.2f}" if not np.isnan(nb) else "       nan"
        print(f"{label:<12} {b} {n} {e} {ebs} {nbs}")
    fig_per_family_density_appendix()
    fig_olmo3_trajectory()
    fig_olmo3_specialized()
    fig_persistence()
    fig_probe()
    fig_probe_matrix_appendix()
    fig_generalization()
    print("Done.")
