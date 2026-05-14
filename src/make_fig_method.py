#!/usr/bin/env python
"""
Method schematic figure (Figure 1): three input conditions + example outputs.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ASSETS = Path("../paper/assets")
ASSETS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.size": 9,
})


def draw_box(ax, x, y, w, h, text, *, facecolor="#f4f4f4",
              edgecolor="#333", fontsize=8, monospace=False,
              text_kwargs=None):
    rect = mpatches.FancyBboxPatch((x, y), w, h,
                                     boxstyle="round,pad=0.02,rounding_size=0.05",
                                     facecolor=facecolor,
                                     edgecolor=edgecolor, lw=0.8)
    ax.add_patch(rect)
    family = "monospace" if monospace else "serif"
    kw = dict(ha="center", va="center", fontsize=fontsize,
              family=family, wrap=True)
    if text_kwargs:
        kw.update(text_kwargs)
    ax.text(x + w / 2, y + h / 2, text, **kw)


def main():
    fig, ax = plt.subplots(figsize=(7.3, 2.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # Title at top
    ax.text(5, 4.85, "Three input conditions probe the chat-default register",
            ha="center", va="top", fontsize=10, weight="bold")

    # Column labels
    cond_labels = [
        ("BOS-only", "no chat template"),
        ("Chat template,\nno generation prompt", "user slot present;\nassistant tag absent"),
        ("Empty user +\ngeneration prompt", "user slot empty;\nassistant tag appended"),
    ]
    inputs = [
        r"$\langle$BOS$\rangle$",
        "<|im_start|>user\n\n<|im_end|>\n",
        "<|im_start|>user\n\n<|im_end|>\n<|im_start|>assistant\n",
    ]
    outputs = [
        '"Three weeks :: Twelve weeks ::\nForty week. Does this sound\nbetter? -- Social Moslem..."',
        '"to provide additional\ndetails about the requested\ntopic without context..."',
        '"Sure, I\'d be happy to help.\nWhat would you like to know?"',
    ]
    arrow_colors = ["#7d7d7d", "#c4924d", "#1f77b4"]

    col_centers = [1.5, 5.0, 8.5]
    for cx, (lbl_top, lbl_sub), inp, out, ac in zip(
            col_centers, cond_labels, inputs, outputs, arrow_colors):
        # Condition title
        ax.text(cx, 4.30, lbl_top, ha="center", va="center",
                fontsize=9.5, weight="bold")
        ax.text(cx, 3.85, lbl_sub, ha="center", va="center",
                fontsize=7.5, style="italic", color="#555")
        # Input box
        draw_box(ax, cx - 1.45, 2.45, 2.9, 1.2, inp,
                 facecolor="#fafafa", edgecolor=ac, monospace=True,
                 fontsize=7.5)
        # Arrow
        ax.annotate("", xy=(cx, 2.10), xytext=(cx, 2.45),
                     arrowprops=dict(arrowstyle="->", lw=1.2, color=ac))
        # Output box
        draw_box(ax, cx - 1.45, 0.50, 2.9, 1.5, out,
                 facecolor="#f0f4fa" if ac == "#1f77b4" else "#fafafa",
                 edgecolor=ac, fontsize=7.5)

    # Section labels at left
    ax.text(-0.05, 3.05, "Input", ha="right", va="center",
            fontsize=8.5, style="italic", color="#555")
    ax.text(-0.05, 1.25, "Output", ha="right", va="center",
            fontsize=8.5, style="italic", color="#555")

    # Bottom annotation
    ax.text(5, 0.05,
            "Surface register signals (assistant phrasing, brevity, "
            "hedging) emerge sharply only under the third condition.",
            ha="center", va="bottom", fontsize=7.5,
            style="italic", color="#444")

    plt.tight_layout()
    out_path = ASSETS / "fig_method.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"  wrote {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
