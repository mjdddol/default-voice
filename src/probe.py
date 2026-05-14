#!/usr/bin/env python
"""
Linear probe on hidden states (E6, supplement).

Subcommands:

  family
    Train a logistic-regression probe on hidden states (one NPZ per
    (model, condition) tuple) to predict the model FAMILY from the
    activations alone. Reports CV accuracy + per-family F1.

  transfer
    Train a probe on activations from one condition (e.g. empty_user),
    test on activations from another (e.g. bos_only). High transfer
    accuracy = the family signature is encoded in hidden states
    independently of the surface format that elicited it.

Usage:
    # 1. Aggregate NPZs into one big array
    python -m src.probe family \
        --hidden_dir outputs/hidden \
        --condition empty_user \
        --out_dir analysis/

    # 2. Transfer test
    python -m src.probe transfer \
        --hidden_dir outputs/hidden \
        --train_condition empty_user \
        --test_condition bos_only \
        --out_dir analysis/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# Hidden-state aggregation
# --------------------------------------------------------------------------

def load_hidden(hidden_dir: Path, condition: str,
                 hidden_dim: int | None = None
                 ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Walk `hidden_dir` for files named like
        <model_name>__<condition>.npz
    and aggregate the `hidden` arrays plus their per-sample family / model
    labels.

    Because different model families have different hidden dimensions
    (e.g. 4096 for 8 B, 2560 for ~3 B), this function groups by hidden_dim
    and either:
      * If `hidden_dim` is specified, keeps only that dim.
      * Otherwise, returns the most common dim's group (logs the dropped models).

    Returns:
        X      [N, D]  stacked hidden vectors of a single hidden_dim
        y_fam  [N]     family labels (str)
        models [N]     per-row model name (str)
    """
    by_dim: dict[int, list[tuple[np.ndarray, str, str]]] = {}
    for npz in sorted(hidden_dir.glob(f"*__{condition}.npz")):
        data = np.load(npz, allow_pickle=True)
        H = data["hidden"]
        d = H.shape[1]
        fam = str(data["model_family"])
        mod = str(data["model_name"])
        by_dim.setdefault(d, []).append((H, fam, mod))

    if not by_dim:
        raise RuntimeError(f"no hidden NPZs found for condition={condition} "
                           f"in {hidden_dir}")

    if hidden_dim is None:
        # Pick the dim that covers the most distinct families (>= 3 to make
        # classification meaningful).
        scored = sorted(
            by_dim.items(),
            key=lambda kv: (len({m[1] for m in kv[1]}), sum(m[0].shape[0] for m in kv[1])),
            reverse=True,
        )
        hidden_dim = scored[0][0]
        if len(by_dim) > 1:
            dropped = [(d, len({m[1] for m in items}))
                        for d, items in by_dim.items() if d != hidden_dim]
            print(f"[load] multiple hidden_dims found; using d={hidden_dim} "
                  f"with {len({m[1] for m in by_dim[hidden_dim]})} families. "
                  f"Dropped: {dropped}")
    items = by_dim[hidden_dim]
    Xs   = [it[0] for it in items]
    fams = [it[1] for it in items for _ in range(it[0].shape[0])]
    mods = [it[2] for it in items for _ in range(it[0].shape[0])]
    return np.concatenate(Xs, axis=0), np.asarray(fams), np.asarray(mods)


# --------------------------------------------------------------------------
# Probes
# --------------------------------------------------------------------------

def cmd_family(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report

    X, y, models = load_hidden(Path(args.hidden_dir), args.condition)
    print(f"[family] loaded {len(X)} vectors, "
          f"{len(set(y))} families, {len(set(models))} models, "
          f"dim={X.shape[1]}, condition={args.condition}")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=2000, C=args.C, n_jobs=-1)
    cv = cross_val_score(clf, Xs, y, cv=5, n_jobs=-1)
    print(f"[family] 5-fold CV accuracy: {cv.mean():.3f} +/- {cv.std():.3f}")

    # Train on full set for per-class report
    clf.fit(Xs, y)
    yhat = clf.predict(Xs)
    report = classification_report(y, yhat, output_dict=True, zero_division=0)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(
        {"condition": args.condition,
         "cv_mean": float(cv.mean()),
         "cv_std":  float(cv.std()),
         "n_samples": len(X),
         "n_families": int(len(set(y))),
         "report": report},
        open(out_dir / f"probe_family_{args.condition}.json", "w"),
        indent=2)
    print(f"[family] wrote probe_family_{args.condition}.json")
    return 0


def cmd_transfer(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    Xtr, ytr, _ = load_hidden(Path(args.hidden_dir), args.train_condition)
    Xte, yte, _ = load_hidden(Path(args.hidden_dir), args.test_condition)
    print(f"[transfer] train: {len(Xtr)} samples on '{args.train_condition}'")
    print(f"[transfer] test:  {len(Xte)} samples on '{args.test_condition}'")

    # Same scaler for both: fit on train only
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xte_s = scaler.transform(Xte)

    clf = LogisticRegression(max_iter=2000, C=args.C, n_jobs=-1)
    clf.fit(Xtr_s, ytr)

    # In-distribution
    in_acc = clf.score(Xtr_s, ytr)
    # Transfer
    common = set(ytr) & set(yte)
    mask = np.isin(yte, list(common))
    if mask.sum() == 0:
        raise RuntimeError("no overlapping families between conditions")
    out_acc = accuracy_score(yte[mask], clf.predict(Xte_s[mask]))

    print(f"[transfer] in-distribution accuracy:  {in_acc:.3f}")
    print(f"[transfer] transfer accuracy:        {out_acc:.3f}")
    print(f"[transfer] retention ratio:          {out_acc / max(in_acc, 1e-9):.3f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(
        {"train_condition": args.train_condition,
         "test_condition":  args.test_condition,
         "in_acc": float(in_acc),
         "transfer_acc": float(out_acc),
         "retention": float(out_acc / max(in_acc, 1e-9)),
         "n_train": len(Xtr),
         "n_test_overlap": int(mask.sum()),
         "shared_families": sorted(common)},
        open(out_dir / f"probe_transfer_{args.train_condition}_to_{args.test_condition}.json", "w"),
        indent=2)
    print(f"[transfer] wrote probe_transfer_{args.train_condition}_to_{args.test_condition}.json")
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("family")
    pf.add_argument("--hidden_dir", default=str(PROJECT_ROOT / "outputs" / "hidden"))
    pf.add_argument("--condition", required=True)
    pf.add_argument("--out_dir", default=str(PROJECT_ROOT / "analysis"))
    pf.add_argument("--C", type=float, default=1.0)
    pf.set_defaults(func=cmd_family)

    pt = sub.add_parser("transfer")
    pt.add_argument("--hidden_dir", default=str(PROJECT_ROOT / "outputs" / "hidden"))
    pt.add_argument("--train_condition", required=True)
    pt.add_argument("--test_condition",  required=True)
    pt.add_argument("--out_dir", default=str(PROJECT_ROOT / "analysis"))
    pt.add_argument("--C", type=float, default=1.0)
    pt.set_defaults(func=cmd_transfer)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
