#!/usr/bin/env python
"""
Register-direction ablation (Arditi-style).

For each instruction-tuned family that activates the chat-conditioned
register (Llama 3.1, Qwen 3, OLMo 3, Aya Expanse, Granite 4.1):

  1. Re-encode the existing empty_user and bos_only generations
     (outputs/generations/en/<condition>/<name>.jsonl) through the model and
     extract last-token hidden state at every decoder layer.
  2. Compute the per-layer difference-of-means
       d(L) = mu_empty(L) - mu_bos(L)
     and the per-layer separation score
       s(L) = ||d(L)|| / pooled_std(L)
  3. Select the ablation source layer = argmax of s over the middle/late
     band (default: 25-85 % of depth).
  4. Ablate the unit direction d(L*)/||d(L*)|| from the residual stream
     output of every layer L in [L*, n_layers).  This is the Arditi
     "directional ablation" recipe.
  5. Generate under four conditions, each N=N_GENERATE:
       empty_baseline       no ablation, neutral system prompt
       empty_ablate         direction ablated, neutral system prompt
       empty_anti           no ablation, anti-assistant system prompt
       empty_anti_ablate    direction ablated + anti-assistant system prompt

Outputs:
    outputs/ablation/generations/<name>/<cond>.jsonl
    outputs/ablation/directions/<name>.npz
    outputs/ablation/diag/<name>_layer_probe.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = PROJECT_ROOT / "outputs" / "ablation"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
GEN_ROOT = PROJECT_ROOT / "outputs" / "generations" / "en"


# -------------------- model + tokenizer --------------------

def load_model(hf_id: str, device: str = "cuda", dtype=torch.float16):
    tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tok


def get_decoder_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Cannot locate decoder layers")


# -------------------- prompt construction --------------------

def build_prompt(tok, condition: str, anti: bool = False) -> str:
    anti_system = (
        "Respond directly without opening pleasantries. Do not greet, "
        "apologize, hedge, or describe what you are about to do. Do not "
        "use phrases like 'I'd be happy to', 'Sure', 'Of course', 'Let me "
        "explain', or 'I cannot'. Start with the substance immediately."
    )
    if condition == "bos_only":
        bos = tok.bos_token or "<|endoftext|>"
        return bos
    if condition == "empty_user":
        msgs = []
        if anti:
            msgs.append({"role": "system", "content": anti_system})
        msgs.append({"role": "user", "content": ""})
        try:
            return tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return msgs[-1]["content"]
    raise ValueError(condition)


# -------------------- multi-layer extraction --------------------

@torch.no_grad()
def extract_multilayer_from_jsonl(
    model, tok, jsonl_path: Path, n_samples: int = 200,
    max_chars: int = 1500, device: str = "cuda",
) -> np.ndarray:
    """
    Read texts from jsonl, encode each through model, return per-layer
    last-token hidden states of shape (n_used, n_layers, d_model).
    """
    out = []
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if len(out) >= n_samples:
                break
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = d.get("text") or d.get("output") or d.get("generation") or ""
            text = text.strip()
            if len(text) < 5:
                continue
            text = text[:max_chars]
            ids = tok(
                text, return_tensors="pt", add_special_tokens=False,
                truncation=True, max_length=512,
            ).to(device)
            if ids["input_ids"].shape[1] < 2:
                continue
            res = model(**ids, output_hidden_states=True, use_cache=False)
            layers = res.hidden_states[1:]
            last_tok = torch.stack([h[0, -1] for h in layers], dim=0)
            out.append(last_tok.float().cpu().numpy())
            del res
    if not out:
        return np.zeros((0, 0, 0), dtype=np.float32)
    return np.stack(out, axis=0)


def compute_per_layer_directions(empty_h_ml: np.ndarray, bos_h_ml: np.ndarray):
    """
    empty_h_ml, bos_h_ml: (n_samples, n_layers, d_model)
    Returns:
        directions: (n_layers, d_model)  unit vectors (zero where invalid)
        norms:      (n_layers,)         ||mu_diff||
        scores:     (n_layers,)         ||mu_diff|| / pooled_std
    """
    # Robust to nan/inf in hidden states (can happen for FP16 hybrid
    # architectures): drop non-finite samples from each layer's mean/var.
    empty_h_ml = np.where(np.isfinite(empty_h_ml), empty_h_ml, np.nan)
    bos_h_ml   = np.where(np.isfinite(bos_h_ml),   bos_h_ml,   np.nan)
    mu_e = np.nanmean(empty_h_ml, axis=0)
    mu_b = np.nanmean(bos_h_ml, axis=0)
    diff = mu_e - mu_b
    diff = np.where(np.isfinite(diff), diff, 0.0)
    norms = np.linalg.norm(diff, axis=-1)
    directions = diff / np.maximum(norms[:, None], 1e-8)
    var_e = np.nanvar(empty_h_ml, axis=0).mean(axis=-1)
    var_b = np.nanvar(bos_h_ml, axis=0).mean(axis=-1)
    pooled = np.sqrt(0.5 * (var_e + var_b))
    pooled = np.where(np.isfinite(pooled) & (pooled > 0), pooled, 1e-8)
    scores = norms / pooled
    scores = np.where(np.isfinite(scores), scores, 0.0)
    return directions, norms, scores


def pick_best_layer(scores: np.ndarray, n_layers: int,
                    lo_frac: float = 0.25, hi_frac: float = 0.85) -> int:
    lo = max(1, int(lo_frac * n_layers))
    hi = max(lo + 1, int(hi_frac * n_layers))
    band = scores[lo:hi]
    # nanargmax for robustness; if entire band is nan/zero, fall back to
    # midpoint
    if not np.any(np.isfinite(band)) or np.all(band == 0):
        return (lo + hi) // 2
    return lo + int(np.nanargmax(band))


# -------------------- ablation hook --------------------

def make_ablation_hook(direction: torch.Tensor):
    d = direction.detach()

    def hook(module, args, output):
        if isinstance(output, tuple):
            h = output[0]; rest = output[1:]
        else:
            h = output; rest = None
        d_local = d.to(dtype=h.dtype, device=h.device)
        scalar = (h * d_local).sum(dim=-1, keepdim=True)
        h_new = h - scalar * d_local
        if rest is not None:
            return (h_new,) + rest
        return h_new

    return hook


# -------------------- generation --------------------

@torch.no_grad()
def generate_batch(
    model, tok, prompts: list[str], max_new_tokens: int = 256,
    temperature: float = 1.0, top_p: float = 1.0,
    seed: int = 0, device: str = "cuda",
):
    out = []
    torch.manual_seed(seed)
    for i, p in enumerate(prompts):
        ids = tok(p, return_tensors="pt", add_special_tokens=False).to(device)
        gen = model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tok.eos_token_id,
        )
        text = tok.decode(
            gen[0, ids["input_ids"].shape[1]:], skip_special_tokens=True
        )
        out.append({"sample_id": i, "prompt_index": i, "text": text})
    return out


# -------------------- per-family driver --------------------

def run_family(
    name: str,
    hf_id: str,
    n_extract: int = 150,
    n_generate: int = 500,
    max_new_tokens: int = 256,
    lo_frac: float = 0.25,
    hi_frac: float = 0.85,
    device: str = "cuda",
    seed: int = 0,
    diag_only: bool = False,
):
    print(f"\n=== {name} ({hf_id}) ===")
    out_dir = OUT_ROOT / "generations" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    dir_dir = OUT_ROOT / "directions"
    dir_dir.mkdir(parents=True, exist_ok=True)
    diag_dir = OUT_ROOT / "diag"
    diag_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model, tok = load_model(hf_id, device=device)
    n_layers = len(get_decoder_layers(model))
    d_model = model.config.hidden_size
    print(f"  loaded: {n_layers} layers, d_model={d_model} "
          f"({time.time()-t0:.1f}s)")

    # ---- step 1: re-encode existing generations at every layer ----
    empty_jsonl = GEN_ROOT / "empty_user" / f"{name}.jsonl"
    bos_jsonl = GEN_ROOT / "bos_only" / f"{name}.jsonl"
    if not (empty_jsonl.exists() and bos_jsonl.exists()):
        print(f"  missing source jsonl(s): {empty_jsonl} or {bos_jsonl}")
        return

    print(f"  extracting multi-layer activations (N<={n_extract} per cond)")
    t1 = time.time()
    empty_ml = extract_multilayer_from_jsonl(
        model, tok, empty_jsonl, n_samples=n_extract, device=device
    )
    bos_ml = extract_multilayer_from_jsonl(
        model, tok, bos_jsonl, n_samples=n_extract, device=device
    )
    print(f"    shapes: empty {empty_ml.shape}, bos {bos_ml.shape} "
          f"({time.time()-t1:.1f}s)")
    if empty_ml.shape[0] == 0 or bos_ml.shape[0] == 0:
        print("  empty extraction; abort")
        return

    directions, norms, scores = compute_per_layer_directions(empty_ml, bos_ml)
    best_layer = pick_best_layer(scores, n_layers, lo_frac, hi_frac)
    direction_np = directions[best_layer]
    print(f"  best layer = {best_layer} / {n_layers-1}  "
          f"(||diff||={norms[best_layer]:.2f}, score={scores[best_layer]:.2f})")
    print(f"  scores by layer (top-5): "
          f"{[(i, round(float(scores[i]),2)) for i in np.argsort(-scores)[:5]]}")

    ablate_layers = list(range(best_layer, n_layers))
    print(f"  ablating direction at layers {best_layer}..{n_layers-1} "
          f"({len(ablate_layers)} layers)")

    np.savez(
        dir_dir / f"{name}.npz",
        directions=directions.astype(np.float32),
        norms=norms,
        scores=scores,
        best_layer=best_layer,
        ablate_layers=np.array(ablate_layers),
        n_layers=n_layers,
        d_model=d_model,
        hf_id=hf_id,
    )
    diag = {
        "name": name,
        "hf_id": hf_id,
        "n_layers": n_layers,
        "best_layer": best_layer,
        "best_score": float(scores[best_layer]),
        "best_norm": float(norms[best_layer]),
        "ablate_layers": ablate_layers,
        "scores_by_layer": [float(s) for s in scores],
        "norms_by_layer": [float(n) for n in norms],
        "n_used_empty": int(empty_ml.shape[0]),
        "n_used_bos":   int(bos_ml.shape[0]),
    }
    json.dump(diag, open(diag_dir / f"{name}_layer_probe.json", "w"), indent=2)

    if diag_only:
        del model, tok
        torch.cuda.empty_cache(); gc.collect()
        return

    # ---- step 2: ablated generation ----
    chosen_direction = torch.tensor(
        direction_np, device=device, dtype=torch.float32
    )
    decoder_layers = get_decoder_layers(model)
    hook_handles: list = []

    def attach_hooks():
        for li in ablate_layers:
            h = decoder_layers[li].register_forward_hook(
                make_ablation_hook(chosen_direction)
            )
            hook_handles.append(h)

    def remove_hooks():
        nonlocal hook_handles
        for h in hook_handles:
            h.remove()
        hook_handles = []

    base_prompts = [build_prompt(tok, "empty_user")] * n_generate
    anti_prompts = [build_prompt(tok, "empty_user", anti=True)] * n_generate

    conditions = [
        ("empty_baseline",    base_prompts, False),
        ("empty_ablate",      base_prompts, True),
        ("empty_anti",        anti_prompts, False),
        ("empty_anti_ablate", anti_prompts, True),
    ]

    for cond_name, prompts, do_ablate in conditions:
        cond_path = out_dir / f"{cond_name}.jsonl"
        if cond_path.exists() and cond_path.stat().st_size > 0:
            print(f"  [{cond_name}] exists, skipping")
            continue
        if do_ablate:
            attach_hooks()
        t2 = time.time()
        print(f"  [{cond_name}] generating N={n_generate} (ablate={do_ablate})")
        gens = generate_batch(
            model, tok, prompts,
            max_new_tokens=max_new_tokens,
            temperature=1.0, top_p=1.0,
            seed=seed + abs(hash(cond_name)) % 10000,
            device=device,
        )
        remove_hooks()
        with open(cond_path, "w") as f:
            for g in gens:
                g["model"] = name
                g["condition"] = cond_name
                f.write(json.dumps(g) + "\n")
        print(f"    wrote {len(gens)} -> {cond_path} ({time.time()-t2:.1f}s)")

    del model, tok
    torch.cuda.empty_cache(); gc.collect()


# -------------------- entrypoint --------------------

FAMILIES = [
    ("llama31_it",       "meta-llama/Llama-3.1-8B-Instruct"),
    ("qwen3_it",         "Qwen/Qwen3-8B"),
    ("olmo3_it",         "allenai/Olmo-3-7B-Instruct"),
    ("aya_expanse",      "CohereLabs/aya-expanse-8b"),
    ("granite41_8b_it",  "ibm-granite/granite-4.1-8b"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default=None,
                    help="single family name; default = all")
    ap.add_argument("--n_extract", type=int, default=150)
    ap.add_argument("--n_generate", type=int, default=500)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--lo_frac", type=float, default=0.25)
    ap.add_argument("--hi_frac", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--diag_only", action="store_true")
    args = ap.parse_args()

    chosen = FAMILIES
    if args.family is not None:
        chosen = [(n, h) for (n, h) in FAMILIES if n == args.family]
        if not chosen:
            print(f"unknown family {args.family}", file=sys.stderr)
            return 1

    for (name, hf_id) in chosen:
        try:
            run_family(
                name=name,
                hf_id=hf_id,
                n_extract=args.n_extract,
                n_generate=args.n_generate,
                max_new_tokens=args.max_new_tokens,
                lo_frac=args.lo_frac,
                hi_frac=args.hi_frac,
                seed=args.seed,
                diag_only=args.diag_only,
            )
        except Exception as e:
            print(f"ERROR on {name}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main() or 0)
