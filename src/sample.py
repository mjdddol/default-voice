#!/usr/bin/env python
"""
Default Voice: sampling pipeline.

Generates N samples from a single model under a single condition,
saves to JSONL. Designed to be called per (model, condition, language)
triple by an outer scheduler.

Usage:
    python -m src.sample \
        --model_name qwen3_it \
        --condition bos_only \
        --language en \
        --n 2000 \
        --out_dir outputs

Each output JSONL file has rows:
    {"sample_id": int, "text": str, "n_input_tokens": int, "n_output_tokens": int,
     "input_text": str, "stop_reason": str, "model": str, "condition": str,
     "language": str, "seed": int}
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"


# --------------------------------------------------------------------------
# Config helpers
# --------------------------------------------------------------------------

def load_yaml(p: Path) -> dict[str, Any]:
    with open(p) as f:
        return yaml.safe_load(f)


def find_model_spec(name: str, models_cfg: dict) -> dict:
    """Look up a model by short name across all tier sections."""
    tier_keys = (
        "cross_family",
        "trajectory_olmo3",
        "trajectory_ministral3",
        "scale_qwen3",
        "reasoning_pairs",
    )
    for tier_key in tier_keys:
        for entry in models_cfg.get(tier_key, []):
            if entry["name"] == name:
                return entry
    raise KeyError(f"model '{name}' not found in models.yaml")


def find_condition_spec(cid: str, conds_cfg: dict) -> dict:
    for c in conds_cfg["conditions"]:
        if c["id"] == cid:
            return c
    for c in conds_cfg.get("suppression", []):
        if c["id"] == cid:
            return c
    raise KeyError(f"condition '{cid}' not found in conditions.yaml")


def find_language_spec(code: str, langs_cfg: dict) -> dict:
    for lang in langs_cfg["languages"]:
        if lang["code"] == code:
            return lang
    raise KeyError(f"language '{code}' not found in languages.yaml")


# --------------------------------------------------------------------------
# Prompt construction per condition
# --------------------------------------------------------------------------

def build_prompts(condition: dict, language: dict, n: int,
                  tokenizer, model_stage: str, rng: random.Random) -> list[str]:
    """Returns a list of n prompt strings ready to feed vLLM."""
    cid = condition["id"]

    # Suppression conditions (E4) use chat-template + system + user
    if "system_prompt" in condition:
        prompts = []
        for _ in range(n):
            messages = [
                {"role": "system", "content": condition["system_prompt"]},
                {"role": "user",   "content": condition["user_prompt"]},
            ]
            prompts.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True))
        return prompts

    # Standard conditions (E1)
    # Determine the most minimal "start-of-sequence" token that is valid for
    # this tokenizer.  Order of preference:
    #   1. tokenizer.bos_token  (Llama, Mistral, OLMo, Gemma)
    #   2. tokenizer.pad_token  (Qwen uses <|endoftext|> here: same role)
    #   3. tokenizer.eos_token  (some models share BOS/EOS)
    #   4. literal " "  (last resort: non-empty single space)
    def _minimal_seed() -> str:
        for tok_str in (tokenizer.bos_token,
                        tokenizer.pad_token,
                        tokenizer.eos_token):
            if tok_str:
                return tok_str
        return " "

    if cid == "bos_only":
        return [_minimal_seed()] * n

    if cid == "empty_user":
        if model_stage == "base":
            return [_minimal_seed()] * n
        try:
            templated = tokenizer.apply_chat_template(
                [{"role": "user", "content": ""}],
                tokenize=False, add_generation_prompt=True)
            return [templated] * n
        except Exception as e:
            print(f"  [warn] empty_user chat template failed: {e}; falling back to BOS")
            return [_minimal_seed()] * n

    if cid == "chat_template_no_gen_prompt":
        # Role-token ablation: chat template applied but assistant
        # generation prompt NOT appended. For base models, fall back to
        # BOS-only (they have no chat template).
        if model_stage == "base":
            return [_minimal_seed()] * n
        try:
            templated = tokenizer.apply_chat_template(
                [{"role": "user", "content": ""}],
                tokenize=False, add_generation_prompt=False)
            if not templated:                # some tokenizers return ""
                templated = _minimal_seed()
            return [templated] * n
        except Exception as e:
            print(f"  [warn] no_gen_prompt chat template failed: {e}; falling back to BOS")
            return [_minimal_seed()] * n

    if cid == "continue_cue":
        cue = language.get("continue_cue", "Continue.")
        if model_stage != "base":
            try:
                return [tokenizer.apply_chat_template(
                    [{"role": "user", "content": cue}],
                    tokenize=False, add_generation_prompt=True)] * n
            except Exception:
                pass
        return [cue] * n

    if cid == "random_prefix":
        # Sample n_random_tokens uniformly at random from vocab id space
        n_rand = condition.get("n_random_tokens", 5)
        vocab_size = tokenizer.vocab_size
        # Avoid special tokens: pick from [100, vocab_size) typically
        lo, hi = 100, vocab_size
        prompts = []
        for _ in range(n):
            ids = [rng.randrange(lo, hi) for _ in range(n_rand)]
            text = tokenizer.decode(ids, skip_special_tokens=True)
            prompts.append(text)
        return prompts

    if cid == "document_opening":
        opening = language.get("document_opening", "The following is a passage.")
        return [opening] * n

    raise ValueError(f"Unknown condition id: {cid}")


# --------------------------------------------------------------------------
# Output I/O
# --------------------------------------------------------------------------

def output_path(out_dir: Path, model_name: str, condition_id: str,
                language: str, label: str | None = None) -> Path:
    """Path to generation JSONL.

    If `label` is provided (e.g. for robustness sweeps "T07_qwen3"), the
    output path nests under generations/.../<label>/ to avoid clobbering
    main-run outputs at the default settings.
    """
    base = out_dir / "generations" / language / condition_id
    if label:
        base = base / label
    return base / f"{model_name}.jsonl"


def existing_count(p: Path) -> int:
    if not p.exists():
        return 0
    n = 0
    with open(p) as f:
        for _ in f:
            n += 1
    return n


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True,
                    help="short name from models.yaml, e.g. qwen3_it")
    ap.add_argument("--condition", required=True,
                    help="condition id from conditions.yaml")
    ap.add_argument("--language", default="en")
    ap.add_argument("--n", type=int, default=None,
                    help="number of samples (default: from conditions.yaml)")
    ap.add_argument("--out_dir", default=str(PROJECT_ROOT / "outputs"))
    ap.add_argument("--max_new_tokens", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pilot", action="store_true",
                    help="use small pilot N (override)")
    ap.add_argument("--gpu_mem_util", type=float, default=None,
                    help="override gpu_memory_utilization for vLLM")
    ap.add_argument("--max_model_len", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None,
                    help="override temperature (default from conditions.yaml)")
    ap.add_argument("--top_p", type=float, default=None,
                    help="override top_p")
    ap.add_argument("--n_random_tokens", type=int, default=None,
                    help="override n_random_tokens for random_prefix condition")
    ap.add_argument("--label", default=None,
                    help="optional run label, appended to output dirname for sweeps")
    args = ap.parse_args()

    models_cfg = load_yaml(CONFIG_DIR / "models.yaml")
    conds_cfg  = load_yaml(CONFIG_DIR / "conditions.yaml")
    langs_cfg  = load_yaml(CONFIG_DIR / "languages.yaml")

    model_spec = find_model_spec(args.model_name, models_cfg)
    cond_spec  = find_condition_spec(args.condition, conds_cfg)
    lang_spec  = find_language_spec(args.language, langs_cfg)

    n = args.n or (conds_cfg["sampling"]["pilot_n"] if args.pilot
                   else conds_cfg["sampling"]["n_samples_per_model_per_condition"])
    max_new = (args.max_new_tokens
               or conds_cfg["sampling"]["max_new_tokens"])
    temperature = (args.temperature
                   if args.temperature is not None
                   else conds_cfg["sampling"]["temperature"])
    top_p = (args.top_p if args.top_p is not None
             else conds_cfg["sampling"]["top_p"])
    top_k       = conds_cfg["sampling"]["top_k"]
    min_new     = conds_cfg["sampling"]["min_new_tokens"]
    # Honor random-prefix override if user supplied it
    if args.n_random_tokens is not None and cond_spec.get("id") == "random_prefix":
        cond_spec = dict(cond_spec)  # don't mutate cached YAML
        cond_spec["n_random_tokens"] = args.n_random_tokens

    vllm_defaults = models_cfg["vllm_defaults"]
    out_path = output_path(Path(args.out_dir), args.model_name,
                           args.condition, args.language, label=args.label)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    already = existing_count(out_path)
    if already >= n:
        print(f"[skip] {out_path} already has {already} samples (>= {n})")
        return 0
    remaining = n - already
    print(f"[sample] model={args.model_name} cond={args.condition} "
          f"lang={args.language} n={n} (remaining={remaining})")

    # Lazy import vLLM only after argparse to give fast --help.
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    quant = model_spec.get("quantization")

    # Resolution order: CLI arg > per-model spec > vllm_defaults
    gpu_mem = (args.gpu_mem_util
               if args.gpu_mem_util is not None
               else model_spec.get("gpu_memory_utilization",
                                    vllm_defaults["gpu_memory_utilization"]))
    max_len = (args.max_model_len
               if args.max_model_len is not None
               else model_spec.get("max_model_len",
                                    vllm_defaults["max_model_len"]))

    llm_kwargs = dict(
        model=model_spec["hf_id"],
        dtype=vllm_defaults["dtype"],
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_len,
        trust_remote_code=vllm_defaults.get("trust_remote_code", True),
        enforce_eager=vllm_defaults.get("enforce_eager", False),
        swap_space=vllm_defaults.get("swap_space", 4),
    )
    if quant:
        llm_kwargs["quantization"] = quant

    print(f"[init] vLLM loading {model_spec['hf_id']} (quant={quant or 'none'})")
    t0 = time.time()
    llm = LLM(**llm_kwargs)
    tok = AutoTokenizer.from_pretrained(model_spec["hf_id"], trust_remote_code=True)
    print(f"[init] loaded in {time.time()-t0:.1f}s")

    rng = random.Random(args.seed)
    prompts = build_prompts(cond_spec, lang_spec, remaining,
                            tok, model_spec.get("stage", "instruct"), rng)

    # Per-prompt different seed for genuine diversity
    sampling_params = [
        SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k != -1 else -1,
            min_tokens=min_new,
            max_tokens=max_new,
            seed=args.seed + already + i,
        )
        for i in range(remaining)
    ]

    print(f"[generate] starting {remaining} samples")
    t1 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t1
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"[generate] done in {elapsed:.1f}s "
          f"({total_tokens} new tokens, {total_tokens/max(elapsed,1):.1f} tok/s)")

    drop_below_words = conds_cfg["sampling"].get("drop_below_words", 0)
    n_dropped = 0

    # Write append-style so resume works
    with open(out_path, "a") as f:
        for i, out in enumerate(outputs):
            sample_id = already + i
            o = out.outputs[0]
            n_words = len(o.text.split())
            if drop_below_words and n_words < drop_below_words:
                n_dropped += 1
                continue
            row = {
                "sample_id": sample_id,
                "model": args.model_name,
                "model_hf_id": model_spec["hf_id"],
                "model_family": model_spec.get("family"),
                "model_stage": model_spec.get("stage"),
                "condition": args.condition,
                "language": args.language,
                "input_text": prompts[i],
                "n_input_tokens": len(out.prompt_token_ids),
                "text": o.text,
                "n_output_tokens": len(o.token_ids),
                "stop_reason": str(o.finish_reason),
                "seed": sampling_params[i].seed,
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new,
                "min_new_tokens": min_new,
                "label": args.label,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if n_dropped:
        print(f"[done] dropped {n_dropped} samples below {drop_below_words}-word threshold")

    print(f"[done] wrote {remaining} rows to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
