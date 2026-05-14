#!/usr/bin/env python
"""
Linguistic feature extraction for "Default Voice" paper.

Reads JSONL of generations (one per row), computes per-sample features
across five axes (lexical, syntactic, discourse, register, diversity).

Aggregates per-sample features into a per-(model, condition, language)
profile vector for downstream classification and trajectory analysis.

Usage:
    # Per-file extraction
    python -m src.features extract \
        --input outputs/generations/en/bos_only/qwen3_it.jsonl \
        --output outputs/features/en/bos_only/qwen3_it.parquet

    # Aggregate all
    python -m src.features aggregate \
        --root outputs --out outputs/profiles.parquet
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEXICON_DIR = PROJECT_ROOT / "data" / "lexicons"


# --------------------------------------------------------------------------
# Lexicon loading
# --------------------------------------------------------------------------

def load_lexicon(name: str) -> set[str]:
    p = LEXICON_DIR / f"{name}.txt"
    if not p.exists():
        warnings.warn(f"missing lexicon: {p}")
        return set()
    out = set()
    with open(p) as f:
        for line in f:
            line = line.strip().lower()
            if not line or line.startswith("#"):
                continue
            out.add(line)
    return out


_LEX_CACHE: dict[str, set[str]] = {}

def lex(name: str) -> set[str]:
    if name not in _LEX_CACHE:
        _LEX_CACHE[name] = load_lexicon(name)
    return _LEX_CACHE[name]


# --------------------------------------------------------------------------
# Tokenization (simple, language-agnostic fallback)
# --------------------------------------------------------------------------

WORD_RE = re.compile(r"[A-Za-z']+|\d+")
SENT_END_RE = re.compile(r"[.!?]+\s*")
PUNCT_RE = re.compile(r"[^\w\s]")

def simple_words(text: str) -> list[str]:
    return [t.lower() for t in WORD_RE.findall(text)]

def simple_sentences(text: str) -> list[str]:
    parts = SENT_END_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


# --------------------------------------------------------------------------
# Phrase counter (for multi-word lexicon entries)
# --------------------------------------------------------------------------

def count_phrase_hits(text: str, lexicon: set[str]) -> int:
    """Count occurrences of any phrase from `lexicon` in `text` (lowercased)."""
    if not lexicon:
        return 0
    text_low = text.lower()
    n = 0
    # Sort by phrase length desc so longer matches don't get shadowed
    for phrase in sorted(lexicon, key=len, reverse=True):
        if " " in phrase:
            n += text_low.count(phrase)
        else:
            # Whole-word match for single tokens
            n += sum(1 for _ in re.finditer(rf"\b{re.escape(phrase)}\b", text_low))
    return n


# --------------------------------------------------------------------------
# Lexical features
# --------------------------------------------------------------------------

def feat_lexical(text: str, words: list[str]) -> dict[str, float]:
    n = len(words)
    if n == 0:
        return {k: 0.0 for k in [
            "ttr", "mtld", "hapax_ratio", "mean_word_length",
            "function_word_ratio", "content_word_ratio",
            "hedging_density", "assertive_density", "modal_density",
            "first_person_density", "second_person_density",
            "third_person_density"]}
    types = set(words)
    counts = Counter(words)
    hapax = sum(1 for c in counts.values() if c == 1)
    fw = lex("function_words")
    fw_count = sum(1 for w in words if w in fw)
    modals = {"can", "could", "will", "would", "shall", "should", "may",
              "might", "must", "ought", "need", "dare"}
    first_person = {"i", "me", "my", "mine", "myself",
                    "we", "us", "our", "ours", "ourselves"}
    second_person = {"you", "your", "yours", "yourself", "yourselves"}
    third_person = {"he", "him", "his", "himself",
                    "she", "her", "hers", "herself",
                    "it", "its", "itself",
                    "they", "them", "their", "theirs", "themselves"}

    return {
        "ttr": len(types) / n,
        "mtld": _mtld(words),
        "hapax_ratio": hapax / n,
        "mean_word_length": sum(len(w) for w in words) / n,
        "function_word_ratio": fw_count / n,
        "content_word_ratio": 1.0 - fw_count / n,
        "hedging_density": count_phrase_hits(text, lex("hedging")) / n,
        "assertive_density": count_phrase_hits(text, lex("assertive")) / n,
        "modal_density": sum(1 for w in words if w in modals) / n,
        "first_person_density": sum(1 for w in words if w in first_person) / n,
        "second_person_density": sum(1 for w in words if w in second_person) / n,
        "third_person_density": sum(1 for w in words if w in third_person) / n,
    }


def _mtld(words: list[str], threshold: float = 0.72) -> float:
    """Measure of Textual Lexical Diversity (McCarthy & Jarvis, 2010).
    Length-robust diversity. Returns 0 for very short texts."""
    if len(words) < 50:
        return 0.0
    def _one_dir(seq: list[str]) -> float:
        factor = 0
        types: set[str] = set()
        tokens = 0
        for w in seq:
            tokens += 1
            types.add(w)
            ttr = len(types) / tokens
            if ttr <= threshold:
                factor += 1
                types.clear()
                tokens = 0
        # Partial factor for trailing segment
        if tokens > 0:
            ttr = len(types) / tokens
            factor += (1 - ttr) / (1 - threshold) if threshold < 1 else 0
        if factor == 0:
            return float("inf")
        return len(seq) / factor
    forward = _one_dir(words)
    backward = _one_dir(list(reversed(words)))
    return (forward + backward) / 2.0 if math.isfinite(forward + backward) else 0.0


# --------------------------------------------------------------------------
# Syntactic features (spaCy when available, else heuristics)
# --------------------------------------------------------------------------

_SPACY_NLP = {}

def _get_spacy(lang_code: str):
    if lang_code in _SPACY_NLP:
        return _SPACY_NLP[lang_code]
    spacy_models = {
        "en": "en_core_web_sm",
        "ko": "ko_core_news_sm",
        "zh": "zh_core_web_sm",
        "es": "es_core_news_sm",
    }
    try:
        import spacy
        nlp = spacy.load(spacy_models[lang_code], disable=["ner", "lemmatizer"])
        _SPACY_NLP[lang_code] = nlp
        return nlp
    except Exception as e:
        warnings.warn(f"spaCy unavailable for {lang_code}: {e}; using heuristic syntactic features")
        _SPACY_NLP[lang_code] = None
        return None


def feat_syntactic(text: str, lang_code: str = "en") -> dict[str, float]:
    nlp = _get_spacy(lang_code)
    if nlp is None:
        # Heuristic fallback
        sents = simple_sentences(text)
        words = simple_words(text)
        if not sents or not words:
            return {"mean_sentence_length": 0.0, "mean_dep_depth": 0.0,
                    "passive_ratio": 0.0, "subordinate_ratio": 0.0,
                    "punctuation_density": 0.0, "pos_dist_kl": 0.0}
        return {
            "mean_sentence_length": len(words) / len(sents),
            "mean_dep_depth": 0.0,
            "passive_ratio": 0.0,
            "subordinate_ratio": 0.0,
            "punctuation_density": len(PUNCT_RE.findall(text)) / max(len(words), 1),
            "pos_dist_kl": 0.0,
        }

    doc = nlp(text)
    sents = list(doc.sents)
    if not sents or len(doc) == 0:
        return {"mean_sentence_length": 0.0, "mean_dep_depth": 0.0,
                "passive_ratio": 0.0, "subordinate_ratio": 0.0,
                "punctuation_density": 0.0, "pos_dist_kl": 0.0}

    lengths = [len([t for t in s if not t.is_punct]) for s in sents]
    n_clauses = sum(1 for t in doc if t.dep_ in {"ROOT", "ccomp", "advcl", "relcl", "xcomp"})
    n_passive = sum(1 for t in doc if t.dep_ in {"nsubjpass", "auxpass"})
    n_subord = sum(1 for t in doc if t.dep_ in {"advcl", "ccomp", "relcl", "xcomp"})

    def _depth(token):
        d = 0
        cur = token
        while cur.head != cur:
            d += 1
            cur = cur.head
            if d > 100: break
        return d
    depths = [_depth(t) for t in doc if not t.is_space]

    # POS distribution KL vs uniform reference (rough proxy)
    pos_counts = Counter(t.pos_ for t in doc if not t.is_space)
    total = sum(pos_counts.values())
    pos_dist = {k: v / total for k, v in pos_counts.items()}
    # KL vs uniform over observed POSes
    if pos_dist:
        u = 1.0 / len(pos_dist)
        kl = sum(p * math.log(p / u) for p in pos_dist.values() if p > 0)
    else:
        kl = 0.0

    return {
        "mean_sentence_length": sum(lengths) / len(lengths),
        "mean_dep_depth": sum(depths) / max(len(depths), 1),
        "passive_ratio": n_passive / max(n_clauses, 1),
        "subordinate_ratio": n_subord / max(n_clauses, 1),
        "punctuation_density": sum(1 for t in doc if t.is_punct) / max(len(doc), 1),
        "pos_dist_kl": kl,
    }


# --------------------------------------------------------------------------
# Discourse / register markers
# --------------------------------------------------------------------------

def feat_discourse(text: str, words: list[str]) -> dict[str, float]:
    n = max(len(words), 1)
    n_sents = max(len(simple_sentences(text)), 1)
    n_chars = max(len(text), 1)

    return {
        "discourse_marker_density": count_phrase_hits(text, lex("discourse_markers")) / n_sents,
        "assistant_phrase_ratio": count_phrase_hits(text, lex("assistant_phrases")) / n_sents,
        "apology_density": count_phrase_hits(text, lex("apology")) / n_sents,
        "refusal_density": count_phrase_hits(text, lex("refusal")) / n_sents,
        "politeness_score": count_phrase_hits(text, lex("politeness")) / n,
        "formality_score": (
            count_phrase_hits(text, lex("formal"))
            - count_phrase_hits(text, lex("informal"))
        ) / n,
        "emoji_density": _emoji_count(text) / n_chars * 1000,
        "markdown_density": (text.count("#") + text.count("**") + text.count("|")
                             + text.count("```")) / n_chars * 1000,
    }


_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001F9FF\U0001FA70-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F02F]"
)
def _emoji_count(text: str) -> int:
    return len(_EMOJI_RE.findall(text))


# --------------------------------------------------------------------------
# Per-sample feature extraction
# --------------------------------------------------------------------------

def extract_one(text: str, lang_code: str = "en") -> dict[str, float]:
    words = simple_words(text)
    out = {}
    out.update(feat_lexical(text, words))
    out.update(feat_syntactic(text, lang_code))
    out.update(feat_discourse(text, words))
    out["n_words"] = float(len(words))
    out["n_chars"] = float(len(text))
    return out


# --------------------------------------------------------------------------
# Position-split extraction (defends against opening-boilerplate confound)
# --------------------------------------------------------------------------

def slice_by_position(text: str, position: str, slice_words: int = 100) -> str:
    """Return text containing approximately `slice_words` words from the
    requested position. Chooses on word boundaries.

    `position`  in  {"full", "first", "middle", "last"}.
    """
    if position == "full":
        return text
    words = WORD_RE.findall(text)
    n = len(words)
    if n <= slice_words:
        return text
    if position == "first":
        sel_words = words[:slice_words]
    elif position == "last":
        sel_words = words[-slice_words:]
    elif position == "middle":
        start = max(0, (n - slice_words) // 2)
        sel_words = words[start:start + slice_words]
    else:
        raise ValueError(f"unknown position {position}")

    # Rough reconstruction: find first occurrence of the boundary words in
    # the original text. Good enough for feature extraction.
    if not sel_words:
        return ""
    first = sel_words[0]
    last  = sel_words[-1]
    try:
        idx_first = text.lower().index(first)
    except ValueError:
        idx_first = 0
    # Find the *last* occurrence of `last` after idx_first
    tail_low = text[idx_first:].lower()
    try:
        idx_last_rel = tail_low.rindex(last) + len(last)
    except ValueError:
        idx_last_rel = len(text) - idx_first
    return text[idx_first: idx_first + idx_last_rel]


def extract_one_position(text: str, lang_code: str, position: str,
                          slice_words: int = 100) -> dict[str, float]:
    """Extract features on a positional slice of `text`."""
    sliced = slice_by_position(text, position, slice_words)
    return extract_one(sliced, lang_code=lang_code)


# --------------------------------------------------------------------------
# Corpus-level diversity (computed across many samples)
# --------------------------------------------------------------------------

def corpus_diversity(texts: list[str]) -> dict[str, float]:
    all_words: list[str] = []
    bigrams: Counter = Counter()
    trigrams: Counter = Counter()
    for t in texts:
        ws = simple_words(t)
        all_words.extend(ws)
        bigrams.update(zip(ws[:-1], ws[1:]))
        trigrams.update(zip(ws[:-2], ws[1:-1], ws[2:]))
    n1 = len(all_words)
    out = {
        "distinct_1": len(set(all_words)) / max(n1, 1),
        "distinct_2": len(bigrams) / max(sum(bigrams.values()), 1),
        "distinct_3": len(trigrams) / max(sum(trigrams.values()), 1),
        "corpus_n_words": float(n1),
    }
    if n1 > 0:
        # entropy of unigram distribution
        c = Counter(all_words)
        total = sum(c.values())
        ent = -sum((v / total) * math.log(v / total) for v in c.values())
        out["sample_entropy"] = ent
    else:
        out["sample_entropy"] = 0.0
    return out


# --------------------------------------------------------------------------
# CLI: extract / aggregate
# --------------------------------------------------------------------------

def cmd_extract(args: argparse.Namespace) -> int:
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    texts_for_corpus = []
    with open(in_path) as f:
        for line in f:
            row = json.loads(line)
            text = row["text"]
            lang = row.get("language", "en")
            if args.position == "full":
                feats = extract_one(text, lang_code=lang)
            else:
                feats = extract_one_position(text, lang, args.position,
                                              args.slice_words)
            feats["sample_id"] = row["sample_id"]
            feats["model"] = row["model"]
            feats["model_family"] = row.get("model_family")
            feats["model_stage"] = row.get("model_stage")
            feats["condition"] = row["condition"]
            feats["language"] = lang
            feats["position"] = args.position
            rows.append(feats)
            texts_for_corpus.append(text)

    df = pd.DataFrame(rows)
    df.to_parquet(out_path)
    print(f"[features] wrote {len(rows)} rows (position={args.position}) to {out_path}")

    # Also write a corpus-level summary alongside
    div = corpus_diversity(texts_for_corpus)
    summary_path = out_path.with_suffix(".summary.json")
    summary = {
        "model": rows[0]["model"] if rows else None,
        "condition": rows[0]["condition"] if rows else None,
        "language": rows[0]["language"] if rows else None,
        "n_samples": len(rows),
        "diversity": div,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[features] wrote summary to {summary_path}")
    return 0


def cmd_aggregate(args: argparse.Namespace) -> int:
    """Combine all per-file feature parquets into one big table for analysis."""
    root = Path(args.root)
    feat_dir = root / "features"
    if not feat_dir.exists():
        print(f"[aggregate] no features dir: {feat_dir}")
        return 1
    parquets = list(feat_dir.rglob("*.parquet"))
    if not parquets:
        print(f"[aggregate] no parquet files under {feat_dir}")
        return 1
    print(f"[aggregate] merging {len(parquets)} parquet files")
    dfs = [pd.read_parquet(p) for p in parquets]
    big = pd.concat(dfs, ignore_index=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    big.to_parquet(out)
    print(f"[aggregate] wrote {len(big)} rows to {out}")
    print(f"[aggregate] models: {big['model'].nunique()}, "
          f"conditions: {big['condition'].nunique()}, "
          f"languages: {big['language'].nunique()}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ex = sub.add_parser("extract", help="extract features for one JSONL file")
    p_ex.add_argument("--input", required=True)
    p_ex.add_argument("--output", required=True)
    p_ex.add_argument("--position", choices=["full", "first", "middle", "last"],
                       default="full",
                       help="extract features on a positional slice of the text")
    p_ex.add_argument("--slice_words", type=int, default=100,
                       help="word count for non-full position slices")
    p_ex.set_defaults(func=cmd_extract)

    p_ag = sub.add_parser("aggregate", help="merge all feature parquets")
    p_ag.add_argument("--root", default=str(PROJECT_ROOT / "outputs"))
    p_ag.add_argument("--out",  default=str(PROJECT_ROOT / "outputs" / "all_features.parquet"))
    p_ag.set_defaults(func=cmd_aggregate)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
