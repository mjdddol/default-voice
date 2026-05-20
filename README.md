# Default Voice -- Code and Data

Anonymous supplementary material for the EMNLP submission "The Default Voice of Language Models: How Chat Format Gates Model Behavior."

## Layout

```
.
+-- src/                  Sampling, feature extraction, probes, ablation, figures
+-- configs/              YAML: models, conditions, languages, features, robustness
+-- data/lexicons/        Hedging, refusal, assistant-phrase, politeness, formality, etc.
+-- analysis/             JSON results: probe accuracies, lexicon ablation, position split
\-- outputs/
    +-- all_features.parquet            Per-sample features used in every figure and table
    +-- summary.csv                     Ablation experiment per-cell means (Appendix R, Table 14)
    \-- raw_vs_dedup_body_full.csv      Raw-sample-weighted vs deduplicated means
                                        on the seven-family x three-condition main panel
                                        (Appendix R, Table 6)
```

## Requirements

Python 3.11+. Install:

```
pip install -r requirements.txt
```

Generation requires vLLM and a CUDA GPU; feature extraction and analysis run on CPU.

## Reproducing paper numbers

The single file `outputs/all_features.parquet` contains per-sample features for every (model, condition, language) cell used in the paper. Reproduce all main-text numbers with:

```
python -m src.paper_numbers --features outputs/all_features.parquet
```

Probe-transfer accuracies are in `analysis/probe_family_*.json` and `analysis/probe_transfer_*.json`.

The raw-vs-deduplicated sensitivity check (Appendix R, Table 6) is pre-computed in `outputs/raw_vs_dedup_body_full.csv`. To re-derive it from the raw generations (not bundled in this archive), run:

```
python -m src.raw_vs_dedup_compare
```

## Re-running from scratch

```
# Sample (one cell per call)
python -m src.sample --model_name MODEL --condition COND --language en --n 2000

# Extract features
python -m src.features --input outputs/generations --output outputs/all_features.parquet

# Probe and ablation
python -m src.probe --features outputs/all_features.parquet --out analysis/
python -m src.ablation --model MODEL --out outputs/ablation/

# Figures
python -m src.make_figures
```

Model identifiers and HuggingFace paths are in `configs/models.yaml`.

## License

Apache-2.0 (see LICENSE).
