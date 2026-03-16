# CEFR-based Morphology-Aware Lexical Complexity Prediction for Kazakh


## Abstract
CEFR-graded lexical resources and lexical complexity prediction remain limited for low-resource Turkic languages, leaving it unclear how well existing approaches transfer to agglutinative settings. We introduce the first CEFR-graded lexicon for Kazakh, containing 4,561 lemma–POS entries across A1–C1, and use it to test whether explicit morphology improves lexical complexity prediction. We compare handcrafted morphological features, contextual embeddings, and models that combine both signal types on held-out CEFR classification. Our results show that morphology provides useful information beyond character-level lexical cues, contextual representations are strong on their own, and dual-encoder models that combine morphology with context achieve the best overall performance.

## Setup

```bash
conda env create -f environment.yml
conda activate kazakh_cefr_env
```

## Usage

### Preprocessing

```bash
python scripts/preprocess.py --data-type lexicon
```

### Training baselines

```bash
python scripts/train_baselines.py --variant majority
python scripts/train_baselines.py --variant freq_only
python scripts/train_baselines.py --variant full_feature
```

### Training neural models

```bash
# Context-only encoder (mBERT)
python scripts/train_context.py --variant mbert_context

# Dual encoder with concatenation fusion
python scripts/train_dual.py --variant dual_concat

# Dual encoder with gated fusion
python scripts/train_dual.py --variant dual_gated
```

### Evaluation

```bash
python scripts/evaluate.py --model dual_gated --generate-report
```

### Multi-seed ablation

Repeat training across seeds `42, 123, 456, 789, 2024` to reproduce Table 4:

```bash
for seed in 42 123 456 789 2024; do
  python scripts/train_dual.py --variant dual_gated --seed $seed
done
```

## Configuration

Hyperparameters are in `config/training.yaml`:

- **Epochs:** 10
- **Batch size:** 16
- **Learning rate:** 2e-5
- **Early stopping patience:** 3
- **Transformer:** `bert-base-multilingual-cased`
- **Fusion:** gated

Kazakh suffix inventory used for morphological feature extraction is defined in `config/kaz_suffixes.yaml`.

## Citation

```bibtex
@article{anonymous2026cefr,
  title   = {CEFR-based Morphology-Aware Lexical Complexity Prediction for Kazakh Language},
  author  = {Anonymous},
  journal = {ACM Transactions on Asian and Low-Resource Language Information Processing},
  year    = {2026}
}
```
