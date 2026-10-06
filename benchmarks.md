# Fake News Detection Benchmarks

## Overview
This document tracks the performance of 4 models across 2 datasets (ISOT and LIAR).

**Status:** Classical models trained; GloVe CNN-BiLSTM is unavailable until the configured embedding file is supplied.

## Datasets
- **ISOT**: Fake/True binary classification.
- **LIAR**: 6-class mapped to binary (real/fake).

## Methodology
- **Preprocessing**: Lowercase, URL removal, whitespace normalization.
- **Cross-Validation**: 5-fold stratified CV on training data. TF-IDF fitted per fold.
- **Hyperparameter Search**: Conducted within CV folds.
- **Final Evaluation**: Retrained on full train set, evaluated once on untouched test set.

## Models
1. TF-IDF + Passive Aggressive Classifier
2. TF-IDF + Random Forest
3. TF-IDF + Logistic Regression
4. GloVe + CNN-BiLSTM + Dense Softmax

## Results (ISOT)
See `ml/results/isot/benchmark.json` for the authoritative metrics and model availability status.

## Results (LIAR)
See `ml/results/liar/benchmark.json` for the authoritative metrics and model availability status.

## Artifacts
- **Models**: `ml/models/<dataset>/<model_name>/`
- **Predictions**: `ml/results/<dataset>/predictions/`
- **Metrics**: `ml/results/<dataset>/benchmark.json` and `benchmarks.csv`

