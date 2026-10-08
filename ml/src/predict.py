import json
from typing import Any, Dict, Optional

import joblib
import numpy as np

from src.config import (
    MODEL_NAMES,
    get_benchmark_json_path,
    get_metadata_path,
    get_model_path,
    get_vectorizer_path,
)
from src.data_preprocessing import clean_text, clean_text_liar


class NewsPredictor:
    def __init__(self):
        self.models: Dict[str, Dict[str, Any]] = {}
        self.vectorizers: Dict[str, Dict[str, Any]] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}

    def load_models(self, dataset: str) -> None:
        if dataset in self.models:
            return
        self.models[dataset] = {}
        self.vectorizers[dataset] = {}
        self.metadata[dataset] = {}
        for model_name in MODEL_NAMES:
            try:
                self.models[dataset][model_name] = joblib.load(get_model_path(dataset, model_name))
                self.vectorizers[dataset][model_name] = joblib.load(get_vectorizer_path(dataset, model_name))
                with get_metadata_path(dataset, model_name).open("r", encoding="utf-8") as file:
                    self.metadata[dataset][model_name] = json.load(file)
            except FileNotFoundError:
                continue

    def predict(self, text: str, dataset: str = "isot", model_name: Optional[str] = None) -> Dict[str, Any]:
        self.load_models(dataset)
        if not self.models[dataset]:
            raise RuntimeError(f"No trained models are available for dataset {dataset}.")
        cleaned_text = clean_text_liar(text) if dataset.lower() == "liar" else clean_text(text)
        if model_name is not None:
            if model_name not in self.models[dataset]:
                raise ValueError(f"Model {model_name} is not trained for dataset {dataset}.")
            return self._predict_classical(cleaned_text, dataset, model_name)
        predictions = [self._predict_classical(cleaned_text, dataset, name) for name in self.models[dataset]]
        fake_probability = float(np.mean([item["probabilities"]["FAKE"] for item in predictions]))
        prediction = "FAKE" if fake_probability >= 0.5 else "REAL"
        return {
            "prediction": prediction,
            "confidence": fake_probability if prediction == "FAKE" else 1.0 - fake_probability,
            "model": "Ensemble",
            "probabilities": {"FAKE": fake_probability, "REAL": 1.0 - fake_probability},
        }

    def _predict_classical(self, cleaned_text: str, dataset: str, model_name: str) -> Dict[str, Any]:
        model = self.models[dataset][model_name]
        features = self.vectorizers[dataset][model_name].transform([cleaned_text])
        raw_prediction = int(model.predict(features)[0])
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            positive_index = list(model.classes_).index(1)
            real_probability = float(probabilities[positive_index])
        else:
            score = float(model.decision_function(features)[0])
            real_probability = float(1.0 / (1.0 + np.exp(-np.clip(score, -50, 50))))
        fake_probability = 1.0 - real_probability
        prediction = "REAL" if raw_prediction == 1 else "FAKE"
        return {
            "prediction": prediction,
            "confidence": real_probability if prediction == "REAL" else fake_probability,
            "model": model_name,
            "probabilities": {"FAKE": fake_probability, "REAL": real_probability},
        }

    def predict_all(self, text: str, dataset: str, title: str = "") -> Dict[str, Any]:
        self.load_models(dataset)
        benchmark = {}
        benchmark_path = get_benchmark_json_path(dataset)
        if benchmark_path.exists():
            with benchmark_path.open("r", encoding="utf-8") as file:
                benchmark = json.load(file)
        benchmark_models = {item["id"]: item for item in benchmark.get("models", [])}
        results = []
        raw_input = f"{title} {text}".strip()
        cleaned_input = clean_text_liar(raw_input) if dataset.lower() == "liar" else clean_text(raw_input)
        for model_name in self.models[dataset]:
            result = self._predict_classical(cleaned_input, dataset, model_name)
            metrics = benchmark_models.get(model_name, {}).get("metrics", {})
            results.append({
                "model_id": model_name,
                "model_name": benchmark_models.get(model_name, {}).get("name", model_name),
                "vectorizer": "Word TF-IDF (1,2) with Sublinear TF" if dataset.lower() == "liar" else "TF-IDF (unigrams + bigrams)",
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                "probabilities": result["probabilities"],
                "val_weight": None,
                "test_dataset_metrics": metrics,
            })
        fake_probabilities = [item["probabilities"]["FAKE"] for item in results]
        fake_probability = float(np.mean(fake_probabilities))
        soft_prediction = "FAKE" if fake_probability >= 0.5 else "REAL"
        fake_votes = sum(probability >= 0.5 for probability in fake_probabilities)
        real_votes = len(results) - fake_votes
        hard_prediction = "FAKE" if fake_votes > real_votes else "REAL"
        return {
            "candidate_models": results,
            "ensembles": {
                "weighted_soft_ensemble": {
                    "name": "Mean Probability Ensemble",
                    "prediction": soft_prediction,
                    "confidence": fake_probability if soft_prediction == "FAKE" else 1.0 - fake_probability,
                    "probabilities": {"FAKE": fake_probability, "REAL": 1.0 - fake_probability},
                },
                "majority_voting_hard_ensemble": {
                    "name": "Majority Voting",
                    "prediction": hard_prediction,
                    "real_votes": real_votes,
                    "fake_votes": fake_votes,
                    "confidence": max(real_votes, fake_votes) / len(results) if results else 0.0,
                },
            },
        }


predictor = NewsPredictor()
