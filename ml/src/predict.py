import json
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.config import (
    MODEL_TFIDF_PAC,
    MODEL_TFIDF_RF,
    MODEL_TFIDF_LR,
    MODEL_GLOVE_CNN_BILSTM,
    get_model_path,
    get_keras_model_path,
    get_vectorizer_path,
    get_tokenizer_path,
    get_metadata_path,
    KERAS_CONFIG,
    get_benchmark_json_path
)
from src.data_preprocessing import clean_text

class NewsPredictor:
    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        self.tokenizers = {}
        self.metadata = {}
        
    def load_models(self, dataset: str):
        if dataset in self.models:
            return
            
        self.models[dataset] = {}
        self.vectorizers[dataset] = {}
        self.tokenizers[dataset] = {}
        self.metadata[dataset] = {}

        available_models = set()
        benchmark_path = get_benchmark_json_path(dataset)
        if benchmark_path.exists():
            with open(benchmark_path, "r", encoding="utf-8") as file:
                benchmark = json.load(file)
            available_models = {
                model["id"] for model in benchmark.get("models", [])
                if model.get("status") == "trained"
            }
        
        classical_models = [MODEL_TFIDF_PAC, MODEL_TFIDF_RF, MODEL_TFIDF_LR]
        
        for model_name in classical_models:
            if available_models and model_name not in available_models:
                continue
            try:
                self.models[dataset][model_name] = joblib.load(get_model_path(dataset, model_name))
                self.vectorizers[dataset][model_name] = joblib.load(get_vectorizer_path(dataset, model_name))
                with open(get_metadata_path(dataset, model_name), "r") as f:
                    self.metadata[dataset][model_name] = json.load(f)
            except Exception as e:
                print(f"Failed to load {model_name} for {dataset}: {e}")
                
        if not available_models or MODEL_GLOVE_CNN_BILSTM in available_models:
            try:
                self.models[dataset][MODEL_GLOVE_CNN_BILSTM] = tf.keras.models.load_model(get_keras_model_path(dataset, MODEL_GLOVE_CNN_BILSTM))
                self.tokenizers[dataset][MODEL_GLOVE_CNN_BILSTM] = joblib.load(get_tokenizer_path(dataset, MODEL_GLOVE_CNN_BILSTM))
                with open(get_metadata_path(dataset, MODEL_GLOVE_CNN_BILSTM), "r") as f:
                    self.metadata[dataset][MODEL_GLOVE_CNN_BILSTM] = json.load(f)
            except Exception as e:
                print(f"Failed to load {MODEL_GLOVE_CNN_BILSTM} for {dataset}: {e}")
            
    def predict(self, text: str, dataset: str = "isot", model_name: str = None) -> Dict[str, Any]:
        self.load_models(dataset)
        
        cleaned_text = clean_text(text)
        
        if model_name is None:
            # Ensemble
            return self._predict_ensemble(cleaned_text, dataset)
            
        if model_name not in self.models[dataset]:
            raise ValueError(f"Model {model_name} not loaded for dataset {dataset}")
            
        if model_name == MODEL_GLOVE_CNN_BILSTM:
            return self._predict_keras(cleaned_text, dataset)
        else:
            return self._predict_classical(cleaned_text, dataset, model_name)
            
    def _predict_classical(self, cleaned_text: str, dataset: str, model_name: str) -> Dict[str, Any]:
        model = self.models[dataset][model_name]
        vectorizer = self.vectorizers[dataset][model_name]
        
        vec_features = vectorizer.transform([cleaned_text])
        raw_pred = int(model.predict(vec_features)[0])
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(vec_features)[0]
            prob_fake = float(probs[0])
            prob_real = float(probs[1])
        elif hasattr(model, "decision_function"):
            decision = model.decision_function(vec_features)[0]
            prob_real = 1 / (1 + np.exp(-decision))
            prob_fake = 1 - prob_real
        else:
            prob_fake = 1.0 if raw_pred == 0 else 0.0
            prob_real = 1.0 if raw_pred == 1 else 0.0
            
        confidence = prob_fake if raw_pred == 1 else prob_real
        
        return {
            "prediction": "FAKE" if raw_pred == 0 else "REAL",
            "confidence": confidence,
            "model": model_name,
            "probabilities": {"FAKE": prob_fake, "REAL": prob_real}
        }
        
    def _predict_keras(self, cleaned_text: str, dataset: str) -> Dict[str, Any]:
        model = self.models[dataset][MODEL_GLOVE_CNN_BILSTM]
        tokenizer = self.tokenizers[dataset][MODEL_GLOVE_CNN_BILSTM]
        
        seq = tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(seq, maxlen=KERAS_CONFIG["sequence_length"])
        
        prob_real = float(model.predict(padded, verbose=0)[0][0])
        prob_fake = 1.0 - prob_real
        
        raw_pred = 0 if prob_fake > 0.5 else 1
        confidence = prob_fake if raw_pred == 1 else prob_real
        
        return {
            "prediction": "FAKE" if raw_pred == 1 else "REAL",
            "confidence": confidence,
            "model": MODEL_GLOVE_CNN_BILSTM,
            "probabilities": {"FAKE": prob_fake, "REAL": prob_real}
        }
        
    def _predict_ensemble(self, cleaned_text: str, dataset: str) -> Dict[str, Any]:
        probs_fake = []
        
        for model_name in self.models[dataset]:
            if model_name == MODEL_GLOVE_CNN_BILSTM:
                res = self._predict_keras(cleaned_text, dataset)
            else:
                res = self._predict_classical(cleaned_text, dataset, model_name)
            probs_fake.append(res["probabilities"]["FAKE"])
            
        avg_prob_fake = np.mean(probs_fake)
        avg_prob_real = 1.0 - avg_prob_fake
        
        raw_pred = 1 if avg_prob_fake > 0.5 else 0
        confidence = avg_prob_fake if raw_pred == 1 else avg_prob_real
        
        return {
            "prediction": "FAKE" if raw_pred == 1 else "REAL",
            "confidence": confidence,
            "model": "Ensemble",
            "probabilities": {"FAKE": avg_prob_fake, "REAL": avg_prob_real}
        }

    def predict_all(self, text: str, dataset: str, title: str = "") -> Dict[str, Any]:
        self.load_models(dataset)
        cleaned_text = clean_text(title + " " + text)

        benchmark_models = {}
        benchmark_ensemble = {}
        benchmark_path = get_benchmark_json_path(dataset)
        if benchmark_path.exists():
            with benchmark_path.open("r", encoding="utf-8") as file:
                benchmark = json.load(file)
            benchmark_models = {model["id"]: model for model in benchmark.get("models", [])}
            benchmark_ensemble = benchmark.get("ensemble", {})

        # Derive weights from the current four-model CV results so stale ensemble
        # metadata cannot exclude the newly trained GloVe model.
        f1_scores = {
            model_id: model.get("crossValidation", {}).get("mean", {}).get("f1")
            for model_id, model in benchmark_models.items()
            if model.get("status") == "trained" and model_id in self.models[dataset]
        }
        total_f1 = sum(score for score in f1_scores.values() if isinstance(score, (int, float)))
        benchmark_weights = {
            model_id: score / total_f1
            for model_id, score in f1_scores.items()
            if isinstance(score, (int, float))
        } if total_f1 else {}
        
        candidate_models = []
        probs_fake = []
        model_weights = []
        
        for model_name in self.models[dataset]:
            if model_name == MODEL_GLOVE_CNN_BILSTM:
                res = self._predict_keras(cleaned_text, dataset)
            else:
                res = self._predict_classical(cleaned_text, dataset, model_name)
            
            benchmark_model = benchmark_models.get(model_name, {})
            benchmark_metrics = benchmark_model.get("metrics", {})
            candidate_models.append({
                "model_id": model_name,
                "model_name": benchmark_model.get("name", model_name),
                "vectorizer": benchmark_model.get("representation"),
                "prediction": res["prediction"],
                "confidence": res["confidence"],
                "probabilities": res["probabilities"],
                "val_weight": benchmark_weights.get(model_name),
                "test_dataset_metrics": {
                    "accuracy": benchmark_metrics.get("accuracy"),
                    "precision": benchmark_metrics.get("precision"),
                    "recall": benchmark_metrics.get("recall"),
                    "f1": benchmark_metrics.get("f1"),
                    "roc_auc": benchmark_metrics.get("rocAuc")
                } if benchmark_metrics else None
            })
            probs_fake.append(res["probabilities"]["FAKE"])
            model_weights.append(benchmark_weights.get(model_name, 0.0))
            
        weight_total = sum(model_weights)
        avg_prob_fake = float(np.average(probs_fake, weights=model_weights)) if weight_total else float(np.mean(probs_fake))
        avg_prob_real = 1.0 - avg_prob_fake
        soft_pred = "FAKE" if avg_prob_fake > 0.5 else "REAL"
        soft_conf = avg_prob_fake if soft_pred == "FAKE" else avg_prob_real
        
        fake_votes = sum(1 for p in probs_fake if p > 0.5)
        real_votes = len(probs_fake) - fake_votes
        hard_pred = "FAKE" if fake_votes > real_votes else "REAL"
        hard_conf = float(max(fake_votes, real_votes) / len(probs_fake))
        
        return {
            "candidate_models": candidate_models,
            "ensembles": {
                "weighted_soft_ensemble": {
                    "prediction": soft_pred,
                    "confidence": soft_conf,
                    "probabilities": {"FAKE": avg_prob_fake, "REAL": avg_prob_real},
                    "test_dataset_metrics": benchmark_ensemble.get("soft", {}).get("metrics")
                },
                "majority_voting_hard_ensemble": {
                    "prediction": hard_pred,
                    "confidence": hard_conf,
                    "test_dataset_metrics": benchmark_ensemble.get("hard", {}).get("metrics")
                }
            }
        }

predictor = NewsPredictor()

