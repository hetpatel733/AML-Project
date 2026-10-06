import os
import json
import time
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

from src.config import (
    MODEL_TFIDF_PAC,
    MODEL_TFIDF_RF,
    MODEL_TFIDF_LR,
    MODEL_GLOVE_CNN_BILSTM,
    get_model_dir,
    get_model_path,
    get_keras_model_path,
    get_vectorizer_path,
    get_tokenizer_path,
    get_metadata_path,
    get_predictions_dir,
    KERAS_CONFIG,
    GLOVE_PATH,
    ensure_glove_embeddings
)
from src.data_preprocessing import load_isot_dataset, load_liar_dataset
from src.evaluate import calculate_metrics, calculate_cv_statistics, save_predictions, write_dataset_benchmark

def train_classical_model(model_name, X_train, y_train, X_val, y_val, X_test, y_test, dataset_name):
    print(f"Training {model_name} on {dataset_name}...")
    start_time = time.time()
    
    # 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train_fold_vec = vectorizer.fit_transform(X_train_fold)
        X_val_fold_vec = vectorizer.transform(X_val_fold)
        
        if model_name == MODEL_TFIDF_PAC:
            model = PassiveAggressiveClassifier(max_iter=1000, random_state=42, C=1.0)
        elif model_name == MODEL_TFIDF_RF:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_name == MODEL_TFIDF_LR:
            model = LogisticRegression(max_iter=1000, random_state=42)
            
        model.fit(X_train_fold_vec, y_train_fold)
        y_pred = model.predict(X_val_fold_vec)
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_val_fold_vec)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_val_fold_vec)
        else:
            y_score = None
        
        metrics = calculate_metrics(y_val_fold, y_pred, y_score)
        fold_metrics.append(metrics)
        
    cv_stats = calculate_cv_statistics(fold_metrics)
    
    # Final training on full train set
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    if model_name == MODEL_TFIDF_PAC:
        model = PassiveAggressiveClassifier(max_iter=1000, random_state=42, C=1.0)
    elif model_name == MODEL_TFIDF_RF:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == MODEL_TFIDF_LR:
        model = LogisticRegression(max_iter=1000, random_state=42)
        
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test_vec)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test_vec)
    else:
        y_score = None
    
    test_metrics = calculate_metrics(y_test, y_pred, y_score)
    training_time = time.time() - start_time
    
    # Save artifacts
    model_dir = get_model_dir(dataset_name, model_name)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, get_model_path(dataset_name, model_name))
    joblib.dump(vectorizer, get_vectorizer_path(dataset_name, model_name))
    
    metadata = {
        "dataset": dataset_name,
        "model": model_name,
        "training_time": training_time,
        "cv_stats": cv_stats,
        "test_metrics": test_metrics,
        "hyperparameters": {"max_features": 5000, "ngram_range": [1, 1], "random_state": 42}
    }
    with open(get_metadata_path(dataset_name, model_name), "w") as f:
        json.dump(metadata, f, indent=4)
        
    # Save predictions
    save_predictions(dataset_name, model_name, "v1", X_test, y_test, y_pred, y_score if y_score is not None else [None]*len(y_pred), get_predictions_dir(dataset_name))
    
    return metadata, fold_metrics, cv_stats, test_metrics, y_pred, y_score

def load_glove_embedding_matrix(tokenizer, glove_path, vocab_size, embedding_dim):
    embeddings_index = {}
    with open(glove_path, "r", encoding="utf-8") as glove_file:
        for line in glove_file:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = vector

    embedding_matrix = np.zeros((vocab_size + 1, embedding_dim), dtype="float32")
    for word, index in tokenizer.word_index.items():
        if index <= vocab_size:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector
    return embedding_matrix


def build_keras_model(vocab_size, embedding_dim, sequence_length, embedding_matrix=None):
    model = Sequential()
    model.add(Input(shape=(sequence_length,), dtype="int32"))
    if embedding_matrix is not None:
        model.add(Embedding(vocab_size + 1, embedding_dim, weights=[embedding_matrix], trainable=False))
    else:
        model.add(Embedding(vocab_size, embedding_dim))
    model.add(Conv1D(filters=128, kernel_size=5, activation="relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def train_keras_model(model_name, X_train, y_train, X_val, y_val, X_test, y_test, dataset_name):
    print(f"Training {model_name} on {dataset_name}...")
    ensure_glove_embeddings()
    start_time = time.time()
    
    vocab_size = KERAS_CONFIG["vocab_size"]
    sequence_length = KERAS_CONFIG["sequence_length"]
    embedding_dim = KERAS_CONFIG["embedding_dim"]
    
    # 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        
        tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
        tokenizer.fit_on_texts(X_train_fold)
        embedding_matrix = load_glove_embedding_matrix(tokenizer, GLOVE_PATH, vocab_size, embedding_dim)
        
        X_train_fold_seq = pad_sequences(tokenizer.texts_to_sequences(X_train_fold), maxlen=sequence_length)
        X_val_fold_seq = pad_sequences(tokenizer.texts_to_sequences(X_val_fold), maxlen=sequence_length)
        
        model = build_keras_model(vocab_size, embedding_dim, sequence_length, embedding_matrix)
        
        early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        
        model.fit(X_train_fold_seq, y_train_fold, validation_data=(X_val_fold_seq, y_val_fold), epochs=10, batch_size=32, callbacks=[early_stopping], verbose=0)
        
        y_score = model.predict(X_val_fold_seq).flatten()
        y_pred = (y_score > 0.5).astype(int)
        
        metrics = calculate_metrics(y_val_fold, y_pred, y_score)
        fold_metrics.append(metrics)
        
    cv_stats = calculate_cv_statistics(fold_metrics)
    
    # Final training on full train set
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    embedding_matrix = load_glove_embedding_matrix(tokenizer, GLOVE_PATH, vocab_size, embedding_dim)
    
    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=sequence_length)
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=sequence_length)
    
    model = build_keras_model(vocab_size, embedding_dim, sequence_length, embedding_matrix)
    
    early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    
    # We use X_val for early stopping in final training
    X_val_seq = pad_sequences(tokenizer.texts_to_sequences(X_val), maxlen=sequence_length)
    
    history = model.fit(X_train_seq, y_train, validation_data=(X_val_seq, y_val), epochs=10, batch_size=32, callbacks=[early_stopping], verbose=1)
    
    y_score = model.predict(X_test_seq).flatten()
    y_pred = (y_score > 0.5).astype(int)
    
    test_metrics = calculate_metrics(y_test, y_pred, y_score)
    training_time = time.time() - start_time
    
    # Save artifacts
    model_dir = get_model_dir(dataset_name, model_name)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model.save(get_keras_model_path(dataset_name, model_name))
    joblib.dump(tokenizer, get_tokenizer_path(dataset_name, model_name))
    
    metadata = {
        "dataset": dataset_name,
        "model": model_name,
        "training_time": training_time,
        "cv_stats": cv_stats,
        "test_metrics": test_metrics,
        "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
        "training_history": history.history,
        "hyperparameters": KERAS_CONFIG
    }
    with open(get_metadata_path(dataset_name, model_name), "w") as f:
        json.dump(metadata, f, indent=4)
        
    # Save predictions
    save_predictions(dataset_name, model_name, "v1", X_test, y_test, y_pred, y_score, get_predictions_dir(dataset_name))
    
    return metadata, fold_metrics, cv_stats, test_metrics, y_pred, y_score

def main():
    print("Loading ISOT dataset...")
    X_train_isot, X_val_isot, X_test_isot, y_train_isot, y_val_isot, y_test_isot = load_isot_dataset()
    
    print("Loading LIAR dataset...")
    X_train_liar, X_val_liar, X_test_liar, y_train_liar, y_val_liar, y_test_liar = load_liar_dataset()
    
    datasets = {
        "isot": (X_train_isot, y_train_isot, X_val_isot, y_val_isot, X_test_isot, y_test_isot),
        "liar": (X_train_liar, y_train_liar, X_val_liar, y_val_liar, X_test_liar, y_test_liar)
    }
    
    classical_models = [MODEL_TFIDF_PAC, MODEL_TFIDF_RF, MODEL_TFIDF_LR]
    
    for dataset_name, (X_train, y_train, X_val, y_val, X_test, y_test) in datasets.items():
        model_results = []
        test_predictions = []
        for model_name in classical_models:
            metadata, fold_metrics, cv_stats, test_metrics, y_pred, y_score = train_classical_model(model_name, X_train, y_train, X_val, y_val, X_test, y_test, dataset_name)
            test_predictions.append((model_name, cv_stats.get("f1_mean", 0.0), y_pred, y_score))
            model_results.append({
                "model": model_name,
                "name": {MODEL_TFIDF_PAC: "TF-IDF + Passive Aggressive Classifier", MODEL_TFIDF_RF: "TF-IDF + Random Forest", MODEL_TFIDF_LR: "TF-IDF + Logistic Regression"}[model_name],
                "representation": "TF-IDF",
                "cv_folds": fold_metrics,
                "cv_stats": cv_stats,
                "test_metrics": test_metrics,
                "training_time": metadata["training_time"],
                "hyperparameters": metadata["hyperparameters"],
                "artifact_path": str(get_model_dir(dataset_name, model_name)),
                "prediction_archive": f"ml/results/{dataset_name}/predictions/v1_{model_name}_predictions.jsonl"
            })
            
        try:
            metadata, fold_metrics, cv_stats, test_metrics, y_pred, y_score = train_keras_model(MODEL_GLOVE_CNN_BILSTM, X_train, y_train, X_val, y_val, X_test, y_test, dataset_name)
            test_predictions.append((MODEL_GLOVE_CNN_BILSTM, cv_stats.get("f1_mean", 0.0), y_pred, y_score))
            model_results.append({
                "model": MODEL_GLOVE_CNN_BILSTM,
                "name": "GloVe + CNN-BiLSTM + Dense Softmax",
                "type": "deep_learning",
                "representation": "GloVe",
                "cv_folds": fold_metrics,
                "cv_stats": cv_stats,
                "test_metrics": test_metrics,
                "training_time": metadata["training_time"],
                "hyperparameters": metadata["hyperparameters"],
                "training_history": metadata["training_history"],
                "artifact_path": str(get_model_dir(dataset_name, MODEL_GLOVE_CNN_BILSTM)),
                "prediction_archive": f"ml/results/{dataset_name}/predictions/v1_{MODEL_GLOVE_CNN_BILSTM}_predictions.jsonl"
            })
        except Exception as e:
            print(f"Failed to train Keras model on {dataset_name}: {e}")
            model_results.append({
                "model": MODEL_GLOVE_CNN_BILSTM,
                "name": "GloVe + CNN-BiLSTM + Dense Softmax",
                "type": "deep_learning",
                "representation": "GloVe",
                "status": "unavailable",
                "reason": str(e),
                "cv_folds": [],
                "cv_stats": {},
                "test_metrics": {},
                "training_time": None,
                "artifact_path": None,
                "prediction_archive": None
            })

        weights = np.array([item[1] for item in test_predictions], dtype=float)
        weights = weights / weights.sum() if weights.sum() else np.ones(len(test_predictions)) / max(1, len(test_predictions))
        hard_pred = (np.mean([item[2] for item in test_predictions], axis=0) >= 0.5).astype(int)
        soft_scores = []
        for _, _, _, scores in test_predictions:
            values = np.asarray(scores, dtype=float)
            if np.any((values < 0) | (values > 1)):
                values = 1.0 / (1.0 + np.exp(-values))
            soft_scores.append(values)
        soft_score = np.average(soft_scores, axis=0, weights=weights)
        soft_pred = (soft_score >= 0.5).astype(int)
        soft_metrics = calculate_metrics(y_test, soft_pred, soft_score)
        hard_metrics = calculate_metrics(y_test, hard_pred)
        ensemble = {
            "models": [item[0] for item in test_predictions],
            "votingMethod": "validation-weighted soft voting and majority hard voting",
            "weights": {item[0]: float(weights[index]) for index, item in enumerate(test_predictions)},
            "validationBasis": "normalized five-fold CV F1 mean",
            "soft": {"metrics": soft_metrics, "confusionMatrix": {"labels": ["FAKE", "REAL"], "matrix": [[soft_metrics["tn"], soft_metrics["fp"]], [soft_metrics["fn"], soft_metrics["tp"]]]}},
            "hard": {"metrics": hard_metrics, "confusionMatrix": {"labels": ["FAKE", "REAL"], "matrix": [[hard_metrics["tn"], hard_metrics["fp"]], [hard_metrics["fn"], hard_metrics["tp"]]]}}
        }

        write_dataset_benchmark(
            dataset_name,
            "v2",
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            {"training": (X_train, y_train), "validation": (X_val, y_val), "testing": (X_test, y_test)},
            {"false": 0, "barely-true": 0, "pants-fire": 0, "true": 1, "mostly-true": 1, "half-true": 1} if dataset_name == "liar" else {"FAKE": 0, "REAL": 1},
            {"textCleaning": "lowercase, URL/HTML/punctuation/number removal, stopword removal", "tfidf": {"maxFeatures": 5000, "ngramRange": [1, 1]}},
            model_results,
            ensemble
        )

if __name__ == "__main__":
    main()

