import os
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from src.config import (
    MODEL_TFIDF_PAC,
    MODEL_TFIDF_RF,
    MODEL_TFIDF_LR,
    MODEL_GLOVE_CNN_BILSTM,
    get_model_dir,
    get_model_path,
    get_vectorizer_path,
    get_metadata_path,
    KERAS_CONFIG
)
from src.data_preprocessing import load_dataset, clean_text

def get_classical_model(model_name: str):
    if model_name == MODEL_TFIDF_PAC:
        return PassiveAggressiveClassifier(max_iter=50, random_state=42)
    elif model_name == MODEL_TFIDF_RF:
        return RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == MODEL_TFIDF_LR:
        return LogisticRegression(max_iter=1000, random_state=42)
    else:
        raise ValueError(f"Unknown classical model: {model_name}")

def build_keras_model(config: dict):
    """
    Builds the GloVe + CNN-BiLSTM + Dense Softmax model.
    (Infrastructure only - no training)
    """
    # from tensorflow.keras.models import Sequential
    # from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout
    # model = Sequential()
    # model.add(Embedding(config["vocab_size"], config["embedding_dim"], input_length=config["sequence_length"]))
    # model.add(Conv1D(filters=config["cnn_filters"], kernel_size=config["kernel_size"], activation="relu"))
    # model.add(MaxPooling1D(pool_size=2))
    # model.add(Bidirectional(LSTM(config["bilstm_units"])))
    # model.add(Dropout(config["dropout"]))
    # model.add(Dense(config["dense_units"], activation="relu"))
    # model.add(Dense(2, activation="softmax"))
    # model.compile(optimizer=config["optimizer"], loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    # return model
    pass

def run_cross_validation(X, y, model_name: str, n_splits: int = 5):
    """
    Run 5-fold stratified cross-validation.
    TF-IDF is fitted ONLY on the training folds to prevent data leakage.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # 1. Preprocess
        X_train_fold = [clean_text(text) for text in X_train_fold]
        X_val_fold = [clean_text(text) for text in X_val_fold]
        
        # 2. TF-IDF (Fit ONLY on train fold)
        if model_name in [MODEL_TFIDF_PAC, MODEL_TFIDF_RF, MODEL_TFIDF_LR]:
            vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            X_train_vec = vectorizer.fit_transform(X_train_fold)
            X_val_vec = vectorizer.transform(X_val_fold)
            
            # 3. Train Model
            model = get_classical_model(model_name)
            # model.fit(X_train_vec, y_train_fold)
            
            # 4. Evaluate
            # y_pred = model.predict(X_val_vec)
            # metrics = calculate_metrics(y_val_fold, y_pred)
            # fold_metrics.append(metrics)
            
        elif model_name == MODEL_GLOVE_CNN_BILSTM:
            # Keras CV logic here
            pass
            
    # Calculate mean and std of fold_metrics
    return fold_metrics

def train_final_model(X_train, y_train, dataset: str, model_name: str):
    """
    Retrain the selected configuration on the complete training data.
    """
    # 1. Preprocess
    X_train_clean = [clean_text(text) for text in X_train]
    
    if model_name in [MODEL_TFIDF_PAC, MODEL_TFIDF_RF, MODEL_TFIDF_LR]:
        # 2. TF-IDF
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        # X_train_vec = vectorizer.fit_transform(X_train_clean)
        
        # 3. Train
        model = get_classical_model(model_name)
        # model.fit(X_train_vec, y_train)
        
        # 4. Save Artifacts
        # joblib.dump(model, get_model_path(dataset, model_name))
        # joblib.dump(vectorizer, get_vectorizer_path(dataset, model_name))
        
    elif model_name == MODEL_GLOVE_CNN_BILSTM:
        # Keras training logic here
        pass

def run_experiment(dataset: str, model_name: str):
    """
    Main experiment pipeline.
    """
    print(f"Starting experiment for {dataset} with {model_name}")
    
    # 1. Load Data
    # X_train, X_test, y_train, y_test = load_dataset(dataset)
    
    # 2. Cross Validation
    # cv_results = run_cross_validation(X_train, y_train, model_name)
    
    # 3. Train Final Model
    # train_final_model(X_train, y_train, dataset, model_name)
    
    # 4. Evaluate on Test Set
    # evaluate_model(dataset, model_name, X_test, y_test)
    
    print("Experiment infrastructure prepared. (No training executed)")

if __name__ == "__main__":
    # Example usage (commented out to prevent accidental training)
    # run_experiment("isot", MODEL_TFIDF_PAC)
    pass

