# Fake News Detection — Machine Learning & NLP Microservice

An academic and production-ready Natural Language Processing (NLP) text classification system designed to detect lexical and stylistic patterns characteristic of fabricated versus credible news reporting.

---

## 📁 Machine Learning Folder Structure

```
ml/
├── data/
│   ├── Fake.csv                       # Deceptive / sensational news corpus
│   └── True.csv                       # Verified / standard journalistic news corpus
│
├── notebooks/
│   └── fake_news_analysis.ipynb       # Jupyter Notebook for EDA, TF-IDF, training & analysis
│
├── src/
│   ├── config.py                      # Global paths, hyperparameters, and label mappings
│   ├── data_preprocessing.py          # Text cleaning, regex normalization, stopword filtering
│   ├── generate_dataset.py            # Academic corpus generator with balanced classes
│   ├── train.py                       # Stratified train/test split, TF-IDF fitting, model training
│   ├── evaluate.py                    # Metrics calculation, confusion matrices, and ROC curves
│   └── predict.py                     # Inference engine with model caching and probability calibration
│
├── models/
│   ├── logistic_regression.joblib     # Serialized Logistic Regression classifier
│   ├── naive_bayes.joblib             # Serialized Multinomial Naive Bayes classifier
│   ├── svm.joblib                     # Serialized Calibrated Linear Support Vector Machine
│   ├── random_forest.joblib           # Serialized Random Forest ensemble classifier
│   ├── final_model.joblib             # Selected best model
│   ├── tfidf_vectorizer.joblib        # Fitted TF-IDF vectorizer (trained ONLY on X_train)
│   └── model_config.json              # Model selection and metadata configuration
│
├── results/
│   ├── metrics.json                   # Empirically measured quantitative metrics
│   ├── metrics_comparison.csv         # Tabular benchmark comparison across classifiers
│   ├── confusion_matrix.png           # 4-model confusion matrix heatmaps
│   ├── model_comparison.png           # Grouped bar chart (Accuracy, Precision, Recall, F1, ROC-AUC)
│   ├── roc_curve.png                  # Receiver Operating Characteristic curves with AUC scores
│   └── precision_recall_curve.png     # Precision-Recall curves
│
├── api/
│   └── main.py                        # FastAPI REST microservice with CORS & startup model caching
│
├── requirements.txt                   # Python library dependencies
└── README.md                          # Comprehensive ML documentation & academic guide
```

---

## 🛠️ Environment Setup & Installation

### 1. Install Dependencies
Make sure Python 3.9+ is installed:

```bash
cd ml
pip install -r requirements.txt
```

---

## 🏋️ Training & Evaluation Commands

### Step 1: Generate or Load Dataset
```bash
python src/generate_dataset.py
```

### Step 2: Run Model Training Pipeline
Fits the TF-IDF vectorizer strictly on `X_train` (80%), trains all candidate models, and selects the top-performing classifier:
```bash
python src/train.py
```

### Step 3: Run Academic Evaluation & Generate Visualizations
Evaluates out-of-sample test data, creates `results/metrics.json`, and exports high-resolution plots:
```bash
python src/evaluate.py
```

---

## 🚀 Running the FastAPI NLP Microservice

Start the high-performance asynchronous FastAPI server:

```bash
# Using Python directly
python api/main.py

# Or using Uvicorn CLI
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

The microservice will be active at:
- **API Base**: `http://localhost:8000`
- **Swagger Interactive Docs**: `http://localhost:8000/docs`
- **Redoc Documentation**: `http://localhost:8000/redoc`

---

## 📡 API Endpoints & Request/Response Contracts

### 1. Health Check
`GET /health`

#### Response (`200 OK`)
```json
{
  "status": "ok",
  "service": "Fake News Detection NLP Microservice",
  "active_model": "Logistic Regression",
  "uptime_seconds": 124.52
}
```

---

### 2. Classify News Article
`POST /predict`

#### Request Payload
```json
{
  "title": "Senate passes bipartisan clean energy infrastructure investment package",
  "text": "WASHINGTON (Reuters) - The United States Senate passed legislation allocating billions to grid modernization and renewable research after extensive committee reviews.",
  "model": "logistic_regression"
}
```

#### Response Payload (`200 OK`)
```json
{
  "prediction": "REAL",
  "confidence": 0.7393,
  "model": "Logistic Regression",
  "probabilities": {
    "FAKE": 0.2607,
    "REAL": 0.7393
  },
  "wordCount": 30,
  "charCount": 237,
  "topTokens": [
    "senate",
    "bill",
    "infrastructure",
    "investment",
    "legislation"
  ],
  "explanation": "Classified as REAL with 73.9% model confidence. Key lexical tokens identified include 'senate', 'bill', 'infrastructure', 'investment', 'legislation'. The text demonstrates formal journalistic syntax, institutional agency references, and neutral reporting tone."
}
```

---

## 🔗 How the Node.js Backend Communicates with FastAPI

In the full-stack architecture, the **Node.js + Express API server** acts as the gateway:

1. The client sends a `POST /api/predictions` request to Express (`http://localhost:5000/api/predictions`).
2. Express's `mlService.js` makes an HTTP request to FastAPI (`http://localhost:8000/predict`) using Axios:
   ```javascript
   const response = await axios.post('http://localhost:8000/predict', {
     title: req.body.title,
     text: req.body.text,
     model: req.body.model
   });
   ```
3. FastAPI executes inference using the pre-loaded in-memory model and returns the prediction, confidence, probabilities, and lexical explanation.
4. Express saves the prediction document to MongoDB and returns the formatted response back to the React UI.

---

## 🔬 Experimental Benchmark & Academic Analysis

### Quantitative Performance Summary

| Classifier | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 100.00% | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Linear SVM (Calibrated)** | 100.00% | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Multinomial Naive Bayes** | 100.00% | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Random Forest** | 100.00% | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

---

### Key Academic Insights & Methodological Findings

1. **Effects of TF-IDF Vectorization**:
   - Sublinear term frequency scaling ($\text{sublinear\_tf}=\text{True}$) prevents long, sensationalized articles from dominating the feature space with repeated trigger words.
   - Using bi-grams (`ngram_range=(1, 2)`) preserves critical contextual word pairs such as *"breaking exclusive"*, *"unverified whistleblower"*, and *"press briefing"*.

2. **Data Leakage Mitigation**:
   - The TF-IDF vocabulary and IDF weights were fitted **strictly on the training split ($X_{\text{train}}$)**. $X_{\text{test}}$ was transformed using out-of-sample projection, preventing information from the test distribution from leaking into the feature space.

3. **Probability Calibration for SVM**:
   - Standard `LinearSVC` optimizes a maximum-margin hyperplane and outputs uncalibrated decision distances. To provide valid confidence scores and ROC-AUC metrics, `CalibratedClassifierCV` with isotonic/sigmoid regression was applied.

4. **Model Performance Discrepancies**:
   - **Logistic Regression** provides smooth probabilistic boundaries and high interpretability via linear coefficients.
   - **Naive Bayes** operates under conditional feature independence assumptions, which can underestimate uncertainty in correlated phrase structures.

---

## ⚠️ Critical Academic Limitation & Disclaimer

> **Essential Methodological Boundary**:
> The models in this project are **statistical text classifiers trained to recognize stylistic, lexical, and syntactical patterns** correlated with the labeled articles in the dataset.
>
> - The model **does NOT verify real-world factual claims**, check external databases, or prove whether a stated event occurred.
> - A completely false statement written in dry, formal journalistic prose (e.g., mimicking wire agency style) may be classified as `"REAL"`.
> - Conversely, an authentic emergency bulletin written with emotional urgency or colloquial language may trigger `"FAKE"` stylistic markers.
>
> Automated NLP classification should be treated as an **assistive lexical analysis tool**, not an infallible arbiter of objective truth.
