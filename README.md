# Fake News Detection Using Machine Learning & Natural Language Processing

An academic research and production-grade full-stack system for detecting deceptive, manipulated, and fabricated news content using Natural Language Processing (NLP), Machine Learning, React, D3.js data visualizations, Node.js/Express REST API, and MongoDB.

---

## 🏛️ Architecture Overview

The system is organized into decoupled micro-layers:

```
┌────────────────────────────────────────────────────────┐
│                   React 18 + Vite UI                   │
│  (Custom D3.js Charts, Real-time Dashboard, History)   │
└───────────────────────────┬────────────────────────────┘
                            │ HTTP / Axios (Port 5173)
                            ▼
┌────────────────────────────────────────────────────────┐
│              Node.js + Express.js API Server           │
│    (REST Endpoints, MongoDB Mongoose, Validation)      │
└───────────────────────────┬────────────────────────────┘
                            │ HTTP POST (Port 8000)
                            ▼
┌────────────────────────────────────────────────────────┐
│                 Python FastAPI ML Service              │
│ (Six trained NLP models + calibrated ensemble service) │
└────────────────────────────────────────────────────────┘
```

---

## 📦 Project Structure

```
fake-news-detection/
├── client/                     # React 18 + Vite Frontend with D3.js Visualizations
│   ├── public/
│   ├── src/
│   │   ├── charts/             # Pure D3.js Visualization Components
│   │   │   ├── DonutChart.jsx          # Fake vs Real Distribution
│   │   │   ├── ConfidenceHistogram.jsx # Prediction Confidence Spread
│   │   │   ├── TimeSeriesTrend.jsx     # Volume & Daily Detection Trends
│   │   │   ├── ModelComparisonBar.jsx  # Accuracy, Precision, Recall, F1
│   │   │   └── ConfusionMatrix.jsx     # TP, FP, TN, FN Error Matrix
│   │   ├── components/         # Reusable UI Blocks (Navbar, Footer, StatCard, Badge, etc.)
│   │   ├── pages/              # Home, Dashboard, Predict, Analytics, History, About, NotFound
│   │   ├── services/api.js     # Axios API Client
│   │   ├── utils/helpers.js    # Formatting, Confidence Helpers, Color Themes
│   │   ├── App.jsx             # Router Setup & Navigation Container
│   │   ├── index.css           # Modern Academic Dark/Light Theme System
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
│
├── server/                     # Node.js + Express.js + MongoDB Backend
│   ├── config/
│   │   └── db.js               # MongoDB Mongoose Connection with Resilient Retry
│   ├── controllers/
│   │   ├── predictionController.js # Predict, History, Pagination, Filter, Search, Delete
│   │   └── analyticsController.js  # Aggregation and artifact-backed experiment metrics
│   ├── middleware/
│   │   ├── errorMiddleware.js      # Centralized Error & 404 Handlers
│   │   └── validationMiddleware.js # Input & Query Sanitization and Schema Validation
│   ├── models/
│   │   └── Prediction.js       # MongoDB Schema (title, text, prediction, confidence, explanation)
│   ├── routes/
│   │   ├── predictionRoutes.js # /api/predictions & /api/predictions/stats
│   │   └── analyticsRoutes.js  # /api/analytics & /api/analytics/performance
│   ├── services/
│   │   └── mlService.js        # Axios Client for Python ML Microservice
│   ├── utils/
│   │   └── response.js         # Standardized JSON Response Formatter
│   ├── .env.example            # Environment Variable Template
│   ├── app.js                  # Express App, Helmet, CORS, Route Mounts
│   ├── package.json
│   └── server.js               # Server Entry Point & Process Handlers
│
└── package.json                # Root Runner Scripts
```

---

## � Docker Deployment (All-in-One Container)

The project includes an optimized, production-grade **All-in-One Docker Container** that embeds:
1. **Embedded MongoDB 7.0 Server** (with automated storage initialization and volume persistence)
2. **Python 3.11 NLP FastAPI Microservice** (loading all 6 trained machine learning models)
3. **Node.js Express API Server & Static Asset Server** (serving REST endpoints and the compiled React UI)
4. **React 18 + Vite Frontend** (pre-compiled into static assets via multi-stage build)

### 🗜️ Container Compression & Size-Reduction Techniques Used:
- **Multi-Stage Build**: Client assets are built in an ephemeral `node:20-alpine` stage; dev tools and node_modules (~250MB) are discarded.
- **Python `--no-cache-dir`**: Prevents wheel caching in image layers.
- **Single-Layer Apt Cleanup**: Combined `apt-get` installation, GPG key provisioning, and aggressive removal of lists/caches/documentation.
- **Production-Only Dependencies**: Server installs exclusively production dependencies (`npm ci --omit=dev`).
- **Python Bytecode Suppression**: `PYTHONDONTWRITEBYTECODE=1` prevents redundant `.pyc` accumulation.

### 🚀 Quick Start with Docker:

#### Option 1: Docker CLI (Single Command)
```bash
# Build the optimized image
docker build -t fake-news-detection .

# Run the container
docker run -d -p 5000:5000 -p 8000:8000 --name fake-news-app fake-news-detection
```

#### Option 2: Docker Compose
```bash
docker compose up --build -d
```

### 🌐 Access Endpoints
- **Web Application & REST API**: [http://localhost:5000](http://localhost:5000)
- **FastAPI NLP Inference Microservice**: [http://localhost:8000](http://localhost:8000) (Swagger Docs: [http://localhost:8000/docs](http://localhost:8000/docs))
- **Embedded MongoDB**: `mongodb://127.0.0.1:27017/fake_news_detection` (internal)

---

## 🚀 Getting Started Locally

### Prerequisites
- **Node.js**: v18.0.0 or higher
- **npm**: v9.0.0 or higher
- **MongoDB**: Local MongoDB community service (`mongodb://127.0.0.1:27017`) or MongoDB Atlas URI (Optional for initial development)
- **Python**: 3.9+ for local ML training and FastAPI inference

---

### ⚙️ Backend Setup (`/server`)

1. **Navigate to server directory and install dependencies:**
   ```bash
   cd fake-news-detection/server
   npm install
   ```

2. **Configure Environment Variables:**
   Create a `.env` file in `server/` (or copy `.env.example`):
   ```ini
   PORT=5000
   NODE_ENV=development
   MONGO_URI=mongodb://127.0.0.1:27017/fake_news_db
   PYTHON_ML_URL=http://localhost:8000/predict
   CLIENT_ORIGIN=http://localhost:5173
   USE_MOCK_ML=true
   ```

3. **Start the Express API Server:**
   ```bash
   # Development mode with auto-reload (Node.js --watch)
   npm run dev

   # Or standard start
   npm start
   ```
   *The server will start on `http://localhost:5000`.*

---

### 💻 Frontend Setup (`/client`)

1. **Navigate to client directory and install dependencies:**
   ```bash
   cd fake-news-detection/client
   npm install
   ```

2. **Start the Vite Development Server:**
   ```bash
   npm run dev
   ```
   *The frontend dashboard will be available at `http://localhost:5173`.*

---

## 🧪 Experimental Methodology

The website reports the saved ML experiment artifact rather than manually written benchmark values. The source of truth is generated at:

- `ml/models/model_config.json`
- `ml/results/metrics.json`
- `ml/results/metrics_comparison.csv`

### Dataset currently available

The repository currently contains `ml/data/Fake.csv` with 750 records and `ml/data/True.csv` with 750 records. After preprocessing, the current experiment contains 1,500 valid records: 750 `FAKE` and 750 `REAL`. The preprocessing pipeline checks missing values, empty text, duplicates, cleaned text, labels, and class distribution. A larger original ISOT distribution is not included locally, so the application does not claim to have evaluated records that are absent from the repository.

### Leakage-controlled split

The experiment uses a fixed random seed (`42`) and a stratified 70/15/15 split:

| Partition | Records | Purpose |
|---|---:|---|
| Training | 1,050 | Model fitting, vectorizer fitting, and training-only tuning |
| Validation | 225 | Ensemble weighting and validation checks |
| Final test | 225 | One-time internal evaluation |

CountVectorizer and TfidfVectorizer are fitted only on training text. The final test set is not used for hyperparameter selection or ensemble weighting.

### Models and evaluation

The experiment retains six models: BoW + Logistic Regression, TF-IDF + Logistic Regression, TF-IDF + Multinomial Naive Bayes, TF-IDF + calibrated Linear SVM, TF-IDF + Random Forest, and TF-IDF + calibrated Passive Aggressive. Each model uses training-only `GridSearchCV` followed by 5-fold stratified cross-validation on the training partition. The artifact stores CV mean/std for Accuracy, Precision, Recall, and F1, plus validation and final-test Accuracy, Precision, Recall, F1, ROC-AUC, and confusion matrices.

The two ensembles are also retained: majority voting and validation-F1 weighted probability voting. Ensemble weights are calculated from validation F1 only.

### Current result and limitations

The regenerated experiment still records perfect internal scores on this available corpus. These values are not lowered or replaced with invented values. They should be interpreted cautiously because source, topic, and writing-style artifacts may make the classes unusually separable. Prediction confidence is not the same as accuracy for an individual article.

External validation is currently **not performed**. No compatible independent fake/real labeled dataset is included in the repository, so the application does not fabricate external metrics or a generalization gap. Add and document an independent dataset before publishing external performance claims.

Experiment metadata records version `v2`, the UTC training timestamp, dataset counts, selected hyperparameters, data-quality counts, and leakage controls.

### Re-running the experiment

From the repository root:

```bash
cd ml
python -m src.train
```

This retrains all six models and regenerates vectorizers, model artifacts, split arrays, metadata, metrics JSON, and the comparison CSV. The backend and frontend read the resulting metadata dynamically.

---

## 📡 REST API Documentation

### Base URL: `http://localhost:5000/api`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | API health check & server status |
| `POST` | `/predictions` | Classify news article (calls ML service and saves to MongoDB) |
| `GET` | `/predictions` | Retrieve prediction history (supports `page`, `limit`, `prediction`, `model`, `search`, `sortBy`) |
| `GET` | `/predictions/stats` | Retrieve overview statistics for dashboard scorecards |
| `GET` | `/predictions/:id` | Get single prediction record by ID |
| `DELETE` | `/predictions/:id` | Delete prediction record by ID |
| `GET` | `/analytics` | Prediction aggregates plus artifact-backed model metrics and confusion matrix |
| `GET` | `/model-performance` | Artifact-backed test metrics, CV metrics, and external-validation fields |

---

### Example: News Article Prediction

#### Request
```http
POST /api/predictions
Content-Type: application/json

{
  "title": "Government Approves New Environmental Protection Subsidies",
  "text": "The Ministry of Environment has officially announced the implementation of a comprehensive subsidy program aimed at accelerating renewable energy transitions across municipal sectors.",
  "model": "Passive Aggressive"
}
```

#### Response (`201 Created`)
```json
{
  "success": true,
  "data": {
    "_id": "65e6d0a7a4b87c12f0a1b2c3",
    "title": "Government Approves New Environmental Protection Subsidies",
    "text": "The Ministry of Environment has officially announced...",
    "prediction": "REAL",
    "confidence": 0.942,
    "model": "Passive Aggressive",
    "explanation": "High linguistic density, consistent journalistic syntax, verified institutional entities, and low sensationalism marker frequency.",
    "probabilities": {
      "REAL": 0.942,
      "FAKE": 0.058
    },
    "wordCount": 26,
    "charCount": 182,
    "source": "api_v1",
    "createdAt": "2025-03-05T08:30:00.000Z"
  },
  "message": "Prediction generated and saved successfully"
}
```

---

## 🧠 Python ML Service Integration Contract

The Express server dispatches news payload to the Python ML microservice via HTTP `POST` to `process.env.PYTHON_ML_URL` (`http://localhost:8000/predict`).

### Expected Python Request Payload
```json
{
  "title": "String",
  "text": "String",
  "model": "String (optional)"
}
```

### Expected Python Response Payload
```json
{
  "prediction": "FAKE | REAL",
  "confidence": 0.954,
  "model": "Passive Aggressive Classifier",
  "probabilities": {
    "FAKE": 0.954,
    "REAL": 0.046
  },
  "explanation": "Linguistic cues indicate subjective emotional markers."
}
```

The system predicts patterns learned from labeled datasets; it does not independently verify whether an article is factually true. If the ML service or experiment artifact is unavailable, the UI reports unavailable statistics instead of substituting invented benchmark values.
