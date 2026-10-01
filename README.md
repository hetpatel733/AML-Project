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
│             Python FastAPI / Flask ML Service          │
│   (TF-IDF Vectorizer + Passive Aggressive Classifier)  │
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
│   │   ├── services/api.js     # Axios API Client with Resilient Fallback Layer
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
│   │   └── analyticsController.js  # Aggregation, Metrics, Distribution, Model Benchmarks
│   ├── middleware/
│   │   ├── errorMiddleware.js      # Centralized Error & 404 Handlers
│   │   └── validationMiddleware.js # Input & Query Sanitization and Schema Validation
│   ├── models/
│   │   └── Prediction.js       # MongoDB Schema (title, text, prediction, confidence, explanation)
│   ├── routes/
│   │   ├── predictionRoutes.js # /api/predictions & /api/predictions/stats
│   │   └── analyticsRoutes.js  # /api/analytics & /api/analytics/performance
│   ├── services/
│   │   └── mlService.js        # Axios Client for Python ML Microservice (with dev fallback)
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
- **Python**: 3.9+ (Optional, only needed when running the future Python ML model microservice)

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
| `GET` | `/analytics` | Aggregate stats, daily volume trends, confidence spread, confusion matrix |
| `GET` | `/model-performance` | Evaluation benchmark metrics across NLP classifiers |

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

*(If the Python service is offline during testing, Express uses a heuristic NLP analysis fallback seamlessly).*
