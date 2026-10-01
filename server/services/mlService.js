import axios from 'axios';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * ML Microservice Client
 * Communicates with the external Python FastAPI machine learning prediction service.
 */

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';
const USE_MOCK_ML = process.env.USE_MOCK_ML === 'true' || false;

const mlClient = axios.create({
  baseURL: ML_SERVICE_URL,
  headers: {
    'Content-Type': 'application/json'
  },
  timeout: 15000 // 15 seconds timeout for multi-model inference
});

/**
 * Temporary mock prediction algorithm for development when Python service is not running.
 */
const mockPredict = (title, text) => {
  console.log('[ML Service] Running in MOCK mode (Python ML service bypassed).');

  const combined = (title + ' ' + text).toLowerCase();
  const sensationalTerms = [
    'shocking', 'secret bunker', 'alien', 'miracle cure', 'doctors hate', 'conspiracy',
    'mind control', 'nanobots', 'clones', 'whistleblower leaked', '5g waves', 'big pharma',
    'illuminati', 'hoax', 'time traveler', 'bizarre', 'unbelievable', 'you won\'t believe'
  ];

  let score = 0;
  sensationalTerms.forEach(term => {
    if (combined.includes(term)) score += 1.5;
  });

  const uppercaseChars = title.replace(/[^A-Z]/g, '').length;
  if (title.length > 0 && (uppercaseChars / title.length) > 0.4) {
    score += 2;
  }

  const isFake = score >= 1.5;
  const baseConfidence = 0.88 + Math.random() * 0.10;
  const confidence = Number(Math.min(0.995, Math.max(0.70, isFake ? baseConfidence + 0.02 : baseConfidence)).toFixed(3));

  return {
    prediction: isFake ? 'FAKE' : 'REAL',
    confidence,
    model: 'TF-IDF + Logistic Regression',
    explanation: isFake
      ? 'TF-IDF feature weights detected high density of sensational terminology, emotional polarization, and lack of institutional attribution.'
      : 'TF-IDF feature weights indicate balanced vocabulary, standard journalistic syntax, and objective reporting conventions.',
    isMockResult: true
  };
};

/**
 * Mock multi-model simulation response for development fallback
 */
const mockSimulation = (title, text) => {
  const single = mockPredict(title, text);
  const isFake = single.prediction === 'FAKE';
  const words = (title + ' ' + text).split(/\s+/).filter(Boolean);

  const models = [
    {
      id: 'bow_lr',
      name: 'BoW + Logistic Regression',
      description: 'Bag-of-Words word count vectors paired with L2-regularized Logistic Regression.',
      feature_type: 'bow',
      prediction: single.prediction,
      confidence: Number((single.confidence - 0.02).toFixed(4)),
      probabilities: {
        FAKE: isFake ? Number(single.confidence.toFixed(4)) : Number((1 - single.confidence).toFixed(4)),
        REAL: isFake ? Number((1 - single.confidence).toFixed(4)) : Number(single.confidence.toFixed(4))
      },
      test_metrics: { accuracy: 0.941, precision: 0.938, recall: 0.945, f1_score: 0.941, roc_auc: 0.981 },
      validation_weight: 0.165
    },
    {
      id: 'tfidf_lr',
      name: 'TF-IDF + Logistic Regression',
      description: 'Term Frequency-Inverse Document Frequency sublinear weighting with L2 Logistic Regression.',
      feature_type: 'tfidf',
      prediction: single.prediction,
      confidence: single.confidence,
      probabilities: {
        FAKE: isFake ? Number(single.confidence.toFixed(4)) : Number((1 - single.confidence).toFixed(4)),
        REAL: isFake ? Number((1 - single.confidence).toFixed(4)) : Number(single.confidence.toFixed(4))
      },
      test_metrics: { accuracy: 0.952, precision: 0.950, recall: 0.954, f1_score: 0.952, roc_auc: 0.989 },
      validation_weight: 0.170
    },
    {
      id: 'tfidf_nb',
      name: 'TF-IDF + Multinomial Naive Bayes',
      description: 'Probabilistic conditional frequency classifier based on Bayes theorem with Laplace smoothing.',
      feature_type: 'tfidf',
      prediction: single.prediction,
      confidence: Number((single.confidence - 0.05).toFixed(4)),
      probabilities: {
        FAKE: isFake ? Number((single.confidence - 0.05).toFixed(4)) : Number((1 - single.confidence + 0.05).toFixed(4)),
        REAL: isFake ? Number((1 - single.confidence + 0.05).toFixed(4)) : Number((single.confidence - 0.05).toFixed(4))
      },
      test_metrics: { accuracy: 0.908, precision: 0.902, recall: 0.915, f1_score: 0.908, roc_auc: 0.958 },
      validation_weight: 0.158
    },
    {
      id: 'tfidf_svm',
      name: 'TF-IDF + Linear SVM',
      description: 'Maximum-margin hyperplane classifier (LinearSVC) with 3-fold probability calibration.',
      feature_type: 'tfidf',
      prediction: single.prediction,
      confidence: Number((single.confidence + 0.01).toFixed(4)),
      probabilities: {
        FAKE: isFake ? Number((single.confidence + 0.01).toFixed(4)) : Number((1 - single.confidence - 0.01).toFixed(4)),
        REAL: isFake ? Number((1 - single.confidence - 0.01).toFixed(4)) : Number((single.confidence + 0.01).toFixed(4))
      },
      test_metrics: { accuracy: 0.956, precision: 0.958, recall: 0.954, f1_score: 0.956, roc_auc: 0.991 },
      validation_weight: 0.172
    },
    {
      id: 'tfidf_rf',
      name: 'TF-IDF + Random Forest',
      description: 'Ensemble of 100 decorrelated decision trees capturing non-linear feature interactions.',
      feature_type: 'tfidf',
      prediction: single.prediction,
      confidence: Number((single.confidence - 0.03).toFixed(4)),
      probabilities: {
        FAKE: isFake ? Number((single.confidence - 0.03).toFixed(4)) : Number((1 - single.confidence + 0.03).toFixed(4)),
        REAL: isFake ? Number((1 - single.confidence + 0.03).toFixed(4)) : Number((single.confidence - 0.03).toFixed(4))
      },
      test_metrics: { accuracy: 0.924, precision: 0.931, recall: 0.918, f1_score: 0.924, roc_auc: 0.968 },
      validation_weight: 0.162
    },
    {
      id: 'tfidf_pac',
      name: 'TF-IDF + Passive Aggressive Classifier',
      description: 'Online margin-based incremental learning algorithm with 3-fold probability calibration.',
      feature_type: 'tfidf',
      prediction: single.prediction,
      confidence: Number((single.confidence + 0.015).toFixed(4)),
      probabilities: {
        FAKE: isFake ? Number((single.confidence + 0.015).toFixed(4)) : Number((1 - single.confidence - 0.015).toFixed(4)),
        REAL: isFake ? Number((1 - single.confidence - 0.015).toFixed(4)) : Number((single.confidence + 0.015).toFixed(4))
      },
      test_metrics: { accuracy: 0.958, precision: 0.960, recall: 0.956, f1_score: 0.958, roc_auc: 0.992 },
      validation_weight: 0.173
    }
  ];

  const realVotes = isFake ? 0 : 6;
  const fakeVotes = 6 - realVotes;

  const candidate_models = models.map(m => ({
    model_id: m.id,
    model_name: m.name,
    vectorizer: m.feature_type === 'bow' ? 'Bag of Words (1,2-gram)' : 'TF-IDF (1,2-gram)',
    prediction: m.prediction,
    probabilities: m.probabilities,
    confidence: m.confidence,
    inference_time_ms: 1.5,
    val_weight: m.validation_weight,
    test_dataset_metrics: {
      accuracy: m.test_metrics.accuracy,
      precision: m.test_metrics.precision,
      recall: m.test_metrics.recall,
      f1: m.test_metrics.f1_score,
      roc_auc: m.test_metrics.roc_auc
    }
  }));

  const cleanedText = (title + ' ' + text).toLowerCase().replace(/[^a-z0-9\s]/g, ' ').replace(/\s+/g, ' ').trim();
  const wordCount = words.length;
  const charCount = (title + ' ' + text).length;
  const uniqueWords = new Set(words.map(w => w.toLowerCase())).size;
  const ttr = wordCount > 0 ? Number((uniqueWords / wordCount).toFixed(4)) : 0.82;
  const upperChars = title.replace(/[^A-Z]/g, '').length;
  const upperRatio = title.length > 0 ? Number((upperChars / title.length).toFixed(4)) : 0.0;

  const topSalient = words.slice(0, 10).map((w, idx) => ({
    token: w.toLowerCase().replace(/[^a-z]/g, ''),
    term: w.toLowerCase().replace(/[^a-z]/g, ''),
    tfidf_weight: Number((0.45 - idx * 0.035).toFixed(4)),
    class_association: isFake ? 'FAKE' : 'REAL'
  }));

  return {
    prediction_id: 'sim_' + Date.now().toString(36),
    article_analysis: {
      title: title.trim(),
      text: text.trim(),
      preprocessed_text: cleanedText,
      word_count: wordCount,
      char_count: charCount,
      lexical_diversity: ttr,
      stopword_ratio: 0.35,
      uppercase_ratio: upperRatio,
      exclamation_count: (title + ' ' + text).split('!').length - 1,
      question_count: (title + ' ' + text).split('?').length - 1,
      sentiment_heuristic: isFake ? 'Sensationalist / High Emotional Loading' : 'Neutral / Factual Journalistic Register'
    },
    vectorizers: {
      tfidf: {
        name: 'TF-IDF Vectorizer',
        ngram_range: [1, 2],
        max_features: 10000,
        non_zero_features: Math.min(120, Math.max(15, Math.floor(wordCount * 0.6))),
        density: 0.0084
      },
      bow: {
        name: 'Bag of Words (CountVectorizer)',
        ngram_range: [1, 2],
        max_features: 10000,
        non_zero_features: Math.min(135, Math.max(18, Math.floor(wordCount * 0.7))),
        density: 0.0091
      }
    },
    candidate_models: candidate_models,
    individual_models: models,
    ensembles: {
      weighted_soft_ensemble: {
        name: 'Validation-Weighted Soft Ensemble',
        formula: 'P(REAL) = Σ (w_i * P_i(REAL)) where w_i = val_f1_i / Σ val_f1_k',
        prediction: isFake ? 'FAKE' : 'REAL',
        confidence: single.confidence,
        probabilities: {
          REAL: isFake ? Number((1 - single.confidence).toFixed(4)) : Number(single.confidence.toFixed(4)),
          FAKE: isFake ? Number(single.confidence.toFixed(4)) : Number((1 - single.confidence).toFixed(4))
        },
        consensus_strength: 'Very Strong Consensus',
        test_dataset_metrics: { accuracy: 0.968, precision: 0.965, recall: 0.971, f1: 0.968, roc_auc: 0.996 }
      },
      majority_voting_hard_ensemble: {
        name: 'Majority Voting Hard Ensemble',
        formula: 'Verdict = mode(model_verdicts), Vote % = max(N_REAL, N_FAKE) / 6',
        prediction: isFake ? 'FAKE' : 'REAL',
        real_votes: realVotes,
        fake_votes: fakeVotes,
        total_models: 6,
        vote_percentage: 100.0,
        confidence: 1.0,
        test_dataset_metrics: { accuracy: 0.962, precision: 0.960, recall: 0.964, f1: 0.962, roc_auc: 0.993 }
      },
      majority_voting: {
        name: 'Majority Voting Ensemble',
        type: 'Hard Consensus (Democratic Vote)',
        prediction: isFake ? 'FAKE' : 'REAL',
        real_votes: realVotes,
        fake_votes: fakeVotes,
        total_models: 6,
        consensus_percentage: 100.0,
        probabilities: { REAL: isFake ? 0.0 : 1.0, FAKE: isFake ? 1.0 : 0.0 },
        confidence: 1.0,
        test_metrics: { accuracy: 0.962, f1_score: 0.962, roc_auc: 0.993 }
      },
      validation_weighted: {
        name: 'Validation-Weighted Ensemble',
        type: 'Soft Probabilistic Weighting (w_i = Val-F1_i)',
        prediction: isFake ? 'FAKE' : 'REAL',
        confidence: single.confidence,
        probabilities: {
          REAL: isFake ? Number((1 - single.confidence).toFixed(4)) : Number(single.confidence.toFixed(4)),
          FAKE: isFake ? Number(single.confidence.toFixed(4)) : Number((1 - single.confidence).toFixed(4))
        },
        test_metrics: { accuracy: 0.968, f1_score: 0.968, roc_auc: 0.996 }
      }
    },
    primary_verdict: {
      label: single.prediction,
      confidence: single.confidence,
      decision_source: 'Validation-Weighted Soft Ensemble',
      probabilities: {
        REAL: isFake ? Number((1 - single.confidence).toFixed(4)) : Number(single.confidence.toFixed(4)),
        FAKE: isFake ? Number(single.confidence.toFixed(4)) : Number((1 - single.confidence).toFixed(4))
      },
      explanation: single.explanation
    },
    salient_features: topSalient,
    top_salient_tokens: topSalient,
    input_summary: {
      raw_title: title,
      raw_text_preview: text.slice(0, 200) + '...',
      cleaned_text_preview: cleanedText.slice(0, 200) + '...',
      diagnostics: {
        char_count: charCount,
        word_count: wordCount,
        sentence_count: Math.max(1, text.split(/[.!?]+/).filter(Boolean).length),
        cleaned_word_count: Math.max(1, Math.floor(wordCount * 0.75)),
        unique_word_count: uniqueWords,
        stopword_count: Math.floor(wordCount * 0.35),
        stopword_ratio: 0.35,
        lexical_diversity_ttr: ttr,
        avg_word_length: 5.4,
        avg_sentence_length: Math.round(wordCount / Math.max(1, text.split(/[.!?]+/).filter(Boolean).length))
      }
    },
    metadata: {
      dataset: {
        total_samples: 1500,
        train_samples: 1050,
        val_samples: 225,
        test_samples: 225,
        train_ratio: 0.7,
        val_ratio: 0.15,
        test_ratio: 0.15,
        random_state: 42
      },
      champion_model: {
        id: 'tfidf_pac',
        name: 'TF-IDF + Passive Aggressive Classifier',
        val_f1: 0.958
      }
    }
  };
};

/**
 * Request single model prediction from Python ML Service
 */
export const predictWithML = async ({ title, text, model = null }) => {
  if (USE_MOCK_ML) {
    return mockPredict(title, text);
  }

  try {
    const response = await mlClient.post('/predict', {
      title,
      text,
      model
    });

    const data = response.data;
    if (!data || !data.prediction || typeof data.confidence !== 'number') {
      throw new Error('Invalid response structure received from Python ML service.');
    }

    const normalizedVerdict = String(data.prediction).toUpperCase();
    return {
      prediction: normalizedVerdict,
      confidence: Number(data.confidence),
      model: data.model || 'TF-IDF + Logistic Regression',
      explanation: data.explanation || '',
      probabilities: data.probabilities,
      topTokens: data.topTokens || [],
      wordCount: data.wordCount,
      charCount: data.charCount,
      isMockResult: false
    };
  } catch (error) {
    if (process.env.NODE_ENV !== 'production' && (error.code === 'ECONNREFUSED' || error.code === 'ETIMEDOUT')) {
      console.warn(`[ML Service Warning] Python ML Service at ${ML_SERVICE_URL} is unreachable. Falling back to local development mock.`);
      return mockPredict(title, text);
    }
    throw error;
  }
};

/**
 * Request full multi-model simulation from Python ML Service
 */
export const predictSimulationWithML = async ({ title, text }) => {
  if (USE_MOCK_ML) {
    return mockSimulation(title, text);
  }

  try {
    console.log(`[ML Service] Running multi-model simulation at ${ML_SERVICE_URL}/predict/all`);
    const response = await mlClient.post('/predict/all', {
      title,
      text
    });

    return response.data;
  } catch (error) {
    if (process.env.NODE_ENV !== 'production' && (error.code === 'ECONNREFUSED' || error.code === 'ETIMEDOUT')) {
      console.warn(`[ML Service Warning] Python ML Service unreachable. Falling back to mock simulation.`);
      return mockSimulation(title, text);
    }
    throw error;
  }
};

/**
 * Fetch experiment metadata from Python service or local disk
 */
export const getExperimentMetadataFromML = async () => {
  try {
    const response = await mlClient.get('/experiment-metadata');
    return response.data;
  } catch (error) {
    // Attempt local read from ../ml/models/model_config.json
    try {
      const localPath = path.resolve(__dirname, '../../ml/models/model_config.json');
      if (fs.existsSync(localPath)) {
        const raw = fs.readFileSync(localPath, 'utf8');
        return JSON.parse(raw);
      }
    } catch (fsErr) {
      console.warn('[ML Service] Could not read local model_config.json:', fsErr.message);
    }
    return null;
  }
};

export default {
  predictWithML,
  predictSimulationWithML,
  getExperimentMetadataFromML
};
