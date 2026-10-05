import axios from 'axios';
import { normalizeSimulationData } from '../utils/helpers';

/**
 * Axios API client instance with centralized configuration
 */
const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json'
  },
  timeout: 15000
});

// Storage key for mock database in browser local storage
const LOCAL_STORAGE_KEY = 'fake_news_predictions_history';

/**
 * Default Seed Mock Predictions for academic demonstration
 */
const DEFAULT_SEED_DATA = [
  {
    id: 'pred_101',
    title: 'Astronomers Detect Coherent Radio Signal from Gliese 667Cc Star System',
    text: 'A peer-reviewed astrophysics paper from the European Southern Observatory indicates faint narrowband emissions discovered in archival ALMA telescope telemetry data.',
    prediction: 'REAL',
    confidence: 0.942,
    model: 'Logistic Regression (TF-IDF)',
    explanation: 'Article structure exhibits neutral journalistic tone, verified scientific attribution, and standard citation patterns.',
    createdAt: new Date(Date.now() - 1000 * 60 * 35).toISOString()
  },
  {
    id: 'pred_102',
    title: 'SHOCKING: Secret Underground Lab Releases Nanobots in Tap Water!',
    text: 'Viral leaked documents claim military operatives deployed smart microscopic chips to alter human DNA via suburban municipal water supply reservoirs.',
    prediction: 'FAKE',
    confidence: 0.985,
    model: 'Passive Aggressive Classifier',
    explanation: 'High density of sensationalist superlatives, unverified conspiratorial claims, and absence of credible institutional sources.',
    createdAt: new Date(Date.now() - 1000 * 60 * 180).toISOString()
  },
  {
    id: 'pred_103',
    title: 'Federal Reserve Lowers Key Benchmark Lending Rate by 25 Basis Points',
    text: 'The Federal Open Market Committee announced a quarter-point rate reduction following steady improvements in core consumer price index data.',
    prediction: 'REAL',
    confidence: 0.961,
    model: 'Multinomial Naive Bayes',
    explanation: 'Linguistic tokens strongly correlate with standard economic press releases and institutional financial reporting.',
    createdAt: new Date(Date.now() - 1000 * 60 * 60 * 6).toISOString()
  },
  {
    id: 'pred_104',
    title: 'Drinking Boiling Saltwater Completely Cures All Respiratory Illnesses in 2 Hours',
    text: 'Doctors hate this one ancient miracle trick discovered by an anonymous monk in the mountains that big pharma doesn’t want you to know.',
    prediction: 'FAKE',
    confidence: 0.991,
    model: 'Support Vector Machine (SVM)',
    explanation: 'Identified clickbait trigger phrases, exaggerated medical efficacy claims, and lack of clinical trial corroboration.',
    createdAt: new Date(Date.now() - 1000 * 60 * 60 * 14).toISOString()
  },
  {
    id: 'pred_105',
    title: 'International Renewable Energy Agency Reports 22% Surge in Solar Capacity',
    text: 'Grid modernization investments across the EU and North America accelerated solar photovoltaic deployment past initial quarterly forecasts.',
    prediction: 'REAL',
    confidence: 0.918,
    model: 'Logistic Regression (TF-IDF)',
    explanation: 'Terminology aligns with energy sector publications with factual statistical representations.',
    createdAt: new Date(Date.now() - 1000 * 60 * 60 * 24).toISOString()
  },
  {
    id: 'pred_106',
    title: 'Celebrity Found to be Robotic Clone Controlled by AI Satellite',
    text: 'Social media users spot strange facial glitches during live television broadcast, claiming original actor was replaced in 2021.',
    prediction: 'FAKE',
    confidence: 0.978,
    model: 'Passive Aggressive Classifier',
    explanation: 'Wild ungrounded speculation typical of disinformation campaigns and clickbait entertainment blogs.',
    createdAt: new Date(Date.now() - 1000 * 60 * 60 * 36).toISOString()
  },
  {
    id: 'pred_107',
    title: 'Global Semiconductor Alliance Finalizes Next-Gen 1.4nm Lithography Standard',
    text: 'Industry leaders from TSMC, ASML, and Intel established formal testing protocols for high-NA EUV equipment manufacturing.',
    prediction: 'REAL',
    confidence: 0.953,
    model: 'Multinomial Naive Bayes',
    explanation: 'Technical vocabulary and accurate corporate entity references denote reliable trade reporting.',
    createdAt: new Date(Date.now() - 1000 * 60 * 60 * 48).toISOString()
  },
  {
    id: 'pred_108',
    title: 'Leaked Video Proves Time Traveler Visited 1928 with Smart Watch',
    text: 'Bizarre vintage footage uploaded to forum shows mysterious bystander holding what appears to be a digital wrist display device.',
    prediction: 'FAKE',
    confidence: 0.966,
    model: 'Logistic Regression (TF-IDF)',
    explanation: 'High concentration of tabloid buzzwords and speculative historical anomalies.',
    createdAt: new Date(Date.now() - 1000 * 60 * 60 * 60).toISOString()
  }
];

/**
 * Helper to enrich any prediction object with complete 6-model simulation and ensemble voting results
 */
export const enrichPredictionWithSimulation = (item) => {
  if (!item) return null;
  if (item.simulationResults && item.simulationResults.candidate_models && item.simulationResults.ensembles) {
    return normalizeSimulationData(item.simulationResults, item.title, item.text);
  }
  const generatedSim = simulateFullExperimentalPipeline(item.title || '', item.text || item.title || '');
  return normalizeSimulationData(generatedSim, item.title, item.text);
};

/**
 * Initialize local storage mock database if not already present
 */
const getLocalHistory = () => {
  try {
    const saved = localStorage.getItem(LOCAL_STORAGE_KEY);
    if (saved) {
      return JSON.parse(saved);
    }
  } catch (e) {
    console.warn('Could not read from localStorage', e);
  }
  return DEFAULT_SEED_DATA;
};

const saveLocalHistory = (data) => {
  try {
    localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(data));
  } catch (e) {
    console.warn('Could not save to localStorage', e);
  }
};

/**
 * Academic ML Model Benchmark Statistics for comparison
 */
export const MODEL_PERFORMANCE_DATA = [];

/**
 * Mock Prediction Generator with NLP heuristic simulation
 */
const simulateNlpPrediction = (title, text) => {
  const combined = (title + ' ' + text).toLowerCase();
  
  const fakeKeywords = [
    'shocking', 'secret bunker', 'alien', 'miracle cure', 'doctors hate', 'conspiracy',
    'mind control', 'nanobots', 'clones', 'whistleblower leaked', '5g waves', 'big pharma',
    'illuminati', 'hoax', 'time traveler', 'bizarre', 'unbelievable', 'you won\'t believe'
  ];

  let fakeScore = 0;
  fakeKeywords.forEach(keyword => {
    if (combined.includes(keyword)) fakeScore += 1.5;
  });

  // Check uppercase title ratio
  const uppercaseChars = title.replace(/[^A-Z]/g, '').length;
  if (title.length > 0 && (uppercaseChars / title.length) > 0.4) {
    fakeScore += 2;
  }

  // Check exclamation marks
  const exclamationCount = (combined.match(/!/g) || []).length;
  if (exclamationCount >= 2) fakeScore += 1.5;

  const isFake = fakeScore >= 1.5;
  const baseConfidence = 0.82 + Math.random() * 0.16;
  const confidence = Number(Math.min(0.994, Math.max(0.72, isFake ? baseConfidence + 0.05 : baseConfidence)).toFixed(3));

  const models = [
    'Logistic Regression (TF-IDF)',
    'Passive Aggressive Classifier',
    'Multinomial Naive Bayes',
    'Linear SVM'
  ];
  const model = models[Math.floor(Math.random() * models.length)];

  const explanation = isFake
    ? 'NLP feature extraction identified sensationalized syntax, emotional appeal heuristics, and patterns common in unverified text.'
    : 'NLP feature extraction indicated factual sentence structure, objective terminology, and standard journalistic register.';

  return {
    id: 'pred_' + Date.now().toString(36) + Math.random().toString(36).substring(2, 6),
    title: title.trim(),
    text: text.trim(),
    prediction: isFake ? 'FAKE' : 'REAL',
    confidence,
    model,
    explanation,
    createdAt: new Date().toISOString()
  };
};

/**
 * 1. Predict News Article
 * Sends POST /api/predictions with { title, text }
 * Falls back to mock NLP classifier if backend is offline.
 */
export const predictNews = async (newsData) => {
  try {
    const response = await apiClient.post('/predictions', newsData);
    const result = response.data?.data ?? response.data;
    
    // Also store in local history for offline/persistence continuity
    try {
      const history = getLocalHistory();
      const enriched = {
        ...result,
        id: result.id || result.predictionId || ('pred_' + Date.now()),
        title: newsData.title,
        text: newsData.text,
        createdAt: result.createdAt || new Date().toISOString(),
        simulationResults: result.simulationResults || enrichPredictionWithSimulation(result)
      };
      saveLocalHistory([enriched, ...history.filter(h => h.id !== enriched.id)]);
    } catch (e) {
      console.warn('Could not cache prediction to local history', e);
    }

    return { data: result, isMock: false };
  } catch (error) {
    console.info('Backend unreachable or returned error, utilizing client mock fallback:', error.message);
    
    // Simulate brief network delay for realistic UI feedback
    await new Promise(resolve => setTimeout(resolve, 600));

    const mockResult = simulateNlpPrediction(newsData.title, newsData.text);
    const mockSim = simulateFullExperimentalPipeline(newsData.title, newsData.text);
    const enrichedItem = {
      ...mockResult,
      simulationResults: mockSim
    };
    
    // Store in local history
    const history = getLocalHistory();
    const updatedHistory = [enrichedItem, ...history];
    saveLocalHistory(updatedHistory);

    return { data: enrichedItem, isMock: true };
  }
};

/**
 * 2. Get Prediction History with filtering, searching, sorting & pagination
 * Sends GET /api/predictions
 */
export const getPredictionHistory = async (params = {}) => {
  try {
    const response = await apiClient.get('/predictions', { params });
    const result = response.data?.data ?? response.data;
    if (result && Array.isArray(result.predictions)) {
      result.predictions = result.predictions.map(item => ({
        ...item,
        simulationResults: item.simulationResults ? normalizeSimulationData(item.simulationResults, item.title, item.text) : enrichPredictionWithSimulation(item)
      }));
    }
    return { data: result, isMock: false };
  } catch (error) {
    console.info('Using mock prediction history fallback:', error.message);
    
    let list = getLocalHistory();
    const { search = '', filter = 'ALL', consensusFilter = 'ALL', sortBy = 'date_desc', page = 1, limit = 10 } = params;

    // Filter by text search
    if (search.trim()) {
      const q = search.toLowerCase();
      list = list.filter(item => 
        (item.title && item.title.toLowerCase().includes(q)) || 
        (item.text && item.text.toLowerCase().includes(q)) ||
        (item.model && item.model.toLowerCase().includes(q))
      );
    }

    // Filter by status (FAKE / REAL / ALL)
    if (filter !== 'ALL') {
      list = list.filter(item => item.prediction.toUpperCase() === filter.toUpperCase());
    }

    // Filter by consensus if requested
    if (consensusFilter && consensusFilter !== 'ALL') {
      list = list.filter(item => {
        const sim = item.simulationResults ? normalizeSimulationData(item.simulationResults, item.title, item.text) : enrichPredictionWithSimulation(item);
        const hard = sim?.ensembles?.majority_voting_hard_ensemble;
        const maxVotes = Math.max(hard?.real_votes ?? 6, hard?.fake_votes ?? 0);
        if (consensusFilter === 'UNANIMOUS') return maxVotes === 6;
        if (consensusFilter === 'SPLIT') return maxVotes < 6;
        return true;
      });
    }

    // Sort
    list.sort((a, b) => {
      const dateA = new Date(a.createdAt).getTime();
      const dateB = new Date(b.createdAt).getTime();
      if (sortBy === 'date_asc') return dateA - dateB;
      if (sortBy === 'confidence_desc') return b.confidence - a.confidence;
      if (sortBy === 'confidence_asc') return a.confidence - b.confidence;
      return dateB - dateA; // default date_desc
    });

    const total = list.length;
    const startIndex = (page - 1) * limit;
    const paginatedItems = list.slice(startIndex, startIndex + limit).map(item => ({
      ...item,
      simulationResults: item.simulationResults ? normalizeSimulationData(item.simulationResults, item.title, item.text) : enrichPredictionWithSimulation(item)
    }));

    return {
      data: {
        predictions: paginatedItems,
        total,
        page,
        totalPages: Math.ceil(total / limit) || 1
      },
      isMock: true
    };
  }
};

/**
 * 3. Get Prediction Aggregate Statistics
 * Sends GET /api/predictions/stats
 */
export const getPredictionStats = async () => {
  try {
    const response = await apiClient.get('/predictions/stats');
    const result = response.data?.data ?? response.data;
    return { data: result, isMock: false };
  } catch (error) {
    const history = getLocalHistory();
    const totalPredictions = history.length;
    const fakeCount = history.filter(h => h.prediction === 'FAKE').length;
    const realCount = history.filter(h => h.prediction === 'REAL').length;
    
    const avgConfidence = totalPredictions > 0
      ? (history.reduce((acc, curr) => acc + curr.confidence, 0) / totalPredictions)
      : 0;

    return {
      data: {
        totalPredictions,
        fakeCount,
        realCount,
        avgConfidence: Number(avgConfidence.toFixed(3)),
        fakePercentage: totalPredictions > 0 ? Number(((fakeCount / totalPredictions) * 100).toFixed(1)) : 0,
        realPercentage: totalPredictions > 0 ? Number(((realCount / totalPredictions) * 100).toFixed(1)) : 0,
        recentPredictions: history.slice(0, 5)
      },
      isMock: true
    };
  }
};

/**
 * 4. Get Detailed Analytics (Trend, Confidence distribution, Confusion Matrix)
 * Sends GET /api/analytics
 */
export const getAnalytics = async () => {
  try {
    const response = await apiClient.get('/analytics');
    const result = response.data?.data ?? response.data;
    return { data: result, isMock: false };
  } catch (error) {
    const history = getLocalHistory();

    // Generate daily trends for the last 7 days
    const days = 7;
    const trendData = [];
    const now = new Date();

    for (let i = days - 1; i >= 0; i--) {
      const d = new Date(now);
      d.setDate(d.getDate() - i);
      const dateStr = d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
      
      // Filter predictions from this day
      const dayMatches = history.filter(item => {
        const itemDate = new Date(item.createdAt);
        return itemDate.toDateString() === d.toDateString();
      });

      const dayFake = dayMatches.filter(m => m.prediction === 'FAKE').length;
      const dayReal = dayMatches.filter(m => m.prediction === 'REAL').length;

      trendData.push({
        date: dateStr,
        fake: dayFake,
        real: dayReal,
        total: dayFake + dayReal
      });
    }

    return {
      data: {
        trendData,
        confidenceDistribution: [],
        confusionMatrix: null,
        modelPerformance: MODEL_PERFORMANCE_DATA
      },
      isMock: true
    };
  }
};

/**
 * 5. Get Model Performance Comparison Data
 * Sends GET /api/models/performance
 */
export const getModelPerformance = async () => {
  try {
    const response = await apiClient.get('/models/performance');
    const result = response.data?.data ?? response.data;
    return { data: result, isMock: false };
  } catch (error) {
    return {
      data: MODEL_PERFORMANCE_DATA,
      isMock: true
    };
  }
};

/**
 * Delete a prediction from history (supports local mock or backend)
 */
/**
 * Delete a prediction from history (supports local mock or backend)
 */
export const deletePrediction = async (id) => {
  try {
    await apiClient.delete(`/predictions/${id}`);
    return { success: true, isMock: false };
  } catch (error) {
    const history = getLocalHistory();
    const updated = history.filter(item => item.id !== id);
    saveLocalHistory(updated);
    return { success: true, isMock: true };
  }
};

/**
 * Mock Simulation Generator for Offline / Fallback operation
 */
const simulateFullExperimentalPipeline = (title, text) => {
  const combined = (title + ' ' + text).toLowerCase();
  
  const fakeKeywords = [
    'shocking', 'secret bunker', 'alien', 'miracle cure', 'doctors hate', 'conspiracy',
    'mind control', 'nanobots', 'clones', 'whistleblower leaked', '5g waves', 'big pharma',
    'illuminati', 'hoax', 'time traveler', 'bizarre', 'unbelievable', 'you won\'t believe',
    'coverup', 'apocalypse', 'microchip', 'deep state'
  ];

  let fakeScore = 0;
  const matchedTokens = [];
  fakeKeywords.forEach(kw => {
    if (combined.includes(kw)) {
      fakeScore += 1.8;
      matchedTokens.push(kw);
    }
  });

  const upperChars = title.replace(/[^A-Z]/g, '').length;
  const upperRatio = title.length > 0 ? (upperChars / title.length) : 0;
  if (upperRatio > 0.35) fakeScore += 2.0;

  const exclamations = (combined.match(/!/g) || []).length;
  if (exclamations >= 2) fakeScore += 1.5;

  const isFake = fakeScore >= 1.5;
  const words = (title + ' ' + text).trim().split(/\s+/).filter(Boolean);
  const wordCount = words.length;
  const charCount = (title + ' ' + text).length;
  const uniqueWords = new Set(words.map(w => w.toLowerCase()));
  const lexicalDiversity = wordCount > 0 ? Number((uniqueWords.size / wordCount).toFixed(3)) : 0.65;
  const stopwordRatio = 0.38 + Math.random() * 0.08;

  const testMetrics = {
    pac_tfidf: { accuracy: 0.993, precision: 0.992, recall: 0.994, f1: 0.993, roc_auc: 0.999 },
    pac_bow:   { accuracy: 0.988, precision: 0.986, recall: 0.990, f1: 0.988, roc_auc: 0.997 },
    lr_tfidf:  { accuracy: 0.985, precision: 0.984, recall: 0.986, f1: 0.985, roc_auc: 0.998 },
    lr_bow:    { accuracy: 0.987, precision: 0.986, recall: 0.988, f1: 0.987, roc_auc: 0.998 },
    mnb_tfidf: { accuracy: 0.941, precision: 0.938, recall: 0.945, f1: 0.941, roc_auc: 0.982 },
    mnb_bow:   { accuracy: 0.957, precision: 0.952, recall: 0.963, f1: 0.957, roc_auc: 0.989 }
  };

  const weights = {
    pac_tfidf: 0.176,
    pac_bow:   0.175,
    lr_tfidf:  0.174,
    lr_bow:    0.175,
    mnb_tfidf: 0.147,
    mnb_bow:   0.153
  };

  const candidateModels = [
    {
      model_id: 'pac_tfidf',
      model_name: 'Passive Aggressive (TF-IDF)',
      vectorizer: 'TF-IDF (1,2-gram)',
      is_fake: isFake,
      conf: isFake ? 0.97 + Math.random() * 0.025 : 0.96 + Math.random() * 0.035,
      inference_time_ms: 2.1
    },
    {
      model_id: 'pac_bow',
      model_name: 'Passive Aggressive (BoW)',
      vectorizer: 'Bag-of-Words (1,2-gram)',
      is_fake: isFake,
      conf: isFake ? 0.95 + Math.random() * 0.035 : 0.94 + Math.random() * 0.04,
      inference_time_ms: 1.8
    },
    {
      model_id: 'lr_tfidf',
      model_name: 'Logistic Regression (TF-IDF)',
      vectorizer: 'TF-IDF (1,2-gram)',
      is_fake: isFake,
      conf: isFake ? 0.94 + Math.random() * 0.04 : 0.95 + Math.random() * 0.03,
      inference_time_ms: 1.2
    },
    {
      model_id: 'lr_bow',
      model_name: 'Logistic Regression (BoW)',
      vectorizer: 'Bag-of-Words (1,2-gram)',
      is_fake: isFake,
      conf: isFake ? 0.93 + Math.random() * 0.045 : 0.94 + Math.random() * 0.04,
      inference_time_ms: 1.4
    },
    {
      model_id: 'mnb_tfidf',
      model_name: 'Multinomial Naive Bayes (TF-IDF)',
      vectorizer: 'TF-IDF (1,2-gram)',
      is_fake: isFake ? (fakeScore > 3.0 ? true : Math.random() > 0.25) : false,
      conf: 0.88 + Math.random() * 0.08,
      inference_time_ms: 0.8
    },
    {
      model_id: 'mnb_bow',
      model_name: 'Multinomial Naive Bayes (BoW)',
      vectorizer: 'Bag-of-Words (1,2-gram)',
      is_fake: isFake,
      conf: 0.91 + Math.random() * 0.06,
      inference_time_ms: 0.9
    }
  ].map(m => {
    const fakeProb = m.is_fake ? m.conf : (1.0 - m.conf);
    const realProb = 1.0 - fakeProb;
    const pred = fakeProb >= 0.5 ? 'FAKE' : 'REAL';
    const conf = pred === 'FAKE' ? fakeProb : realProb;
    return {
      model_id: m.model_id,
      model_name: m.model_name,
      vectorizer: m.vectorizer,
      prediction: pred,
      probabilities: {
        FAKE: Number(fakeProb.toFixed(4)),
        REAL: Number(realProb.toFixed(4))
      },
      confidence: Number(conf.toFixed(4)),
      inference_time_ms: m.inference_time_ms,
      val_weight: weights[m.model_id],
      test_dataset_metrics: testMetrics[m.model_id]
    };
  });

  let totalW = 0;
  let softReal = 0;
  let fakeVotes = 0;
  let realVotes = 0;

  candidateModels.forEach(m => {
    softReal += m.val_weight * m.probabilities.REAL;
    totalW += m.val_weight;
    if (m.prediction === 'FAKE') fakeVotes++;
    else realVotes++;
  });

  const softRealProb = totalW > 0 ? (softReal / totalW) : 0.5;
  const softFakeProb = 1.0 - softRealProb;
  const softPred = softFakeProb >= 0.5 ? 'FAKE' : 'REAL';
  const softConf = softPred === 'FAKE' ? softFakeProb : softRealProb;

  const hardPred = fakeVotes >= realVotes ? 'FAKE' : 'REAL';
  const hardVotePct = Number(((Math.max(fakeVotes, realVotes) / 6) * 100).toFixed(1));

  const sampleSalient = isFake ? [
    { term: 'shocking', tfidf_weight: 0.428, class_association: 'FAKE' },
    { term: 'unbelievable', tfidf_weight: 0.381, class_association: 'FAKE' },
    { term: 'conspiracy', tfidf_weight: 0.354, class_association: 'FAKE' },
    { term: 'whistleblower', tfidf_weight: 0.312, class_association: 'FAKE' },
    { term: 'secret', tfidf_weight: 0.285, class_association: 'FAKE' },
    { term: 'leaked', tfidf_weight: 0.248, class_association: 'FAKE' }
  ] : [
    { term: 'reuters', tfidf_weight: 0.412, class_association: 'REAL' },
    { term: 'spokesman', tfidf_weight: 0.375, class_association: 'REAL' },
    { term: 'officials', tfidf_weight: 0.341, class_association: 'REAL' },
    { term: 'statement', tfidf_weight: 0.298, class_association: 'REAL' },
    { term: 'department', tfidf_weight: 0.264, class_association: 'REAL' },
    { term: 'reported', tfidf_weight: 0.231, class_association: 'REAL' }
  ];

  return {
    prediction_id: 'sim_' + Date.now().toString(36) + Math.random().toString(36).substring(2, 6),
    article_analysis: {
      title: title.trim(),
      text: text.trim(),
      preprocessed_text: (title + ' ' + text).toLowerCase().replace(/[^a-z0-9\s]/g, ' ').replace(/\s+/g, ' ').trim(),
      word_count: wordCount,
      char_count: charCount,
      lexical_diversity: lexicalDiversity,
      stopword_ratio: Number(stopwordRatio.toFixed(3)),
      uppercase_ratio: Number(upperRatio.toFixed(3)),
      exclamation_count: exclamations,
      question_count: (combined.match(/\?/g) || []).length,
      sentiment_heuristic: isFake ? 'Sensationalist / High Emotional Loading' : 'Neutral / Factual Journalistic Register'
    },
    vectorizers: {
      tfidf: {
        name: 'TF-IDF (Term Frequency - Inverse Document Frequency)',
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
    candidate_models: candidateModels,
    ensembles: {
      weighted_soft_ensemble: {
        name: 'Validation-Weighted Soft Ensemble',
        formula: 'P(REAL) = Σ (w_i * P_i(REAL)) where w_i = val_f1_i / Σ val_f1_k',
        prediction: softPred,
        probabilities: {
          FAKE: Number(softFakeProb.toFixed(4)),
          REAL: Number(softRealProb.toFixed(4))
        },
        confidence: Number(softConf.toFixed(4)),
        consensus_strength: Math.abs(fakeVotes - realVotes) >= 4 ? 'Very Strong Consensus' : 'Moderate Split',
        test_dataset_metrics: { accuracy: 0.991, precision: 0.992, recall: 0.990, f1: 0.991, roc_auc: 0.999 }
      },
      majority_voting_hard_ensemble: {
        name: 'Majority Voting Hard Ensemble',
        formula: 'Verdict = mode(model_verdicts), Vote % = max(N_REAL, N_FAKE) / 6',
        prediction: hardPred,
        fake_votes: fakeVotes,
        real_votes: realVotes,
        vote_percentage: hardVotePct,
        confidence: Number((Math.max(fakeVotes, realVotes) / 6).toFixed(4)),
        test_dataset_metrics: { accuracy: 0.989, precision: 0.988, recall: 0.990, f1: 0.989, roc_auc: 0.998 }
      }
    },
    salient_features: sampleSalient
  };
};

/**
 * 6. Run Comprehensive 6-Model Simulation & Ensemble Inference
 * Sends POST /api/predictions/simulate
 */
export const runSimulation = async (newsData) => {
  try {
    const response = await apiClient.post('/predictions/simulate', newsData);
    const result = response.data?.data ?? response.data;
    const normalized = normalizeSimulationData(result, newsData.title, newsData.text);

    // Persist to local history so that it shows up immediately in the Prediction Archive
    try {
      const primary = normalized.primary_verdict || {};
      const historyItem = {
        id: normalized.prediction_id || ('sim_' + Date.now().toString(36)),
        title: newsData.title || normalized.article_analysis?.title || 'Simulation Run',
        text: newsData.text || normalized.article_analysis?.text || '',
        prediction: primary.label || 'FAKE',
        confidence: primary.confidence || 0.95,
        model: primary.decision_source || 'Validation-Weighted Soft Ensemble',
        explanation: primary.explanation || 'Analyzed via 6 candidate models & dual voting ensembles.',
        createdAt: new Date().toISOString(),
        simulationResults: normalized
      };
      const history = getLocalHistory();
      saveLocalHistory([historyItem, ...history.filter(h => h.id !== historyItem.id)]);
    } catch (e) {
      console.warn('Could not cache simulation to local history:', e);
    }

    return { data: result, isMock: false };
  } catch (error) {
    console.info('Using simulation client fallback:', error.message);
    await new Promise(resolve => setTimeout(resolve, 800));
    const mockSim = simulateFullExperimentalPipeline(newsData.title, newsData.text);
    const normalized = normalizeSimulationData(mockSim, newsData.title, newsData.text);

    // Persist fallback to local history
    try {
      const primary = normalized.primary_verdict || {};
      const historyItem = {
        id: normalized.prediction_id,
        title: newsData.title.trim(),
        text: newsData.text.trim(),
        prediction: primary.label || 'FAKE',
        confidence: primary.confidence || 0.95,
        model: 'Validation-Weighted Soft Ensemble',
        explanation: primary.explanation,
        createdAt: new Date().toISOString(),
        simulationResults: normalized
      };
      const history = getLocalHistory();
      saveLocalHistory([historyItem, ...history]);
    } catch (e) {
      console.warn('Could not cache mock simulation to local history:', e);
    }

    return { data: mockSim, isMock: true };
  }
};

/**
 * 7. Submit Feedback / Correction on a Model Prediction
 * Sends POST /api/predictions/feedback
 */
export const submitPredictionFeedback = async (feedbackData) => {
  try {
    const response = await apiClient.post('/predictions/feedback', feedbackData);
    const result = response.data?.data ?? response.data;
    return { data: result, isMock: false };
  } catch (error) {
    console.info('Using feedback mock response:', error.message);
    return {
      data: {
        feedbackId: 'fb_' + Date.now(),
        message: 'Feedback received and recorded for model calibration evaluation.'
      },
      isMock: true
    };
  }
};

/**
 * 8. Retrieve Academic Experiment Metadata & CV Benchmarks
 * Sends GET /api/predictions/experiments/metadata
 */
export const getExperimentMetadata = async () => {
  try {
    const response = await apiClient.get('/predictions/experiments/metadata');
    const result = response.data?.data ?? response.data;
    return { data: result, isMock: false };
  } catch (error) {
    console.info('Using experiment metadata fallback:', error.message);
    return {
      data: {
        split_ratios: { train: 0.7, val: 0.15, test: 0.15 },
        split_counts: { train: 31428, val: 6735, test: 6735, total: 44898 },
        cv_folds: 5,
        vectorizers: {
          tfidf: { type: 'TfidfVectorizer', ngram_range: [1, 2], max_features: 10000, sublinear_tf: true },
          bow: { type: 'CountVectorizer', ngram_range: [1, 2], max_features: 10000 }
        },
        validation_weights: {
          pac_tfidf: 0.176,
          pac_bow: 0.175,
          lr_tfidf: 0.174,
          lr_bow: 0.175,
          mnb_tfidf: 0.147,
          mnb_bow: 0.153
        },
        model_metrics: {
          pac_tfidf: {
            model_name: 'Passive Aggressive (TF-IDF)',
            cv_metrics: { accuracy_mean: 0.993, accuracy_std: 0.001, f1_mean: 0.993, f1_std: 0.001, roc_auc_mean: 0.999, roc_auc_std: 0.0003 },
            test_metrics: { accuracy: 0.993, precision: 0.992, recall: 0.994, f1: 0.993, roc_auc: 0.999 }
          },
          pac_bow: {
            model_name: 'Passive Aggressive (BoW)',
            cv_metrics: { accuracy_mean: 0.988, accuracy_std: 0.0015, f1_mean: 0.988, f1_std: 0.0014, roc_auc_mean: 0.997, roc_auc_std: 0.0006 },
            test_metrics: { accuracy: 0.988, precision: 0.986, recall: 0.990, f1: 0.988, roc_auc: 0.997 }
          },
          lr_tfidf: {
            model_name: 'Logistic Regression (TF-IDF)',
            cv_metrics: { accuracy_mean: 0.985, accuracy_std: 0.0012, f1_mean: 0.985, f1_std: 0.0012, roc_auc_mean: 0.998, roc_auc_std: 0.0004 },
            test_metrics: { accuracy: 0.985, precision: 0.984, recall: 0.986, f1: 0.985, roc_auc: 0.998 }
          },
          lr_bow: {
            model_name: 'Logistic Regression (BoW)',
            cv_metrics: { accuracy_mean: 0.987, accuracy_std: 0.0011, f1_mean: 0.987, f1_std: 0.001, roc_auc_mean: 0.998, roc_auc_std: 0.0004 },
            test_metrics: { accuracy: 0.987, precision: 0.986, recall: 0.988, f1: 0.987, roc_auc: 0.998 }
          },
          mnb_tfidf: {
            model_name: 'Multinomial Naive Bayes (TF-IDF)',
            cv_metrics: { accuracy_mean: 0.941, accuracy_std: 0.0025, f1_mean: 0.941, f1_std: 0.0024, roc_auc_mean: 0.982, roc_auc_std: 0.0015 },
            test_metrics: { accuracy: 0.941, precision: 0.938, recall: 0.945, f1: 0.941, roc_auc: 0.982 }
          },
          mnb_bow: {
            model_name: 'Multinomial Naive Bayes (BoW)',
            cv_metrics: { accuracy_mean: 0.957, accuracy_std: 0.002, f1_mean: 0.957, f1_std: 0.0019, roc_auc_mean: 0.989, roc_auc_std: 0.0012 },
            test_metrics: { accuracy: 0.957, precision: 0.952, recall: 0.963, f1: 0.957, roc_auc: 0.989 }
          }
        },
        ensembles: {
          weighted_soft_ensemble: {
            name: 'Validation-Weighted Soft Ensemble',
            formula: 'P(REAL) = Σ (w_i * P_i(REAL)) where w_i = val_f1_i / Σ val_f1_k',
            test_metrics: { accuracy: 0.991, precision: 0.992, recall: 0.990, f1: 0.991, roc_auc: 0.999 }
          },
          majority_voting_hard_ensemble: {
            name: 'Majority Voting Hard Ensemble',
            formula: 'Verdict = mode(model_verdicts), Vote % = max(N_REAL, N_FAKE) / 6',
            test_metrics: { accuracy: 0.989, precision: 0.988, recall: 0.990, f1: 0.989, roc_auc: 0.998 }
          }
        }
      },
      isMock: true
    };
  }
};
