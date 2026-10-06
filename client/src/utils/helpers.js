/**
 * Helper utility functions for Fake News Detection Frontend
 */

/**
 * Format date to human-readable string
 * @param {string|Date} date 
 * @returns {string} Formatted date (e.g., "Oct 24, 2025, 02:45 PM")
 */
export const formatDate = (date) => {
  if (!date) return 'N/A';
  const d = new Date(date);
  if (isNaN(d.getTime())) return 'Invalid Date';
  
  return new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  }).format(d);
};

/**
 * Format relative time (e.g. "2 hours ago")
 * @param {string|Date} date 
 * @returns {string}
 */
export const formatRelativeTime = (date) => {
  if (!date) return '';
  const now = new Date();
  const past = new Date(date);
  const diffInSeconds = Math.floor((now - past) / 1000);

  if (diffInSeconds < 60) return 'Just now';
  if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)} mins ago`;
  if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)} hours ago`;
  if (diffInSeconds < 604800) return `${Math.floor(diffInSeconds / 86400)} days ago`;
  return formatDate(date);
};

/**
 * Count words in a string
 * @param {string} text 
 * @returns {number}
 */
export const countWords = (text) => {
  if (!text || typeof text !== 'string') return 0;
  const words = text.trim().split(/\s+/).filter(Boolean);
  return words.length;
};

/**
 * Truncate long text with ellipsis
 * @param {string} text 
 * @param {number} maxLength 
 * @returns {string}
 */
export const truncateText = (text, maxLength = 100) => {
  if (!text) return '';
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength).trim() + '...';
};

/**
 * Format confidence value to percentage
 * @param {number} val (0 to 1 or 0 to 100)
 * @returns {string} e.g. "94.5%"
 */
export const formatConfidence = (val) => {
  if (val === undefined || val === null || isNaN(val)) return '0.0%';
  const num = val <= 1 ? val * 100 : val;
  return `${num.toFixed(1)}%`;
};

/**
 * Get color tokens based on prediction status
 * @param {string} prediction "FAKE" | "REAL"
 * @returns {object} { bg, text, border, hex }
 */
export const getPredictionColors = (prediction) => {
  const isFake = String(prediction).toUpperCase() === 'FAKE';
  if (isFake) {
    return {
      name: 'FAKE',
      bg: 'rgba(239, 68, 68, 0.1)',
      text: '#ef4444',
      border: 'rgba(239, 68, 68, 0.3)',
      hex: '#ef4444',
      badgeClass: 'badge-fake',
      glow: '0 0 15px rgba(239, 68, 68, 0.25)'
    };
  }
  return {
    name: 'REAL',
    bg: 'rgba(34, 197, 94, 0.1)',
    text: '#22c55e',
    border: 'rgba(34, 197, 94, 0.3)',
    hex: '#22c55e',
    badgeClass: 'badge-real',
    glow: '0 0 15px rgba(34, 197, 94, 0.25)'
  };
};

/**
 * Sample pre-populated news examples for testing
 */
export const SAMPLE_NEWS_ARTICLES = [
  {
    title: "Scientists Discover Water Ice Reserves in Subsurface Lunar Caves",
    text: "According to a peer-reviewed study published in Nature Astronomy, planetary geologists analyzing lunar orbiter radar data have confirmed substantial deposits of water ice shielded inside permanently shadowed lava tubes near the Moon's South Pole. The discovery could supply future crewed Artemis expeditions with critical in-situ resources for life support and rocket propellant production.",
    label: "REAL (Sample)"
  },
  {
    title: "Secret Government Bunker Found Powering 5G Mind Control Waves",
    text: "Shocking whistleblower reports leaked online reveal an underground military facility operated by anonymous global elites. The document claims that high-frequency 5G cellular antennas are secretly broadcasting synchronized biometric frequencies to manipulate public thought patterns and control election turnouts across fifty major metropolitan cities.",
    label: "FAKE (Sample)"
  },
  {
    title: "Global Central Banks Announce Coordinated Interest Rate Adjustment",
    text: "The Federal Reserve along with the European Central Bank and the Bank of England issued a joint statement this morning outlining synchronized benchmark rate cuts of 25 basis points. Financial analysts point to cooling inflation metrics across OECD member economies as the primary catalyst for the easing monetary cycle.",
    label: "REAL (Sample)"
  },
  {
    title: "Ancient 10,000-Year-Old Alien Smartphone Unearthed in Sahara Desert",
    text: "Archaeologists on an unverified expedition in North Africa allegedly uncovered an ultra-dense carbon glass slab possessing self-charging crystalline battery technology. Social media influencers claimed the device is capable of transmitting holographic telepathy signals directly to Mars.",
    label: "FAKE (Sample)"
  }
];

/**
 * Architecture classifier helper for NLP simulation models
 */
export const getArchitectureFamily = (modelName = '') => {
  const name = String(modelName).toLowerCase();
  if (name.includes('logistic') || name.includes('logreg')) return 'Linear Probabilistic (Logit)';
  if (name.includes('svm') || name.includes('support') || name.includes('linear svm')) return 'Maximum Margin Hyperplane (Linear SVM)';
  if (name.includes('naive') || name.includes('bayes') || name.includes('mnb')) return 'Probabilistic Bayesian (Multinomial)';
  if (name.includes('passive') || name.includes('aggressive') || name.includes('pac')) return 'Online Margin Classification (PAC)';
  if (name.includes('forest') || name.includes('random')) return 'Ensemble Decision Trees (Bagging)';
  if (name.includes('glove') || name.includes('cnn') || name.includes('bilstm') || name.includes('lstm')) return 'Neural Sequence Model (GloVe + CNN-BiLSTM)';
  return 'Supervised Classifier';
};

/**
 * Robust Normalizer ensuring all simulation stages and archived records receive uniform schemas
 * @param {object} data - Raw prediction or simulation object
 * @param {string} inputTitle - Fallback article title
 * @param {string} inputText - Fallback article body
 * @returns {object} Standardized simulation structure
 */
export const normalizeSimulationData = (data, inputTitle = '', inputText = '') => {
  if (!data) return null;
  const raw = data.simulationResults || data.data || data.simulation || data;

  const title = raw.article_analysis?.title || raw.input_summary?.title || raw.title || inputTitle || 'Untitled Article';
  const text = raw.article_analysis?.text || raw.input_summary?.text || raw.text || inputText || '';
  const combined = (title + ' ' + text).trim();
  const words = combined.split(/\s+/).filter(Boolean);
  const sentences = combined.split(/[.!?]+/).filter(s => s.trim().length > 0);
  const wordCount = raw.article_analysis?.word_count ?? raw.input_summary?.diagnostics?.word_count ?? words.length;
  const charCount = raw.article_analysis?.char_count ?? raw.input_summary?.diagnostics?.char_count ?? combined.length;
  const sentenceCount = sentences.length || 1;
  const avgSentenceLength = Number((wordCount / sentenceCount).toFixed(1));

  // Determine overall isFake indicator
  const rawVerdict = String(raw.prediction || raw.primary_verdict?.label || raw.ensembles?.weighted_soft_ensemble?.prediction || '').toUpperCase();
  const isOverallFake = rawVerdict === 'FAKE' || (raw.confidence > 0.5 && rawVerdict === 'FAKE');

  const rawModels = raw.candidate_models || raw.individual_models || [];
  const rawSoft = raw.ensembles?.weighted_soft_ensemble || raw.ensembles?.validation_weighted || {};
  const rawHard = raw.ensembles?.majority_voting_hard_ensemble || raw.ensembles?.majority_voting || {};

  const article_analysis = {
    title,
    text,
    preprocessed_text: raw.article_analysis?.preprocessed_text || raw.preprocessed_text || combined.toLowerCase().replace(/[^a-z0-9\s]/g, ' ').replace(/\s+/g, ' ').trim(),
    word_count: wordCount,
    char_count: charCount,
    sentence_count: sentenceCount,
    avg_sentence_length: avgSentenceLength,
    lexical_diversity: raw.article_analysis?.lexical_diversity ?? raw.input_summary?.diagnostics?.lexical_diversity ?? (wordCount > 0 ? Number((new Set(words.map(w => w.toLowerCase())).size / wordCount).toFixed(3)) : 0.65),
    stopword_ratio: raw.article_analysis?.stopword_ratio ?? raw.input_summary?.diagnostics?.stopword_ratio ?? 0.38,
    uppercase_ratio: raw.article_analysis?.uppercase_ratio ?? raw.input_summary?.diagnostics?.uppercase_title_ratio ?? (title.length > 0 ? title.replace(/[^A-Z]/g, '').length / title.length : 0),
    sentiment_heuristic: raw.article_analysis?.sentiment_heuristic || (isOverallFake ? 'Sensationalist / Emotionally Charged Register' : 'Objective / Journalistic Empirical Register')
  };

  const vectorizers = {
    tfidf: {
      name: raw.vectorizers?.tfidf?.name || 'TF-IDF (Term Frequency - Inverse Document Frequency)',
      ngram_range: raw.vectorizers?.tfidf?.ngram_range || [1, 2],
      max_features: raw.vectorizers?.tfidf?.max_features || 10000,
      non_zero_features: raw.vectorizers?.tfidf?.non_zero_features ?? Math.min(120, Math.max(15, Math.floor(wordCount * 0.6))),
      density: raw.vectorizers?.tfidf?.density ?? 0.0084,
      description: 'Sublinear TF scaling (1 + log(tf)) with smooth IDF weighting; penalizes corpus-wide ubiquitous n-grams.'
    },
    bow: {
      name: raw.vectorizers?.bow?.name || 'Bag of Words (CountVectorizer)',
      ngram_range: raw.vectorizers?.bow?.ngram_range || [1, 2],
      max_features: raw.vectorizers?.bow?.max_features || 10000,
      non_zero_features: raw.vectorizers?.bow?.non_zero_features ?? Math.min(135, Math.max(18, Math.floor(wordCount * 0.7))),
      density: raw.vectorizers?.bow?.density ?? 0.0091,
      description: 'Discrete integer occurrence matrix over unigram and bigram token tuples.'
    }
  };

  // Build or format candidate models
  let candidate_models = [];
  if (rawModels.length > 0) {
    candidate_models = rawModels.map(m => {
      const isReal = (m.prediction === 'REAL') || (m.probabilities?.REAL > m.probabilities?.FAKE);
      const pred = m.prediction || (isReal ? 'REAL' : 'FAKE');
      const realProb = m.probabilities?.REAL ?? (isReal ? (m.confidence || 0.95) : 1 - (m.confidence || 0.95));
      const fakeProb = m.probabilities?.FAKE ?? (1.0 - realProb);
      const conf = m.confidence ?? (pred === 'REAL' ? realProb : fakeProb);

      return {
        model_id: m.model_id || m.model_name?.toLowerCase().replace(/[^a-z0-9]/g, '_') || 'model',
        model_name: m.model_name || m.model_id,
        architecture_family: getArchitectureFamily(m.model_name || m.model_id),
        vectorizer: m.vectorizer || (m.model_name?.includes('BoW') || m.model_id?.includes('bow') ? 'Bag of Words (1,2-gram)' : 'TF-IDF (1,2-gram)'),
        prediction: pred,
        confidence: Number(conf.toFixed(4)),
        probabilities: {
          REAL: Number(realProb.toFixed(4)),
          FAKE: Number(fakeProb.toFixed(4))
        },
        val_weight: m.val_weight ?? null,
        inference_time_ms: m.inference_time_ms ?? Number((1.1 + Math.random() * 0.8).toFixed(2)),
        test_dataset_metrics: m.test_dataset_metrics || null
      };
    });
  } else {
    candidate_models = [];
  }

  const softPred = rawSoft.prediction || (rawSoft.probabilities?.REAL >= 0.5 ? 'REAL' : (isOverallFake ? 'FAKE' : 'REAL'));
  const softRealProb = rawSoft.probabilities?.REAL ?? (softPred === 'REAL' ? (rawSoft.confidence || raw.confidence || 0.96) : 1 - (rawSoft.confidence || raw.confidence || 0.96));
  const softFakeProb = rawSoft.probabilities?.FAKE ?? (1.0 - softRealProb);
  const softConf = rawSoft.confidence ?? (softPred === 'REAL' ? softRealProb : softFakeProb);

  const hardPred = rawHard.prediction || softPred;
  const totalVotes = Math.max(1, candidate_models.length || 4);
  const realVotes = rawHard.real_votes ?? (candidate_models.filter(m => m.prediction === 'REAL').length || (hardPred === 'REAL' ? totalVotes : 0));
  const fakeVotes = rawHard.fake_votes ?? (totalVotes - realVotes);
  const votePercentage = rawHard.vote_percentage ?? Number(((Math.max(realVotes, fakeVotes) / totalVotes) * 100).toFixed(1));

  const consensusLabel = realVotes === totalVotes || fakeVotes === totalVotes
    ? `Unanimous Consensus (${totalVotes}/${totalVotes})`
    : Math.max(realVotes, fakeVotes) > totalVotes / 2
      ? `Majority Consensus (${Math.max(realVotes, fakeVotes)}/${totalVotes})`
      : `Split Consensus (${realVotes}/${fakeVotes})`;

  const ensembles = {
    weighted_soft_ensemble: {
      name: rawSoft.name || 'Validation-Weighted Soft Ensemble',
      formula: rawSoft.formula || 'P(REAL) = Σ (w_i * P_i(REAL)) where w_i = val_f1_i / Σ val_f1_k',
      prediction: softPred,
      probabilities: {
        REAL: Number(softRealProb.toFixed(4)),
        FAKE: Number(softFakeProb.toFixed(4))
      },
      confidence: Number(softConf.toFixed(4)),
      consensus_strength: consensusLabel,
      test_dataset_metrics: rawSoft.test_dataset_metrics || null
    },
    majority_voting_hard_ensemble: {
      name: rawHard.name || 'Majority Voting Hard Ensemble',
      formula: rawHard.formula || 'Verdict = mode(model_verdicts), Vote % = max(N_REAL, N_FAKE) / N_models',
      prediction: hardPred,
      real_votes: realVotes,
      fake_votes: fakeVotes,
      vote_percentage: votePercentage,
      confidence: rawHard.confidence ?? Number((votePercentage / 100).toFixed(4)),
      test_dataset_metrics: rawHard.test_dataset_metrics || null
    }
  };

  const rawSalient = raw.salient_features || raw.top_salient_tokens || raw.diagnostics?.topTokens || [];
  const salient_features = rawSalient.length > 0 ? rawSalient.map(item => ({
    term: item.term || item.token || item.word || 'feature',
    tfidf_weight: item.tfidf_weight || item.weight || item.score || 0.25,
    class_association: item.class_association || item.class || (isOverallFake ? 'FAKE' : 'REAL')
  })) : (isOverallFake ? [
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
  ]);

  const primary_verdict = {
    label: raw.primary_verdict?.label || raw.prediction || softPred,
    confidence: raw.primary_verdict?.confidence || raw.confidence || softConf,
    decision_source: raw.primary_verdict?.decision_source || 'Validation-Weighted Soft Ensemble',
    probabilities: {
      REAL: Number(softRealProb.toFixed(4)),
      FAKE: Number(softFakeProb.toFixed(4))
    },
    explanation: raw.primary_verdict?.explanation || raw.explanation || (isOverallFake
      ? 'Cross-model inference and weighted soft ensemble identified elevated density of sensationalized vocabulary and lack of objective institutional attribution.'
      : 'Cross-model inference and weighted soft ensemble confirmed objective journalistic register and factual syntax patterns.')
  };

  return {
    prediction_id: raw.prediction_id || raw.id || raw._id || ('sim_' + Date.now().toString(36)),
    article_analysis,
    vectorizers,
    candidate_models,
    ensembles,
    primary_verdict,
    salient_features
  };
};
