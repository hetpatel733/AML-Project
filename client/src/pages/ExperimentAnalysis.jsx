import React, { useEffect, useState } from 'react';
import { 
  BookOpen, 
  Layers, 
  BarChart3, 
  Award, 
  Scale, 
  CheckCircle2, 
  Download, 
  HelpCircle, 
  ChevronDown, 
  ChevronUp, 
  Cpu, 
  FileSpreadsheet,
  Info,
  Sparkles
} from 'lucide-react';
import { getExperimentMetadata } from '../services/api';
import CrossValidationChart from '../charts/CrossValidationChart';
import { useDataset } from '../context/DatasetContext';

export const ExperimentAnalysis = () => {
  const { selectedDataset } = useDataset();
  const [metadata, setMetadata] = useState(null);
  const [loading, setLoading] = useState(true);
  const [openFaq, setOpenFaq] = useState({});

  useEffect(() => {
    const fetchMetadata = async () => {
      setLoading(true);
      try {
        const response = await getExperimentMetadata(selectedDataset);
        if (response?.data?.dataset && Array.isArray(response.data.models)) {
          setMetadata(response.data);
        } else {
          setMetadata(null);
        }
      } catch (err) {
        console.error('Failed to load experiment metadata:', err);
        setMetadata(null);
      } finally {
        setLoading(false);
      }
    };
    fetchMetadata();
  }, [selectedDataset]);

  if (loading) {
    return <div className="page-container"><p className="text-muted">Loading experiment results...</p></div>;
  }

  if (!metadata) {
    return (
      <div className="page-container">
        <div className="card">
          <h2>Training not completed</h2>
          <p className="text-muted">No trained experiment available for this dataset yet.</p>
        </div>
      </div>
    );
  }

  const toggleFaq = (index) => {
    setOpenFaq(prev => ({ ...prev, [index]: !prev[index] }));
  };

  const handleExportJson = () => {
    if (!metadata) return;
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(metadata, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", "experiment_metadata_benchmarks.json");
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  const faqs = [
    {
      q: 'How is data leakage prevented?',
      a: 'The dataset is strictly partitioned into an 80% Training set and a 20% Test hold-out partition (with zero validation tuning or test leakage). Text preprocessing and the TF-IDF feature extractor are fitted solely on the 80% training set and then transformed onto the 20% test partition.'
    },
    {
      q: 'What machine learning models are evaluated in the benchmark?',
      a: 'Following the paper methodology, 5 classical supervised classifiers are trained and evaluated: Logistic Regression (C=1.0), Multinomial Naive Bayes (alpha=1.0), Linear SVM (LinearSVC, C=1.0), Decision Tree (Gini), and Random Forest (100 estimators).'
    },
    {
      q: 'What text feature extraction pipeline is applied?',
      a: 'Sublinear TF-IDF (Term Frequency-Inverse Document Frequency) vectorization using unigrams and bigrams (ngram_range=(1, 2)), capped at a maximum vocabulary of 10,000 features.'
    },
    {
      q: 'What text preprocessing steps are executed?',
      a: 'Raw text is normalized via lowercasing, stripping URLs and HTML markup, eliminating punctuation and digits, filtering NLTK English stopwords, and applying cached WordNet lemmatization.'
    }
  ];

  return (
    <div className="experiment-analysis-page" style={{ padding: '24px 0', maxWidth: '1280px', margin: '0 auto' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '28px', flexWrap: 'wrap', gap: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            background: 'linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(59, 130, 246, 0.2))',
            padding: '10px',
            borderRadius: '12px',
            border: '1px solid rgba(16, 185, 129, 0.3)'
          }}>
            <BookOpen size={28} className="text-primary" />
          </div>
          <div>
            <h1 style={{ margin: 0, fontSize: '26px', fontWeight: 800, color: '#f8fafc' }}>
              Experiment Analysis
            </h1>
            <p style={{ margin: 0, fontSize: '14px', color: '#94a3b8' }}>
              Model benchmarks, validation metrics, and ensemble performance evaluation.
            </p>
          </div>
        </div>

        <button
          onClick={handleExportJson}
          className="btn-secondary btn-sm"
          style={{ display: 'flex', alignItems: 'center', gap: '8px' }}
        >
          <Download size={15} />
          <span>Export Data</span>
        </button>
      </div>

      {/* Section 1: Dataset Partition & Hygiene Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px', marginBottom: '28px' }}>
        <div className="card" style={{ background: 'rgba(30, 41, 59, 0.7)', border: '1px solid rgba(59, 130, 246, 0.3)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <span style={{ fontSize: '12px', fontWeight: 700, color: '#60a5fa', textTransform: 'uppercase' }}>
              Training Set (80%)
            </span>
            <Layers size={16} className="text-primary" />
          </div>
          <div style={{ fontSize: '24px', fontWeight: 800, color: '#f8fafc' }}>
            {(metadata.splits?.training?.count ?? 'N/A').toLocaleString()} <span style={{ fontSize: '13px', fontWeight: 400, color: '#94a3b8' }}>articles</span>
          </div>
          <p style={{ margin: '6px 0 0 0', fontSize: '12px', color: '#94a3b8' }}>
            Used for model fitting and TF-IDF vocabulary extraction.
          </p>
        </div>

        <div className="card" style={{ background: 'rgba(30, 41, 59, 0.7)', border: '1px solid rgba(16, 185, 129, 0.3)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <span style={{ fontSize: '12px', fontWeight: 700, color: '#34d399', textTransform: 'uppercase' }}>
              Hold-Out Test Partition (20%)
            </span>
            <Award size={16} style={{ color: '#34d399' }} />
          </div>
          <div style={{ fontSize: '24px', fontWeight: 800, color: '#f8fafc' }}>
            {(metadata.splits?.testing?.count ?? 'N/A').toLocaleString()} <span style={{ fontSize: '13px', fontWeight: 400, color: '#94a3b8' }}>articles</span>
          </div>
          <p style={{ margin: '6px 0 0 0', fontSize: '12px', color: '#94a3b8' }}>
            Unbiased final empirical benchmark & ROC-AUC verification.
          </p>
        </div>

        <div className="card" style={{ background: 'rgba(30, 41, 59, 0.7)', border: '1px solid rgba(139, 92, 246, 0.3)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <span style={{ fontSize: '12px', fontWeight: 700, color: '#a78bfa', textTransform: 'uppercase' }}>
              Feature Vectorizer
            </span>
            <Cpu size={16} style={{ color: '#a78bfa' }} />
          </div>
          <div style={{ fontSize: '24px', fontWeight: 800, color: '#f8fafc' }}>
            10,000 <span style={{ fontSize: '13px', fontWeight: 400, color: '#94a3b8' }}>features</span>
          </div>
          <p style={{ margin: '6px 0 0 0', fontSize: '12px', color: '#94a3b8' }}>
            Sublinear TF-IDF with unigrams + bigrams. Zero test leakage.
          </p>
        </div>
      </div>

      <div className="card" style={{ marginBottom: '28px', background: 'rgba(30, 41, 59, 0.7)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: '24px', flexWrap: 'wrap' }}>
          <div>
            <h3 style={{ margin: 0, fontSize: '18px', color: '#f8fafc' }}>Experiment Validity & Setup</h3>
            <p style={{ margin: '8px 0 0', color: '#94a3b8', fontSize: '13px' }}>
              Dataset: {metadata.dataset?.name || 'N/A'} | Version: {metadata.experiment?.version || 'Unversioned'} | Seed: {metadata.experiment?.randomSeed ?? 'N/A'} | Trained: {metadata.experiment?.trainingTimestamp || 'N/A'}
            </p>
          </div>
          <div style={{ minWidth: '280px' }}>
            <strong style={{ color: '#fbbf24' }}>Split Strategy: </strong>
            <span style={{ color: '#cbd5e1' }}>80% Train / 20% Test (Zero Leakage)</span>
            <p style={{ margin: '6px 0 0', color: '#94a3b8', fontSize: '12px' }}>
              Text cleaning & TF-IDF fitted solely on training partition.
            </p>
          </div>
        </div>
        <div style={{ marginTop: '16px', color: '#cbd5e1', fontSize: '13px' }}>
          Records: {metadata.dataset?.totalRecords?.toLocaleString() ?? 'N/A'} | Classes: {metadata.dataset?.classes?.join(', ') || 'N/A'} | Preprocessing: {metadata.dataset?.preprocessing?.textCleaning || 'N/A'}
        </div>
      </div>

      {/* Section 2: Comprehensive 5-Model Benchmark Matrix Table */}
      <div className="card" style={{ marginBottom: '28px', background: 'rgba(30, 41, 59, 0.7)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '16px' }}>
          <Award size={20} className="text-primary" />
          <h3 style={{ margin: 0, fontSize: '18px', fontWeight: 700, color: '#f8fafc' }}>
            5-Classifier Benchmark Matrix (Hold-Out Test Split)
          </h3>
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '13px' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.1)', color: '#94a3b8' }}>
                <th style={{ padding: '12px 14px' }}>Model Architecture</th>
                <th style={{ padding: '12px 14px' }}>Feature Vectorizer</th>
                <th style={{ padding: '12px 14px', textAlign: 'center' }}>Training Time</th>
                <th style={{ padding: '12px 14px', textAlign: 'center' }}>Test Accuracy</th>
                <th style={{ padding: '12px 14px', textAlign: 'center' }}>Test Precision</th>
                <th style={{ padding: '12px 14px', textAlign: 'center' }}>Test Recall</th>
                <th style={{ padding: '12px 14px', textAlign: 'center' }}>Test F1 Score</th>
                <th style={{ padding: '12px 14px', textAlign: 'center' }}>ROC-AUC</th>
              </tr>
            </thead>
            <tbody>
              {(metadata.models || []).map((model) => {
                const modelName = model.name || model.id;
                const vectorizerLabel = model.representation || 'TF-IDF (1,2-grams)';
                const trainingTime = typeof model.trainingTime === 'number' ? `${model.trainingTime.toFixed(2)}s` : 'N/A';
                const acc = model.metrics?.accuracy;
                const prec = model.metrics?.precision;
                const rec = model.metrics?.recall;
                const f1 = model.metrics?.f1;
                const roc = model.metrics?.rocAuc;
                const formatMetric = value => typeof value === 'number' ? `${(value * 100).toFixed(2)}%` : 'Unavailable';
                const formatRoc = value => typeof value === 'number' ? value.toFixed(4) : 'Unavailable';

                return (
                  <tr key={model.id} style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)' }}>
                    <td style={{ padding: '12px 14px', fontWeight: 600, color: '#f8fafc' }}>
                      {modelName}
                    </td>
                    <td style={{ padding: '12px 14px', color: '#94a3b8' }}>
                      {vectorizerLabel}
                    </td>
                    <td style={{ padding: '12px 14px', textAlign: 'center', color: '#c4b5fd', fontWeight: 600 }}>
                      {trainingTime}
                    </td>
                    <td style={{ padding: '12px 14px', textAlign: 'center', color: '#38bdf8', fontWeight: 600 }}>
                      {formatMetric(acc)}
                    </td>
                    <td style={{ padding: '12px 14px', textAlign: 'center', color: '#34d399' }}>
                      {formatMetric(prec)}
                    </td>
                    <td style={{ padding: '12px 14px', textAlign: 'center', color: '#a78bfa' }}>
                      {formatMetric(rec)}
                    </td>
                    <td style={{ padding: '12px 14px', textAlign: 'center', color: '#f59e0b', fontWeight: 700 }}>
                      {formatMetric(f1)}
                    </td>
                    <td style={{ padding: '12px 14px', textAlign: 'center', color: '#ec4899', fontWeight: 700 }}>
                      {formatRoc(roc)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Section 3: 5-Fold Stratified Cross Validation D3 Chart */}
      <div className="card" style={{ marginBottom: '28px', background: 'rgba(30, 41, 59, 0.7)' }}>
        <CrossValidationChart
          cvData={metadata?.models || metadata?.model_metrics || metadata || {}}
          height={340}
        />
      </div>

      {/* Section 4: Baseline Paper Comparison Matrix */}
      {metadata.paperComparison && (
        <div className="card" style={{ marginBottom: '28px', background: 'rgba(30, 41, 59, 0.7)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '16px' }}>
            <Scale size={20} className="text-primary" />
            <div>
              <h3 style={{ margin: 0, fontSize: '18px', fontWeight: 700, color: '#f8fafc' }}>
                Replication & Paper Baseline Benchmark Comparison
              </h3>
              <p style={{ margin: '4px 0 0', fontSize: '13px', color: '#94a3b8' }}>
                Comparative evaluation against original literature baseline ({metadata.paperComparison.baselinePaper || 'Published Paper'}).
              </p>
            </div>
          </div>

          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '13px' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.1)', color: '#94a3b8' }}>
                  <th style={{ padding: '12px 14px' }}>Model Architecture</th>
                  <th style={{ padding: '12px 14px', textAlign: 'center' }}>Paper Baseline Accuracy</th>
                  <th style={{ padding: '12px 14px', textAlign: 'center' }}>Our Replicated Test Accuracy</th>
                  <th style={{ padding: '12px 14px', textAlign: 'center' }}>Empirical Improvement Delta</th>
                </tr>
              </thead>
              <tbody>
                {(metadata.models || []).map((model) => {
                  const comp = model.paperComparison || {};
                  const baseAcc = comp.paperBaselineAccuracy;
                  const ourAcc = model.metrics?.accuracy;
                  const delta = comp.improvementDelta;
                  const isPositive = delta >= 0;

                  return (
                    <tr key={model.id} style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)' }}>
                      <td style={{ padding: '12px 14px', fontWeight: 600, color: '#f8fafc' }}>
                        {model.name}
                      </td>
                      <td style={{ padding: '12px 14px', textAlign: 'center', color: '#94a3b8' }}>
                        {typeof baseAcc === 'number' ? `${(baseAcc * 100).toFixed(2)}%` : 'N/A'}
                      </td>
                      <td style={{ padding: '12px 14px', textAlign: 'center', color: '#38bdf8', fontWeight: 700 }}>
                        {typeof ourAcc === 'number' ? `${(ourAcc * 100).toFixed(2)}%` : 'N/A'}
                      </td>
                      <td style={{ padding: '12px 14px', textAlign: 'center' }}>
                        {typeof delta === 'number' ? (
                          <span style={{
                            padding: '3px 8px',
                            borderRadius: '4px',
                            fontWeight: 700,
                            fontSize: '12px',
                            background: isPositive ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.2)',
                            color: isPositive ? '#34d399' : '#f87171'
                          }}>
                            {isPositive ? `+${(delta * 100).toFixed(2)}%` : `${(delta * 100).toFixed(2)}%`}
                          </span>
                        ) : 'N/A'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Section 5: Preprocessing Pipeline Architecture Details */}
      {metadata.dataset?.preprocessing && (
        <div className="card" style={{ marginBottom: '28px', background: 'rgba(30, 41, 59, 0.7)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '16px' }}>
            <FileSpreadsheet size={20} className="text-primary" />
            <h3 style={{ margin: 0, fontSize: '18px', fontWeight: 700, color: '#f8fafc' }}>
              Optimized Preprocessing & Feature Engineering Specifications
            </h3>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '14px' }}>
            <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '12px 16px', borderRadius: '8px', border: '1px solid rgba(255, 255, 255, 0.06)' }}>
              <span style={{ fontSize: '11px', color: '#94a3b8', textTransform: 'uppercase', fontWeight: 600 }}>N-Gram Range</span>
              <div style={{ fontSize: '15px', color: '#f8fafc', fontWeight: 700, marginTop: '4px' }}>
                {metadata.dataset.preprocessing.ngramRange ? `Unigrams + Bigrams (${metadata.dataset.preprocessing.ngramRange.join(', ')})` : 'Unigrams + Bigrams (1, 2)'}
              </div>
            </div>

            <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '12px 16px', borderRadius: '8px', border: '1px solid rgba(255, 255, 255, 0.06)' }}>
              <span style={{ fontSize: '11px', color: '#94a3b8', textTransform: 'uppercase', fontWeight: 600 }}>Sublinear TF Scaling</span>
              <div style={{ fontSize: '15px', color: '#34d399', fontWeight: 700, marginTop: '4px' }}>
                {metadata.dataset.preprocessing.sublinearTf ? 'Enabled (1 + log(tf))' : 'Standard'}
              </div>
            </div>

            <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '12px 16px', borderRadius: '8px', border: '1px solid rgba(255, 255, 255, 0.06)' }}>
              <span style={{ fontSize: '11px', color: '#94a3b8', textTransform: 'uppercase', fontWeight: 600 }}>Document Frequencies</span>
              <div style={{ fontSize: '15px', color: '#f8fafc', fontWeight: 700, marginTop: '4px' }}>
                min_df={metadata.dataset.preprocessing.minDf ?? 2}, max_df={metadata.dataset.preprocessing.maxDf ?? 0.98}
              </div>
            </div>

            <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '12px 16px', borderRadius: '8px', border: '1px solid rgba(255, 255, 255, 0.06)' }}>
              <span style={{ fontSize: '11px', color: '#94a3b8', textTransform: 'uppercase', fontWeight: 600 }}>Stopword Strategy</span>
              <div style={{ fontSize: '15px', color: '#fbbf24', fontWeight: 700, marginTop: '4px' }}>
                {metadata.dataset.preprocessing.removeStopwords || 'Negation-Preserving Filter'}
              </div>
            </div>

            <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '12px 16px', borderRadius: '8px', border: '1px solid rgba(255, 255, 255, 0.06)' }}>
              <span style={{ fontSize: '11px', color: '#94a3b8', textTransform: 'uppercase', fontWeight: 600 }}>Lemmatization</span>
              <div style={{ fontSize: '15px', color: '#a78bfa', fontWeight: 700, marginTop: '4px' }}>
                {metadata.dataset.preprocessing.lemmatization || 'WordNet POS-Aware'}
              </div>
            </div>

            <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '12px 16px', borderRadius: '8px', border: '1px solid rgba(255, 255, 255, 0.06)' }}>
              <span style={{ fontSize: '11px', color: '#94a3b8', textTransform: 'uppercase', fontWeight: 600 }}>Max Vocabulary</span>
              <div style={{ fontSize: '15px', color: '#38bdf8', fontWeight: 700, marginTop: '4px' }}>
                {(metadata.dataset.preprocessing.maxFeatures ?? 10000).toLocaleString()} Features
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Section 6: Generated High-Resolution Visual Assets */}
      <div className="card" style={{ marginBottom: '28px', background: 'rgba(30, 41, 59, 0.7)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '16px' }}>
          <Sparkles size={20} className="text-primary" />
          <div>
            <h3 style={{ margin: 0, fontSize: '18px', fontWeight: 700, color: '#f8fafc' }}>
              Generated Visual Benchmark Figures (300 DPI)
            </h3>
            <p style={{ margin: '4px 0 0', fontSize: '13px', color: '#94a3b8' }}>
              Visual analytics generated from the 20% hold-out evaluation on {metadata.dataset?.name || selectedDataset.toUpperCase()}.
            </p>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '20px' }}>
          <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '16px', borderRadius: '10px', border: '1px solid rgba(255, 255, 255, 0.06)' }}>
            <h4 style={{ margin: '0 0 10px', fontSize: '14px', color: '#f8fafc', fontWeight: 600 }}>
              Accuracy vs. Paper Baseline
            </h4>
            <img 
              src={`/api/figures/${selectedDataset}/${selectedDataset}_accuracy_comparison.png`} 
              alt="Accuracy Comparison" 
              style={{ width: '100%', borderRadius: '6px' }}
              onError={(e) => { e.target.style.display = 'none'; }}
            />
          </div>

          <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '16px', borderRadius: '10px', border: '1px solid rgba(255, 255, 255, 0.06)' }}>
            <h4 style={{ margin: '0 0 10px', fontSize: '14px', color: '#f8fafc', fontWeight: 600 }}>
              Precision-Recall Curves
            </h4>
            <img 
              src={`/api/figures/${selectedDataset}/${selectedDataset}_precision_recall_curves.png`} 
              alt="Precision-Recall Curves" 
              style={{ width: '100%', borderRadius: '6px' }}
              onError={(e) => { e.target.style.display = 'none'; }}
            />
          </div>

          <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '16px', borderRadius: '10px', border: '1px solid rgba(255, 255, 255, 0.06)' }}>
            <h4 style={{ margin: '0 0 10px', fontSize: '14px', color: '#f8fafc', fontWeight: 600 }}>
              Recall Sensitivity vs. Decision Threshold
            </h4>
            <img 
              src={`/api/figures/${selectedDataset}/${selectedDataset}_recall_vs_threshold.png`} 
              alt="Recall Sensitivity vs Threshold" 
              style={{ width: '100%', borderRadius: '6px' }}
              onError={(e) => { e.target.style.display = 'none'; }}
            />
          </div>

          <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '16px', borderRadius: '10px', border: '1px solid rgba(255, 255, 255, 0.06)' }}>
            <h4 style={{ margin: '0 0 10px', fontSize: '14px', color: '#f8fafc', fontWeight: 600 }}>
              5-Model Confusion Matrices Grid
            </h4>
            <img 
              src={`/api/figures/${selectedDataset}/${selectedDataset}_all_confusion_matrices.png`} 
              alt="Confusion Matrices Grid" 
              style={{ width: '100%', borderRadius: '6px' }}
              onError={(e) => { e.target.style.display = 'none'; }}
            />
          </div>
        </div>
      </div>

      {/* Section 7: Academic FAQ Accordion */}
      <div className="card" style={{ marginBottom: '28px', background: 'rgba(30, 41, 59, 0.7)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '18px' }}>
          <HelpCircle size={20} className="text-primary" />
          <h3 style={{ margin: 0, fontSize: '18px', fontWeight: 700, color: '#f8fafc' }}>
            Experimental Methodology & Scientific FAQ
          </h3>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {faqs.map((faq, idx) => (
            <div
              key={idx}
              style={{
                borderRadius: '8px',
                background: 'rgba(15, 23, 42, 0.6)',
                border: '1px solid rgba(255, 255, 255, 0.08)',
                overflow: 'hidden'
              }}
            >
              <button
                type="button"
                onClick={() => toggleFaq(idx)}
                style={{
                  width: '100%',
                  padding: '14px 16px',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  background: 'transparent',
                  border: 'none',
                  color: '#f8fafc',
                  fontSize: '14px',
                  fontWeight: 600,
                  textAlign: 'left',
                  cursor: 'pointer'
                }}
              >
                <span>{faq.q}</span>
                {openFaq[idx] ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
              </button>
              {openFaq[idx] && (
                <div style={{ padding: '0 16px 14px 16px', fontSize: '13px', color: '#cbd5e1', lineHeight: '1.6', borderTop: '1px solid rgba(255, 255, 255, 0.05)', paddingTop: '10px' }}>
                  {faq.a}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ExperimentAnalysis;
