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
      a: 'The dataset is split into three parts: 70% Training, 15% Validation, and 15% Test. Feature extractors are trained only on training data and applied to other sets. Ensemble weights use validation scores without accessing test data.'
    },
    {
      q: 'Why use CalibratedClassifierCV?',
      a: 'Some classifiers output decision scores rather than probabilities. CalibratedClassifierCV converts these to proper probabilities between 0 and 1, enabling accurate ensemble averaging.'
    },
    {
      q: 'How does the weighted ensemble work?',
      a: 'Each model receives a weight based on its validation performance. The final prediction combines all model probabilities using these weights, with higher-performing models having more influence.'
    },
    {
      q: 'Why use both TF-IDF and Bag-of-Words?',
      a: 'TF-IDF emphasizes rare, distinctive words while downweighting common terms. Bag-of-Words uses raw frequency counts. Testing both approaches helps identify which features work best for fake news detection.'
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
              Training Set
            </span>
            <Layers size={16} className="text-primary" />
          </div>
          <div style={{ fontSize: '24px', fontWeight: 800, color: '#f8fafc' }}>
            {(metadata.splits?.training?.count ?? 'N/A').toLocaleString()} <span style={{ fontSize: '13px', fontWeight: 400, color: '#94a3b8' }}>articles</span>
          </div>
          <p style={{ margin: '6px 0 0 0', fontSize: '12px', color: '#94a3b8' }}>
            Used for model training and feature extraction.
          </p>
        </div>

        <div className="card" style={{ background: 'rgba(30, 41, 59, 0.7)', border: '1px solid rgba(139, 92, 246, 0.3)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <span style={{ fontSize: '12px', fontWeight: 700, color: '#a78bfa', textTransform: 'uppercase' }}>
              Validation Set
            </span>
            <Scale size={16} style={{ color: '#a78bfa' }} />
          </div>
          <div style={{ fontSize: '24px', fontWeight: 800, color: '#f8fafc' }}>
            {(metadata.splits?.validation?.count ?? 'N/A').toLocaleString()} <span style={{ fontSize: '13px', fontWeight: 400, color: '#94a3b8' }}>articles</span>
          </div>
          <p style={{ margin: '6px 0 0 0', fontSize: '12px', color: '#94a3b8' }}>
            Used for ensemble weighting and tuning.
          </p>
        </div>

        <div className="card" style={{ background: 'rgba(30, 41, 59, 0.7)', border: '1px solid rgba(16, 185, 129, 0.3)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <span style={{ fontSize: '12px', fontWeight: 700, color: '#34d399', textTransform: 'uppercase' }}>
              Hold-Out Test Partition (15%)
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
      </div>

      <div className="card" style={{ marginBottom: '28px', background: 'rgba(30, 41, 59, 0.7)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: '24px', flexWrap: 'wrap' }}>
          <div>
            <h3 style={{ margin: 0, fontSize: '18px', color: '#f8fafc' }}>Experiment Validity</h3>
            <p style={{ margin: '8px 0 0', color: '#94a3b8', fontSize: '13px' }}>
              Dataset: {metadata.dataset?.name || 'N/A'} | Version: {metadata.experiment?.version || 'Unversioned'} | Seed: {metadata.experiment?.randomSeed ?? 'N/A'} | Trained: {metadata.experiment?.trainingTimestamp || 'N/A'}
            </p>
          </div>
          <div style={{ minWidth: '280px' }}>
            <strong style={{ color: '#fbbf24' }}>External validation: </strong>
            <span style={{ color: '#cbd5e1' }}>{metadata.experiment?.external_validation?.status === 'not_performed' ? 'Not performed' : 'Available'}</span>
            <p style={{ margin: '6px 0 0', color: '#94a3b8', fontSize: '12px' }}>
              {metadata.experiment?.external_validation?.reason || 'No independent external result is reported.'}
            </p>
          </div>
        </div>
        <div style={{ marginTop: '16px', color: '#cbd5e1', fontSize: '13px' }}>
          Records: {metadata.dataset?.totalRecords ?? 'N/A'} | Classes: {metadata.dataset?.classes?.join(', ') || 'N/A'} | Preprocessing: {metadata.dataset?.preprocessing?.textCleaning || 'N/A'}
        </div>
      </div>

      {/* Section 2: Comprehensive 4-Model Benchmark Matrix Table */}
      <div className="card" style={{ marginBottom: '28px', background: 'rgba(30, 41, 59, 0.7)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '16px' }}>
          <Award size={20} className="text-primary" />
          <h3 style={{ margin: 0, fontSize: '18px', fontWeight: 700, color: '#f8fafc' }}>
            Comprehensive Candidate Models & Ensembles Benchmark Matrix (Hold-Out Test Split)
          </h3>
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '13px' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.1)', color: '#94a3b8' }}>
                <th style={{ padding: '12px 14px' }}>Model Architecture</th>
                <th style={{ padding: '12px 14px' }}>Feature Vectorizer</th>
                <th style={{ padding: '12px 14px', textAlign: 'center' }}>Val Weight ($w_i$)</th>
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
                const vectorizerLabel = model.representation || 'Unavailable';
                const acc = model.metrics?.accuracy;
                const prec = model.metrics?.precision;
                const rec = model.metrics?.recall;
                const f1 = model.metrics?.f1;
                const roc = model.metrics?.rocAuc;
                const formatMetric = value => typeof value === 'number' ? `${(value * 100).toFixed(2)}%` : 'Unavailable';

                return (
                  <tr key={model.id} style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)' }}>
                    <td style={{ padding: '12px 14px', fontWeight: 600, color: '#f8fafc' }}>
                      {modelName}
                    </td>
                    <td style={{ padding: '12px 14px', color: '#94a3b8' }}>
                      {vectorizerLabel}
                    </td>
                    <td style={{ padding: '12px 14px', textAlign: 'center', color: '#c4b5fd', fontWeight: 600 }}>
                      {model.status === 'trained' ? 'CV-selected' : 'Unavailable'}
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
                      {formatMetric(roc)}
                    </td>
                  </tr>
                );
              })}

              {/* Ensemble Rows */}
              {metadata?.ensemble?.status === 'trained' && (() => {
                const softEns = metadata?.ensembles?.validation_weighted || metadata?.ensembles?.weighted_soft_ensemble;
                return (
                  <tr style={{ background: 'rgba(139, 92, 246, 0.1)', borderTop: '2px solid rgba(139, 92, 246, 0.4)' }}>
                    <td style={{ padding: '12px 14px', fontWeight: 700, color: '#c4b5fd' }}>
                      ★ {softEns.name || 'Validation-Weighted Soft Ensemble'}
                    </td>
                    <td style={{ padding: '12px 14px', color: '#cbd5e1' }}>
                      Weighted Probabilities (w_i = Val-F1)
                    </td>
                    <td style={{ padding: '12px 14px', textAlign: 'center', color: '#c4b5fd', fontWeight: 700 }}>
                      100.0% (&Sigma; w_i)
                    </td>
                    <td style={{ padding: '12px 14px', textAlign: 'center', color: '#38bdf8', fontWeight: 700 }}>
                      {((softEns.test_metrics?.accuracy ?? 0) * 100).toFixed(2)}%
                    </td>
                    <td style={{ padding: '12px 14px', textAlign: 'center', color: '#34d399', fontWeight: 700 }}>
                      {((softEns.test_metrics?.precision ?? 0) * 100).toFixed(2)}%
                    </td>
                    <td style={{ padding: '12px 14px', textAlign: 'center', color: '#a78bfa', fontWeight: 700 }}>
                      {((softEns.test_metrics?.recall ?? 0) * 100).toFixed(2)}%
                    </td>
                    <td style={{ padding: '12px 14px', textAlign: 'center', color: '#f59e0b', fontWeight: 800 }}>
                      {((softEns.test_metrics?.f1_score ?? softEns.test_metrics?.f1 ?? 0) * 100).toFixed(2)}%
                    </td>
                    <td style={{ padding: '12px 14px', textAlign: 'center', color: '#ec4899', fontWeight: 800 }}>
                      {((softEns.test_metrics?.roc_auc ?? 0) * 100).toFixed(2)}%
                    </td>
                  </tr>
                );
              })()}

              {(metadata?.ensembles?.majority_voting_hard_ensemble || metadata?.ensembles?.majority_voting) && (() => {
                const hardEns = metadata?.ensembles?.majority_voting_hard_ensemble || metadata?.ensembles?.majority_voting;
                return (
                  <tr style={{ background: 'rgba(59, 130, 246, 0.08)' }}>
                    <td style={{ padding: '12px 14px', fontWeight: 700, color: '#93c5fd' }}>
                      ★ {hardEns.name || 'Majority Voting Hard Ensemble'}
                    </td>
                    <td style={{ padding: '12px 14px', color: '#cbd5e1' }}>
                      4-Model Discrete Vote
                    </td>
                    <td style={{ padding: '12px 14px', textAlign: 'center', color: '#93c5fd', fontWeight: 700 }}>
                      Unweighted
                    </td>
                    <td style={{ padding: '12px 14px', textAlign: 'center', color: '#38bdf8', fontWeight: 700 }}>
                      {((hardEns.test_metrics?.accuracy ?? 0) * 100).toFixed(2)}%
                    </td>
                    <td style={{ padding: '12px 14px', textAlign: 'center', color: '#34d399', fontWeight: 700 }}>
                      {((hardEns.test_metrics?.precision ?? 0) * 100).toFixed(2)}%
                    </td>
                    <td style={{ padding: '12px 14px', textAlign: 'center', color: '#a78bfa', fontWeight: 700 }}>
                      {((hardEns.test_metrics?.recall ?? 0) * 100).toFixed(2)}%
                    </td>
                    <td style={{ padding: '12px 14px', textAlign: 'center', color: '#f59e0b', fontWeight: 800 }}>
                      {((hardEns.test_metrics?.f1_score ?? hardEns.test_metrics?.f1 ?? 0) * 100).toFixed(2)}%
                    </td>
                    <td style={{ padding: '12px 14px', textAlign: 'center', color: '#ec4899', fontWeight: 800 }}>
                      {((hardEns.test_metrics?.roc_auc ?? 0) * 100).toFixed(2)}%
                    </td>
                  </tr>
                );
              })()}
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

      {/* Section 4: Academic FAQ Accordion */}
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
