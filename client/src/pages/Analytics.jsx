import React, { useState, useEffect } from 'react';
import ConfidenceChart from '../charts/ConfidenceChart';
import ModelPerformanceChart from '../charts/ModelPerformanceChart';
import PredictionTrendChart from '../charts/PredictionTrendChart';
import PredictionDistribution from '../charts/PredictionDistribution';
import LoadingSpinner from '../components/LoadingSpinner';
import { getAnalytics, getPredictionStats } from '../services/api';
import { 
  BarChart3, 
  PieChart, 
  Activity, 
  Grid, 
  Layers, 
  Award, 
  CheckCircle2, 
  AlertCircle, 
  Info,
  TrendingUp,
  Percent
} from 'lucide-react';
import { formatConfidence } from '../utils/helpers';
import { useDataset } from '../context/DatasetContext';

export const Analytics = () => {
  const { selectedDataset } = useDataset();
  const [analytics, setAnalytics] = useState(null);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchAllAnalytics = async () => {
      setLoading(true);
      try {
        const [anRes, stRes] = await Promise.all([
          getAnalytics(selectedDataset),
          getPredictionStats(selectedDataset)
        ]);
        setAnalytics(anRes.data);
        setStats(stRes.data);
      } catch (err) {
        console.error('Error loading analytics:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchAllAnalytics();
  }, [selectedDataset]);

  if (loading) {
    return (
      <div className="page-container">
        <LoadingSpinner message="Loading analytics..." />
      </div>
    );
  }

  const cm = analytics?.confusionMatrix || {
    truePositive: 0,
    falsePositive: 0,
    trueNegative: 0,
    falseNegative: 0,
    totalSamples: 0
  };
  const confusionMatrices = analytics?.confusionMatrices?.length
    ? analytics.confusionMatrices
    : [cm];

  const calculatedAccuracy = cm.totalSamples ? ((cm.truePositive + cm.trueNegative) / cm.totalSamples) * 100 : 0;
  const calculatedPrecision = cm.truePositive + cm.falsePositive ? (cm.truePositive / (cm.truePositive + cm.falsePositive)) * 100 : 0;
  const calculatedRecall = cm.truePositive + cm.falseNegative ? (cm.truePositive / (cm.truePositive + cm.falseNegative)) * 100 : 0;
  const calculatedF1 = calculatedPrecision + calculatedRecall
    ? (2 * (calculatedPrecision * calculatedRecall)) / (calculatedPrecision + calculatedRecall)
    : 0;

  return (
    <div className="page-container analytics-page">
      {/* Page Header */}
      <div className="page-header">
        <div className="page-header-content">
          <div className="page-badge">
            <BarChart3 size={14} />
            <span>Performance Analysis &bull; {selectedDataset.toUpperCase()} Dataset</span>
          </div>
          <h1 className="page-title">Analytics</h1>
          <p className="page-subtitle">
            Model performance metrics, comparative benchmarks, and classification accuracy analysis.
          </p>
        </div>
      </div>

      {/* Row 1: Model Benchmark Comparison (D3.js) */}
      <div className="analytics-section">
        <div className="card chart-card">
          <div className="card-header-styled">
            <div className="card-header-left">
              <div className="card-icon-badge text-primary">
                <Award size={18} />
              </div>
              <div>
                <h3 className="card-title">Model Benchmark Comparison</h3>
                <p className="card-subtitle">
                  Performance across Accuracy, Precision, Recall, and F1-Score
                </p>
              </div>
            </div>
            <span className="badge-tag">Test Split</span>
          </div>
          <div className="chart-wrapper">
            <ModelPerformanceChart 
              data={analytics?.modelPerformance || []} 
              height={330} 
            />
          </div>
        </div>
      </div>

      {/* Row 2: Confidence Histogram & Class Ratio */}
      <div className="analytics-grid-two">
        {/* Chart A: Confidence Distribution */}
        <div className="card chart-card">
          <div className="card-header-styled">
            <div className="card-header-left">
              <div className="card-icon-badge text-primary">
                <Activity size={18} />
              </div>
              <div>
                <h3 className="card-title">Confidence Distribution</h3>
                <p className="card-subtitle">Distribution of model prediction confidence levels</p>
              </div>
            </div>
          </div>
          <div className="chart-wrapper">
            <ConfidenceChart 
              data={analytics?.confidenceDistribution || []} 
              height={270}
            />
          </div>
        </div>

        {/* Chart B: Fake vs Real Distribution */}
        <div className="card chart-card">
          <div className="card-header-styled">
            <div className="card-header-left">
              <div className="card-icon-badge text-primary">
                <PieChart size={18} />
              </div>
              <div>
                <h3 className="card-title">Classification Distribution</h3>
                <p className="card-subtitle">Proportion of fake vs real predictions</p>
              </div>
            </div>
          </div>
          <div className="chart-wrapper">
            <PredictionDistribution 
              fakeCount={stats?.fakeCount || 0} 
              realCount={stats?.realCount || 0} 
              height={270}
            />
          </div>
        </div>
      </div>

      {/* Row 3: Confusion Matrices for Every Trained Model */}
      <div className="analytics-section">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(520px, 1fr))', gap: '18px', width: '100%' }}>
          {confusionMatrices.map((matrix) => {
            const accuracy = matrix.totalSamples ? ((matrix.truePositive + matrix.trueNegative) / matrix.totalSamples) * 100 : 0;
            const precision = matrix.truePositive + matrix.falsePositive ? (matrix.truePositive / (matrix.truePositive + matrix.falsePositive)) * 100 : 0;
            const recall = matrix.truePositive + matrix.falseNegative ? (matrix.truePositive / (matrix.truePositive + matrix.falseNegative)) * 100 : 0;
            const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0;

            return (
              <div className="card confusion-matrix-card" key={matrix.modelId || matrix.model}>
                <div className="card-header-styled">
                  <div className="card-header-left">
                    <div className="card-icon-badge text-primary">
                      <Grid size={18} />
                    </div>
                    <div>
                      <h3 className="card-title">{matrix.model || 'Confusion Matrix'}</h3>
                      <p className="card-subtitle">Classification accuracy on {matrix.totalSamples} test samples</p>
                    </div>
                  </div>
                </div>

                <div className="matrix-content-layout">
            {/* Visual 2x2 Matrix Grid */}
            <div className="matrix-grid-wrapper">
              <div className="matrix-table-container">
                <div className="matrix-labels-top">
                  <span>PREDICTED FAKE</span>
                  <span>PREDICTED REAL</span>
                </div>

                <div className="matrix-body-row">
                  <div className="matrix-label-side">
                    <span>ACTUAL FAKE</span>
                  </div>
                  <div className="matrix-cell matrix-tp">
                    <span className="matrix-cell-tag">True Negative (TN)</span>
                    <span className="matrix-cell-val font-mono">{matrix.trueNegative}</span>
                    <span className="matrix-cell-sub">Correctly Classified Fake</span>
                  </div>
                  <div className="matrix-cell matrix-fn">
                    <span className="matrix-cell-tag">False Positive (FP)</span>
                    <span className="matrix-cell-val font-mono">{matrix.falsePositive}</span>
                    <span className="matrix-cell-sub">Real Classified as Fake</span>
                  </div>
                </div>

                <div className="matrix-body-row">
                  <div className="matrix-label-side">
                    <span>ACTUAL REAL</span>
                  </div>
                  <div className="matrix-cell matrix-fp">
                    <span className="matrix-cell-tag">False Negative (FN)</span>
                    <span className="matrix-cell-val font-mono">{matrix.falseNegative}</span>
                    <span className="matrix-cell-sub">Real Missed as Fake</span>
                  </div>
                  <div className="matrix-cell matrix-tn">
                    <span className="matrix-cell-tag">True Positive (TP)</span>
                    <span className="matrix-cell-val font-mono">{matrix.truePositive}</span>
                    <span className="matrix-cell-sub">Correctly Classified Real</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Derived Mathematical Metrics */}
            <div className="matrix-metrics-panel">
              <h4 className="metrics-panel-title">Derived Classification Metrics</h4>
              
              <div className="metric-derived-item">
                <div className="metric-header">
                  <span className="metric-name">Overall Accuracy</span>
                  <span className="metric-val font-mono text-primary">{accuracy.toFixed(2)}%</span>
                </div>
                <div className="metric-formula font-mono">(TP + TN) / Total</div>
              </div>

              <div className="metric-derived-item">
                <div className="metric-header">
                  <span className="metric-name">Precision (Real Class)</span>
                  <span className="metric-val font-mono text-success">{precision.toFixed(2)}%</span>
                </div>
                <div className="metric-formula font-mono">TP / (TP + FP)</div>
              </div>

              <div className="metric-derived-item">
                <div className="metric-header">
                  <span className="metric-name">Recall / Sensitivity</span>
                  <span className="metric-val font-mono text-purple">{recall.toFixed(2)}%</span>
                </div>
                <div className="metric-formula font-mono">TP / (TP + FN)</div>
              </div>

              <div className="metric-derived-item">
                <div className="metric-header">
                  <span className="metric-name">F1 Score (Harmonic Mean)</span>
                  <span className="metric-val font-mono text-amber">{f1.toFixed(2)}%</span>
                </div>
                <div className="metric-formula font-mono">2 &times; (Precision &times; Recall) / (Precision + Recall)</div>
              </div>
            </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default Analytics;
