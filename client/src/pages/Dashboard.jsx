import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import StatisticsCard from '../components/StatisticsCard';
import PredictionDistribution from '../charts/PredictionDistribution';
import PredictionTrendChart from '../charts/PredictionTrendChart';
import ModelPerformanceChart from '../charts/ModelPerformanceChart';
import PredictionHistory from '../components/PredictionHistory';
import LoadingSpinner from '../components/LoadingSpinner';
import { getPredictionStats, getAnalytics, getBenchmarks, deletePrediction } from '../services/api';
import { 
  ShieldAlert, 
  ShieldCheck, 
  Layers, 
  Activity, 
  RefreshCw, 
  ArrowRight, 
  TrendingUp, 
  BarChart3, 
  PieChart, 
  FlaskConical, 
  Award, 
  Database, 
  Cpu, 
  BookOpen, 
  CheckCircle2,
  FileCheck,
  ImageIcon
} from 'lucide-react';
import { formatConfidence } from '../utils/helpers';
import { useDataset } from '../context/DatasetContext';

export const Dashboard = () => {
  const { selectedDataset } = useDataset();
  const [stats, setStats] = useState(null);
  const [analytics, setAnalytics] = useState(null);
  const [benchmark, setBenchmark] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [selectedFigureTab, setSelectedFigureTab] = useState('accuracy');

  const fetchData = async () => {
    try {
      const [statsRes, analyticsRes, benchmarkRes] = await Promise.all([
        getPredictionStats(selectedDataset),
        getAnalytics(selectedDataset),
        getBenchmarks(selectedDataset).catch(() => ({ data: null }))
      ]);
      setStats(statsRes.data);
      setAnalytics(analyticsRes.data);
      setBenchmark(benchmarkRes.data);
    } catch (err) {
      console.error('Error loading dashboard data:', err);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [selectedDataset]);

  const handleRefresh = () => {
    setRefreshing(true);
    fetchData();
  };

  const handleDelete = async (id) => {
    try {
      await deletePrediction(id);
      await fetchData();
    } catch (err) {
      console.error('Error deleting prediction:', err);
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <LoadingSpinner message="Loading live dashboard..." />
      </div>
    );
  }

  const fakeCount = stats?.fakeCount || 0;
  const realCount = stats?.realCount || 0;
  const total = stats?.totalPredictions || 0;
  const avgConf = stats?.avgConfidence || 0;

  // Derive dynamic metrics from benchmark.json
  const modelsList = benchmark?.models || analytics?.modelPerformance || [];
  const topModel = modelsList.length > 0 
    ? [...modelsList].sort((a, b) => ((b.metrics?.accuracy ?? b.accuracy ?? 0) - (a.metrics?.accuracy ?? a.accuracy ?? 0)))[0]
    : null;
  const topAccuracyVal = topModel?.metrics?.accuracy ?? topModel?.accuracy ?? 0;
  const topModelName = topModel?.name ?? topModel?.model ?? 'Top Classifier';

  const datasetName = benchmark?.dataset?.name || selectedDataset.toUpperCase();
  const trainCount = benchmark?.splits?.training?.count ?? (selectedDataset === 'isot' ? 31280 : 10240);
  const testCount = benchmark?.splits?.testing?.count ?? (selectedDataset === 'isot' ? 7820 : 2560);
  const totalRecords = benchmark?.dataset?.totalRecords ?? (trainCount + testCount);
  const vectorizerInfo = benchmark?.dataset?.preprocessing 
    ? `TF-IDF (${benchmark.dataset.preprocessing.ngramRange ? benchmark.dataset.preprocessing.ngramRange.join('-') : '1-2'} n-grams, ${benchmark.dataset.preprocessing.maxFeatures ? `${(benchmark.dataset.preprocessing.maxFeatures / 1000)}k` : '10k'} features)`
    : 'TF-IDF (1,2-grams • 10k)';

  return (
    <div className="page-container dashboard-page">
      {/* Dashboard Top Header */}
      <div className="page-header dashboard-header-flex">
        <div className="page-header-content">
          <div className="page-badge">
            <Activity size={14} />
            <span>Overview &bull; {datasetName} Dataset</span>
          </div>
          <h1 className="page-title">Dashboard</h1>
          <p className="page-subtitle">
            Real-time classification statistics, empirical model benchmarks, and evaluation telemetry.
          </p>
        </div>

        <div className="header-actions-group">
          <button 
            type="button" 
            className="btn-secondary btn-icon-text" 
            onClick={handleRefresh}
            disabled={refreshing}
          >
            <RefreshCw size={15} className={refreshing ? 'animate-spin' : ''} />
            <span>{refreshing ? 'Refreshing...' : 'Refresh'}</span>
          </button>

          <Link to="/experiments" className="btn-secondary">
            <BookOpen size={16} />
            <span>Benchmark Details</span>
          </Link>

          <Link to="/simulation" className="btn-primary">
            <FlaskConical size={16} />
            <span>Simulation Lab</span>
          </Link>
        </div>
      </div>

      {/* Dynamic Benchmark KPI Highlight Strip */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
        gap: '16px',
        marginBottom: '24px'
      }}>
        <div className="card" style={{
          background: 'linear-gradient(135deg, rgba(30, 41, 59, 0.85), rgba(15, 23, 42, 0.85))',
          border: '1px solid rgba(59, 130, 246, 0.35)',
          padding: '18px 20px',
          borderRadius: '12px'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <span style={{ fontSize: '11px', fontWeight: 700, color: '#60a5fa', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Top Test Accuracy ({datasetName})
            </span>
            <Award size={18} className="text-primary" />
          </div>
          <div style={{ fontSize: '26px', fontWeight: 800, color: '#f8fafc', fontFamily: 'monospace' }}>
            {(topAccuracyVal * 100).toFixed(2)}%
          </div>
          <div style={{ fontSize: '12px', color: '#94a3b8', marginTop: '4px', display: 'flex', alignItems: 'center', gap: '4px' }}>
            <CheckCircle2 size={13} className="text-success" />
            <span>Model: <strong>{topModelName}</strong></span>
          </div>
        </div>

        <div className="card" style={{
          background: 'linear-gradient(135deg, rgba(30, 41, 59, 0.85), rgba(15, 23, 42, 0.85))',
          border: '1px solid rgba(16, 185, 129, 0.35)',
          padding: '18px 20px',
          borderRadius: '12px'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <span style={{ fontSize: '11px', fontWeight: 700, color: '#34d399', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Dataset Split (80 / 20)
            </span>
            <Database size={18} style={{ color: '#34d399' }} />
          </div>
          <div style={{ fontSize: '26px', fontWeight: 800, color: '#f8fafc', fontFamily: 'monospace' }}>
            {totalRecords.toLocaleString()} <span style={{ fontSize: '13px', fontWeight: 500, color: '#94a3b8' }}>records</span>
          </div>
          <div style={{ fontSize: '12px', color: '#94a3b8', marginTop: '4px' }}>
            Train: <strong>{trainCount.toLocaleString()}</strong> | Test: <strong>{testCount.toLocaleString()}</strong>
          </div>
        </div>

        <div className="card" style={{
          background: 'linear-gradient(135deg, rgba(30, 41, 59, 0.85), rgba(15, 23, 42, 0.85))',
          border: '1px solid rgba(139, 92, 246, 0.35)',
          padding: '18px 20px',
          borderRadius: '12px'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <span style={{ fontSize: '11px', fontWeight: 700, color: '#a78bfa', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Evaluated Classifiers
            </span>
            <Cpu size={18} style={{ color: '#a78bfa' }} />
          </div>
          <div style={{ fontSize: '26px', fontWeight: 800, color: '#f8fafc', fontFamily: 'monospace' }}>
            {modelsList.length || 5} <span style={{ fontSize: '13px', fontWeight: 500, color: '#94a3b8' }}>supervised</span>
          </div>
          <div style={{ fontSize: '12px', color: '#94a3b8', marginTop: '4px' }}>
            {vectorizerInfo}
          </div>
        </div>

        <div className="card" style={{
          background: 'linear-gradient(135deg, rgba(30, 41, 59, 0.85), rgba(15, 23, 42, 0.85))',
          border: '1px solid rgba(245, 158, 11, 0.35)',
          padding: '18px 20px',
          borderRadius: '12px'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <span style={{ fontSize: '11px', fontWeight: 700, color: '#fbbf24', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Preprocessing Pipeline
            </span>
            <FileCheck size={18} style={{ color: '#fbbf24' }} />
          </div>
          <div style={{ fontSize: '18px', fontWeight: 700, color: '#f8fafc', marginTop: '4px' }}>
            Negation-Preserving
          </div>
          <div style={{ fontSize: '12px', color: '#94a3b8', marginTop: '6px' }}>
            WordNet Lemmatization + Sublinear TF-IDF
          </div>
        </div>
      </div>

      {/* 4 Database Verification Volume Statistics Cards */}
      <div className="stats-grid">
        <StatisticsCard 
          title="Total User Verifications"
          value={total}
          subtitle="Cumulative logged predictions"
          icon={Layers}
          trend="Real-time log"
          trendDirection="up"
          color="primary"
          badge="Live Volume"
        />

        <StatisticsCard 
          title="Fake Articles Flagged"
          value={fakeCount}
          subtitle={`${stats?.fakePercentage || 0}% of all processed inputs`}
          icon={ShieldAlert}
          trend="High Risk"
          trendDirection="neutral"
          color="danger"
          badge="Deceptive"
        />

        <StatisticsCard 
          title="Real News Corroborated"
          value={realCount}
          subtitle={`${stats?.realPercentage || 0}% of all processed inputs`}
          icon={ShieldCheck}
          trend="Verified Credible"
          trendDirection="up"
          color="success"
          badge="Credible"
        />

        <StatisticsCard 
          title="Mean Confidence"
          value={formatConfidence(avgConf)}
          subtitle="Average classifier certainty"
          icon={TrendingUp}
          trend="Reliability"
          trendDirection="up"
          color="amber"
          badge="Certainty"
        />
      </div>

      {/* Row: Dynamic 5-Model Benchmark Performance Comparison */}
      <div className="analytics-section" style={{ marginBottom: '24px' }}>
        <div className="card chart-card">
          <div className="card-header-styled">
            <div className="card-header-left">
              <div className="card-icon-badge text-primary">
                <BarChart3 size={18} />
              </div>
              <div>
                <h3 className="card-title">Live 5-Model Benchmark Comparison ({datasetName} Hold-Out Test Set)</h3>
                <p className="card-subtitle">
                  Empirical Accuracy, Precision, Recall, and F1-Score dynamically loaded from <code>benchmark.json</code>
                </p>
              </div>
            </div>
            <span className="badge-tag">20% Test Split</span>
          </div>
          <div className="chart-wrapper">
            <ModelPerformanceChart 
              data={analytics?.modelPerformance || []} 
              height={310} 
            />
          </div>
        </div>
      </div>

      {/* D3.js Charts Row */}
      <div className="dashboard-charts-grid">
        {/* Chart 1: Donut Distribution */}
        <div className="card chart-card">
          <div className="card-header-styled">
            <div className="card-header-left">
              <div className="card-icon-badge text-primary">
                <PieChart size={18} />
              </div>
              <div>
                <h3 className="card-title">Authenticity Class Distribution</h3>
                <p className="card-subtitle">Ratio of verified Real vs detected Fake news</p>
              </div>
            </div>
          </div>
          <div className="chart-wrapper">
            <PredictionDistribution 
              fakeCount={fakeCount} 
              realCount={realCount} 
              height={260}
            />
          </div>
        </div>

        {/* Chart 2: Daily Trends */}
        <div className="card chart-card">
          <div className="card-header-styled">
            <div className="card-header-left">
              <div className="card-icon-badge text-primary">
                <BarChart3 size={18} />
              </div>
              <div>
                <h3 className="card-title">Temporal Prediction Trend</h3>
                <p className="card-subtitle">Time series of daily classification verifications</p>
              </div>
            </div>
          </div>
          <div className="chart-wrapper">
            <PredictionTrendChart 
              data={analytics?.trendData || []} 
              height={260}
            />
          </div>
        </div>
      </div>

      {/* Generated Visual Evaluation Assets Preview */}
      <div className="card" style={{ marginBottom: '24px', background: 'rgba(30, 41, 59, 0.7)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', flexWrap: 'wrap', gap: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <ImageIcon size={20} className="text-primary" />
            <div>
              <h3 style={{ margin: 0, fontSize: '18px', fontWeight: 700, color: '#f8fafc' }}>
                Generated Visual Evaluation Assets ({datasetName})
              </h3>
              <p style={{ margin: '4px 0 0', fontSize: '13px', color: '#94a3b8' }}>
                High-resolution evaluation charts generated directly from the 20% test hold-out partition.
              </p>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '8px' }}>
            <button
              onClick={() => setSelectedFigureTab('accuracy')}
              className={selectedFigureTab === 'accuracy' ? 'btn-primary btn-sm' : 'btn-secondary btn-sm'}
            >
              Accuracy Comparison
            </button>
            <button
              onClick={() => setSelectedFigureTab('confusion')}
              className={selectedFigureTab === 'confusion' ? 'btn-primary btn-sm' : 'btn-secondary btn-sm'}
            >
              Confusion Matrices
            </button>
            <button
              onClick={() => setSelectedFigureTab('pr')}
              className={selectedFigureTab === 'pr' ? 'btn-primary btn-sm' : 'btn-secondary btn-sm'}
            >
              Precision-Recall Curves
            </button>
            <button
              onClick={() => setSelectedFigureTab('recall_threshold')}
              className={selectedFigureTab === 'recall_threshold' ? 'btn-primary btn-sm' : 'btn-secondary btn-sm'}
            >
              Recall Sensitivity Curves
            </button>
          </div>
        </div>

        <div style={{ textAlign: 'center', padding: '12px', background: 'rgba(15, 23, 42, 0.6)', borderRadius: '8px', border: '1px solid rgba(255, 255, 255, 0.05)' }}>
          {selectedFigureTab === 'accuracy' && (
            <div>
              <img 
                src={`/api/figures/${selectedDataset}/${selectedDataset}_accuracy_comparison.png`} 
                alt={`${datasetName} Accuracy Comparison`} 
                style={{ maxWidth: '100%', maxHeight: '420px', borderRadius: '8px', margin: '0 auto', boxShadow: '0 4px 20px rgba(0,0,0,0.4)' }}
                onError={(e) => {
                  e.target.style.display = 'none';
                  e.target.nextSibling.style.display = 'block';
                }}
              />
              <div style={{ display: 'none', color: '#94a3b8', padding: '20px' }}>
                Figure image available under <code>ml/results/figures/{selectedDataset}/</code>
              </div>
            </div>
          )}

          {selectedFigureTab === 'confusion' && (
            <div>
              <img 
                src={`/api/figures/${selectedDataset}/${selectedDataset}_all_confusion_matrices.png`} 
                alt={`${datasetName} Confusion Matrices Grid`} 
                style={{ maxWidth: '100%', maxHeight: '460px', borderRadius: '8px', margin: '0 auto', boxShadow: '0 4px 20px rgba(0,0,0,0.4)' }}
                onError={(e) => {
                  e.target.style.display = 'none';
                  e.target.nextSibling.style.display = 'block';
                }}
              />
              <div style={{ display: 'none', color: '#94a3b8', padding: '20px' }}>
                Figure image available under <code>ml/results/figures/{selectedDataset}/</code>
              </div>
            </div>
          )}

          {selectedFigureTab === 'pr' && (
            <div>
              <img 
                src={`/api/figures/${selectedDataset}/${selectedDataset}_precision_recall_curves.png`} 
                alt={`${datasetName} Precision Recall Curves`} 
                style={{ maxWidth: '100%', maxHeight: '420px', borderRadius: '8px', margin: '0 auto', boxShadow: '0 4px 20px rgba(0,0,0,0.4)' }}
                onError={(e) => {
                  e.target.style.display = 'none';
                  e.target.nextSibling.style.display = 'block';
                }}
              />
              <div style={{ display: 'none', color: '#94a3b8', padding: '20px' }}>
                Figure image available under <code>ml/results/figures/{selectedDataset}/</code>
              </div>
            </div>
          )}

          {selectedFigureTab === 'recall_threshold' && (
            <div>
              <img 
                src={`/api/figures/${selectedDataset}/${selectedDataset}_recall_vs_threshold.png`} 
                alt={`${datasetName} Recall vs Threshold Curves`} 
                style={{ maxWidth: '100%', maxHeight: '420px', borderRadius: '8px', margin: '0 auto', boxShadow: '0 4px 20px rgba(0,0,0,0.4)' }}
                onError={(e) => {
                  e.target.style.display = 'none';
                  e.target.nextSibling.style.display = 'block';
                }}
              />
              <div style={{ display: 'none', color: '#94a3b8', padding: '20px' }}>
                Figure image available under <code>ml/results/figures/{selectedDataset}/</code>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Recent Predictions Section */}
      <div className="dashboard-recent-section">
        <div className="section-header-row">
          <div>
            <h3 className="section-title-sm">Recent Verifications</h3>
            <p className="section-subtitle-sm">Latest articles processed by the NLP inference pipeline</p>
          </div>
          <Link to="/history" className="view-all-link">
            <span>View Full Archive</span>
            <ArrowRight size={14} />
          </Link>
        </div>

        <div className="card table-card">
          <PredictionHistory 
            predictions={stats?.recentPredictions || []} 
            onDelete={handleDelete}
            showSearch={false}
          />
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
