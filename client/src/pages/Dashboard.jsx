import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import StatisticsCard from '../components/StatisticsCard';
import PredictionDistribution from '../charts/PredictionDistribution';
import PredictionTrendChart from '../charts/PredictionTrendChart';
import PredictionHistory from '../components/PredictionHistory';
import LoadingSpinner from '../components/LoadingSpinner';
import { getPredictionStats, getAnalytics, deletePrediction } from '../services/api';
import { 
  ShieldAlert, 
  ShieldCheck, 
  Layers, 
  Activity, 
  Search, 
  RefreshCw, 
  ArrowRight,
  TrendingUp,
  BarChart3,
  PieChart,
  FlaskConical
} from 'lucide-react';
import { formatConfidence } from '../utils/helpers';
import { useDataset } from '../context/DatasetContext';

export const Dashboard = () => {
  const { selectedDataset } = useDataset();
  const [stats, setStats] = useState(null);
  const [analytics, setAnalytics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const fetchData = async () => {
    try {
      const [statsRes, analyticsRes] = await Promise.all([
        getPredictionStats(selectedDataset),
        getAnalytics(selectedDataset)
      ]);
      setStats(statsRes.data);
      setAnalytics(analyticsRes.data);
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
        <LoadingSpinner message="Loading dashboard..." />
      </div>
    );
  }

  const fakeCount = stats?.fakeCount || 0;
  const realCount = stats?.realCount || 0;
  const total = stats?.totalPredictions || 0;
  const avgConf = stats?.avgConfidence || 0;

  return (
    <div className="page-container dashboard-page">
      {/* Dashboard Top Header */}
      <div className="page-header dashboard-header-flex">
        <div className="page-header-content">
          <div className="page-badge">
            <Activity size={14} />
            <span>Overview</span>
          </div>
          <h1 className="page-title">Dashboard</h1>
          <p className="page-subtitle">
            Real-time classification statistics and model performance insights.
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
            <span>{refreshing ? 'Refreshing...' : 'Refresh Data'}</span>
          </button>

          <Link to="/simulation" className="btn-primary">
            <FlaskConical size={16} />
            <span>Simulation Lab</span>
          </Link>
        </div>
      </div>

      {/* 4 Metric Statistics Cards Grid */}
      <div className="stats-grid">
        <StatisticsCard 
          title="Total Analyzed Articles"
          value={total}
          subtitle="Cumulative verification logs"
          icon={Layers}
          trend="+12% this week"
          trendDirection="up"
          color="primary"
          badge="Total Volume"
        />

        <StatisticsCard 
          title="Fake News Identified"
          value={fakeCount}
          subtitle={`${stats?.fakePercentage || 0}% of all analyzed inputs`}
          icon={ShieldAlert}
          trend="Flagged deceptive"
          trendDirection="neutral"
          color="danger"
          badge="High Risk"
        />

        <StatisticsCard 
          title="Real News Verified"
          value={realCount}
          subtitle={`${stats?.realPercentage || 0}% of all analyzed inputs`}
          icon={ShieldCheck}
          trend="Corroborated sources"
          trendDirection="up"
          color="success"
          badge="Credible"
        />

        <StatisticsCard 
          title="Average Confidence"
          value={formatConfidence(avgConf)}
          subtitle="Mean model certainty score"
          icon={TrendingUp}
          trend="Optimal reliability"
          trendDirection="up"
          color="amber"
          badge="Metric"
        />
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
                <p className="card-subtitle">D3.js proportion of verified Real vs detected Fake news</p>
              </div>
            </div>
          </div>
          <div className="chart-wrapper">
            <PredictionDistribution 
              fakeCount={fakeCount} 
              realCount={realCount} 
              height={270}
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
                <p className="card-subtitle">D3.js multi-line time series of daily classification counts</p>
              </div>
            </div>
          </div>
          <div className="chart-wrapper">
            <PredictionTrendChart 
              data={analytics?.trendData || []} 
              height={270}
            />
          </div>
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
