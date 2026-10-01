import React, { useState, useEffect } from 'react';
import PredictionHistory from '../components/PredictionHistory';
import LoadingSpinner from '../components/LoadingSpinner';
import { getPredictionHistory, deletePrediction } from '../services/api';
import { 
  History as HistoryIcon, 
  Search, 
  Filter, 
  ArrowUpDown, 
  ChevronLeft, 
  ChevronRight, 
  RotateCcw,
  Sparkles,
  Layers,
  Vote,
  Download,
  CheckCircle2,
  AlertCircle
} from 'lucide-react';

export const History = () => {
  const [predictions, setPredictions] = useState([]);
  const [total, setTotal] = useState(0);
  const [totalPages, setTotalPages] = useState(1);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);

  // Filter & Search states
  const [search, setSearch] = useState('');
  const [filter, setFilter] = useState('ALL');
  const [consensusFilter, setConsensusFilter] = useState('ALL');
  const [sortBy, setSortBy] = useState('date_desc');

  const fetchHistory = async () => {
    setLoading(true);
    try {
      const res = await getPredictionHistory({
        search,
        filter,
        consensusFilter,
        sortBy,
        page,
        limit: 8
      });
      setPredictions(res.data.predictions || []);
      setTotal(res.data.total || 0);
      setTotalPages(res.data.totalPages || 1);
    } catch (err) {
      console.error('Error fetching prediction history:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, [search, filter, consensusFilter, sortBy, page]);

  const handleResetFilters = () => {
    setSearch('');
    setFilter('ALL');
    setConsensusFilter('ALL');
    setSortBy('date_desc');
    setPage(1);
  };

  const handleDelete = async (id) => {
    await deletePrediction(id);
    fetchHistory();
  };

  const handleExportJSON = () => {
    if (!predictions.length) return;
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(predictions, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", `prediction_archive_export_${Date.now()}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  return (
    <div className="page-container history-page">
      {/* Page Header */}
      <div className="page-header">
        <div className="page-header-content">
          <div className="page-badge">
            <HistoryIcon size={14} />
            <span>Prediction Archive</span>
          </div>
          <h1 className="page-title">Prediction Archive</h1>
          <p className="page-subtitle">
            Browse, inspect, and evaluate past predictions, multi-model consensus, and detailed text metrics.
          </p>
        </div>

        <div className="page-header-actions">
          <button
            type="button"
            className="btn-secondary"
            onClick={handleExportJSON}
            disabled={predictions.length === 0}
            title="Export full historical records and simulation data"
          >
            <Download size={15} />
            <span>Export Archive (JSON)</span>
          </button>
        </div>
      </div>

      {/* Filter and Search Toolbar */}
      <div className="card history-filter-card">
        <div className="filter-toolbar">
          {/* Search Input */}
          <div className="search-input-wrapper">
            <Search size={17} className="search-icon" />
            <input
              type="text"
              className="search-input"
              placeholder="Search by headline, keyword, or model..."
              value={search}
              onChange={(e) => {
                setSearch(e.target.value);
                setPage(1);
              }}
            />
          </div>

          {/* Status Filter Dropdown */}
          <div className="filter-group">
            <label className="filter-label">
              <Filter size={15} />
              <span>Verdict:</span>
            </label>
            <select
              className="filter-select"
              value={filter}
              onChange={(e) => {
                setFilter(e.target.value);
                setPage(1);
              }}
            >
              <option value="ALL">All Verdicts</option>
              <option value="REAL">Real News Only</option>
              <option value="FAKE">Fake News Only</option>
            </select>
          </div>

          {/* Consensus Split Filter Dropdown */}
          <div className="filter-group">
            <label className="filter-label">
              <Vote size={15} />
              <span>Consensus:</span>
            </label>
            <select
              className="filter-select"
              value={consensusFilter}
              onChange={(e) => {
                setConsensusFilter(e.target.value);
                setPage(1);
              }}
            >
              <option value="ALL">All Vote Profiles</option>
              <option value="UNANIMOUS">Unanimous (6/6 Agree)</option>
              <option value="SPLIT">Split Decisions</option>
            </select>
          </div>

          {/* Sort By Dropdown */}
          <div className="filter-group">
            <label className="filter-label">
              <ArrowUpDown size={15} />
              <span>Sort By:</span>
            </label>
            <select
              className="filter-select"
              value={sortBy}
              onChange={(e) => {
                setSortBy(e.target.value);
                setPage(1);
              }}
            >
              <option value="date_desc">Newest First</option>
              <option value="date_asc">Oldest First</option>
              <option value="confidence_desc">Highest Confidence</option>
              <option value="confidence_asc">Lowest Confidence</option>
            </select>
          </div>

          {/* Reset Filters */}
          {(search || filter !== 'ALL' || consensusFilter !== 'ALL' || sortBy !== 'date_desc') && (
            <button 
              type="button" 
              className="btn-secondary btn-sm"
              onClick={handleResetFilters}
            >
              <RotateCcw size={14} />
              <span>Reset</span>
            </button>
          )}
        </div>

        {/* Results Counter Summary */}
        <div className="filter-results-info">
          <span>Showing <strong>{predictions.length}</strong> of <strong>{total}</strong> records</span>
          {filter !== 'ALL' && <span className="active-filter-pill">Verdict: {filter}</span>}
          {consensusFilter !== 'ALL' && <span className="active-filter-pill">Consensus: {consensusFilter}</span>}
        </div>
      </div>

      {/* Table Section */}
      <div className="card table-card history-results-card">
        <PredictionHistory 
          predictions={predictions} 
          onDelete={handleDelete}
          loading={loading}
        />

        {/* Pagination Bar */}
        {totalPages > 1 && (
          <div className="pagination-bar">
            <div className="pagination-info font-mono">
              Page {page} of {totalPages}
            </div>

            <div className="pagination-controls">
              <button
                type="button"
                className="btn-pagination"
                disabled={page <= 1}
                onClick={() => setPage(p => Math.max(1, p - 1))}
              >
                <ChevronLeft size={16} />
                <span>Previous</span>
              </button>

              <div className="pagination-pages">
                {Array.from({ length: totalPages }, (_, i) => i + 1).map(p => (
                  <button
                    key={p}
                    type="button"
                    className={`btn-page-num ${p === page ? 'btn-page-active' : ''}`}
                    onClick={() => setPage(p)}
                  >
                    {p}
                  </button>
                ))}
              </div>

              <button
                type="button"
                className="btn-pagination"
                disabled={page >= totalPages}
                onClick={() => setPage(p => Math.min(totalPages, p + 1))}
              >
                <span>Next</span>
                <ChevronRight size={16} />
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default History;
