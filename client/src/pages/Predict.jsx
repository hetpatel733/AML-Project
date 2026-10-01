import React from 'react';
import NewsInput from '../components/NewsInput';
import PredictionCard from '../components/PredictionCard';
import LoadingSpinner from '../components/LoadingSpinner';
import { usePrediction } from '../hooks/usePrediction';
import { 
  Sparkles, 
  HelpCircle, 
  CheckCircle2, 
  AlertTriangle, 
  BrainCircuit, 
  FileText,
  Layers,
  Clock
} from 'lucide-react';

export const Predict = () => {
  const { prediction, loading, error, isMock, analyzeNews, clearPrediction } = usePrediction();

  return (
    <div className="page-container predict-page">
      {/* Page Header */}
      <div className="page-header">
        <div className="page-header-content">
          <div className="page-badge">
            <Sparkles size={14} />
            <span>Interactive NLP Classifier</span>
          </div>
          <h1 className="page-title">News Authenticity Verification</h1>
          <p className="page-subtitle">
            Submit a news headline and article body to evaluate vocabulary patterns, semantic coherence, and authenticity probabilities.
          </p>
        </div>
      </div>

      <div className="predict-layout-grid">
        {/* Left / Main Column: Input or Result */}
        <div className="predict-main-column">
          {loading ? (
            <div className="card loading-card-container">
              <LoadingSpinner 
                message="Vectorizing Text &amp; Calculating Class Probabilities..."
                submessage="Running TF-IDF N-gram feature matrix through trained supervised classification models."
              />
            </div>
          ) : prediction ? (
            <PredictionCard 
              prediction={prediction} 
              onReset={clearPrediction}
              isMock={isMock}
            />
          ) : (
            <NewsInput 
              onAnalyze={analyzeNews}
              loading={loading}
              error={error}
            />
          )}
        </div>

        {/* Right / Auxiliary Column: Academic Guidelines & Model Insights */}
        <div className="predict-sidebar-column">
          {/* Card 1: NLP Guidelines */}
          <div className="card sidebar-info-card">
            <div className="sidebar-card-header">
              <HelpCircle size={18} className="text-primary" />
              <h4 className="sidebar-card-heading">Optimization Tips</h4>
            </div>
            <ul className="guidelines-list">
              <li>
                <strong>Include Headline &amp; Body:</strong> Headlines alone may lack sufficient token density for optimal TF-IDF vectorization.
              </li>
              <li>
                <strong>Unedited Original Text:</strong> Retain original phrasing and punctuation to enable accurate linguistic anomaly detection.
              </li>
              <li>
                <strong>Minimum Length:</strong> Articles with 40+ words produce significantly more stable probabilistic confidence scores.
              </li>
            </ul>
          </div>

          {/* Card 2: Model Architecture Specs */}
          <div className="card sidebar-info-card">
            <div className="sidebar-card-header">
              <BrainCircuit size={18} className="text-primary" />
              <h4 className="sidebar-card-heading">Active Pipeline Specs</h4>
            </div>
            <div className="specs-list">
              <div className="spec-row">
                <span className="spec-label">Feature Extractor</span>
                <span className="spec-val font-mono">TF-IDF (1,2-grams)</span>
              </div>
              <div className="spec-row">
                <span className="spec-label">Vocabulary Size</span>
                <span className="spec-val font-mono">10,000 features</span>
              </div>
              <div className="spec-row">
                <span className="spec-label">Primary Classifier</span>
                <span className="spec-val font-mono">Passive Aggressive</span>
              </div>
              <div className="spec-row">
                <span className="spec-label">Decision Boundary</span>
                <span className="spec-val font-mono">Hinge Loss (C=1.0)</span>
              </div>
            </div>
          </div>

          {/* Card 3: Model Reliability Disclaimer */}
          <div className="card sidebar-info-card disclaimer-card">
            <div className="sidebar-card-header">
              <AlertTriangle size={18} className="text-amber" />
              <h4 className="sidebar-card-heading">Academic Note</h4>
            </div>
            <p className="disclaimer-text">
              Natural Language Processing models classify statistical associations with prior training corpora. 
              Always cross-verify high-consequence information with primary institutional sources.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Predict;
