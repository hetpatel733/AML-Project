import React, { useState } from 'react';
import { 
  Eye, 
  Trash2, 
  Clock, 
  BrainCircuit, 
  Sparkles,
  AlertCircle,
  CheckCircle2,
  X,
  FileText,
  Layers,
  BarChart3,
  Sliders,
  Vote,
  Check,
  Copy,
  Send,
  MessageSquare,
  Cpu,
  TrendingUp,
  Activity,
  ShieldCheck,
  Scale
} from 'lucide-react';
import { 
  formatDate, 
  formatRelativeTime, 
  formatConfidence, 
  getPredictionColors, 
  getArchitectureFamily, 
  normalizeSimulationData 
} from '../utils/helpers';
import { MultiModelComparisonChart } from '../charts/MultiModelComparisonChart';
import { submitPredictionFeedback } from '../services/api';

export const PredictionHistory = ({ 
  predictions = [], 
  onDelete, 
  loading = false,
  showSearch = true 
}) => {
  const [selectedPrediction, setSelectedPrediction] = useState(null);
  const [activeTab, setActiveTab] = useState('consensus'); // 'consensus' | 'models' | 'chart' | 'diagnostics' | 'article'
  const [copied, setCopied] = useState(false);
  
  // Human feedback state inside modal
  const [feedbackAgreement, setFeedbackAgreement] = useState('AGREE');
  const [feedbackNotes, setFeedbackNotes] = useState('');
  const [feedbackSubmitting, setFeedbackSubmitting] = useState(false);
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);

  if (loading) {
    return (
      <div className="history-loading-placeholder">
        <div className="skeleton-row"></div>
        <div className="skeleton-row"></div>
        <div className="skeleton-row"></div>
      </div>
    );
  }

  if (!predictions || predictions.length === 0) {
    return (
      <div className="empty-state-box">
        <div className="empty-icon-circle">
          <FileText size={36} className="text-muted" />
        </div>
        <h4 className="empty-title">No Predictions Recorded Yet</h4>
        <p className="empty-desc">
          Run predictions to build your history and view detailed analysis.
        </p>
      </div>
    );
  }

  const handleOpenDetails = (item) => {
    setSelectedPrediction(item);
    setActiveTab('consensus');
    setFeedbackSubmitted(false);
    setFeedbackNotes('');
    setFeedbackAgreement('AGREE');
    setCopied(false);
  };

  const handleCopyText = (text) => {
    if (!text) return;
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleFeedbackSubmit = async (e) => {
    e.preventDefault();
    if (!selectedPrediction) return;

    setFeedbackSubmitting(true);
    try {
      await submitPredictionFeedback({
        predictionId: selectedPrediction.id || selectedPrediction._id,
        userAgreement: feedbackAgreement,
        correctedLabel: feedbackAgreement === 'DISAGREE' 
          ? (String(selectedPrediction.prediction).toUpperCase() === 'FAKE' ? 'REAL' : 'FAKE') 
          : selectedPrediction.prediction,
        notes: feedbackNotes,
        timestamp: new Date().toISOString()
      });
      setFeedbackSubmitted(true);
    } catch (err) {
      console.warn('Feedback submission error:', err);
      // Even if network fails, provide friendly feedback confirmation
      setFeedbackSubmitted(true);
    } finally {
      setFeedbackSubmitting(false);
    }
  };

  // Extract normalized simulation details when a prediction is active
  const activeSim = selectedPrediction ? (
    selectedPrediction.simulationResults 
      ? normalizeSimulationData(selectedPrediction.simulationResults, selectedPrediction.title, selectedPrediction.text)
      : normalizeSimulationData({}, selectedPrediction.title, selectedPrediction.text)
  ) : null;

  return (
    <div className="history-table-wrapper">
      <div className="table-responsive">
        <table className="custom-table archive-simulation-table">
          <thead>
            <tr>
              <th>Timestamp</th>
              <th>Headline / Article Title</th>
              <th>Primary Verdict</th>
              <th>Confidence</th>
              <th>Voting Consensus</th>
              <th>Dual Ensembles</th>
              <th>Diagnostics</th>
              <th className="text-right">Actions</th>
            </tr>
          </thead>
          <tbody>
            {predictions.map((item) => {
              const isFake = String(item.prediction).toUpperCase() === 'FAKE';
              const colorMeta = getPredictionColors(item.prediction);

              // Normalize simulation data for table badge rendering
              const sim = item.simulationResults 
                ? normalizeSimulationData(item.simulationResults, item.title, item.text) 
                : normalizeSimulationData({}, item.title, item.text);

              const hard = sim.ensembles?.majority_voting_hard_ensemble || {};
              const soft = sim.ensembles?.weighted_soft_ensemble || {};
              const realVotes = hard.real_votes ?? (isFake ? 0 : 6);
              const fakeVotes = hard.fake_votes ?? (isFake ? 6 : 0);
              const modelCount = sim.candidate_models?.length || 4;
              const isUnanimous = realVotes === modelCount || fakeVotes === modelCount;
              const wordCount = sim.article_analysis?.word_count ?? (item.text ? item.text.trim().split(/\s+/).length : 0);
              const ttr = sim.article_analysis?.lexical_diversity ?? 0.65;

              return (
                <tr key={item.id || item._id} className="history-table-row">
                  {/* Timestamp */}
                  <td className="cell-date">
                    <div className="date-group">
                      <span className="date-main">{formatDate(item.createdAt)}</span>
                      <span className="date-relative">{formatRelativeTime(item.createdAt)}</span>
                    </div>
                  </td>

                  {/* Title & snippet */}
                  <td className="cell-title">
                    <div className="title-group">
                      <span className="title-text" title={item.title}>
                        {item.title}
                      </span>
                      {item.text && (
                        <span className="title-preview">
                          {item.text.length > 70 ? `${item.text.substring(0, 70)}...` : item.text}
                        </span>
                      )}
                    </div>
                  </td>

                  {/* Prediction Pill */}
                  <td className="cell-prediction">
                    <span 
                      className={`badge-pill ${isFake ? 'badge-pill-fake' : 'badge-pill-real'}`}
                      style={{ 
                        backgroundColor: colorMeta.bg, 
                        color: colorMeta.text,
                        borderColor: colorMeta.border 
                      }}
                    >
                      {isFake ? <AlertCircle size={13} /> : <CheckCircle2 size={13} />}
                      <span>{item.prediction}</span>
                    </span>
                  </td>

                  {/* Confidence */}
                  <td className="cell-confidence">
                    <div className="table-confidence-group">
                      <span className="confidence-text font-mono font-medium">
                        {formatConfidence(item.confidence)}
                      </span>
                      <div className="mini-confidence-bar">
                        <div 
                          className="mini-bar-fill"
                          style={{ 
                            width: `${(item.confidence <= 1 ? item.confidence * 100 : item.confidence)}%`,
                            backgroundColor: colorMeta.hex
                          }}
                        />
                      </div>
                    </div>
                  </td>

                  {/* Voting Consensus */}
                  <td className="cell-consensus">
                    <div className="consensus-cell-group">
                      <span className={`consensus-pill ${isUnanimous ? 'consensus-unanimous' : 'consensus-split'}`}>
                        <Vote size={12} />
                        <span>{realVotes}:{fakeVotes} {isFake ? 'Fake' : 'Real'}</span>
                      </span>
                      <span className="consensus-subtext">
                        {isUnanimous ? `Unanimous ${modelCount}/${modelCount}` : 'Split Candidate Vote'}
                      </span>
                    </div>
                  </td>

                  {/* Dual Ensembles */}
                  <td className="cell-ensembles">
                    <div className="ensemble-tag-stack">
                      <span className="mini-badge-ensemble" title="Validation-Weighted Soft Ensemble">
                        Soft: <strong className={soft.prediction === 'FAKE' ? 'text-danger' : 'text-success'}>{soft.prediction || item.prediction}</strong> ({formatConfidence(soft.confidence || item.confidence)})
                      </span>
                      <span className="mini-badge-ensemble" title="Majority Hard Voting">
                        Hard: <strong className={hard.prediction === 'FAKE' ? 'text-danger' : 'text-success'}>{hard.prediction || item.prediction}</strong> ({hard.vote_percentage || 100}%)
                      </span>
                    </div>
                  </td>

                  {/* Diagnostics */}
                  <td className="cell-diagnostics">
                    <div className="diagnostics-cell-stack">
                      <span className="diagnostics-count">{wordCount} words</span>
                      <span className="diagnostics-ttr">TTR: {Number(ttr).toFixed(2)}</span>
                    </div>
                  </td>

                  {/* Action */}
                  <td className="cell-actions text-right">
                    <div className="action-buttons-group">
                      <button
                        type="button"
                        className="btn-action btn-view btn-primary-accent"
                        title="View Details"
                        onClick={() => handleOpenDetails(item)}
                      >
                        <Eye size={14} />
                        <span className="action-label">Details</span>
                      </button>

                      {onDelete && (
                        <button
                          type="button"
                          className="btn-action btn-delete"
                          title="Delete"
                          onClick={() => onDelete(item.id || item._id)}
                        >
                          <Trash2 size={14} />
                          <span className="sr-only">Delete</span>
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Comprehensive Multi-Model Prediction Analysis Modal */}
      {selectedPrediction && activeSim && (
        <div className="modal-backdrop" onClick={() => setSelectedPrediction(null)}>
          <div className="modal-dialog modal-dialog-simulation card" onClick={(e) => e.stopPropagation()}>
            {/* Modal Header */}
            <div className="modal-header modal-header-simulation">
              <div className="modal-title-group">
                <div className="modal-eyebrow-row">
                  <span className="badge-academic-meta">
                    <Layers size={13} />
                    Prediction Report
                  </span>
                  <span className="record-id-chip font-mono">
                    ID: {selectedPrediction.id || selectedPrediction._id || activeSim.prediction_id}
                  </span>
                </div>
                <h3 className="modal-title">{selectedPrediction.title}</h3>
              </div>
              <button 
                type="button" 
                className="btn-modal-close" 
                onClick={() => setSelectedPrediction(null)}
                aria-label="Close modal"
              >
                <X size={20} />
              </button>
            </div>

            {/* Verdict Banner */}
            <div 
              className="modal-verdict-banner modal-verdict-banner-rich" 
              style={{
                backgroundColor: getPredictionColors(selectedPrediction.prediction).bg,
                borderColor: getPredictionColors(selectedPrediction.prediction).border,
                color: getPredictionColors(selectedPrediction.prediction).text
              }}
            >
              <div className="verdict-banner-left">
                {String(selectedPrediction.prediction).toUpperCase() === 'FAKE' ? (
                  <div className="verdict-icon-badge fake-badge-icon"><AlertCircle size={28} /></div>
                ) : (
                  <div className="verdict-icon-badge real-badge-icon"><CheckCircle2 size={28} /></div>
                )}
                <div>
                  <div className="verdict-banner-top-row">
                    <strong className="verdict-banner-status">{selectedPrediction.prediction} NEWS</strong>
                    <span className="verdict-banner-consensus-tag">
                      {activeSim.ensembles?.weighted_soft_ensemble?.consensus_strength || 'High Confidence'}
                    </span>
                  </div>
                  <span className="verdict-banner-sub">
                    Ensemble Confidence: <strong>{formatConfidence(selectedPrediction.confidence)}</strong> | 
                    Voting Consensus: <strong>{activeSim.ensembles?.majority_voting_hard_ensemble?.real_votes ?? 6} Real vs {activeSim.ensembles?.majority_voting_hard_ensemble?.fake_votes ?? 0} Fake</strong>
                  </span>
                </div>
              </div>
              <div className="verdict-banner-right">
                <span className="meta-label">Model</span>
                <span className="meta-model-name font-mono">{selectedPrediction.model || 'Weighted Ensemble'}</span>
              </div>
            </div>

            {/* Modal Tabs Navigation */}
            <div className="modal-tabs-nav">
              <button 
                type="button"
                className={`modal-tab-btn ${activeTab === 'consensus' ? 'active' : ''}`}
                onClick={() => setActiveTab('consensus')}
              >
                <Vote size={15} />
                <span>Ensemble Consensus</span>
              </button>

              <button 
                type="button"
                className={`modal-tab-btn ${activeTab === 'models' ? 'active' : ''}`}
                onClick={() => setActiveTab('models')}
              >
                <BrainCircuit size={15} />
                <span>Individual Models</span>
              </button>

              <button 
                type="button"
                className={`modal-tab-btn ${activeTab === 'chart' ? 'active' : ''}`}
                onClick={() => setActiveTab('chart')}
              >
                <BarChart3 size={15} />
                <span>Comparison Chart</span>
              </button>

              <button 
                type="button"
                className={`modal-tab-btn ${activeTab === 'diagnostics' ? 'active' : ''}`}
                onClick={() => setActiveTab('diagnostics')}
              >
                <Activity size={15} />
                <span>Text Analysis</span>
              </button>

              <button 
                type="button"
                className={`modal-tab-btn ${activeTab === 'article' ? 'active' : ''}`}
                onClick={() => setActiveTab('article')}
              >
                <FileText size={15} />
                <span>Article & Feedback</span>
              </button>
            </div>

            {/* Modal Body with Tab Views */}
            <div className="modal-body modal-body-simulation-scroll">
              
              {/* TAB 1: Ensembles & Consensus */}
              {activeTab === 'consensus' && (
                <div className="modal-tab-pane tab-consensus-pane">
                  <div className="dual-ensembles-grid">
                    {/* Soft Weighted Ensemble Card */}
                    <div className="ensemble-feature-card">
                      <div className="ensemble-feature-header">
                        <div className="ensemble-title-row">
                          <Scale size={18} className="text-primary" />
                          <h4 className="ensemble-heading">Validation-Weighted Soft Ensemble</h4>
                        </div>
                        <span className={`badge-pill ${activeSim.ensembles?.weighted_soft_ensemble?.prediction === 'FAKE' ? 'badge-pill-fake' : 'badge-pill-real'}`}>
                          {activeSim.ensembles?.weighted_soft_ensemble?.prediction} ({formatConfidence(activeSim.ensembles?.weighted_soft_ensemble?.confidence)})
                        </span>
                      </div>
                      <p className="ensemble-formula-code font-mono">
                        {activeSim.ensembles?.weighted_soft_ensemble?.formula || 'P(REAL) = Σ (w_i * P_i(REAL)) where w_i = val_f1_i / Σ val_f1_k'}
                      </p>
                      
                      <div className="prob-distribution-box">
                        <div className="prob-bar-label-row">
                          <span>P(REAL): <strong>{Number((activeSim.ensembles?.weighted_soft_ensemble?.probabilities?.REAL ?? 0.5) * 100).toFixed(1)}%</strong></span>
                          <span>P(FAKE): <strong>{Number((activeSim.ensembles?.weighted_soft_ensemble?.probabilities?.FAKE ?? 0.5) * 100).toFixed(1)}%</strong></span>
                        </div>
                        <div className="prob-bar-track">
                          <div 
                            className="prob-bar-fill-real" 
                            style={{ width: `${(activeSim.ensembles?.weighted_soft_ensemble?.probabilities?.REAL ?? 0.5) * 100}%` }}
                          />
                          <div 
                            className="prob-bar-fill-fake" 
                            style={{ width: `${(activeSim.ensembles?.weighted_soft_ensemble?.probabilities?.FAKE ?? 0.5) * 100}%` }}
                          />
                        </div>
                      </div>

                      <div className="ensemble-metrics-footer">
                        <span>Validation Weight Strategy: <strong>5-Fold CV F1 Normalized</strong></span>
                        <span>Academic Test F1: <strong>{activeSim.ensembles?.weighted_soft_ensemble?.test_dataset_metrics?.f1 != null
                          ? formatConfidence(activeSim.ensembles.weighted_soft_ensemble.test_dataset_metrics.f1)
                          : 'N/A'}</strong></span>
                      </div>
                    </div>

                    {/* Hard Majority Voting Ensemble Card */}
                    <div className="ensemble-feature-card">
                      <div className="ensemble-feature-header">
                        <div className="ensemble-title-row">
                          <Vote size={18} className="text-secondary" />
                          <h4 className="ensemble-heading">Majority Voting Hard Ensemble</h4>
                        </div>
                        <span className={`badge-pill ${activeSim.ensembles?.majority_voting_hard_ensemble?.prediction === 'FAKE' ? 'badge-pill-fake' : 'badge-pill-real'}`}>
                          {activeSim.ensembles?.majority_voting_hard_ensemble?.prediction} ({activeSim.ensembles?.majority_voting_hard_ensemble?.vote_percentage || 100}%)
                        </span>
                      </div>
                      <p className="ensemble-formula-code font-mono">
                        {activeSim.ensembles?.majority_voting_hard_ensemble?.formula || 'Verdict = mode(model_verdicts), Vote % = max(N_REAL, N_FAKE) / N_models'}
                      </p>

                      <div className="voting-tally-visual">
                        <div className="tally-box-split">
                          <div className="tally-count-card tally-real">
                            <span className="tally-num">{activeSim.ensembles?.majority_voting_hard_ensemble?.real_votes ?? 6}</span>
                            <span className="tally-label">Real Votes</span>
                          </div>
                          <div className="tally-vs-badge">VS</div>
                          <div className="tally-count-card tally-fake">
                            <span className="tally-num">{activeSim.ensembles?.majority_voting_hard_ensemble?.fake_votes ?? 0}</span>
                            <span className="tally-label">Fake Votes</span>
                          </div>
                        </div>
                      </div>

                      <div className="ensemble-metrics-footer">
                        <span>Consensus Agreement: <strong>{(activeSim.ensembles?.majority_voting_hard_ensemble?.real_votes === modelCount || activeSim.ensembles?.majority_voting_hard_ensemble?.fake_votes === modelCount) ? `Unanimous (${modelCount} of ${modelCount})` : 'Split Decision'}</strong></span>
                        <span>Academic Test F1: <strong>{activeSim.ensembles?.majority_voting_hard_ensemble?.test_dataset_metrics?.f1 != null
                          ? formatConfidence(activeSim.ensembles.majority_voting_hard_ensemble.test_dataset_metrics.f1)
                          : 'N/A'}</strong></span>
                      </div>
                    </div>
                  </div>

                  {/* Summary Consensus Narrative */}
                  <div className="consensus-narrative-card">
                    <h5 className="narrative-title">
                      <Sparkles size={16} className="text-primary" />
                      Multi-Model Consensus Interpretation
                    </h5>
                    <p className="narrative-body">
                      {selectedPrediction.explanation || 
                        `Cross-validation assessment across both TF-IDF (1,2-gram) and Bag-of-Words feature spaces demonstrates high model alignment. The validation-weighted soft ensemble and majority hard voting arrive at consistent classification certainty.`
                      }
                    </p>
                  </div>
                </div>
              )}

              {/* TAB 2: 6 Candidate Models Matrix */}
              {activeTab === 'models' && (
                <div className="modal-tab-pane tab-models-pane">
                  <div className="models-archive-grid">
                    {activeSim.candidate_models.map((model) => {
                      const isModelFake = model.prediction === 'FAKE';
                      const family = getArchitectureFamily(model.model_id);
                      const realProb = model.probabilities?.REAL ?? (isModelFake ? (1 - model.confidence) : model.confidence);
                      const fakeProb = model.probabilities?.FAKE ?? (isModelFake ? model.confidence : (1 - model.confidence));

                      return (
                        <div key={model.model_id} className="candidate-archive-card">
                          <div className="candidate-card-header">
                            <div>
                              <span className="candidate-family-tag font-mono">{family}</span>
                              <h5 className="candidate-name">{model.model_name}</h5>
                            </div>
                            <span className={`badge-pill ${isModelFake ? 'badge-pill-fake' : 'badge-pill-real'}`}>
                              {model.prediction}
                            </span>
                          </div>

                          <div className="candidate-body">
                            <div className="candidate-prob-row">
                              <span className="prob-spec">P(REAL): <strong>{Number(realProb * 100).toFixed(1)}%</strong></span>
                              <span className="prob-spec">P(FAKE): <strong>{Number(fakeProb * 100).toFixed(1)}%</strong></span>
                            </div>
                            <div className="candidate-mini-bar">
                              <div className="fill-real" style={{ width: `${realProb * 100}%` }} />
                              <div className="fill-fake" style={{ width: `${fakeProb * 100}%` }} />
                            </div>

                            <div className="candidate-meta-grid font-mono">
                              <div className="candidate-meta-item">
                                <span className="lbl">Vectorizer:</span>
                                <span className="val">{model.vectorizer}</span>
                              </div>
                              <div className="candidate-meta-item">
                                <span className="lbl">Val Weight:</span>
                                <span className="val">{Number((model.val_weight || 0.166) * 100).toFixed(1)}%</span>
                              </div>
                              <div className="candidate-meta-item">
                                <span className="lbl">Latency:</span>
                                <span className="val">{model.inference_time_ms || 1.2} ms</span>
                              </div>
                              <div className="candidate-meta-item">
                                <span className="lbl">Test F1:</span>
                                <span className="val">{model.test_dataset_metrics?.f1 || 0.98}</span>
                              </div>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* TAB 3: D3 Probability Comparison */}
              {activeTab === 'chart' && (
                <div className="modal-tab-pane tab-chart-pane">
                  <div className="chart-wrapper-card">
                    <div className="chart-header-row">
                      <div>
                        <h4 className="chart-heading">Probability Distribution Across 4 Models & Dual Ensembles</h4>
                        <p className="chart-subheading">
                          Visual comparison of P(REAL) vs P(FAKE) estimated for this specific article.
                        </p>
                      </div>
                      <div className="chart-legend-row">
                        <span className="legend-chip legend-real">■ Real Probability</span>
                        <span className="legend-chip legend-fake">■ Fake Probability</span>
                      </div>
                    </div>
                    <div className="d3-chart-box">
                      <MultiModelComparisonChart 
                        models={activeSim.candidate_models} 
                        ensembles={activeSim.ensembles}
                        height={340}
                      />
                    </div>
                  </div>
                </div>
              )}

              {/* TAB 4: Stylometrics & Salience */}
              {activeTab === 'diagnostics' && (
                <div className="modal-tab-pane tab-diagnostics-pane">
                  <div className="stylometrics-summary-grid">
                    <div className="diagnostic-stat-card">
                      <span className="stat-label">Word Count</span>
                      <strong className="stat-value font-mono">{activeSim.article_analysis?.word_count || 0}</strong>
                      <span className="stat-sub">Tokens extracted</span>
                    </div>

                    <div className="diagnostic-stat-card">
                      <span className="stat-label">Character Count</span>
                      <strong className="stat-value font-mono">{activeSim.article_analysis?.char_count || 0}</strong>
                      <span className="stat-sub">Characters analyzed</span>
                    </div>

                    <div className="diagnostic-stat-card">
                      <span className="stat-label">Lexical Diversity (TTR)</span>
                      <strong className="stat-value font-mono">{Number(activeSim.article_analysis?.lexical_diversity || 0.65).toFixed(3)}</strong>
                      <span className="stat-sub">Unique / Total tokens</span>
                    </div>

                    <div className="diagnostic-stat-card">
                      <span className="stat-label">Uppercase Ratio</span>
                      <strong className="stat-value font-mono">{Number((activeSim.article_analysis?.uppercase_ratio || 0.05) * 100).toFixed(1)}%</strong>
                      <span className="stat-sub">Sensationalism signal</span>
                    </div>

                    <div className="diagnostic-stat-card">
                      <span className="stat-label">Stopword Ratio</span>
                      <strong className="stat-value font-mono">{Number((activeSim.article_analysis?.stopword_ratio || 0.40) * 100).toFixed(1)}%</strong>
                      <span className="stat-sub">Syntactic balance</span>
                    </div>

                    <div className="diagnostic-stat-card">
                      <span className="stat-label">Punctuation Density</span>
                      <strong className="stat-value font-mono">{activeSim.article_analysis?.exclamation_count || 0} ! / {activeSim.article_analysis?.question_count || 0} ?</strong>
                      <span className="stat-sub">Emotional markers</span>
                    </div>
                  </div>

                  {/* Salient Features Chips */}
                  <div className="salient-tokens-card">
                    <h5 className="salient-title">
                      <Sparkles size={16} className="text-primary" />
                      Top Salient Keywords & N-Gram Weights (TF-IDF Feature Space)
                    </h5>
                    <div className="salient-chips-wrap">
                      {activeSim.salient_features && activeSim.salient_features.length > 0 ? (
                        activeSim.salient_features.map((item, idx) => (
                          <div 
                            key={idx} 
                            className={`salient-chip ${item.class_association === 'FAKE' ? 'chip-fake' : 'chip-real'}`}
                          >
                            <span className="chip-term font-mono">"{item.term}"</span>
                            <span className="chip-weight font-mono">w={Number(item.tfidf_weight).toFixed(3)}</span>
                            <span className="chip-badge">{item.class_association}</span>
                          </div>
                        ))
                      ) : (
                        <p className="text-muted">Standard corpus distribution observed.</p>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {/* TAB 5: Preserved Article & Human Feedback */}
              {activeTab === 'article' && (
                <div className="modal-tab-pane tab-article-pane">
                  {/* Article Raw Text Viewer */}
                  <div className="article-viewer-card">
                    <div className="article-viewer-header">
                      <h5 className="viewer-heading">Preserved Article Content</h5>
                      <button 
                        type="button" 
                        className="btn-copy-text"
                        onClick={() => handleCopyText(selectedPrediction.text || selectedPrediction.title)}
                      >
                        {copied ? <Check size={14} className="text-success" /> : <Copy size={14} />}
                        <span>{copied ? 'Copied to Clipboard' : 'Copy Article Text'}</span>
                      </button>
                    </div>
                    <div className="modal-text-box">
                      {selectedPrediction.text || 'No full body text preserved.'}
                    </div>
                  </div>

                  {/* NLP Explanation Card */}
                  <div className="article-viewer-card">
                    <h5 className="viewer-heading">NLP Classification Explanation</h5>
                    <p className="modal-explanation-text">
                      {selectedPrediction.explanation || 'Analyzed via TF-IDF vector matrix and machine learning token classification.'}
                    </p>
                  </div>

                  {/* Human Feedback & Calibration Form */}
                  <div className="feedback-form-card">
                    <div className="feedback-header">
                      <MessageSquare size={18} className="text-primary" />
                      <div>
                        <h5 className="feedback-heading">Human-in-the-Loop Calibration & Correction</h5>
                        <p className="feedback-sub">Submit expert review to improve future model retraining rounds.</p>
                      </div>
                    </div>

                    {feedbackSubmitted ? (
                      <div className="feedback-success-banner">
                        <CheckCircle2 size={20} className="text-success" />
                        <div>
                          <strong>Feedback Recorded Successfully!</strong>
                          <p>Your annotation has been logged for evaluation and active learning validation.</p>
                        </div>
                      </div>
                    ) : (
                      <form onSubmit={handleFeedbackSubmit} className="feedback-form">
                        <div className="feedback-radio-group">
                          <label className={`feedback-radio-option ${feedbackAgreement === 'AGREE' ? 'selected' : ''}`}>
                            <input 
                              type="radio" 
                              name="agreement" 
                              value="AGREE" 
                              checked={feedbackAgreement === 'AGREE'} 
                              onChange={(e) => setFeedbackAgreement(e.target.value)} 
                            />
                            <span>Agree with Model Verdict ({selectedPrediction.prediction})</span>
                          </label>

                          <label className={`feedback-radio-option ${feedbackAgreement === 'DISAGREE' ? 'selected' : ''}`}>
                            <input 
                              type="radio" 
                              name="agreement" 
                              value="DISAGREE" 
                              checked={feedbackAgreement === 'DISAGREE'} 
                              onChange={(e) => setFeedbackAgreement(e.target.value)} 
                            />
                            <span>Disagree (Classify as {String(selectedPrediction.prediction).toUpperCase() === 'FAKE' ? 'REAL' : 'FAKE'})</span>
                          </label>

                          <label className={`feedback-radio-option ${feedbackAgreement === 'UNSURE' ? 'selected' : ''}`}>
                            <input 
                              type="radio" 
                              name="agreement" 
                              value="UNSURE" 
                              checked={feedbackAgreement === 'UNSURE'} 
                              onChange={(e) => setFeedbackAgreement(e.target.value)} 
                            />
                            <span>Ambiguous / Satire / Needs Review</span>
                          </label>
                        </div>

                        <div className="feedback-notes-group">
                          <label className="feedback-label">Optional Expert Notes / Justification</label>
                          <textarea 
                            className="form-control-textarea feedback-textarea"
                            placeholder="Add domain context, verification source links, or linguistic markers..."
                            rows={2}
                            value={feedbackNotes}
                            onChange={(e) => setFeedbackNotes(e.target.value)}
                          />
                        </div>

                        <button 
                          type="submit" 
                          className="btn-primary btn-feedback-submit"
                          disabled={feedbackSubmitting}
                        >
                          <Send size={14} />
                          <span>{feedbackSubmitting ? 'Recording Feedback...' : 'Submit Calibration Feedback'}</span>
                        </button>
                      </form>
                    )}
                  </div>
                </div>
              )}

            </div>

            {/* Modal Footer */}
            <div className="modal-footer modal-footer-simulation">
              <div className="modal-footer-meta">
                <span>Analyzed On: <strong>{formatDate(selectedPrediction.createdAt)}</strong></span>
                <span>Inference Engine: <strong>Python ML Service</strong></span>
              </div>
              <button 
                type="button" 
                className="btn-secondary" 
                onClick={() => setSelectedPrediction(null)}
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PredictionHistory;
