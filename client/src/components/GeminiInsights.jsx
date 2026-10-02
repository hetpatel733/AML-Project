import React, { useState } from 'react';
import { Sparkles, AlertTriangle, CheckCircle, Shield, TrendingUp, ChevronDown, ChevronUp } from 'lucide-react';

const GeminiInsights = ({ insights }) => {
  const [expanded, setExpanded] = useState(true);

  if (!insights) return null;

  // If Gemini is disabled
  if (!insights.enabled) {
    return (
      <div className="gemini-insights-card disabled">
        <div className="insights-header">
          <Sparkles size={18} className="icon-gemini" />
          <h4>Gemini AI Insights</h4>
          <span className="badge-disabled">Not Configured</span>
        </div>
        <p className="insights-message">{insights.message}</p>
      </div>
    );
  }

  // If there was an error
  if (insights.error) {
    return (
      <div className="gemini-insights-card error">
        <div className="insights-header">
          <Sparkles size={18} className="icon-gemini" />
          <h4>Gemini AI Insights</h4>
          <span className="badge-error">Error</span>
        </div>
        <p className="insights-message">{insights.message}</p>
      </div>
    );
  }

  const getScoreColor = (score) => {
    if (score >= 75) return '#10b981';
    if (score >= 50) return '#f59e0b';
    return '#ef4444';
  };

  return (
    <div className="gemini-insights-card">
      <div className="insights-header clickable" onClick={() => setExpanded(!expanded)}>
        <Sparkles size={18} className="icon-gemini" />
        <h4>Gemini AI Credibility Analysis</h4>
        <span className="badge-gemini">Powered by Google AI</span>
        {expanded ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
      </div>

      {expanded && (
        <div className="insights-body">
          {/* Credibility Score */}
          {insights.credibilityScore !== null && (
            <div className="credibility-score-section">
              <div className="score-header">
                <Shield size={20} />
                <span>Credibility Score</span>
              </div>
              <div className="score-meter">
                <div 
                  className="score-fill" 
                  style={{ 
                    width: `${insights.credibilityScore}%`,
                    backgroundColor: getScoreColor(insights.credibilityScore)
                  }}
                />
              </div>
              <div className="score-value" style={{ color: getScoreColor(insights.credibilityScore) }}>
                {insights.credibilityScore}/100
              </div>
            </div>
          )}

          {/* Red Flags */}
          {insights.redFlags && insights.redFlags.length > 0 && (
            <div className="insight-section red-flags">
              <div className="section-title">
                <AlertTriangle size={16} />
                <span>Warning Signs Detected</span>
              </div>
              <ul className="flags-list">
                {insights.redFlags.map((flag, idx) => (
                  <li key={idx}>
                    <span className="flag-bullet">•</span>
                    {flag}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Fact Check Suggestions */}
          {insights.factCheckSuggestions && insights.factCheckSuggestions.length > 0 && (
            <div className="insight-section fact-checks">
              <div className="section-title">
                <CheckCircle size={16} />
                <span>Claims to Verify</span>
              </div>
              <ol className="fact-check-list">
                {insights.factCheckSuggestions.slice(0, 3).map((claim, idx) => (
                  <li key={idx}>{claim}</li>
                ))}
              </ol>
            </div>
          )}

          {/* Source Analysis */}
          {insights.sourceAnalysis && (
            <div className="insight-section source-analysis">
              <div className="section-title">
                <TrendingUp size={16} />
                <span>Source Assessment</span>
              </div>
              <p className="analysis-text">{insights.sourceAnalysis}</p>
            </div>
          )}

          {/* Verification Tips */}
          {insights.verificationTips && insights.verificationTips.length > 0 && (
            <div className="insight-section verification-tips">
              <div className="section-title">
                <Shield size={16} />
                <span>How to Verify</span>
              </div>
              <ul className="tips-list">
                {insights.verificationTips.slice(0, 3).map((tip, idx) => (
                  <li key={idx}>
                    <span className="tip-number">{idx + 1}</span>
                    {tip}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default GeminiInsights;
