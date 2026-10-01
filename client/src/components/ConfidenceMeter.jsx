import React from 'react';
import { formatConfidence } from '../utils/helpers';
import { CheckCircle, AlertTriangle, HelpCircle } from 'lucide-react';

export const ConfidenceMeter = ({ confidence = 0, prediction = 'REAL' }) => {
  // Normalize confidence to 0-100 percentage
  const percent = confidence <= 1 ? confidence * 100 : confidence;
  const isFake = String(prediction).toUpperCase() === 'FAKE';

  let certaintyLevel = 'Moderate';
  let certaintyIcon = HelpCircle;
  let statusColor = '#eab308'; // yellow

  if (percent >= 90) {
    certaintyLevel = 'Extremely High Certainty';
    certaintyIcon = isFake ? AlertTriangle : CheckCircle;
    statusColor = isFake ? '#ef4444' : '#22c55e';
  } else if (percent >= 75) {
    certaintyLevel = 'High Certainty';
    certaintyIcon = isFake ? AlertTriangle : CheckCircle;
    statusColor = isFake ? '#f97316' : '#10b981';
  } else {
    certaintyLevel = 'Moderate / Borderline Certainty';
    certaintyIcon = HelpCircle;
    statusColor = '#eab308';
  }

  const CertaintyIconComponent = certaintyIcon;

  return (
    <div className="confidence-meter-container">
      <div className="confidence-header">
        <div className="confidence-label-group">
          <span className="confidence-title">Classification Confidence</span>
          <div className="confidence-badge" style={{ color: statusColor, borderColor: statusColor + '40', background: statusColor + '15' }}>
            <CertaintyIconComponent size={14} />
            <span>{certaintyLevel}</span>
          </div>
        </div>
        <div className="confidence-score" style={{ color: statusColor }}>
          {formatConfidence(confidence)}
        </div>
      </div>

      {/* Visual Meter Bar */}
      <div className="confidence-bar-track">
        <div 
          className="confidence-bar-fill" 
          style={{ 
            width: `${Math.min(100, Math.max(5, percent))}%`,
            backgroundColor: statusColor,
            boxShadow: `0 0 12px ${statusColor}60`
          }}
        />
      </div>

      {/* Scale Markers */}
      <div className="confidence-markers">
        <span>50% (Uncertain)</span>
        <span>75% (Substantial)</span>
        <span>100% (Definitive)</span>
      </div>

      {/* Class Probabilities Distribution */}
      <div className="probability-breakdown">
        <div className="prob-item">
          <span className="prob-label">P(REAL):</span>
          <span className="prob-val font-mono">
            {isFake ? formatConfidence(1 - (percent / 100)) : formatConfidence(percent / 100)}
          </span>
        </div>
        <div className="prob-item">
          <span className="prob-label">P(FAKE):</span>
          <span className="prob-val font-mono">
            {isFake ? formatConfidence(percent / 100) : formatConfidence(1 - (percent / 100))}
          </span>
        </div>
      </div>
    </div>
  );
};

export default ConfidenceMeter;
