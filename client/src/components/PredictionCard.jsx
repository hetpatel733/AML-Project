import React, { useState } from 'react';
import { 
  CheckCircle2, 
  AlertOctagon, 
  BrainCircuit, 
  Clock, 
  Copy, 
  Check, 
  RotateCcw, 
  Share2, 
  Sparkles,
  Info,
  Layers,
  ChevronRight
} from 'lucide-react';
import { Link } from 'react-router-dom';
import ConfidenceMeter from './ConfidenceMeter';
import GeminiInsights from './GeminiInsights';

    navigator.clipboard.writeText(textToCopy);
    setCopied(true);
    setTimeout(() => setCopied(false), 2500);
  };

  return (
    <div className={`card prediction-result-card ${isFake ? 'result-fake' : 'result-real'}`}>
      {/* Top Banner Ribbon */}
      <div className="result-ribbon" style={{ backgroundColor: colorMeta.hex }}>
        <div className="result-ribbon-content">
          <Sparkles size={16} />
          <span>Natural Language Processing Classification Completed</span>
          {isMock && <span className="mock-badge">Client ML Simulation Active</span>}
        </div>
      </div>

      <div className="result-body">
        {/* Main Verdict Header */}
        <div className="verdict-section">
          <div className="verdict-icon-container" style={{ backgroundColor: colorMeta.bg, borderColor: colorMeta.border, color: colorMeta.hex }}>
            {isFake ? <AlertOctagon size={48} /> : <CheckCircle2 size={48} />}
          </div>

          <div className="verdict-info">
            <span className="verdict-label">Classification Verdict</span>
            <div className="verdict-badge-row">
              <span className={`verdict-badge ${isFake ? 'badge-verdict-fake' : 'badge-verdict-real'}`}>
                {prediction.prediction} NEWS
              </span>
              <span className="verdict-model-tag">
                <BrainCircuit size={14} />
                <span>{prediction.model || 'TF-IDF Classifier'}</span>
              </span>
            </div>
            <p className="verdict-summary-text">
              {isFake 
                ? 'High probability of deceptive or unverified linguistic patterns identified.' 
                : 'Linguistic structure indicates standard verified factual journalism.'}
            </p>
          </div>
        </div>

        {/* Confidence Meter Component */}
        <div className="result-meter-wrapper">
          <ConfidenceMeter 
            confidence={prediction.confidence} 
            prediction={prediction.prediction} 
          />
        </div>

        {/* Article Reviewed Summary */}
        <div className="article-preview-box">
          <div className="article-preview-header">
            <span className="preview-tag">Target Headline</span>
            <div className="preview-timestamp">
              <Clock size={13} />
              <span>{formatDate(prediction.createdAt || new Date())}</span>
            </div>
          </div>
          <h4 className="preview-title">"{prediction.title}"</h4>
          {prediction.text && (
            <p className="preview-snippet">
              {prediction.text.length > 220 
                ? `${prediction.text.substring(0, 220)}...` 
                : prediction.text}
            </p>
          )}
        </div>

        {/* NLP Model Explanation */}
        <div className="nlp-insights-card">
          <div className="insights-header">
            <Info size={16} className="text-primary" />
            <span className="insights-title">NLP Feature Analysis &amp; Reasoning</span>
          </div>
          <p className="insights-text">
            {prediction.explanation || (
              isFake
                ? 'TF-IDF weighted vectors detected atypical word frequency distributions, clickbait patterns, and excessive sensational adjectives.'
                : 'TF-IDF weighted vectors reflect objective lexical syntax, coherent entity references, and consistent topic distribution.'
            )}
          </p>
          
          <div className="insights-tags-list">
            <span className="insight-tag"><Layers size={12} /> TF-IDF N-grams (1,2)</span>
            <span className="insight-tag"><BrainCircuit size={12} /> Stop-words Filtered</span>
            <span className="insight-tag"><CheckCircle2 size={12} /> Lemmatized Tokens</span>
          </div>
        </div>

        {/* Gemini AI Insights Section */}
        {prediction.geminiInsights && (
          <GeminiInsights insights={prediction.geminiInsights} />
        )}

        {/* Action Controls */}
        <div className="result-actions-footer">
          <button 
            type="button" 
            className="btn-secondary btn-copy" 
            onClick={handleCopy}
          >
            {copied ? <Check size={16} className="text-success" /> : <Copy size={16} />}
            <span>{copied ? 'Copied Summary!' : 'Copy Summary'}</span>
          </button>

          <div className="result-actions-right">
            <Link to="/history" className="btn-secondary">
              <span>View In History</span>
              <ChevronRight size={15} />
            </Link>

            {onReset && (
              <button 
                type="button" 
                className="btn-primary" 
                onClick={onReset}
              >
                <RotateCcw size={16} />
                <span>Analyze Another Article</span>
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default PredictionCard;
