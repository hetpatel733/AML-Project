import React, { useState } from 'react';
import { 
  Send, 
  Trash2, 
  Sparkles, 
  FileText, 
  Heading, 
  AlertCircle,
  HelpCircle,
  CheckCircle2
} from 'lucide-react';
import { countWords, SAMPLE_NEWS_ARTICLES } from '../utils/helpers';

export const NewsInput = ({ onAnalyze, loading = false, error = null }) => {
  const [title, setTitle] = useState('');
  const [text, setText] = useState('');
  const [validationError, setValidationError] = useState('');

  const wordCount = countWords(text);
  const charCount = text.length;

  const handleSubmit = (e) => {
    e.preventDefault();
    setValidationError('');

    if (!title.trim()) {
      setValidationError('Please enter a news headline or title.');
      return;
    }

    if (!text.trim()) {
      setValidationError('Please paste or write the news article text content.');
      return;
    }

    if (wordCount < 10) {
      setValidationError('For reliable NLP extraction, article content must be at least 10 words.');
      return;
    }

    onAnalyze({ title, text });
  };

  const handleClear = () => {
    setTitle('');
    setText('');
    setValidationError('');
  };

  const loadSample = (sample) => {
    setTitle(sample.title);
    setText(sample.text);
    setValidationError('');
  };

  return (
    <div className="card news-input-card">
      <div className="card-header-styled">
        <div className="card-header-left">
          <div className="card-icon-badge">
            <FileText size={20} className="text-primary" />
          </div>
          <div>
            <h3 className="card-title">Article Verification Input</h3>
            <p className="card-subtitle">
              Enter the news headline and body text to execute natural language feature extraction
            </p>
          </div>
        </div>

        {/* Quick Sample Selector */}
        <div className="sample-buttons-wrapper">
          <span className="sample-label">
            <Sparkles size={14} className="text-amber" />
            <span>Load Demo Sample:</span>
          </span>
          <div className="sample-buttons-list">
            <button
              type="button"
              className="btn-sample btn-sample-real"
              onClick={() => loadSample(SAMPLE_NEWS_ARTICLES[0])}
              disabled={loading}
            >
              <CheckCircle2 size={13} />
              <span>Real News</span>
            </button>
            <button
              type="button"
              className="btn-sample btn-sample-fake"
              onClick={() => loadSample(SAMPLE_NEWS_ARTICLES[1])}
              disabled={loading}
            >
              <AlertCircle size={13} />
              <span>Fake News</span>
            </button>
          </div>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="news-input-form">
        {/* Validation or API Error banner */}
        {(validationError || error) && (
          <div className="alert-box alert-error">
            <AlertCircle size={18} />
            <div className="alert-content">
              <span>{validationError || error}</span>
            </div>
          </div>
        )}

        {/* Title Input */}
        <div className="form-group">
          <label htmlFor="news-title" className="form-label">
            <Heading size={16} />
            <span>News Headline / Title</span>
            <span className="required-mark">*</span>
          </label>
          <input
            id="news-title"
            type="text"
            className="form-control form-control-title"
            placeholder="e.g. Scientists Discover Water Ice Reserves in Lunar Caves..."
            value={title}
            onChange={(e) => {
              setTitle(e.target.value);
              if (validationError) setValidationError('');
            }}
            disabled={loading}
            maxLength={300}
          />
          <div className="form-helper-text">
            <span>Enter the exact published headline</span>
            <span className="char-indicator">{title.length}/300 chars</span>
          </div>
        </div>

        {/* Article Textarea */}
        <div className="form-group">
          <label htmlFor="news-text" className="form-label">
            <FileText size={16} />
            <span>Article Body Content</span>
            <span className="required-mark">*</span>
          </label>
          <textarea
            id="news-text"
            className="form-control form-control-textarea"
            placeholder="Paste the full paragraph or news text here for TF-IDF tokenization and vocabulary weighting..."
            rows={8}
            value={text}
            onChange={(e) => {
              setText(e.target.value);
              if (validationError) setValidationError('');
            }}
            disabled={loading}
          />
          <div className="form-helper-text">
            <span>Linguistic features analyze vocabulary density, sentiment polarity, and syntactic cues</span>
            <span className="stats-pill">
              <strong>{wordCount}</strong> words &bull; <strong>{charCount}</strong> chars
            </span>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="form-actions-bar">
          <button
            type="button"
            className="btn-secondary"
            onClick={handleClear}
            disabled={loading || (!title && !text)}
          >
            <Trash2 size={16} />
            <span>Clear Fields</span>
          </button>

          <button
            type="submit"
            className="btn-primary btn-submit-analyze"
            disabled={loading || !title.trim() || !text.trim()}
          >
            {loading ? (
              <>
                <div className="btn-spinner"></div>
                <span>Analyzing NLP Tokens...</span>
              </>
            ) : (
              <>
                <Send size={16} />
                <span>Analyze News Authenticity</span>
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
};

export default NewsInput;
