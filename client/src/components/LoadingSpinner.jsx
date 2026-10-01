import React from 'react';
import { Loader2, Sparkles, Cpu } from 'lucide-react';

export const LoadingSpinner = ({ 
  size = 32, 
  message = 'Processing Natural Language Tokens...', 
  submessage = 'Applying TF-IDF vectorizer and executing classification model',
  fullScreen = false 
}) => {
  const content = (
    <div className="spinner-wrapper">
      <div className="spinner-animation-container">
        <div className="spinner-ring"></div>
        <div className="spinner-core-icon">
          <Cpu size={Math.max(16, Math.floor(size * 0.6))} className="spinner-cpu-icon" />
        </div>
      </div>
      {message && <p className="spinner-text">{message}</p>}
      {submessage && <p className="spinner-subtext">{submessage}</p>}
    </div>
  );

  if (fullScreen) {
    return <div className="spinner-fullscreen">{content}</div>;
  }

  return content;
};

export default LoadingSpinner;
