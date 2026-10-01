import { useState, useCallback } from 'react';
import { predictNews } from '../services/api';

/**
 * Custom React Hook to manage news prediction logic and UI states
 */
export const usePrediction = () => {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isMock, setIsMock] = useState(false);

  const analyzeNews = useCallback(async ({ title, text }) => {
    if (!title?.trim() || !text?.trim()) {
      setError('Please provide both a news title and article content.');
      return null;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await predictNews({ title: title.trim(), text: text.trim() });
      setPrediction(response.data);
      setIsMock(response.isMock);
      return response.data;
    } catch (err) {
      const message = err.response?.data?.message || err.message || 'An unexpected error occurred during analysis.';
      setError(message);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const clearPrediction = useCallback(() => {
    setPrediction(null);
    setError(null);
  }, []);

  return {
    prediction,
    loading,
    error,
    isMock,
    analyzeNews,
    clearPrediction
  };
};

export default usePrediction;
