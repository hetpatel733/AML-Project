import axios from 'axios';
import { normalizeSimulationData } from '../utils/helpers';

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: { 'Content-Type': 'application/json' },
  timeout: 15000
});

export const predictNews = async (newsData) => {
  const response = await apiClient.post('/predictions', newsData);
  return { data: response.data?.data ?? response.data, isMock: false };
};

export const getPredictionHistory = async (params = {}) => {
  const response = await apiClient.get('/predictions', { params });
  const result = response.data?.data ?? response.data;
  if (result && Array.isArray(result.predictions)) {
    result.predictions = result.predictions.map(item => ({
      ...item,
      simulationResults: item.simulationResults
        ? normalizeSimulationData(item.simulationResults, item.title, item.text)
        : null
    }));
  }
  return { data: result, isMock: false };
};

export const getPredictionStats = async (dataset = 'isot') => {
  const response = await apiClient.get(`/predictions/stats?dataset=${dataset}`);
  return { data: response.data?.data ?? response.data, isMock: false };
};

export const getAnalytics = async (dataset = 'isot') => {
  const response = await apiClient.get(`/analytics?dataset=${dataset}`);
  return { data: response.data?.data ?? response.data, isMock: false };
};

export const getModelPerformance = async (dataset = 'isot') => {
  const response = await apiClient.get(`/models/performance?dataset=${dataset}`);
  return { data: response.data?.data ?? response.data, isMock: false };
};

export const deletePrediction = async (id) => {
  await apiClient.delete(`/predictions/${id}`);
  return { success: true, isMock: false };
};

export const runSimulation = async (newsData) => {
  const response = await apiClient.post('/predictions/simulate', newsData);
  const result = response.data?.data ?? response.data;
  return {
    data: normalizeSimulationData(result, newsData.title, newsData.text),
    isMock: false
  };
};

export const submitPredictionFeedback = async (feedbackData) => {
  const response = await apiClient.post('/predictions/feedback', feedbackData);
  return { data: response.data?.data ?? response.data, isMock: false };
};

export const getExperimentMetadata = async (dataset = 'isot') => {
  const response = await apiClient.get(`/benchmarks/${dataset}`);
  return { data: response.data?.data ?? response.data, isMock: false };
};

export const getBenchmarks = async (dataset = 'isot') => {
  const response = await apiClient.get(`/benchmarks/${dataset}`);
  return { data: response.data?.data ?? response.data, isMock: false };
};

export const getAllBenchmarks = async () => {
  const response = await apiClient.get('/benchmarks');
  return { data: response.data?.data ?? response.data, isMock: false };
};

export const getDatasetInfo = async (dataset = 'isot') => {
  const response = await apiClient.get(`/dataset/${dataset}`);
  return { data: response.data?.data ?? response.data, isMock: false };
};

export const getFigures = async (dataset = 'isot') => {
  const response = await apiClient.get(`/figures/${dataset}`);
  return { data: response.data?.data ?? response.data, isMock: false };
};
