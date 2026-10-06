import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { Prediction } from '../models/Prediction.js';
import { predictWithML } from '../services/mlService.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const getBenchmarkPath = (dataset) => path.resolve(__dirname, `../../ml/results/${dataset}/benchmark.json`);

const loadBenchmark = (dataset) => {
  const benchmarkPath = getBenchmarkPath(dataset);
  if (!fs.existsSync(benchmarkPath)) return null;
  return JSON.parse(fs.readFileSync(benchmarkPath, 'utf8'));
};

export const getDatasetInfo = async (req, res) => {
  const { dataset } = req;
  try {
    const benchmark = loadBenchmark(dataset);
    if (benchmark) {
      return res.json({ success: true, data: benchmark });
    }
    return res.json({
      success: true,
      data: {
        name: dataset,
        description: 'Dataset information unavailable. Training not completed.',
        records: null,
        classDistribution: null,
        trainingSize: null,
        validationSize: null,
        testSize: null,
        preprocessingInformation: null,
        labelMapping: null,
        experimentVersion: null,
        trainingDate: null,
        modelList: []
      }
    });
  } catch (error) {
    return res.status(500).json({ success: false, error: error.message });
  }
};

export const getBenchmarks = async (req, res) => {
  const { dataset } = req;
  try {
    const benchmark = loadBenchmark(dataset);
    if (benchmark) {
      return res.json({ success: true, data: benchmark });
    }
    return res.status(404).json({ success: false, error: 'Training not completed. Benchmarks not available.' });
  } catch (error) {
    return res.status(500).json({ success: false, error: error.message });
  }
};

export const getModels = async (req, res) => {
  const { dataset } = req;
  try {
    const benchmark = loadBenchmark(dataset);
    if (benchmark) {
      return res.json({ success: true, data: benchmark.models || [] });
    }
    return res.status(404).json({ success: false, error: 'Training not completed. Models not available.' });
  } catch (error) {
    return res.status(500).json({ success: false, error: error.message });
  }
};

export const getAnalytics = async (req, res) => {
  const { dataset } = req;
  try {
    const total = await Prediction.countDocuments({ dataset });
    const fake = await Prediction.countDocuments({ dataset, prediction: 'FAKE' });
    const real = await Prediction.countDocuments({ dataset, prediction: 'REAL' });
    return res.json({ success: true, data: { total, fake, real } });
  } catch (error) {
    return res.status(500).json({ success: false, error: error.message });
  }
};

export const getArchive = async (req, res) => {
  const { dataset } = req;
  try {
    const predictions = await Prediction.find({ dataset }).sort({ createdAt: -1 }).limit(100);
    const formatted = predictions.map(p => ({
      dataset: p.dataset,
      model: p.model,
      experimentVersion: p.experimentVersion || 'unknown',
      prediction: p.prediction,
      confidence: p.confidence,
      timestamp: p.createdAt,
      trueLabel: p.trueLabel || null,
      correct: p.correct || null
    }));
    return res.json({ success: true, data: formatted });
  } catch (error) {
    return res.status(500).json({ success: false, error: error.message });
  }
};

export const predict = async (req, res) => {
  const { dataset } = req;
  const { text } = req.body;
  
  if (!text) {
    return res.status(400).json({ success: false, error: 'Text is required' });
  }

  try {
    // Call ML service
    const mlResult = await predictWithML({ title: '', text, dataset });
    
    // The ML service should return predictions from the 4 models and an ensemble
    // For now, we adapt the existing mlResult structure
    const responseData = {
      dataset,
      experimentVersion: mlResult.experimentVersion || 'unknown',
      predictions: mlResult.predictions || [{
        model: mlResult.model || 'unknown',
        prediction: mlResult.prediction,
        confidence: mlResult.confidence,
        dataset,
        experimentVersion: mlResult.experimentVersion || 'unknown'
      }],
      ensemble: mlResult.ensemble || {
        prediction: mlResult.prediction,
        confidence: mlResult.confidence
      },
      timestamp: new Date().toISOString()
    };

    // Store in archive
    await Prediction.create({
      title: '',
      text,
      dataset,
      prediction: responseData.ensemble.prediction,
      confidence: responseData.ensemble.confidence,
      model: 'ensemble',
      experimentVersion: responseData.experimentVersion
    });

    return res.json({ success: true, data: responseData });
  } catch (error) {
    return res.status(500).json({ success: false, error: error.message });
  }
};