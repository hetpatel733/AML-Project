import { Prediction } from '../models/Prediction.js';
import { sendSuccess } from '../utils/response.js';

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const EXPERIMENT_PATH = path.resolve(__dirname, '../../ml/models/model_config.json');

const loadExperiment = () => {
  try {
    return JSON.parse(fs.readFileSync(EXPERIMENT_PATH, 'utf8'));
  } catch (error) {
    console.warn('[Experiment] Could not load model_config.json:', error.message);
    return null;
  }
};

const toBenchmarkRows = (experiment) => Object.values(experiment?.models || {}).map((model) => ({
  model: model.name,
  accuracy: model.test_metrics?.accuracy,
  precision: model.test_metrics?.precision,
  recall: model.test_metrics?.recall,
  f1Score: model.test_metrics?.f1_score,
  rocAuc: model.test_metrics?.roc_auc,
  cvMeanF1: model.cv_metrics?.f1_mean ?? model.cv_f1_mean,
  cvStdF1: model.cv_metrics?.f1_std ?? model.cv_f1_std,
  externalF1: model.external_metrics?.f1_score ?? null,
  description: model.description,
  featureType: model.feature_type
}));

/**
 * @desc    Get aggregate analytics, prediction counts, confidence metrics, and daily trends
 * @route   GET /api/analytics
 * @access  Public
 */
export const getAnalytics = async (req, res, next) => {
  try {
    const experiment = loadExperiment();
    const modelPerformance = toBenchmarkRows(experiment);
    let totalPredictions = 0;
    let fakeCount = 0;
    let realCount = 0;
    let avgConfidence = 0;
    let trendData = [];
    let confidenceDistribution = [
      { range: '50-60%', count: 0, label: 'Low' },
      { range: '60-70%', count: 0, label: 'Moderate' },
      { range: '70-80%', count: 0, label: 'High' },
      { range: '80-90%', count: 0, label: 'Very High' },
      { range: '90-100%', count: 0, label: 'Extremely High' }
    ];

    try {
      totalPredictions = await Prediction.countDocuments();
      fakeCount = await Prediction.countDocuments({ prediction: 'FAKE' });
      realCount = await Prediction.countDocuments({ prediction: 'REAL' });

      // Aggregate average confidence
      const avgResult = await Prediction.aggregate([
        {
          $group: {
            _id: null,
            avgConf: { $avg: '$confidence' }
          }
        }
      ]);
      if (avgResult.length > 0 && avgResult[0].avgConf) {
        avgConfidence = Number(avgResult[0].avgConf.toFixed(3));
      }

      // Group predictions by day for the last 7 days
      const sevenDaysAgo = new Date();
      sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);

      const dailyAgg = await Prediction.aggregate([
        { $match: { createdAt: { $gte: sevenDaysAgo } } },
        {
          $group: {
            _id: {
              date: { $dateToString: { format: '%b %d', date: '$createdAt' } },
              prediction: '$prediction'
            },
            count: { $sum: 1 }
          }
        }
      ]);

      // Map daily aggregation into structured chart format
      const dateMap = {};
      dailyAgg.forEach(item => {
        const dateKey = item._id.date;
        if (!dateMap[dateKey]) {
          dateMap[dateKey] = { date: dateKey, fake: 0, real: 0, total: 0 };
        }
        if (item._id.prediction === 'FAKE') dateMap[dateKey].fake += item.count;
        if (item._id.prediction === 'REAL') dateMap[dateKey].real += item.count;
        dateMap[dateKey].total += item.count;
      });

      trendData = Object.values(dateMap);

      // Bucket confidence scores
      const allPredictions = await Prediction.find({}, 'confidence').lean();
      allPredictions.forEach(p => {
        const c = p.confidence <= 1 ? p.confidence * 100 : p.confidence;
        if (c >= 50 && c < 60) confidenceDistribution[0].count++;
        else if (c >= 60 && c < 70) confidenceDistribution[1].count++;
        else if (c >= 70 && c < 80) confidenceDistribution[2].count++;
        else if (c >= 80 && c < 90) confidenceDistribution[3].count++;
        else if (c >= 90) confidenceDistribution[4].count++;
      });
    } catch (dbErr) {
      console.warn('[Database Notice] Database aggregation unavailable. Using initial metrics structure.');
    }

    const fakePercentage = totalPredictions > 0
      ? Number(((fakeCount / totalPredictions) * 100).toFixed(1))
      : 0;

    const realPercentage = totalPredictions > 0
      ? Number(((realCount / totalPredictions) * 100).toFixed(1))
      : 0;

    const championMetrics = experiment?.champion_model?.test_metrics
      || Object.values(experiment?.models || {})[0]?.test_metrics;
    const championMatrix = championMetrics?.confusion_matrix;
    const confusionMatrix = championMatrix ? {
      truePositive: championMatrix[1][1],
      falsePositive: championMatrix[0][1],
      trueNegative: championMatrix[0][0],
      falseNegative: championMatrix[1][0],
      totalSamples: championMatrix.flat().reduce((sum, value) => sum + value, 0)
    } : null;

    return sendSuccess(res, {
      totalPredictions,
      fakeCount,
      realCount,
      fakePercentage,
      realPercentage,
      averageConfidence: avgConfidence,
      predictionsPerDay: trendData,
      trendData,
      confidenceDistribution,
      confusionMatrix,
      modelPerformance,
      experiment
    });
  } catch (error) {
    next(error);
  }
};

/**
 * @desc    Get aggregate stats specifically for dashboard cards
 * @route   GET /api/predictions/stats
 * @access  Public
 */
export const getPredictionStats = async (req, res, next) => {
  try {
    let totalPredictions = 0;
    let fakeCount = 0;
    let realCount = 0;
    let avgConfidence = 0;
    let recentPredictions = [];

    try {
      [totalPredictions, fakeCount, realCount, recentPredictions] = await Promise.all([
        Prediction.countDocuments(),
        Prediction.countDocuments({ prediction: 'FAKE' }),
        Prediction.countDocuments({ prediction: 'REAL' }),
        Prediction.find().sort({ createdAt: -1 }).limit(5).lean()
      ]);

      const avgResult = await Prediction.aggregate([
        { $group: { _id: null, avgConf: { $avg: '$confidence' } } }
      ]);
      if (avgResult.length > 0 && avgResult[0].avgConf) {
        avgConfidence = Number(avgResult[0].avgConf.toFixed(3));
      }
    } catch (dbErr) {
      console.warn('[Database Notice] Database query fallback for stats.');
    }

    const fakePercentage = totalPredictions > 0 ? Number(((fakeCount / totalPredictions) * 100).toFixed(1)) : 0;
    const realPercentage = totalPredictions > 0 ? Number(((realCount / totalPredictions) * 100).toFixed(1)) : 0;

    return sendSuccess(res, {
      totalPredictions,
      fakeCount,
      realCount,
      avgConfidence,
      fakePercentage,
      realPercentage,
      recentPredictions: recentPredictions.map(p => ({ id: p._id, ...p }))
    });
  } catch (error) {
    next(error);
  }
};

/**
 * @desc    Get model performance benchmarks
 * @route   GET /api/model-performance (and /api/models/performance)
 * @access  Public
 */
export const getModelPerformance = async (req, res, next) => {
  try {
    return sendSuccess(res, toBenchmarkRows(loadExperiment()));
  } catch (error) {
    next(error);
  }
};

/**
 * @desc    API Health Check endpoint
 * @route   GET /api/health
 * @access  Public
 */
export const getHealth = async (req, res) => {
  return res.status(200).json({
    success: true,
    message: 'API is running',
    timestamp: new Date().toISOString(),
    service: 'Fake News Detection Backend',
    environment: process.env.NODE_ENV || 'development'
  });
};
