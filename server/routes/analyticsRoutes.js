import express from 'express';
import {
  getAnalytics,
  getModelPerformance,
  getPredictionStats
} from '../controllers/analyticsController.js';

const router = express.Router();

/**
 * @route   GET /api/analytics
 * @desc    Comprehensive aggregate metrics (fake/real breakdown, trends, distribution, matrix)
 */
router.get('/', getAnalytics);

/**
 * @route   GET /api/analytics/performance
 * @desc    Evaluation benchmarks for all trained NLP classifiers
 */
router.get('/performance', getModelPerformance);

/**
 * @route   GET /api/analytics/stats
 * @desc    Quick summary statistics
 */
router.get('/stats', getPredictionStats);

export default router;
