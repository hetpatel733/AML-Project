import express from 'express';
import {
  createPrediction,
  getPredictions,
  getPredictionById,
  deletePrediction,
  simulatePrediction,
  submitFeedback,
  getFeedbackList,
  getExperimentMetadata
} from '../controllers/predictionController.js';
import { getPredictionStats } from '../controllers/analyticsController.js';
import {
  validatePredictionInput,
  validateQueryPagination
} from '../middleware/validationMiddleware.js';

const router = express.Router();

/**
 * @route   GET /api/predictions/stats
 * @desc    Get quick aggregate stats for overview dashboard
 */
router.get('/stats', getPredictionStats);

/**
 * @route   GET /api/predictions/experiments/metadata
 * @desc    Get complete experimental metadata, 6 model benchmarks, CV splits, vectorizer params
 */
router.get('/experiments/metadata', getExperimentMetadata);

/**
 * @route   POST /api/predictions/simulate
 * @desc    Run full 6-model inference + dual ensembles + salient keywords + diagnostic pipeline
 */
router.post('/simulate', validatePredictionInput, simulatePrediction);

/**
 * @route   POST /api/predictions/feedback
 * @desc    Submit user feedback/correction on an inference result
 */
router.post('/feedback', submitFeedback);

/**
 * @route   GET /api/predictions/feedback
 * @desc    Get list of feedback items
 */
router.get('/feedback', getFeedbackList);

/**
 * @route   POST /api/predictions
 * @desc    Submit news title & text for NLP classification & persistence
 */
router.post('/', validatePredictionInput, createPrediction);

/**
 * @route   GET /api/predictions
 * @desc    Get paginated, filtered, and searched prediction history
 */
router.get('/', validateQueryPagination, getPredictions);

/**
 * @route   GET /api/predictions/:id
 * @desc    Retrieve individual prediction record by MongoDB ObjectId
 */
router.get('/:id', getPredictionById);

/**
 * @route   DELETE /api/predictions/:id
 * @desc    Delete a prediction record by MongoDB ObjectId
 */
router.delete('/:id', deletePrediction);

export default router;
