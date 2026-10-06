import express from 'express';
import {
  getDatasetInfo,
  getBenchmarks,
  getModels,
  getAnalytics,
  getArchive,
  predict
} from '../controllers/datasetController.js';

const router = express.Router();

// Dataset validation middleware
const validateDataset = (req, res, next) => {
  const dataset = req.params.dataset || req.body.dataset;
  if (!dataset || !['isot', 'liar'].includes(dataset.toLowerCase())) {
    return res.status(400).json({
      success: false,
      error: 'Invalid or missing dataset. Allowed values: "isot", "liar"'
    });
  }
  req.dataset = dataset.toLowerCase();
  next();
};

router.get('/dataset/:dataset', validateDataset, getDatasetInfo);
router.get('/benchmarks/:dataset', validateDataset, getBenchmarks);
router.get('/models/:dataset', validateDataset, getModels);
router.get('/analytics/:dataset', validateDataset, getAnalytics);
router.get('/archive/:dataset', validateDataset, getArchive);
router.post('/predict', validateDataset, predict);

export default router;