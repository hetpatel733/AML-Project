import { Prediction } from '../models/Prediction.js';
import { Feedback } from '../models/Feedback.js';
import { predictWithML, predictSimulationWithML, getExperimentMetadataFromML } from '../services/mlService.js';
import { sendSuccess, sendError, sendPaginated } from '../utils/response.js';

/**
 * @desc    Submit news headline & article for ML prediction & store in database
 * @route   POST /api/predictions
 * @access  Public
 */
export const createPrediction = async (req, res, next) => {
  try {
    const { title, text, model } = req.body;

    // 1. Run full simulation / prediction
    let simResult = null;
    try {
      simResult = await predictSimulationWithML({ title, text });
    } catch (simErr) {
      console.warn('[Simulation Notice] Auto-simulation fallback on single prediction:', simErr.message);
    }

    // 2. Send title/text to the ML service (or configured dev mock)
    const mlResult = await predictWithML({ title, text, model });

    // 3. Prepare database document with full simulation payload
    const newPredictionData = {
      title,
      text,
      prediction: mlResult.prediction,
      confidence: mlResult.confidence,
      model: mlResult.model,
      explanation: mlResult.explanation || '',
      geminiInsights: mlResult.geminiInsights || null,
      diagnostics: {
        wordCount: mlResult.wordCount,
        charCount: mlResult.charCount,
        topTokens: mlResult.topTokens
      },
      simulationResults: simResult
    };

    let savedDoc;
    try {
      // 4. Store the prediction in MongoDB
      savedDoc = await Prediction.create(newPredictionData);
    } catch (dbError) {
      console.warn('[Database Notice] Could not persist prediction to MongoDB (database may be offline). Returning prediction without persistent ID.');
      savedDoc = {
        _id: 'temp_' + Date.now().toString(36),
        ...newPredictionData,
        createdAt: new Date()
      };
    }

    // 5. Return formatted result to frontend
    return sendSuccess(
      res,
      {
        predictionId: savedDoc._id,
        id: savedDoc._id,
        title: savedDoc.title,
        text: savedDoc.text,
        prediction: savedDoc.prediction,
        confidence: savedDoc.confidence,
        model: savedDoc.model,
        probabilities: mlResult.probabilities,
        topTokens: mlResult.topTokens,
        explanation: savedDoc.explanation,
        simulationResults: savedDoc.simulationResults || simResult,
        createdAt: savedDoc.createdAt
      },
      201,
      'News article analyzed successfully.'
    );
  } catch (error) {
    next(error);
  }
};

/**
 * @desc    Run full multi-model simulation across all 6 models + 2 ensembles
 * @route   POST /api/predictions/simulate
 * @access  Public
 */
export const simulatePrediction = async (req, res, next) => {
  try {
    const { title, text } = req.body;

    if (!text || text.trim().length < 5) {
      return sendError(res, 'Article text must be at least 5 characters long for simulation.', 400);
    }

    const simResult = await predictSimulationWithML({ title, text });

    // Persist to database if possible
    let savedDocId = 'sim_' + Date.now().toString(36);
    try {
      const primary = simResult.primary_verdict || {};
      const doc = await Prediction.create({
        title: title || 'Untitled Simulation',
        text: text,
        prediction: primary.label || 'FAKE',
        confidence: primary.confidence || 0.5,
        model: primary.decision_source || 'Validation-Weighted Soft Ensemble',
        explanation: primary.explanation || '',
        geminiInsights: simResult.geminiInsights || null,
        diagnostics: simResult.input_summary?.diagnostics,
        simulationResults: simResult
      });
      savedDocId = doc._id;
    } catch (dbErr) {
      console.warn('[Database Notice] MongoDB offline or write skipped for simulation run.');
    }

    return sendSuccess(
      res,
      {
        id: savedDocId,
        predictionId: savedDocId,
        ...simResult
      },
      200,
      'Multi-model simulation completed successfully.'
    );
  } catch (error) {
    next(error);
  }
};

/**
 * @desc    Submit user feedback on model prediction
 * @route   POST /api/predictions/feedback
 * @access  Public
 */
export const submitFeedback = async (req, res, next) => {
  try {
    const {
      predictionId,
      articleTitle,
      articleSnippet,
      modelPrediction,
      modelConfidence,
      userVerdict,
      userCorrection,
      userComments,
      rating
    } = req.body;

    if (!userVerdict || !['AGREE', 'DISAGREE', 'UNCERTAIN'].includes(userVerdict)) {
      return sendError(res, 'Invalid user verdict. Allowed values: AGREE, DISAGREE, UNCERTAIN', 400);
    }

    let savedFeedback;
    try {
      savedFeedback = await Feedback.create({
        predictionId: predictionId || null,
        articleTitle: articleTitle || '',
        articleSnippet: articleSnippet ? articleSnippet.slice(0, 300) : '',
        modelPrediction: modelPrediction || 'UNKNOWN',
        modelConfidence: typeof modelConfidence === 'number' ? modelConfidence : 0,
        userVerdict,
        userCorrection: userCorrection || 'NONE',
        userComments: userComments || '',
        rating: rating || 5
      });
    } catch (dbErr) {
      console.warn('[Database Notice] Feedback not saved to MongoDB (DB may be offline).');
      savedFeedback = {
        _id: 'fb_' + Date.now().toString(36),
        userVerdict,
        createdAt: new Date()
      };
    }

    return sendSuccess(
      res,
      {
        feedbackId: savedFeedback._id,
        userVerdict,
        message: 'Feedback received. Thank you for helping evaluate our experimental models!'
      },
      201,
      'Feedback recorded successfully.'
    );
  } catch (error) {
    next(error);
  }
};

/**
 * @desc    Get user feedback list
 * @route   GET /api/predictions/feedback
 * @access  Public
 */
export const getFeedbackList = async (req, res, next) => {
  try {
    let feedbacks = [];
    try {
      feedbacks = await Feedback.find().sort({ createdAt: -1 }).limit(50).lean();
    } catch (dbErr) {
      feedbacks = [];
    }

    return sendSuccess(res, feedbacks);
  } catch (error) {
    next(error);
  }
};

/**
 * @desc    Get experiment metadata and benchmarks
 * @route   GET /api/predictions/experiments/metadata
 * @access  Public
 */
export const getExperimentMetadata = async (req, res, next) => {
  try {
    const meta = await getExperimentMetadataFromML();
    if (meta) {
      return sendSuccess(res, meta);
    }
    return sendError(res, 'Experiment metadata could not be loaded.', 404);
  } catch (error) {
    next(error);
  }
};

/**
 * @desc    Get prediction history with search, filtering, and pagination
 * @route   GET /api/predictions
 * @access  Public
 */
export const getPredictions = async (req, res, next) => {
  try {
    const page = parseInt(req.query.page, 10) || 1;
    const limit = parseInt(req.query.limit, 10) || 10;
    const search = req.query.search ? req.query.search.trim() : '';
    const predictionFilter = req.query.prediction ? req.query.prediction.trim().toUpperCase() : 'ALL';
    const sortBy = req.query.sortBy || 'date_desc';

    // Build query filter
    const query = {};

    if (predictionFilter && predictionFilter !== 'ALL') {
      query.prediction = predictionFilter;
    }

    if (search) {
      query.$or = [
        { title: { $regex: search, $options: 'i' } },
        { text: { $regex: search, $options: 'i' } },
        { model: { $regex: search, $options: 'i' } }
      ];
    }

    // Determine sort ordering
    let sortOptions = { createdAt: -1 };
    if (sortBy === 'date_asc') sortOptions = { createdAt: 1 };
    if (sortBy === 'confidence_desc') sortOptions = { confidence: -1 };
    if (sortBy === 'confidence_asc') sortOptions = { confidence: 1 };

    const skip = (page - 1) * limit;

    let predictions = [];
    let total = 0;

    try {
      [predictions, total] = await Promise.all([
        Prediction.find(query).sort(sortOptions).skip(skip).limit(limit).lean(),
        Prediction.countDocuments(query)
      ]);
    } catch (dbErr) {
      console.warn('[Database Notice] MongoDB query failed. Returning empty list fallback.');
      predictions = [];
      total = 0;
    }

    const totalPages = Math.ceil(total / limit) || 1;

    // Standardize _id to id for frontend convenience
    const formatted = predictions.map(p => ({
      id: p._id,
      ...p
    }));

    return res.status(200).json({
      success: true,
      data: {
        predictions: formatted,
        total,
        page,
        totalPages
      }
    });
  } catch (error) {
    next(error);
  }
};

/**
 * @desc    Get single prediction by ID
 * @route   GET /api/predictions/:id
 * @access  Public
 */
export const getPredictionById = async (req, res, next) => {
  try {
    const { id } = req.params;

    const prediction = await Prediction.findById(id).lean();

    if (!prediction) {
      return sendError(res, `Prediction not found with ID: ${id}`, 404);
    }

    return sendSuccess(res, {
      id: prediction._id,
      ...prediction
    });
  } catch (error) {
    next(error);
  }
};

/**
 * @desc    Delete a prediction record by ID
 * @route   DELETE /api/predictions/:id
 * @access  Public
 */
export const deletePrediction = async (req, res, next) => {
  try {
    const { id } = req.params;

    const deleted = await Prediction.findByIdAndDelete(id);

    if (!deleted) {
      return sendError(res, `Prediction record not found with ID: ${id}`, 404);
    }

    return sendSuccess(res, { deletedId: id }, 200, 'Prediction record removed successfully.');
  } catch (error) {
    next(error);
  }
};
