import { sendError } from '../utils/response.js';

/**
 * Validate prediction POST request body
 * Checks for presence, types, and string lengths of title and text
 */
export const validatePredictionInput = (req, res, next) => {
  const { title, text } = req.body;

  if (!title || typeof title !== 'string' || !title.trim()) {
    return sendError(res, 'Validation Error: "title" is required and cannot be empty.', 400);
  }

  if (title.trim().length > 500) {
    return sendError(res, 'Validation Error: "title" cannot exceed 500 characters.', 400);
  }

  if (!text || typeof text !== 'string' || !text.trim()) {
    return sendError(res, 'Validation Error: "text" is required and cannot be empty.', 400);
  }

  if (text.trim().length < 10) {
    return sendError(res, 'Validation Error: "text" must contain at least 10 characters for NLP feature extraction.', 400);
  }

  // Normalize inputs on request body
  req.body.title = title.trim();
  req.body.text = text.trim();

  next();
};

/**
 * Validate query parameters for prediction history filtering
 */
export const validatePredictionQuery = (req, res, next) => {
  const { page, limit, prediction } = req.query;

  if (page && (isNaN(Number(page)) || Number(page) < 1)) {
    return sendError(res, 'Query parameter "page" must be a positive integer.', 400);
  }

  if (limit && (isNaN(Number(limit)) || Number(limit) < 1 || Number(limit) > 100)) {
    return sendError(res, 'Query parameter "limit" must be an integer between 1 and 100.', 400);
  }

  if (prediction && !['FAKE', 'REAL', 'ALL'].includes(String(prediction).toUpperCase())) {
    return sendError(res, 'Query parameter "prediction" must be "FAKE", "REAL", or "ALL".', 400);
  }

  next();
};

export const validateQueryPagination = validatePredictionQuery;
