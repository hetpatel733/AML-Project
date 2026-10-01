import { sendError } from '../utils/response.js';

/**
 * Handle 404 Route Not Found
 */
export const notFoundHandler = (req, res, next) => {
  return sendError(res, `Endpoint Not Found: ${req.method} ${req.originalUrl}`, 404);
};

export const notFound = notFoundHandler;

/**
 * Global centralized error handling middleware
 */
export const errorHandler = (err, req, res, next) => {
  console.error(`[Error] ${err.name || 'Server Error'}: ${err.message}`);

  // Mongoose CastError (invalid ObjectId format)
  if (err.name === 'CastError' && err.kind === 'ObjectId') {
    return sendError(res, `Resource not found with id of ${err.value}`, 404);
  }

  // Mongoose ValidationError
  if (err.name === 'ValidationError') {
    const messages = Object.values(err.errors).map(val => val.message);
    return sendError(res, 'Mongoose Validation Error', 400, messages);
  }

  // Axios HTTP Errors from external ML Service
  if (err.isAxiosError) {
    if (err.code === 'ECONNREFUSED' || err.code === 'ETIMEDOUT') {
      return sendError(
        res,
        `ML Microservice is unreachable at ${err.config?.baseURL || err.config?.url}. Please verify Python ML service is running.`,
        503
      );
    }
    const mlStatus = err.response?.status || 502;
    const mlMessage = err.response?.data?.message || err.response?.data?.detail || 'Error received from Python ML Service';
    return sendError(res, `ML Service Error: ${mlMessage}`, mlStatus);
  }

  // Default internal server error
  const statusCode = err.statusCode || 500;
  const message = process.env.NODE_ENV === 'production' && statusCode === 500
    ? 'Internal Server Error'
    : err.message || 'Internal Server Error';

  return sendError(res, message, statusCode);
};
