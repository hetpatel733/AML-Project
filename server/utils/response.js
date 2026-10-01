/**
 * Standardized API Response Utilities
 */

/**
 * Send a success response
 * @param {object} res Express response object
 * @param {any} data Payload data
 * @param {number} statusCode HTTP status code (default 200)
 * @param {string} message Optional success message
 */
export const sendSuccess = (res, data, statusCode = 200, message = null) => {
  const response = {
    success: true,
    data
  };
  if (message) response.message = message;
  return res.status(statusCode).json(response);
};

/**
 * Send an error response
 * @param {object} res Express response object
 * @param {string} message Error message
 * @param {number} statusCode HTTP status code (default 500)
 * @param {any} errors Optional detailed errors (e.g. validation errors array)
 */
export const sendError = (res, message = 'Internal Server Error', statusCode = 500, errors = null) => {
  const response = {
    success: false,
    message
  };
  if (errors) response.errors = errors;
  return res.status(statusCode).json(response);
};

/**
 * Send a paginated collection response
 * @param {object} res Express response object
 * @param {Array} items List of items
 * @param {object} pagination Pagination metadata { total, page, limit, totalPages }
 */
export const sendPaginated = (res, items, pagination, statusCode = 200) => {
  return res.status(statusCode).json({
    success: true,
    data: items,
    pagination: {
      total: pagination.total,
      page: pagination.page,
      limit: pagination.limit,
      totalPages: pagination.totalPages,
      hasNextPage: pagination.page < pagination.totalPages,
      hasPrevPage: pagination.page > 1
    }
  });
};
