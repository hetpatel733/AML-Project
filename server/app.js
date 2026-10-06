import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import predictionRoutes from './routes/predictionRoutes.js';
import analyticsRoutes from './routes/analyticsRoutes.js';
import datasetRoutes from './routes/datasetRoutes.js';
import { getHealth, getModelPerformance } from './controllers/analyticsController.js';
import { notFound, errorHandler } from './middleware/errorMiddleware.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const clientDistPath = path.resolve(__dirname, '../client/dist');

const app = express();

// Security HTTP headers (configured for SPA compatibility)
app.use(
  helmet({
    contentSecurityPolicy: false,
    crossOriginEmbedderPolicy: false
  })
);

// CORS configuration supporting Vite dev server and production clients
const allowedOrigins = [
  process.env.CLIENT_ORIGIN || 'http://localhost:5173',
  'http://127.0.0.1:5173',
  'http://localhost:3000',
  'http://localhost:5000'
];

app.use(
  cors({
    origin: (origin, callback) => {
      // Allow requests with no origin (like mobile apps, curl, Postman)
      if (!origin || allowedOrigins.includes(origin)) {
        return callback(null, true);
      }
      return callback(null, true); // Permissive in dev to avoid CORS blocking
    },
    credentials: true,
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With']
  })
);

// Body parser middleware
app.use(express.json({ limit: '2mb' }));
app.use(express.urlencoded({ extended: true, limit: '2mb' }));

// Health check endpoint
app.get('/api/health', getHealth);

// Standalone model performance route for direct access
app.get('/api/model-performance', getModelPerformance);
app.get('/api/models/performance', getModelPerformance);

// API route mount points
app.use('/api/predictions', predictionRoutes);
app.use('/api/analytics', analyticsRoutes);
app.use('/api', datasetRoutes);

// Frontend static asset serving and SPA fallback (when built client exists)
if (fs.existsSync(clientDistPath)) {
  console.log(`[Static] Serving frontend static assets from: ${clientDistPath}`);
  app.use(express.static(clientDistPath));

  // SPA fallback for client-side routing on all non-API GET requests
  app.get('*', (req, res, next) => {
    if (req.path.startsWith('/api')) {
      return next();
    }
    res.sendFile(path.join(clientDistPath, 'index.html'));
  });
} else {
  // Root route for API-only / development mode
  app.get('/', (req, res) => {
    res.status(200).json({
      name: 'Fake News Detection System - API Server',
      version: '1.0.0',
      description: 'RESTful API for Natural Language Processing and Fake News Detection',
      endpoints: {
        health: 'GET /api/health',
        predict: 'POST /api/predictions',
        history: 'GET /api/predictions',
        stats: 'GET /api/predictions/stats',
        analytics: 'GET /api/analytics',
        modelPerformance: 'GET /api/model-performance'
      }
    });
  });
}

// 404 Not Found Middleware
app.use(notFound);

// Centralized Error Handling Middleware
app.use(errorHandler);

export default app;
