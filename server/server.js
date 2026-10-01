import dotenv from 'dotenv';
import app from './app.js';
import { connectDB } from './config/db.js';

// Load environment variables from .env file
dotenv.config();

const PORT = process.env.PORT || 5000;
const NODE_ENV = process.env.NODE_ENV || 'development';

/**
 * Initialize MongoDB connection and launch Express HTTP server
 */
const startServer = async () => {
  // Connect to MongoDB
  await connectDB();

  const server = app.listen(PORT, () => {
    console.log('====================================================');
    console.log(`🚀 Fake News Detection API Server is running!`);
    console.log(`📡 Port: ${PORT}`);
    console.log(`🌐 Mode: ${NODE_ENV}`);
    console.log(`🔗 API Base: http://localhost:${PORT}/api`);
    console.log(`❤️  Health Check: http://localhost:${PORT}/api/health`);
    console.log(`🧠 ML Microservice Target: ${process.env.PYTHON_ML_URL || 'http://localhost:8000/predict'}`);
    console.log('====================================================');
  });

  // Handle unhandled promise rejections
  process.on('unhandledRejection', (err) => {
    console.error(`[UnhandledRejection Error]: ${err.message}`);
    // Keep server running in development mode
    if (NODE_ENV === 'production') {
      server.close(() => process.exit(1));
    }
  });

  // Handle uncaught exceptions
  process.on('uncaughtException', (err) => {
    console.error(`[UncaughtException Error]: ${err.message}`);
    if (NODE_ENV === 'production') {
      server.close(() => process.exit(1));
    }
  });
};

startServer();
