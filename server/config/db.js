import mongoose from 'mongoose';
import { MongoMemoryServer } from 'mongodb-memory-server';

let mongoServer;

/**
 * Connect to MongoDB instance using Mongoose
 */
export const connectDB = async () => {
  try {
    const mongoUri = process.env.MONGO_URI || 'mongodb://localhost:27017/fake_news_detection';

    const conn = await mongoose.connect(mongoUri, {
      serverSelectionTimeoutMS: 5000 // 5 seconds timeout
    });

    console.log(`[Database] MongoDB Connected: ${conn.connection.host}/${conn.connection.name}`);
  } catch (error) {
    console.error(`[Database Error] Failed to connect to MongoDB: ${error.message}`);
    console.log('[Database Notice] Attempting to start in-memory MongoDB fallback...');
    
    try {
      mongoServer = await MongoMemoryServer.create();
      const inMemoryUri = mongoServer.getUri();
      
      const conn = await mongoose.connect(inMemoryUri);
      console.log(`[Database] In-Memory MongoDB Connected: ${conn.connection.host}/${conn.connection.name}`);
    } catch (fallbackError) {
      console.error(`[Database Error] Failed to start in-memory MongoDB: ${fallbackError.message}`);
      console.warn('[Database Notice] Backend will operate, but database-dependent operations will fail without a running MongoDB server.');
    }
  }
};

export const closeDB = async () => {
  await mongoose.disconnect();
  if (mongoServer) {
    await mongoServer.stop();
  }
};

export default connectDB;
