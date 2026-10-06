import { connectDB, closeDB } from './config/db.js';
import { Prediction } from './models/Prediction.js';

const testDB = async () => {
  try {
    await connectDB();
    const count = await Prediction.countDocuments();
    console.log('Count:', count);
    await closeDB();
  } catch (err) {
    console.error('Error:', err);
  }
};

testDB();