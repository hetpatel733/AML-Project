import mongoose from 'mongoose';

/**
 * Prediction Mongoose Schema for Fake News Detection
 */
const predictionSchema = new mongoose.Schema(
  {
    title: {
      type: String,
      required: [true, 'News headline/title is required'],
      trim: true,
      maxlength: [500, 'Title cannot exceed 500 characters']
    },
    text: {
      type: String,
      required: [true, 'News article text is required'],
      trim: true,
      minlength: [10, 'Article text must contain at least 10 characters']
    },
    prediction: {
      type: String,
      required: [true, 'Classification verdict is required'],
      enum: {
        values: ['FAKE', 'REAL'],
        message: '{VALUE} is not a valid prediction verdict. Allowed values are FAKE or REAL'
      },
      uppercase: true
    },
    confidence: {
      type: Number,
      required: [true, 'Confidence score is required'],
      min: [0, 'Confidence cannot be less than 0.0'],
      max: [1, 'Confidence cannot be greater than 1.0']
    },
    model: {
      type: String,
      required: [true, 'Model identifier is required'],
      default: 'Logistic Regression',
      trim: true
    },
    explanation: {
      type: String,
      default: ''
    },
    geminiInsights: {
      type: mongoose.Schema.Types.Mixed,
      default: null
    },
    diagnostics: {
      type: mongoose.Schema.Types.Mixed,
      default: null
    },
    simulationResults: {
      type: mongoose.Schema.Types.Mixed,
      default: null
    }
  },
  {
    timestamps: true
  }
);

// Indexes for fast querying, filtering, and time-series aggregation
predictionSchema.index({ createdAt: -1 });
predictionSchema.index({ prediction: 1, createdAt: -1 });
predictionSchema.index({ title: 'text', text: 'text' });

export const Prediction = mongoose.model('Prediction', predictionSchema);
export default Prediction;
