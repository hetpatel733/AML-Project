import mongoose from 'mongoose';

/**
 * Feedback Mongoose Schema for User Evaluation of ML Predictions
 */
const feedbackSchema = new mongoose.Schema(
  {
    predictionId: {
      type: String,
      trim: true,
      default: null
    },
    articleTitle: {
      type: String,
      trim: true,
      default: ''
    },
    articleSnippet: {
      type: String,
      trim: true,
      default: ''
    },
    modelPrediction: {
      type: String,
      enum: ['FAKE', 'REAL', 'UNKNOWN'],
      default: 'UNKNOWN'
    },
    modelConfidence: {
      type: Number,
      min: 0,
      max: 1,
      default: 0
    },
    userVerdict: {
      type: String,
      required: [true, 'User agreement verdict is required'],
      enum: ['AGREE', 'DISAGREE', 'UNCERTAIN']
    },
    userCorrection: {
      type: String,
      enum: ['FAKE', 'REAL', 'UNCERTAIN', 'NONE'],
      default: 'NONE'
    },
    userComments: {
      type: String,
      trim: true,
      maxlength: [1000, 'Comments cannot exceed 1000 characters'],
      default: ''
    },
    rating: {
      type: Number,
      min: 1,
      max: 5,
      default: 5
    }
  },
  {
    timestamps: true
  }
);

feedbackSchema.index({ createdAt: -1 });
feedbackSchema.index({ userVerdict: 1 });

export const Feedback = mongoose.model('Feedback', feedbackSchema);
export default Feedback;
