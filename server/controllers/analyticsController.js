import { Prediction } from "../models/Prediction.js";
import { sendSuccess } from "../utils/response.js";

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const loadExperiment = (dataset = "isot") => {
  try {
    const configPath = path.resolve(__dirname, `../../ml/results/${dataset}/benchmark.json`);
    if (fs.existsSync(configPath)) {
      return JSON.parse(fs.readFileSync(configPath, "utf8"));
    }
    return null;
  } catch (error) {
    console.warn(`[Experiment] Could not load benchmark.json for dataset ${dataset}:`, error.message);
    return null;
  }
};

const toBenchmarkRows = (dataset) => {
  const experiment = loadExperiment(dataset);
  return (experiment?.models || []).map(model => ({
    model: model.name,
    modelId: model.id,
    accuracy: model.metrics?.accuracy,
    precision: model.metrics?.precision,
    recall: model.metrics?.recall,
    f1Score: model.metrics?.f1,
    rocAuc: model.metrics?.rocAuc,
    cvMeanF1: model.crossValidation?.mean?.f1,
    cvStdF1: model.crossValidation?.std?.f1,
    description: model.name,
    featureType: model.representation
  }));
};

export const getAnalytics = async (req, res, next) => {
  try {
    const dataset = req.query.dataset || "isot";
    const modelPerformance = toBenchmarkRows(dataset);
    let totalPredictions = 0;
    let fakeCount = 0;
    let realCount = 0;
    let avgConfidence = 0;
    let trendData = [];
    let confidenceDistribution = {
      "90-100%": 0,
      "80-90%": 0,
      "70-80%": 0,
      "60-70%": 0,
      "50-60%": 0
    };

    try {
      const predictions = await Prediction.find({ dataset });
      totalPredictions = predictions.length;
      
      if (totalPredictions > 0) {
        let totalConf = 0;
        
        predictions.forEach(p => {
          if (p.prediction === "FAKE") fakeCount++;
          else realCount++;
          
          totalConf += p.confidence;
          
          const conf = p.confidence * 100;
          if (conf >= 90) confidenceDistribution["90-100%"]++;
          else if (conf >= 80) confidenceDistribution["80-90%"]++;
          else if (conf >= 70) confidenceDistribution["70-80%"]++;
          else if (conf >= 60) confidenceDistribution["60-70%"]++;
          else confidenceDistribution["50-60%"]++;
        });
        
        avgConfidence = totalConf / totalPredictions;
        
        // Group by date for trend
        const grouped = {};
        predictions.forEach(p => {
          const date = p.createdAt.toISOString().split("T")[0];
          if (!grouped[date]) grouped[date] = { date, fake: 0, real: 0, total: 0 };
          grouped[date].total++;
          if (p.prediction === "FAKE") grouped[date].fake++;
          else grouped[date].real++;
        });
        
        trendData = Object.values(grouped).sort((a, b) => a.date.localeCompare(b.date)).slice(-30);
      }
    } catch (dbErr) {
      console.warn("[Analytics] DB error:", dbErr.message);
    }

    // Expose confusion matrices for every trained benchmark model.
    const experiment = loadExperiment(dataset);
    const confusionMatrices = (experiment?.models || [])
      .filter(model => model.status === "trained" && model.confusionMatrix)
      .map(model => ({
        modelId: model.id,
        model: model.name,
        ...model.confusionMatrix,
        totalSamples: ["truePositive", "falsePositive", "trueNegative", "falseNegative"]
          .reduce((sum, key) => sum + (model.confusionMatrix[key] || 0), 0)
      }));

    // Keep the first matrix as a backward-compatible aggregate field.
    let confusionMatrix = {
      truePositive: 0,
      falsePositive: 0,
      trueNegative: 0,
      falseNegative: 0,
      totalSamples: 0
    };
    
    if (confusionMatrices.length > 0) {
      confusionMatrix = confusionMatrices[0];
    }

    sendSuccess(res, {
      totalPredictions,
      fakeCount,
      realCount,
      avgConfidence,
      trendData,
      confidenceDistribution,
      modelPerformance,
      confusionMatrices,
      confusionMatrix
    });
  } catch (error) {
    next(error);
  }
};

export const getPredictionStats = async (req, res, next) => {
  try {
    const dataset = req.query.dataset || "isot";
    const totalPredictions = await Prediction.countDocuments({ dataset });
    const fakeCount = await Prediction.countDocuments({ dataset, prediction: "FAKE" });
    const realCount = await Prediction.countDocuments({ dataset, prediction: "REAL" });
    
    sendSuccess(res, {
      totalPredictions,
      fakeCount,
      realCount
    });
  } catch (error) {
    next(error);
  }
};

export const getModelPerformance = async (req, res, next) => {
  try {
    const dataset = req.query.dataset || "isot";
    const modelPerformance = toBenchmarkRows(dataset);
    sendSuccess(res, modelPerformance);
  } catch (error) {
    next(error);
  }
};

export const getHealth = (req, res) => {
  res.status(200).json({
    success: true,
    status: 'UP',
    timestamp: new Date().toISOString()
  });
};

