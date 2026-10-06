import axios from "axios";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || "http://localhost:8000";
const mlClient = axios.create({
  baseURL: ML_SERVICE_URL,
  headers: {
    "Content-Type": "application/json"
  },
  timeout: 15000
});

export const predictWithML = async (payload) => {
  try {
    const response = await mlClient.post("/predict", payload);
    return response.data;
  } catch (error) {
    console.error("[ML Service] Error calling Python ML service:", error.message);
    throw new Error("Failed to get prediction from ML service");
  }
};

export const predictSimulationWithML = async (payload) => {
  try {
    const response = await mlClient.post("/predict/all", payload);
    return response.data;
  } catch (error) {
    console.error("[ML Service] Error calling Python ML simulation service:", error.message);
    throw new Error("Failed to get simulation from ML service");
  }
};

export const getExperimentMetadataFromML = async (dataset = "isot") => {
  try {
    const response = await mlClient.get(`/experiment-metadata?dataset=${dataset}`);
    return response.data;
  } catch (error) {
    console.error("[ML Service] Error fetching metadata:", error.message);
    return {
      dataset,
      status: "offline",
      message: "ML service is currently unavailable."
    };
  }
};

