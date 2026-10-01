#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "🚀 Initializing Full-Stack Fake News Detection Container"
echo "=========================================================="

# Ensure MongoDB directories exist with correct permissions
mkdir -p /data/db /data/log

# 1. Start MongoDB daemon in background
echo "📦 [1/3] Starting Embedded MongoDB Server..."
mongod --dbpath /data/db --logpath /data/log/mongodb.log --fork --bind_ip 127.0.0.1 --logappend

# Wait for MongoDB to become ready
echo "⏳ Waiting for MongoDB on 127.0.0.1:27017..."
max_mongo_attempts=30
attempt=0
while ! (python3 -c "import socket; s = socket.socket(); s.settimeout(1); s.connect(('127.0.0.1', 27017)); s.close()" 2>/dev/null); do
  attempt=$((attempt+1))
  if [ $attempt -ge $max_mongo_attempts ]; then
    echo "❌ [Error] MongoDB failed to start within 30 seconds."
    if [ -f /data/log/mongodb.log ]; then
      tail -n 25 /data/log/mongodb.log
    fi
    exit 1
  fi
  sleep 1
done
echo "✅ MongoDB is running and accepting connections."

# 2. Start Python FastAPI NLP Microservice in background
echo "🧠 [2/3] Starting Python FastAPI NLP Microservice (port 8000)..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 --app-dir /app/ml &
ML_PID=$!

# Wait for ML microservice to become ready
echo "⏳ Waiting for ML Microservice on 127.0.0.1:8000..."
max_ml_attempts=30
attempt=0
while ! (python3 -c "import socket; s = socket.socket(); s.settimeout(1); s.connect(('127.0.0.1', 8000)); s.close()" 2>/dev/null); do
  attempt=$((attempt+1))
  if [ $attempt -ge $max_ml_attempts ]; then
    echo "⚠️ [Notice] ML service initialization check timed out, proceeding..."
    break
  fi
  sleep 1
done
echo "✅ ML Microservice is running (PID: $ML_PID)."

# 3. Start Node.js Express Application Server
echo "🌐 [3/3] Starting Express.js Application & Web Server (port 5000)..."
node /app/server/server.js &
NODE_PID=$!
echo "✅ Express Server is running (PID: $NODE_PID)."

echo "=========================================================="
echo "🎉 System Ready!"
echo "📡 Full-Stack Web App:  http://localhost:5000"
echo "🧠 NLP ML Microservice: http://localhost:8000"
echo "📦 Embedded MongoDB:    mongodb://127.0.0.1:27017"
echo "=========================================================="

# Graceful shutdown handler
cleanup() {
  echo ""
  echo "🛑 Caught termination signal. Gracefully shutting down all services..."
  
  if [ -n "$NODE_PID" ] && kill -0 "$NODE_PID" 2>/dev/null; then
    echo "Stopping Node.js server (PID: $NODE_PID)..."
    kill -TERM "$NODE_PID" 2>/dev/null || true
  fi
  
  if [ -n "$ML_PID" ] && kill -0 "$ML_PID" 2>/dev/null; then
    echo "Stopping Python ML Microservice (PID: $ML_PID)..."
    kill -TERM "$ML_PID" 2>/dev/null || true
  fi
  
  echo "Stopping MongoDB daemon..."
  mongod --dbpath /data/db --shutdown 2>/dev/null || true
  
  wait "$NODE_PID" 2>/dev/null || true
  wait "$ML_PID" 2>/dev/null || true
  echo "🏁 All services stopped cleanly."
  exit 0
}

trap cleanup SIGINT SIGTERM SIGHUP

# Continuously monitor critical processes
while true; do
  if ! kill -0 "$NODE_PID" 2>/dev/null; then
    echo "❌ Node.js server process terminated unexpectedly!"
    cleanup
    exit 1
  fi
  if ! kill -0 "$ML_PID" 2>/dev/null; then
    echo "❌ Python ML Microservice process terminated unexpectedly!"
    cleanup
    exit 1
  fi
  sleep 2
done
