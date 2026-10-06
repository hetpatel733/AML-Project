# ==============================================================================
# Multi-Stage Optimized All-in-One Dockerfile for Fake News Detection System
# Incorporates embedded MongoDB 7.0, Python NLP Microservice, Express API & React UI
# Utilizes container compression & size-reduction best practices.
# ==============================================================================

# ------------------------------------------------------------------------------
# STAGE 1: Client Build Stage (Alpine Node builder - discarded after build)
# ------------------------------------------------------------------------------
FROM node:20-alpine AS client-builder

WORKDIR /app/client

# Install frontend dependencies with strict lockfile
COPY client/package*.json ./
RUN npm ci --no-audit --no-fund

# Copy frontend source and build optimized production bundle
COPY client/ ./
ARG VITE_API_URL=/api
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build \
    && rm -rf node_modules

# ------------------------------------------------------------------------------
# STAGE 2: Compressed All-in-One Production Runtime Image
# ------------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS production

LABEL maintainer="Academic Project Team"
LABEL description="Full-stack AI Fake News Detection Platform with Embedded MongoDB & NLP Microservice"

# Environment configuration for optimal performance and minimal footprint
ENV DEBIAN_FRONTEND=noninteractive \
    NODE_ENV=production \
    PORT=5000 \
    MONGO_URI=mongodb://127.0.0.1:27017/fake_news_detection \
    ML_SERVICE_URL=http://127.0.0.1:8000 \
    PYTHON_ML_URL=http://127.0.0.1:8000/predict \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Layer Compression Technique: Combined apt installation and aggressive cleanup
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        gnupg \
        ca-certificates \
        procps \
    # Add MongoDB 7.0 Official Debian Bookworm Repository
    && curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | gpg --dearmor -o /usr/share/keyrings/mongodb-server-7.0.gpg \
    && echo "deb [ signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] http://repo.mongodb.org/apt/debian bookworm/mongodb-org/7.0 main" > /etc/apt/sources.list.d/mongodb-org-7.0.list \
    # Add NodeSource Node.js 20.x Repository
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    # Install MongoDB Server, Tools, and Node.js
    && apt-get install -y --no-install-recommends \
        mongodb-org-server \
        mongodb-mongosh \
        nodejs \
    # Clean up build tools, caches, and repository lists to minimize image layer size
    && apt-get purge -y --auto-remove gnupg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /var/cache/apt/* /usr/share/doc/* /usr/share/man/*

# Initialize MongoDB data and log directories
RUN mkdir -p /data/db /data/log

WORKDIR /app

# Install Python ML dependencies with --no-cache-dir and pre-download NLTK corpora
COPY ml/requirements.txt ./ml/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir -r ./ml/requirements.txt \
    && python -m nltk.downloader stopwords

# Install Node.js Express server production dependencies only (no devDependencies)
COPY server/package*.json ./server/
RUN cd /app/server \
    && npm ci --omit=dev --no-audit --no-fund \
    && npm cache clean --force

# Copy ML service source code, trained model artifacts, and datasets
COPY ml/ ./ml/

# Copy Express server source code
COPY server/ ./server/

# Copy compiled static frontend SPA from Stage 1 builder
COPY --from=client-builder /app/client/dist ./client/dist

# Copy entrypoint orchestration supervisor script
COPY entrypoint.sh /app/entrypoint.sh
RUN sed -i 's/\r$//' /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Persist MongoDB data across container restarts
VOLUME ["/data/db"]

# Expose primary application port (Web UI + REST API), ML microservice port, and MongoDB port
EXPOSE 5000 8000 27017

# Container healthcheck testing both Node Express API and FastAPI ML Microservice
HEALTHCHECK --interval=20s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5000/api/health && curl -f http://localhost:8000/health || exit 1

# Launch embedded supervisor managing MongoDB, FastAPI, and Express
ENTRYPOINT ["/app/entrypoint.sh"]
