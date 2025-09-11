# Multi-stage build for IFC Semantic Agent
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code
COPY . .

# Change ownership to app user
RUN chown -R app:app /app

# Switch to non-root user
USER app

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/cache /app/models /app/output

# Download required models
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Default command
CMD ["python", "run_agent.py", "--interactive"]

# Development stage
FROM base as development

# Switch back to root for development tools installation
USER root

# Install development dependencies
RUN pip install -e ".[dev]"

# Install additional development tools
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    tree \
    && rm -rf /var/lib/apt/lists/*

# Switch back to app user
USER app

# Development command
CMD ["python", "run_agent.py", "--interactive", "--debug"]

# Production stage
FROM base as production

# Install only production dependencies
RUN pip install --no-dev .

# Remove unnecessary files
RUN rm -rf /app/tests /app/examples /app/docs /app/.git

# Production command
CMD ["python", "run_agent.py", "--config", "config/config.yaml"]