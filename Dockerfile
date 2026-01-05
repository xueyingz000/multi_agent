# Stage 1: Build Frontend
FROM node:18-alpine as frontend-build
WORKDIR /app/frontend

# Copy dependency definitions
COPY frontend/package.json frontend/package-lock.json ./
RUN npm install --legacy-peer-deps

# Copy source code and build
COPY frontend ./
RUN npm run build

# Stage 2: Final Image (Python + Nginx)
FROM python:3.10-slim

# Install system dependencies
# - nginx & supervisor for process management
# - libgl1 & libglib2.0-0 for ifcopenshell/graphics dependencies
RUN apt-get update && apt-get install -y \
    nginx \
    supervisor \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Setup Backend
WORKDIR /app/backend
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend ./

# Create temp directory
RUN mkdir -p temp

# Setup Frontend (Copy built assets from Stage 1)
COPY --from=frontend-build /app/frontend/dist /usr/share/nginx/html

# Setup Nginx Config
RUN rm /etc/nginx/sites-enabled/default
COPY nginx_root.conf /etc/nginx/conf.d/default.conf

# Setup Supervisor Config
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose port 80
EXPOSE 80

# Start Supervisor (which starts Nginx and Uvicorn)
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
