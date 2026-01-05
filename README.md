# Multi-Agent Building Area Calculation

This project contains a multi-agent system for calculating building areas from IFC files.

## Deployment to Zeabur

This project is configured to be deployed using **Docker Compose**.

### Prerequisites

- Ensure you are deploying the **root** of the repository (do not set "Root Directory" to `frontend` or `backend` unless deploying services individually).
- Zeabur should automatically detect `docker-compose.yaml` (or `docker-compose.yml`) in the root.

### Troubleshooting

If Zeabur fails to detect `docker-compose.yaml` and tries to build as a static site (using `zeabur/caddy-static`):

1. **Check Service Settings**: Ensure you haven't accidentally selected "Static Site" or set a subdirectory as the Root Directory.
2. **Re-deploy**: Sometimes Zeabur caches the build configuration. Try triggering a new deployment or creating a new Project in Zeabur.
3. **Docker Compose Support**: Zeabur supports Docker Compose v3.8. The file has been renamed to `docker-compose.yaml` to follow best practices.

## Local Development

1. **Backend**:
   ```bash
   cd backend
   pip install -r requirements.txt
   uvicorn main:app --reload
   ```

2. **Frontend**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

3. **Docker Compose**:
   ```bash
   docker-compose up --build
   ```
