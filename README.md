# Cloud Run API Boilerplate

A minimal FastAPI application template for deployment to Google Cloud Run.

## Features

- Basic FastAPI application structure
- Containerized with Docker
- Ready for deployment to Cloud Run
- Minimal dependencies

## Getting Started

### Local Development

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   uvicorn app.main:app --reload
   ```
4. Visit `http://localhost:8000` in your browser

### Docker Build and Run

Build the Docker image:
```
docker build -t cloud-run-boilerplate .
```

Run the container:
```
docker run -p 8000:8000 cloud-run-boilerplate
```

### Deploying to Cloud Run

```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/cloud-run-boilerplate

# Deploy to Cloud Run
gcloud run deploy cloud-run-api --image gcr.io/YOUR_PROJECT_ID/cloud-run-boilerplate --platform managed
```

Replace `YOUR_PROJECT_ID` with your Google Cloud project ID.