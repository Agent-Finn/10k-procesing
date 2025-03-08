# Cloud Run API Boilerplate

A minimal FastAPI application template for deployment to Google Cloud Run.

## Features

- Basic FastAPI application structure
- Containerized with Docker
- Ready for deployment to Cloud Run
- Minimal dependencies
- 10-K Processing API with Google Generative AI

## Getting Started

### Local Development

1. Clone this repository
2. Install dependencies and start virtual environment:
   ```
   pip install -r requirements.txt
   source venv/bin/activate
   gcloud auth application-default login  # For Google Cloud authentication
   ```
3. Run the application:
   ```
   uvicorn app.main:app --reload
   ```
4. Visit `http://localhost:8000` in your browser
5. API documentation is available at `http://localhost:8000/docs`

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

# 10-K Processing System

This system downloads 10-K reports for S&P 500 companies from the SEC EDGAR database, processes them, and stores semantically meaningful chunks in a Pinecone vector database for financial analysis.

## Components

1. **SEC Scraper**: Downloads and parses 10-K filings from the SEC EDGAR database
2. **Processing API**: FastAPI backend that processes 10-K reports and stores them in Pinecone
3. **Vector Database**: Pinecone vector database for semantic search and retrieval

## Requirements

- Python 3.9+
- FastAPI
- Google Cloud (for GCS and Vertex AI with Gemini)
- Pinecone account with an API key

## Installation

```bash
pip install -r requirements.txt
```

## Environment Variables

Export the following environment variables or use a `.env` file:

```
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your-google-credentials.json
PINECONE_API_KEY=your-pinecone-api-key
```

## Usage

### Starting the API Server

```bash
uvicorn app.main:app --reload
```

The API will be available at http://localhost:8000.

### API Endpoints

#### 1. Process by Ticker Symbol

Process 10-K reports directly by ticker symbol(s):

```bash
curl -X POST http://localhost:8000/process-by-ticker \
  -H "Content-Type: application/json" \
  -d '{"tickers": ["AAPL", "MSFT", "GOOGL"]}'
```

This endpoint handles the entire process:
- Fetches the 10-K report from SEC EDGAR
- Cleans and processes the content
- Generates embeddings
- Stores the data in Pinecone

You can also skip the embedding step and just fetch the 10-K:

```bash
curl -X POST http://localhost:8000/process-by-ticker \
  -H "Content-Type: application/json" \
  -d '{"tickers": "AAPL", "skip_embedding": true}'
```

#### 2. Process from File (GCS)

Process a 10-K report JSON file from Google Cloud Storage:

```bash
curl -X POST http://localhost:8000/process-10k
```

### Processing S&P 500 Companies (Using the Script)

If you prefer to use the Python script instead of the API:

```bash
python sp500_10k_processor.py
```

To process specific companies:

```bash
python sp500_10k_processor.py --symbols AAPL MSFT GOOGL
```

## Folder Structure

- `app/`: FastAPI application
  - `routes/process_10k.py`: API endpoints for processing 10-K reports
  - `main.py`: FastAPI entry point
- `sp500_10k_processor.py`: Script to download and process 10-K reports
- `sp500_10k/`: Directory where downloaded 10-K reports are stored

## Notes

- The SEC EDGAR API has a rate limit of 10 requests per second, which the system respects.
- You should update the User-Agent header in the code with your company name, app name, and email address.
- Google Vertex AI and Pinecone API keys are hard-coded in the script for simplicity. In a production environment, these should be stored in environment variables or a secure secret management system.