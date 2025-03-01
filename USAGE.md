# 10-K Processing API

This document provides instructions on how to run and use the 10-K processing API.

## Running the Application Locally

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

   Note: The application uses the `pinecone` package (not `pinecone-client` which is deprecated) for vector database operations.

2. Make sure you have authenticated with Google Cloud:
   ```
   gcloud auth application-default login
   ```

   This will authenticate your local environment to use Google Cloud services through the Application Default Credentials (ADC) mechanism. The API is configured to use this authentication for the Gemini API calls and Google Cloud Storage access.

3. Run the FastAPI application:
   ```
   uvicorn app.main:app --reload
   ```

4. The application will be available at `http://localhost:8000`

## Using the API Endpoint

The API provides an endpoint to process 10-K reports and store them in Pinecone. This endpoint now reads directly from a hardcoded Google Cloud Storage path.

### POST /api/v1/process-10k

This endpoint reads the 10-K report from the hardcoded GCS path (`gs://finn-cleaned-data/10k_files/aapl_10k.json`), processes it, and stores the embeddings in Pinecone.

**Request:**
- Method: POST
- URL: `http://localhost:8000/api/v1/process-10k`
- No body parameters required

**Sample Request using curl:**
```bash
curl -X POST "http://localhost:8000/api/v1/process-10k" \
  -H "accept: application/json"
```

**Sample Request using Python:**
```python
import requests

url = "http://localhost:8000/api/v1/process-10k"
response = requests.post(url)
print(response.json())
```

**Response:**
```json
{
  "success": true,
  "gcs_path": "gs://finn-cleaned-data/10k_files/aapl_10k.json",
  "symbol": "AAPL",
  "period": "2023-09-30",
  "total_vectors_added": 4,
  "processed_sections": [
    {
      "section_name": "ITEM 1. BUSINESS",
      "vectors_added": 2
    },
    {
      "section_name": "ITEM 1A. RISK FACTORS",
      "vectors_added": 2
    }
  ]
}
```

## Interactive Documentation

FastAPI provides interactive API documentation. You can access it at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Notes

- The API currently reads from a hardcoded Google Cloud Storage path. This is configured in the `app/routes/process_10k.py` file.
- The API uses Google Cloud authentication through Application Default Credentials (ADC).
- The Pinecone settings are also hardcoded in the `app/routes/process_10k.py` file.

- The API currently uses hardcoded values for Google Cloud and Pinecone settings. For production use, these should be moved to environment variables.
- The current implementation processes 10-K reports in the format shown in the `test_10k.json` file.
- The API is configured to run locally, but can be deployed to Google Cloud Run using the provided Dockerfile. 