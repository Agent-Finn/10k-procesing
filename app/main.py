"""
FastAPI application for 10-K processing.
"""
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Union

from app.processor import Processor

# Create FastAPI application
app = FastAPI(title="SEC 10-K Processing API")

# Initialize processor
processor = Processor()

class ProcessTickerRequest(BaseModel):
    """Request model for processing 10-K reports by ticker symbol."""
    tickers: Union[str, List[str]]
    skip_embedding: bool = False
    fiscal_year: Optional[str] = "2023"  # Default to fiscal year 2023 (2024 filings)

@app.get("/")
def read_root():
    """Root endpoint."""
    return {"message": "Welcome to SEC 10-K Processing API"}

@app.post("/api/v1/process-10k")
async def process_10k_gcs():
    """Process a 10-K report from Google Cloud Storage."""
    try:
        result = await processor.process_from_gcs()
        return result
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/process-by-ticker")
async def process_by_ticker(request: ProcessTickerRequest):
    """Process 10-K reports by ticker symbol(s)."""
    try:
        result = await processor.process_by_ticker(
            tickers=request.tickers,
            skip_embedding=request.skip_embedding,
            fiscal_year=request.fiscal_year
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Processing failed"))
            
        return result
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/process-10k-json")
async def process_10k_json(data: Dict[str, Any] = Body(...)):
    """Process a 10-K report from JSON data."""
    try:
        result = await processor.process_from_json(data)
        return result
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))