from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
import json
import time
import unicodedata
import sys
from typing import Optional, Dict, Any, List, Union
import os
from google.cloud import storage
from google import genai
from pinecone import Pinecone, ServerlessSpec
import asyncio
from concurrent.futures import ThreadPoolExecutor
import requests
from bs4 import BeautifulSoup
import re
from pydantic import BaseModel
from datetime import datetime, timedelta
import random

router = APIRouter()

# Global counters for tracking failed API calls
failed_cleaning_requests = 0
failed_tagging_requests = 0 
failed_embedding_requests = 0

# Rate limiter for Gemini API to respect Google's quotas (200 requests per minute)
from datetime import datetime, timedelta
import asyncio

from datetime import datetime, timedelta
import asyncio

class GeminiRateLimiter:
    def __init__(self, max_requests=180, period=60, min_interval=0.1):
        """
        Initialize the rate limiter.
        
        Args:
            max_requests (int): Maximum number of requests allowed in the period (default: 180).
            period (int): Time period in seconds (default: 60 for 1 minute).
            min_interval (float): Minimum time in seconds between consecutive requests (default: 0.1).
        """
        self.max_requests = max_requests
        self.period = period  # in seconds
        self.min_interval = min_interval  # minimum seconds between requests
        self.request_times = []  # List to store timestamps of requests
        self.last_request_time = 0  # Timestamp of the last request
        self.lock = asyncio.Lock()  # Ensure thread-safe updates
    
    async def acquire(self):
        """
        Acquire permission to make an API request, waiting if necessary to respect rate limits.
        """
        async with self.lock:
            now = datetime.now()
            # Remove requests older than the period
            cutoff = now - timedelta(seconds=self.period)
            self.request_times = [t for t in self.request_times if t > cutoff]
            
            # Calculate wait time for the rate limit
            if len(self.request_times) >= self.max_requests:
                oldest = self.request_times[0]
                rate_limit_wait = (oldest + timedelta(seconds=self.period) - now).total_seconds()
            else:
                rate_limit_wait = 0
            
            # Calculate wait time for the minimum interval
            time_since_last = now.timestamp() - self.last_request_time
            if time_since_last < self.min_interval:
                min_interval_wait = self.min_interval - time_since_last
            else:
                min_interval_wait = 0
            
            # Wait for the longer of the two constraints
            wait_time = max(rate_limit_wait, min_interval_wait)
            if wait_time > 0:
                print(f"[RATE LIMITER] Waiting {wait_time:.2f} seconds (Rate limit: {rate_limit_wait:.2f}s, Min interval: {min_interval_wait:.2f}s)")
                await asyncio.sleep(wait_time)
            
            # Update state for this request
            self.last_request_time = datetime.now().timestamp()
            self.request_times.append(datetime.now())
    
    def get_quota_usage(self):
        """
        Return current quota usage as a percentage.
        """
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.period)
        current_requests = len([t for t in self.request_times if t > cutoff])
        return (current_requests / self.max_requests) * 100

# Initialize the rate limiter
# Initialize the rate limiter with a buffer (180 instead of 200) and a small min_interval
gemini_limiter = GeminiRateLimiter(max_requests=180, period=60, min_interval=0.1)

# Google Vertex AI settings
PROJECT_ID = "nimble-chess-449208-f3"
LOCATION = "us-central1"

# Initialize the Google Generative AI client
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

# Google Cloud Storage settings
GCS_BUCKET = "finn-cleaned-data"
GCS_FILE_PATH = "10k_files/aapl_10k.json"

# Pinecone settings
PINECONE_API_KEY = "pcsk_6rBfZw_6oEhbN34NsgDsq57Gcj6CZhmQnXBFujB33XUcEsmrgVWCvBC5Lyv3KKgBd7cweP"
INDEX_NAME = "10k"
NAMESPACE = "AAPL"
EMBEDDING_MODEL = "llama-text-embed-v2"
EMBEDDING_DIMENSIONS = 1024
INDEX_HOST = "https://10k-gsf4yiq.svc.gcp-us-central1-4a9f.pinecone.io"

# SEC API headers
SEC_HEADERS = {
    "User-Agent": "YourCompanyName YourAppName (your.email@example.com)"
}

class ProcessTickerRequest(BaseModel):
    """Request model for processing 10-K reports by ticker symbol."""
    tickers: Union[str, List[str]]
    skip_embedding: bool = False
    fiscal_year: Optional[str] = "2023"  # Default to fiscal year 2023 (2024 filings)

# Helper function to normalize vector IDs to ASCII
def normalize_vector_id(raw_id: str) -> str:
    return unicodedata.normalize('NFKD', raw_id).encode('ascii', 'ignore').decode('ascii')

async def clean_text_with_gemini_async(text: str, max_retries=5, initial_delay=4) -> str:
    """Asynchronously clean 10-K text using Gemini API with retry logic for rate limiting."""
    global failed_cleaning_requests
    
    prompt = (
        "Clean the following financial report text for analysis. Exclude legalese, addresses, filing info, signatures, "
        "checkmarks, tables of contents, and any tables or sections with numerical financial data. Remove page numbers, "
        "bullet points, extraneous headings, random characters, and any formatting that isn't relevant. For sections that "
        "are only headers or titles with no content, return an empty string. Also omit the 'Exhibit and Financial Statement "
        "Schedule' section, any parts talking about the 10k document itself. Do not summarize, add commentary or analysis. "
        "Either return the cleaned, meaningful text, or nothing at all. Here is the text:" + text
    )
    
    loop = asyncio.get_event_loop()
    
    for attempt in range(max_retries):
        try:
            # Wait for rate limiter approval before proceeding
            await gemini_limiter.acquire()
            
            quota_usage = gemini_limiter.get_quota_usage()
            print(f"\n[GEMINI REQUEST - CLEANING] Processing text of length: {len(text)} (Quota usage: {quota_usage:.1f}%)")
            print(f"First 100 chars: {text[:100]}...")
            
            response = await loop.run_in_executor(None, lambda: client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            ))
            cleaned_text = response.text.strip()
            
            print(f"[GEMINI RESPONSE - CLEANING] Cleaned text length: {len(cleaned_text)}")
            print(f"First 100 chars of cleaned text: {cleaned_text[:100]}...")
            
            return cleaned_text
        
        except Exception as e:
            # Check if this is a rate limit error
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                # Calculate exponential backoff delay with jitter (randomness)
                # Jitter helps prevent thundering herd problem
                base_delay = initial_delay * (2 ** attempt)
                jitter = base_delay * 0.2 * (0.5 - random.random())  # ±10% randomness
                delay = base_delay + jitter
                
                print(f"[RATE LIMIT] Gemini API rate limited. Retrying in {delay:.2f} seconds... (Attempt {attempt+1}/{max_retries})")
                print(f"[RATE LIMIT] Error details: {str(e)}")
                await asyncio.sleep(delay)
                
                # If this was the last attempt, return the text as is
                if attempt == max_retries - 1:
                    print(f"Error cleaning text with Gemini API after {max_retries} attempts: {e}", file=sys.stderr)
                    failed_cleaning_requests += 1
                    print(f"[FAILED REQUESTS] Cleaning failures: {failed_cleaning_requests}")
                    return text
            else:
                # For non-rate-limit errors, don't retry
                print(f"Error cleaning text with Gemini API: {e}", file=sys.stderr)
                failed_cleaning_requests += 1
                print(f"[FAILED REQUESTS] Cleaning failures: {failed_cleaning_requests}")
                return text
    
    # If we get here, all retries failed
    failed_cleaning_requests += 1
    print(f"[FAILED REQUESTS] Cleaning failures: {failed_cleaning_requests}")
    return text

async def create_tags_with_gemini_async(text: str, section_name: str, symbol: str, max_retries=5, initial_delay=4) -> list:
    """Asynchronously create tags for a 10-K section using Gemini API with retry logic for rate limiting."""
    global failed_tagging_requests
    
    prompt = (
        "You are a financial analyst. You are given a section of a 10-K report. Your job is to create a list of exactly "
        "two tags for the section. Return the tags in a list in this format: ['tag1', 'tag2']. Do not return any other "
        "text. Here is the 10k section:" + text
    )
    
    loop = asyncio.get_event_loop()
    
    for attempt in range(max_retries):
        try:
            # Wait for rate limiter approval before proceeding
            await gemini_limiter.acquire()
            
            quota_usage = gemini_limiter.get_quota_usage()
            print(f"\n[GEMINI REQUEST - TAGGING] Creating tags for section: {section_name} (Quota usage: {quota_usage:.1f}%)")
            
            response = await loop.run_in_executor(None, lambda: client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            ))
            tags_text = response.text.strip()
            print(f"[GEMINI RESPONSE - TAGGING] Raw tags response: {tags_text}")
            
            try:
                tags_list = eval(tags_text)
                print(f"Parsed tags: {tags_list}")
                return tags_list
            except:
                print(f"Failed to parse tags: {tags_text}")
                failed_tagging_requests += 1
                print(f"[FAILED REQUESTS] Tagging failures: {failed_tagging_requests}")
                return ["financial_report", "10k_filing"]
        
        except Exception as e:
            # Check if this is a rate limit error
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                # Calculate exponential backoff delay with jitter (randomness)
                base_delay = initial_delay * (2 ** attempt)
                jitter = base_delay * 0.2 * (0.5 - random.random())  # ±10% randomness
                delay = base_delay + jitter
                
                print(f"[RATE LIMIT] Gemini API rate limited. Retrying in {delay:.2f} seconds... (Attempt {attempt+1}/{max_retries})")
                print(f"[RATE LIMIT] Error details: {str(e)}")
                await asyncio.sleep(delay)
                
                # If this was the last attempt, use default tags
                if attempt == max_retries - 1:
                    print(f"Error generating tags with Gemini API after {max_retries} attempts: {e}", file=sys.stderr)
                    failed_tagging_requests += 1
                    print(f"[FAILED REQUESTS] Tagging failures: {failed_tagging_requests}")
                    return ["financial_report", "10k_filing"]
            else:
                # For non-rate-limit errors, don't retry
                print(f"Error generating tags with Gemini API: {e}", file=sys.stderr)
                failed_tagging_requests += 1
                print(f"[FAILED REQUESTS] Tagging failures: {failed_tagging_requests}")
                return ["financial_report", "10k_filing"]
    
    # If we get here, all retries failed
    failed_tagging_requests += 1
    print(f"[FAILED REQUESTS] Tagging failures: {failed_tagging_requests}")
    return ["financial_report", "10k_filing"]

def init_pinecone():
    """Initialize Pinecone connection and ensure index exists."""
    print("Initializing Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating new index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSIONS,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="gcp",
                region="us-central1"
            )
        )
        while not pc.describe_index(INDEX_NAME).status['ready']:
            time.sleep(1)
    
    return pc

def chunk_text(text, chunk_size=1000, chunk_overlap=200, min_chunk_length=100):
    """Split text into chunks with specified size and overlap.
    Skips chunks with fewer than min_chunk_length characters."""
    if not text:
        return []
    step = chunk_size - chunk_overlap
    chunks = []
    for i in range(0, len(text), step):
        chunk = text[i:i+chunk_size].strip()
        if chunk and len(chunk) >= min_chunk_length:
            chunks.append((i, chunk))
    
    print(f"\n[CHUNKING] Generated {len(chunks)} chunks from text of length {len(text)}")
    if chunks:
        print(f"Sample chunk: {chunks[0][1][:100]}...")
    else:
        print(f"No chunks generated: all chunks were under minimum length of {min_chunk_length} characters")
    
    return chunks

async def process_section_to_pinecone(pc, section_text, section_name, symbol, period, namespace=None, fiscal_year=None):
    """Process a section: clean, chunk, embed in batches, and upload to Pinecone."""
    global failed_embedding_requests
    
    print(f"\n{'='*80}")
    print(f"PROCESSING SECTION: {section_name} (Length: {len(section_text)})")
    print(f"{'='*80}")
    
    # Clean the text asynchronously
    cleaned_text = await clean_text_with_gemini_async(section_text)
    if not cleaned_text:
        print(f"Skipping empty cleaned section: {section_name}")
        return 0
    
    cleaned_text = ' '.join([para.strip() for para in cleaned_text.split('\n') if para.strip()])
    cleaned_text = ' '.join(cleaned_text.split())
    
    # Chunk the cleaned text - skip chunks with fewer than 100 characters
    chunks = chunk_text(cleaned_text, min_chunk_length=100)
    if not chunks:
        print(f"No chunks generated for section: {section_name}")
        return 0
    
    # Generate tags asynchronously - only if we have chunks to process
    tags = await create_tags_with_gemini_async(cleaned_text, section_name, symbol)
    
    print(f"Generated {len(chunks)} chunks for section {section_name}")
    
    # Use the provided fiscal year if available, otherwise extract year from period (first 4 characters)
    if fiscal_year is None:
        year = period[:4] if period and len(period) >= 4 else "unknown"
    else:
        year = fiscal_year
    
    # Format namespace as {ticker}-{year}
    updated_namespace = f"{symbol}-{year}"
    
    print(f"Using namespace: {updated_namespace} (Fiscal Year: {year})")
    
    index = pc.Index(INDEX_NAME, host=INDEX_HOST)
    records = []
    
    # Batch embedding with maximum batch size of 96 (llama-text-embed-v2 limit)
    MAX_EMBEDDING_BATCH_SIZE = 96
    chunk_texts = [chunk for _, chunk in chunks]
    
    try:
        print(f"\n[EMBEDDING] Processing {len(chunk_texts)} chunks in batches of {MAX_EMBEDDING_BATCH_SIZE}")
        
        # Process embeddings in batches of 96 or fewer
        all_embeddings = []
        for i in range(0, len(chunk_texts), MAX_EMBEDDING_BATCH_SIZE):
            batch = chunk_texts[i:i+MAX_EMBEDDING_BATCH_SIZE]
            print(f"[EMBEDDING] Generating embeddings for batch {i//MAX_EMBEDDING_BATCH_SIZE + 1} with {len(batch)} chunks")
            
            try:
                # Generate embeddings for this batch
                batch_embeddings = pc.inference.embed(
                    model=EMBEDDING_MODEL,
                    inputs=batch,
                    parameters={"input_type": "passage", "truncate": "END"}
                )
                
                all_embeddings.extend(batch_embeddings)
                print(f"[EMBEDDING] Successfully generated {len(batch_embeddings)} embeddings for batch {i//MAX_EMBEDDING_BATCH_SIZE + 1}")
            except Exception as e:
                print(f"[EMBEDDING ERROR] Failed to generate embeddings for batch {i//MAX_EMBEDDING_BATCH_SIZE + 1}: {str(e)}")
                failed_embedding_requests += 1
                print(f"[FAILED REQUESTS] Embedding failures: {failed_embedding_requests}")
                # Continue with the next batch
                continue
        
        print(f"[EMBEDDING] Total embeddings generated: {len(all_embeddings)}")
        
        # Process embeddings and create records
        for (start, chunk), embedding in zip(chunks, all_embeddings):
            try:
                raw_id = f"{symbol}-{period}-{section_name}-{start}"
                vector_id = normalize_vector_id(raw_id)
                
                record = {
                    "id": vector_id,
                    "values": embedding['values'],
                    "metadata": {
                        "chunk_start": start,
                        "metatags": tags,
                        "text": chunk,  # Store the chunk text here
                    }
                }
                records.append(record)
            except Exception as e:
                print(f"Error processing chunk at start {start}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Error during batch embedding for section {section_name}: {e}", file=sys.stderr)
        failed_embedding_requests += 1
        print(f"[FAILED REQUESTS] Embedding failures: {failed_embedding_requests}")
        return 0
    
    # Upsert in batches using ThreadPoolExecutor
    if records:
        batch_size = 100
        total_batches = (len(records) + batch_size - 1) // batch_size
        
        print(f"\n[PINECONE] Uploading {len(records)} vectors in {total_batches} batches")
        
        async def upsert_batch(batch):
            loop = asyncio.get_event_loop()
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: index.upsert(vectors=batch, namespace=updated_namespace)),
                    timeout=30.0  # 30-second timeout
                )
            except asyncio.TimeoutError:
                print(f"Upsert batch timed out after 30 seconds", file=sys.stderr)
                raise
            except Exception as e:
                print(f"Error upserting batch: {e}", file=sys.stderr)
                raise
        
        tasks = []
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            tasks.append(upsert_batch(batch))
        
        await asyncio.gather(*tasks)
        print(f"[PINECONE] Successfully uploaded {len(records)} vectors for section {section_name}")
    
    return len(records)

def read_from_gcs(bucket_name, file_path):
    """Read a file from Google Cloud Storage."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    return blob.download_as_string()

@router.post("/process-10k")
async def process_10k_gcs():
    """Process a 10-K report JSON file from Google Cloud Storage and store it in Pinecone."""
    try:
        pc = init_pinecone()
        gcs_path = f"gs://{GCS_BUCKET}/{GCS_FILE_PATH}"
        print(f"Reading 10-K report from {gcs_path}")
        
        content = read_from_gcs(GCS_BUCKET, GCS_FILE_PATH)
        data = json.loads(content)
        
        return await process_10k_data(pc, data, gcs_path)
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@router.post("/process-by-ticker")
async def process_by_ticker(request: ProcessTickerRequest):
    """
    Process 10-K reports for specified tickers, handling fetching from SEC EDGAR, 
    cleaning, and loading into Pinecone.
    """
    global failed_cleaning_requests, failed_tagging_requests, failed_embedding_requests
    
    # Reset counters at the start of processing
    failed_cleaning_requests = 0
    failed_tagging_requests = 0
    failed_embedding_requests = 0
    
    try:
        tickers = request.tickers
        if isinstance(tickers, str):
            tickers = [tickers]
        
        tickers = [ticker.upper() for ticker in tickers]
        
        pc = init_pinecone()
        
        results = []
        for ticker in tickers:
            try:
                # Get CIK for ticker
                cik = await get_cik_for_ticker(ticker)
                if not cik:
                    results.append({
                        "ticker": ticker,
                        "success": False,
                        "error": "CIK not found for ticker"
                    })
                    continue
                
                # Fetch 10-K filing
                filing_data = await fetch_10k_filing(cik, ticker, request.fiscal_year)
                if not filing_data:
                    results.append({
                        "ticker": ticker,
                        "success": False,
                        "error": "Failed to fetch 10-K filing"
                    })
                    continue
                
                # Skip embedding if requested (useful for just fetching the data)
                if request.skip_embedding:
                    results.append({
                        "ticker": ticker,
                        "success": True,
                        "message": "10-K filing fetched successfully, embedding skipped"
                    })
                    continue
                
                # Process the 10-K data - use ticker as namespace
                result = await process_10k_data(pc, filing_data, namespace=ticker)
                results.append({
                    "ticker": ticker,
                    "success": True,
                    "total_vectors_added": result["total_vectors_added"],
                    "sections_processed": len(result["processed_sections"]),
                    "failed_requests": {
                        "cleaning": failed_cleaning_requests,
                        "tagging": failed_tagging_requests,
                        "embedding": failed_embedding_requests
                    }
                })
            
            except Exception as e:
                results.append({
                    "ticker": ticker,
                    "success": False,
                    "error": str(e)
                })
        
        # Log summary of failed requests
        print(f"\n{'='*80}")
        print(f"PROCESSING SUMMARY - FAILED REQUESTS")
        print(f"{'='*80}")
        print(f"Failed cleaning requests: {failed_cleaning_requests}")
        print(f"Failed tagging requests: {failed_tagging_requests}")
        print(f"Failed embedding requests: {failed_embedding_requests}")
        print(f"Total failed requests: {failed_cleaning_requests + failed_tagging_requests + failed_embedding_requests}")
        print(f"{'='*80}")
        
        return {
            "success": True,
            "results": results,
            "failed_requests_summary": {
                "cleaning": failed_cleaning_requests,
                "tagging": failed_tagging_requests,
                "embedding": failed_embedding_requests,
                "total": failed_cleaning_requests + failed_tagging_requests + failed_embedding_requests
            }
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False, 
                "error": str(e),
                "failed_requests_summary": {
                    "cleaning": failed_cleaning_requests,
                    "tagging": failed_tagging_requests,
                    "embedding": failed_embedding_requests,
                    "total": failed_cleaning_requests + failed_tagging_requests + failed_embedding_requests
                }
            }
        )

async def get_cik_for_ticker(ticker: str) -> Optional[str]:
    """Get the CIK number for a ticker symbol from SEC."""
    try:
        sec_cik_url = "https://www.sec.gov/files/company_tickers.json"
        response = requests.get(sec_cik_url, headers=SEC_HEADERS)
        
        if response.status_code != 200:
            print(f"Error fetching CIK data: {response.status_code}")
            return None
            
        cik_data = response.json()
        
        for entry in cik_data.values():
            if entry['ticker'] == ticker:
                return str(entry['cik_str']).zfill(10)
        
        return None
    
    except Exception as e:
        print(f"Error getting CIK for {ticker}: {str(e)}")
        return None

async def fetch_10k_filing(cik: str, symbol: str, fiscal_year: str = "2023") -> Optional[dict]:
    """Fetch and process 10-K filing for a company.
    
    Args:
        cik: The CIK number for the company
        symbol: The ticker symbol
        fiscal_year: The fiscal year to fetch (e.g., "2023" for 2024 filings, "2022" for 2023 filings)
    """
    try:
        # Fetch submission history
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        response = requests.get(url, headers=SEC_HEADERS)
        
        if response.status_code != 200:
            print(f"Error fetching submission data for {symbol}: {response.status_code}")
            return None
            
        data = response.json()
        recent_filings = data['filings']['recent']
        
        forms = recent_filings['form']
        accession_numbers = recent_filings['accessionNumber']
        primary_documents = recent_filings['primaryDocument']
        filing_dates = recent_filings['filingDate']
        report_dates = recent_filings['reportDate']
        
        # Find all 10-K filings and their dates
        ten_k_indices = [i for i, form in enumerate(forms) if form == "10-K"]
        
        if not ten_k_indices:
            print(f"No 10-K filing found for {symbol}")
            return None
        
        # Calculate the expected filing year based on fiscal year
        # Generally, companies file their 10-K for fiscal year X in year X+1
        expected_filing_year = str(int(fiscal_year) + 1)
        fallback_filing_year = str(int(fiscal_year) + 2)  # Sometimes there are late filings
        
        # Try to find a 10-K filing for the requested fiscal year
        index = None
        filing_year = None
        
        # First pass: look for exact match in expected filing year
        for i in ten_k_indices:
            filing_date = filing_dates[i]
            
            # Check for filings in the expected filing year
            if filing_date.startswith(expected_filing_year):
                index = i
                filing_year = expected_filing_year
                break
        
        # Second pass: if not found, check for late filings in the following year
        if index is None:
            for i in ten_k_indices:
                filing_date = filing_dates[i]
                
                if filing_date.startswith(fallback_filing_year):
                    index = i
                    filing_year = fallback_filing_year
                    break
        
        if index is None:
            print(f"No 10-K filing found for {symbol} fiscal year {fiscal_year}")
            return None
            
        accession = accession_numbers[index]
        primary_doc = primary_documents[index]
        
        # Build the filing URL
        accession_no = accession.replace("-", "")
        base_url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession_no}"
        html_url = f"{base_url}/{primary_doc}"
        
        # Fetch the filing
        html_response = requests.get(html_url, headers=SEC_HEADERS)
        
        if html_response.status_code != 200:
            print(f"Error downloading HTML content for {symbol}: {html_response.status_code}")
            return None
            
        soup = BeautifulSoup(html_response.text, 'html.parser')
        
        # Extract text content by sections
        filing_content = {
            "metadata": {
                "symbol": symbol,
                "cik": cik,
                "filing_type": "10-K",
                "filing_year": filing_year,
                "fiscal_year": fiscal_year,
                "accession_number": accession,
                "filing_date": filing_dates[index],
                "period_of_report": report_dates[index],
                "html_url": html_url
            },
            "content": {}
        }
        
        current_section = "GENERAL"
        filing_content["content"][current_section] = []
        
        # Process each element in the document
        for element in soup.find_all(['p', 'div', 'span', 'table']):
            if not element.get_text(strip=True):
                continue
                
            text = clean_text(element.get_text())
            
            # Look for section headers
            if re.match(r'^ITEM\s+\d+[A-Z]?\.?', text, re.IGNORECASE) or \
               re.match(r'^PART\s+[IVX]+\.?', text, re.IGNORECASE):
                current_section = text
                filing_content["content"][current_section] = []
            else:
                filing_content["content"][current_section].append(text)
                
        return filing_content
    
    except Exception as e:
        print(f"Error processing {symbol}: {str(e)}")
        return None

def clean_text(text):
    """Clean text by removing extra whitespace and newlines."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

async def process_10k_data(pc, data, source_path=None, namespace=None):
    """Process a 10-K report data and store it in Pinecone."""
    global failed_cleaning_requests, failed_tagging_requests, failed_embedding_requests
    
    meta = data.get("metadata", {})
    symbol = meta.get("symbol", "unknown")
    period = meta.get("period_of_report", "unknown")
    fiscal_year = meta.get("fiscal_year", None)
    content_sections = data.get("content", {})
    
    # No need to use provided namespace as we'll build it based on symbol and year
    print(f"Processing data for symbol: {symbol}, period: {period}, fiscal year: {fiscal_year}")
    
    MIN_CHARS = 200
    total_vectors = 0
    processed_sections = []
    
    # Process sections in parallel
    async def process_section(section_name, section_text):
        if section_name.upper() == "GENERAL" or not section_text or len(section_text) < MIN_CHARS:
            return None
        vectors_added = await process_section_to_pinecone(
            pc, 
            section_text, 
            section_name, 
            symbol, 
            period, 
            namespace=namespace,
            fiscal_year=fiscal_year
        )
        return {"section_name": section_name, "vectors_added": vectors_added} if vectors_added > 0 else None
    
    tasks = []
    for section_name, paragraphs in content_sections.items():
        section_text = "\n".join([str(p) for p in paragraphs]).strip()
        tasks.append(process_section(section_name, section_text))
    
    results = await asyncio.gather(*tasks)
    
    for result in results:
        if result:
            total_vectors += result["vectors_added"]
            processed_sections.append(result)
    
    # Use fiscal year for namespace if available, otherwise extract from period
    if fiscal_year:
        updated_namespace = f"{symbol}-{fiscal_year}"
    else:
        # Extract year from period for response
        year = period[:4] if period and len(period) >= 4 else "unknown"
        updated_namespace = f"{symbol}-{year}"
    
    # Log summary of failed requests
    print(f"\n{'='*80}")
    print(f"PROCESSING SUMMARY FOR {symbol}")
    print(f"{'='*80}")
    print(f"Total vectors added: {total_vectors}")
    print(f"Sections processed: {len(processed_sections)}")
    print(f"Failed cleaning requests: {failed_cleaning_requests}")
    print(f"Failed tagging requests: {failed_tagging_requests}")
    print(f"Failed embedding requests: {failed_embedding_requests}")
    print(f"Total failed requests: {failed_cleaning_requests + failed_tagging_requests + failed_embedding_requests}")
    print(f"{'='*80}")
    
    response = {
        "success": True,
        "symbol": symbol,
        "period": period,
        "total_vectors_added": total_vectors,
        "processed_sections": processed_sections,
        "namespace": updated_namespace,
        "failed_requests": {
            "cleaning": failed_cleaning_requests,
            "tagging": failed_tagging_requests,
            "embedding": failed_embedding_requests,
            "total": failed_cleaning_requests + failed_tagging_requests + failed_embedding_requests
        }
    }
    
    if source_path:
        response["source_path"] = source_path
    
    return response