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
import logging
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("10k_processing")

router = APIRouter()

# Global counters for tracking failed API calls
failed_cleaning_requests = 0
failed_tagging_requests = 0 
failed_embedding_requests = 0

# Global counters for tracking embeddings and storage operations
embeddings_created = 0
embeddings_pushed_to_pinecone = 0
records_pushed_to_gcs = 0

# Rate limiter for Gemini API to respect Google's quotas (200 requests per minute)
def upload_to_gcs_sync(bucket_name, file_path, data):
    global records_pushed_to_gcs
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_path)
        blob.upload_from_string(json.dumps(data), content_type='application/json')
        records_pushed_to_gcs += 1
        logger.info(f"Successfully uploaded file to GCS: {file_path}")
    except Exception as e:
        logger.error(f"Failed to upload file to GCS: {file_path}. Error: {str(e)}")
        pass

class GeminiRateLimiter:
    def __init__(self, max_requests=30, period=60, min_interval=0.1):
        self.max_requests = max_requests
        self.period = period
        self.min_interval = min_interval
        self.request_times = []
        self.last_request_time = 0
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        async with self.lock:
            now = datetime.now()
            cutoff = now - timedelta(seconds=self.period)
            self.request_times = [t for t in self.request_times if t > cutoff]
            if len(self.request_times) >= self.max_requests:
                oldest = self.request_times[0]
                rate_limit_wait = (oldest + timedelta(seconds=self.period) - now).total_seconds()
            else:
                rate_limit_wait = 0
            time_since_last = now.timestamp() - self.last_request_time
            min_interval_wait = self.min_interval - time_since_last if time_since_last < self.min_interval else 0
            wait_time = max(rate_limit_wait, min_interval_wait)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self.last_request_time = datetime.now().timestamp()
            self.request_times.append(datetime.now())
    
    def get_quota_usage(self):
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.period)
        current_requests = len([t for t in self.request_times if t > cutoff])
        return (current_requests / self.max_requests) * 100

gemini_limiter = GeminiRateLimiter(max_requests=30, period=60, min_interval=0.1)

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

SEC_HEADERS = {
    "User-Agent": "YourCompanyName YourAppName (your.email@example.com)"
}

class ProcessTickerRequest(BaseModel):
    tickers: Union[str, List[str]]
    skip_embedding: bool = False
    fiscal_year: Optional[str] = "2023"

def normalize_vector_id(raw_id: str) -> str:
    return unicodedata.normalize('NFKD', raw_id).encode('ascii', 'ignore').decode('ascii')

async def clean_text_with_gemini_async(text: str, max_retries=5, initial_delay=4) -> str:
    global failed_cleaning_requests
    # Create a truncated version of the text for logging (first 100 chars)
    truncated_text = text[:100] + "..." if len(text) > 100 else text
    
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
            await gemini_limiter.acquire()
            quota_usage = gemini_limiter.get_quota_usage()
            response = await loop.run_in_executor(None, lambda: client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            ))
            # Check if response.text is None before calling strip()
            if response.text is None:
                logger.warning(f"Gemini returned None response for text: {truncated_text}")
                return ""  # Return empty string to skip this section
            cleaned_text = response.text.strip()
            return cleaned_text
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                base_delay = initial_delay * (2 ** attempt)
                jitter = base_delay * 0.2 * (0.5 - random.random())
                delay = base_delay + jitter
                await asyncio.sleep(delay)
                if attempt == max_retries - 1:
                    failed_cleaning_requests += 1
                    logger.error(f"Failed to clean text after {max_retries} attempts due to rate limiting. Error: {str(e)}. Text: {truncated_text}")
                    return text
            else:
                failed_cleaning_requests += 1
                logger.error(f"Failed to clean text due to error: {str(e)}. Text: {truncated_text}")
                return ""  # Return empty string to skip this section on error
    failed_cleaning_requests += 1
    logger.error(f"Failed to clean text after exhausting all retries. Text: {truncated_text}")
    return ""  # Return empty string to skip this section after all retries

async def create_tags_with_gemini_async(text: str, section_name: str, symbol: str, num_tags: int = 2, max_retries=5, initial_delay=4) -> list:
    global failed_tagging_requests
    # Create a truncated version of the text for logging (first 100 chars)
    truncated_text = text[:100] + "..." if len(text) > 100 else text
    
    prompt = (
        "You are a financial analyst. You are given a part of a 10-K report. Your job is to create a list of exactly "
        f"{num_tags} tags that capture the key topics or points in this text. Return the tags in a list in this format: "
        f"['tag1', 'tag2', ...], where each tag is no longer than 3-4 words. Do not return any other text. Here is the text:" + text
    )
    loop = asyncio.get_event_loop()
    for attempt in range(max_retries):
        try:
            await gemini_limiter.acquire()
            quota_usage = gemini_limiter.get_quota_usage()
            response = await loop.run_in_executor(None, lambda: client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            ))
            tags_text = response.text.strip()
            try:
                tags_list = eval(tags_text)
                return tags_list
            except:
                failed_tagging_requests += 1
                logger.error(f"Failed to parse tags response for section '{section_name}' of {symbol}. Response: {tags_text}. Text: {truncated_text}")
                return [f"default_tag{i+1}" for i in range(num_tags)]
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                base_delay = initial_delay * (2 ** attempt)
                jitter = base_delay * 0.2 * (0.5 - random.random())
                delay = base_delay + jitter
                await asyncio.sleep(delay)
                if attempt == max_retries - 1:
                    failed_tagging_requests += 1
                    logger.error(f"Failed to create tags for section '{section_name}' of {symbol} after {max_retries} attempts due to rate limiting. Error: {str(e)}. Text: {truncated_text}")
                    return [f"default_tag{i+1}" for i in range(num_tags)]
            else:
                failed_tagging_requests += 1
                logger.error(f"Failed to create tags for section '{section_name}' of {symbol} due to error: {str(e)}. Text: {truncated_text}")
                return [f"default_tag{i+1}" for i in range(num_tags)]
    failed_tagging_requests += 1
    logger.error(f"Failed to create tags for section '{section_name}' of {symbol} after exhausting all retries. Text: {truncated_text}")
    return [f"default_tag{i+1}" for i in range(num_tags)]

def split_section_into_chunks(text, max_length=2048, min_length=100):
    """
    Split a section of text into chunks of maximum length, ensuring that sentences stay together.
    Uses a non-recursive approach to avoid maximum recursion depth errors.
    
    Args:
        text: The text to split
        max_length: Maximum length of each chunk
        min_length: Minimum length for a chunk to be considered valid
        
    Returns:
        List of tuples (index, chunk_text)
    """
    # If text is already small enough, return it as a single chunk
    if len(text) <= max_length:
        if len(text) >= min_length:
            return [(0, text)]
        return []
    
    # Split text into sentences
    sentences = re.split(r'(?<=\.)\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return []
    
    # If there's only one sentence and it's too long, return it anyway
    if len(sentences) <= 1:
        if len(text) >= min_length:
            return [(0, text)]
        return []
    
    # Calculate how many chunks we need based on total text length
    total_length = len(text)
    num_chunks_needed = max(1, math.ceil(total_length / max_length))
    
    # Calculate target sentences per chunk
    total_sentences = len(sentences)
    target_sentences_per_chunk = math.ceil(total_sentences / num_chunks_needed)
    
    chunks = []
    current_chunk = []
    current_length = 0
    chunk_index = 0
    
    for sentence in sentences:
        # If adding this sentence would exceed max_length and we already have content,
        # finalize the current chunk and start a new one
        if current_length + len(sentence) > max_length and current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= min_length:
                chunks.append((chunk_index, chunk_text))
                chunk_index += 1
            current_chunk = []
            current_length = 0
        
        # Add the sentence to the current chunk
        current_chunk.append(sentence)
        current_length += len(sentence) + 1  # +1 for the space
    
    # Don't forget the last chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        if len(chunk_text) >= min_length:
            chunks.append((chunk_index, chunk_text))
    
    return chunks

def init_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if INDEX_NAME not in pc.list_indexes().names():
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

async def process_section_to_pinecone(pc, section_text, section_name, symbol, period, namespace=None, fiscal_year=None, number_of_chunks: Optional[int] = None):
    global failed_embedding_requests, embeddings_created, embeddings_pushed_to_pinecone
    
    # Log the section being processed
    truncated_text = section_text[:100] + "..." if len(section_text) > 100 else section_text
    logger.info(f"Processing section '{section_name}' for {symbol}, period {period}. Text length: {len(section_text)}")
    
    cleaned_text = await clean_text_with_gemini_async(section_text)
    if not cleaned_text:
        logger.warning(f"Section '{section_name}' for {symbol} returned empty after cleaning. Original text: {truncated_text}")
        return 0
    cleaned_text = ' '.join([para.strip() for para in cleaned_text.split('\n') if para.strip()])
    cleaned_text = ' '.join(cleaned_text.split())
    chunks = split_section_into_chunks(cleaned_text, max_length=2048, min_length=100)
    if not chunks:
        logger.warning(f"No chunks created for section '{section_name}' of {symbol} after cleaning. Cleaned text length: {len(cleaned_text)}")
        return 0
    section_tags = await create_tags_with_gemini_async(cleaned_text, section_name, symbol, num_tags=1)
    section_tag = section_tags[0] if section_tags else "default_section_tag"
    chunk_tags_tasks = [create_tags_with_gemini_async(chunk, section_name, symbol, num_tags=2) for _, chunk in chunks]
    all_chunk_tags = await asyncio.gather(*chunk_tags_tasks)
    if fiscal_year is None:
        year = period[:4] if period and len(period) >= 4 else "unknown"
    else:
        year = fiscal_year
    updated_namespace = f"{symbol}-{year}"
    index = pc.Index(INDEX_NAME, host=INDEX_HOST)
    records = []
    MAX_EMBEDDING_BATCH_SIZE = 96
    chunk_texts = [chunk for _, chunk in chunks]
    try:
        all_embeddings = []
        for i in range(0, len(chunk_texts), MAX_EMBEDDING_BATCH_SIZE):
            batch = chunk_texts[i:i + MAX_EMBEDDING_BATCH_SIZE]
            try:
                batch_embeddings = pc.inference.embed(
                    model=EMBEDDING_MODEL,
                    inputs=batch,
                    parameters={"input_type": "passage", "truncate": "END"}
                )
                all_embeddings.extend(batch_embeddings)
                embeddings_created += len(batch_embeddings)
            except Exception as e:
                failed_embedding_requests += 1
                logger.error(f"Failed to create embeddings for batch {i//MAX_EMBEDDING_BATCH_SIZE + 1} of section '{section_name}' for {symbol}. Error: {str(e)}")
                continue
        for (chunk_idx, chunk), embedding, chunk_tags in zip(chunks, all_embeddings, all_chunk_tags):
            try:
                metatags = [section_tag] + chunk_tags
                raw_id = f"{symbol}-{period}-{section_name}-chunk_{chunk_idx}"
                vector_id = normalize_vector_id(raw_id)
                json_data = {
                    "symbol": symbol,
                    "period": period,
                    "section_name": section_name,
                    "chunk_index": chunk_idx,
                    "text": chunk,
                    "tags": metatags
                }
                # *** Updated Line ***
                file_name = f"llm_processed_10k/{vector_id}.json"  # Changed to include '10k_files/' prefix
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, upload_to_gcs_sync, GCS_BUCKET, file_name, json_data)
                record = {
                    "id": vector_id,
                    "values": embedding['values'],
                    "metadata": {
                        "metatags": metatags,
                        "text": chunk,
                    }
                }
                records.append(record)
            except Exception as e:
                pass
    except Exception as e:
        failed_embedding_requests += 1
        return 0
    if records:
        batch_size = 100
        total_batches = (len(records) + batch_size - 1) // batch_size
        async def upsert_batch(batch):
            global embeddings_pushed_to_pinecone
            loop = asyncio.get_event_loop()
            try:
                logger.info(f"Upserting batch of {len(batch)} vectors to Pinecone namespace '{updated_namespace}'")
                await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: index.upsert(vectors=batch, namespace=updated_namespace)),
                    timeout=30.0
                )
                embeddings_pushed_to_pinecone += len(batch)
                logger.info(f"Successfully upserted {len(batch)} vectors to Pinecone namespace '{updated_namespace}'")
            except asyncio.TimeoutError:
                logger.error(f"Timeout error while upserting batch of {len(batch)} vectors to Pinecone namespace '{updated_namespace}'")
                raise
            except Exception as e:
                logger.error(f"Error upserting batch of {len(batch)} vectors to Pinecone namespace '{updated_namespace}': {str(e)}")
                raise
        tasks = []
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            tasks.append(upsert_batch(batch))
        await asyncio.gather(*tasks)
    return len(records)

def read_from_gcs(bucket_name, file_path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    return blob.download_as_string()

@router.post("/process-10k")
async def process_10k_gcs():
    global failed_cleaning_requests, failed_tagging_requests, failed_embedding_requests
    global embeddings_created, embeddings_pushed_to_pinecone, records_pushed_to_gcs
    failed_cleaning_requests = 0
    failed_tagging_requests = 0
    failed_embedding_requests = 0
    embeddings_created = 0
    embeddings_pushed_to_pinecone = 0
    records_pushed_to_gcs = 0
    try:
        pc = init_pinecone()
        gcs_path = f"gs://{GCS_BUCKET}/{GCS_FILE_PATH}"
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
    global failed_cleaning_requests, failed_tagging_requests, failed_embedding_requests
    global embeddings_created, embeddings_pushed_to_pinecone, records_pushed_to_gcs
    failed_cleaning_requests = 0
    failed_tagging_requests = 0
    failed_embedding_requests = 0
    embeddings_created = 0
    embeddings_pushed_to_pinecone = 0
    records_pushed_to_gcs = 0
    try:
        tickers = request.tickers
        if isinstance(tickers, str):
            tickers = [tickers]
        tickers = [ticker.upper() for ticker in tickers]
        pc = init_pinecone()
        
        # Define an async function to process a single ticker completely
        async def process_single_ticker(ticker):
            try:
                cik = await get_cik_for_ticker(ticker)
                if not cik:
                    return {
                        "ticker": ticker,
                        "success": False,
                        "error": "CIK not found for ticker"
                    }
                
                filing_data = await fetch_10k_filing(cik, ticker, request.fiscal_year)
                if not filing_data:
                    return {
                        "ticker": ticker,
                        "success": False,
                        "error": "Failed to fetch 10-K filing"
                    }
                
                if request.skip_embedding:
                    return {
                        "ticker": ticker,
                        "success": True,
                        "message": "10-K filing fetched successfully, embedding skipped"
                    }
                
                result = await process_10k_data(pc, filing_data, namespace=ticker)
                
                return {
                    "ticker": ticker,
                    "success": True,
                    "total_vectors_added": result["total_vectors_added"],
                    "sections_processed": len(result["processed_sections"]),
                    "failed_requests": {
                        "cleaning": failed_cleaning_requests,
                        "tagging": failed_tagging_requests,
                        "embedding": failed_embedding_requests
                    }
                }
            except Exception as e:
                return {
                    "ticker": ticker,
                    "success": False,
                    "error": str(e)
                }
        
        # Process all tickers and wait for all of them to complete
        results = await asyncio.gather(*[process_single_ticker(ticker) for ticker in tickers])
        
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
    try:
        sec_cik_url = "https://www.sec.gov/files/company_tickers.json"
        response = requests.get(sec_cik_url, headers=SEC_HEADERS)
        if response.status_code != 200:
            return None
        cik_data = response.json()
        for entry in cik_data.values():
            if entry['ticker'] == ticker:
                return str(entry['cik_str']).zfill(10)
        return None
    except Exception as e:
        return None

async def fetch_10k_filing(cik: str, symbol: str, fiscal_year: str = "2023") -> Optional[dict]:
    try:
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        response = requests.get(url, headers=SEC_HEADERS)
        if response.status_code != 200:
            return None
        data = response.json()
        recent_filings = data['filings']['recent']
        forms = recent_filings['form']
        accession_numbers = recent_filings['accessionNumber']
        primary_documents = recent_filings['primaryDocument']
        filing_dates = recent_filings['filingDate']
        report_dates = recent_filings['reportDate']
        ten_k_indices = [i for i, form in enumerate(forms) if form == "10-K"]
        if not ten_k_indices:
            return None
        expected_filing_year = str(int(fiscal_year) + 1)
        fallback_filing_year = str(int(fiscal_year) + 2)
        index = None
        filing_year = None
        for i in ten_k_indices:
            filing_date = filing_dates[i]
            if filing_date.startswith(expected_filing_year):
                index = i
                filing_year = expected_filing_year
                break
        if index is None:
            for i in ten_k_indices:
                filing_date = filing_dates[i]
                if filing_date.startswith(fallback_filing_year):
                    index = i
                    filing_year = fallback_filing_year
                    break
        if index is None:
            return None
        accession = accession_numbers[index]
        primary_doc = primary_documents[index]
        accession_no = accession.replace("-", "")
        base_url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession_no}"
        html_url = f"{base_url}/{primary_doc}"
        html_response = requests.get(html_url, headers=SEC_HEADERS)
        if html_response.status_code != 200:
            return None
        soup = BeautifulSoup(html_response.text, 'html.parser')
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
        for element in soup.find_all(['p', 'div', 'span', 'table']):
            if not element.get_text(strip=True):
                continue
            text = clean_text(element.get_text())
            if re.match(r'^ITEM\s+\d+[A-Z]?\.?', text, re.IGNORECASE) or \
               re.match(r'^PART\s+[IVX]+\.?', text, re.IGNORECASE):
                current_section = text
                filing_content["content"][current_section] = []
            else:
                filing_content["content"][current_section].append(text)
        return filing_content
    except Exception as e:
        return None


def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

async def process_10k_data(pc, data, source_path=None, namespace=None):
    global failed_cleaning_requests, failed_tagging_requests, failed_embedding_requests
    global embeddings_created, embeddings_pushed_to_pinecone, records_pushed_to_gcs
    meta = data.get("metadata", {})
    symbol = meta.get("symbol", "unknown")
    period = meta.get("period_of_report", "unknown")
    fiscal_year = meta.get("fiscal_year", None)
    content_sections = data.get("content", {})
    MIN_CHARS = 200
    total_vectors = 0
    processed_sections = []
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
    if fiscal_year:
        updated_namespace = f"{symbol}-{fiscal_year}"
    else:
        year = period[:4] if period and len(period) >= 4 else "unknown"
        updated_namespace = f"{symbol}-{year}"
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
        },
        "metrics": {
            "embeddings_created": embeddings_created,
            "embeddings_pushed_to_pinecone": embeddings_pushed_to_pinecone,
            "records_pushed_to_gcs": records_pushed_to_gcs
        }
    }
    if source_path:
        response["source_path"] = source_path
    return response