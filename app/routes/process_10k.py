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
def upload_to_gcs_sync(bucket_name, file_path, data):
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_path)
        blob.upload_from_string(json.dumps(data), content_type='application/json')
        print(f"Uploaded {file_path} to GCS")
    except Exception as e:
        print(f"Error uploading {file_path} to GCS: {e}", file=sys.stderr)

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
                print(f"[RATE LIMITER] Waiting {wait_time:.2f} seconds (Rate limit: {rate_limit_wait:.2f}s, Min interval: {min_interval_wait:.2f}s)")
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
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                base_delay = initial_delay * (2 ** attempt)
                jitter = base_delay * 0.2 * (0.5 - random.random())
                delay = base_delay + jitter
                print(f"[RATE LIMIT] Gemini API rate limited. Retrying in {delay:.2f} seconds... (Attempt {attempt+1}/{max_retries})")
                print(f"[RATE LIMIT] Error details: {str(e)}")
                await asyncio.sleep(delay)
                if attempt == max_retries - 1:
                    print(f"Error cleaning text with Gemini API after {max_retries} attempts: {e}", file=sys.stderr)
                    failed_cleaning_requests += 1
                    print(f"[FAILED REQUESTS] Cleaning failures: {failed_cleaning_requests}")
                    return text
            else:
                print(f"Error cleaning text with Gemini API: {e}", file=sys.stderr)
                failed_cleaning_requests += 1
                print(f"[FAILED REQUESTS] Cleaning failures: {failed_cleaning_requests}")
                return text
    failed_cleaning_requests += 1
    print(f"[FAILED REQUESTS] Cleaning failures: {failed_cleaning_requests}")
    return text

async def create_tags_with_gemini_async(text: str, section_name: str, symbol: str, num_tags: int = 2, max_retries=5, initial_delay=4) -> list:
    global failed_tagging_requests
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
            print(f"\n[GEMINI REQUEST - TAGGING] Creating {num_tags} tags for section: {section_name} (Quota usage: {quota_usage:.1f}%)")
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
                return [f"default_tag{i+1}" for i in range(num_tags)]
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                base_delay = initial_delay * (2 ** attempt)
                jitter = base_delay * 0.2 * (0.5 - random.random())
                delay = base_delay + jitter
                print(f"[RATE LIMIT] Gemini API rate limited. Retrying in {delay:.2f} seconds... (Attempt {attempt+1}/{max_retries})")
                print(f"[RATE LIMIT] Error details: {str(e)}")
                await asyncio.sleep(delay)
                if attempt == max_retries - 1:
                    print(f"Error generating tags with Gemini API after {max_retries} attempts: {e}", file=sys.stderr)
                    failed_tagging_requests += 1
                    print(f"[FAILED REQUESTS] Tagging failures: {failed_tagging_requests}")
                    return [f"default_tag{i+1}" for i in range(num_tags)]
            else:
                print(f"Error generating tags with Gemini API: {e}", file=sys.stderr)
                failed_tagging_requests += 1
                print(f"[FAILED REQUESTS] Tagging failures: {failed_tagging_requests}")
                return [f"default_tag{i+1}" for i in range(num_tags)]
    failed_tagging_requests += 1
    print(f"[FAILED REQUESTS] Tagging failures: {failed_tagging_requests}")
    return [f"default_tag{i+1}" for i in range(num_tags)]

def split_section_into_chunks(text, max_length=2048, min_length=100):
    if len(text) <= max_length:
        if len(text) >= min_length:
            return [(0, text)]
        return []
    sentences = re.split(r'(?<=\.)\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return []
    num_sentences = len(sentences)
    if num_sentences <= 1:
        if len(text) >= min_length:
            return [(0, text)]
        return []
    def split_into_two(start_idx, end_idx):
        mid = (start_idx + end_idx) // 2
        left_chunk = ' '.join(sentences[start_idx:mid + 1])
        right_chunk = ' '.join(sentences[mid + 1:end_idx + 1])
        return left_chunk, right_chunk
    def recursive_split(chunk_text, chunk_idx):
        if len(chunk_text) <= max_length:
            if len(chunk_text) >= min_length:
                return [(chunk_idx, chunk_text)]
            return []
        local_sentences = re.split(r'(?<=\.)\s+', chunk_text.strip())
        local_sentences = [s.strip() for s in local_sentences if s.strip()]
        if len(local_sentences) <= 1:
            if len(chunk_text) >= min_length:
                return [(chunk_idx, chunk_text)]
            return []
        mid = len(local_sentences) // 2
        left_chunk = ' '.join(local_sentences[:mid + 1])
        right_chunk = ' '.join(local_sentences[mid + 1:])
        result = []
        result.extend(recursive_split(left_chunk, chunk_idx))
        if right_chunk:
            result.extend(recursive_split(right_chunk, chunk_idx + 1))
        return result
    left_chunk, right_chunk = split_into_two(0, num_sentences - 1)
    chunks = recursive_split(left_chunk, 0)
    if right_chunk:
        chunks.extend(recursive_split(right_chunk, len(chunks)))
    return [(i, chunk) for i, (_, chunk) in enumerate(chunks) if len(chunk) >= min_length]

def init_pinecone():
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

async def process_section_to_pinecone(pc, section_text, section_name, symbol, period, namespace=None, fiscal_year=None, number_of_chunks: Optional[int] = None):
    global failed_embedding_requests
    print(f"\n{'='*80}")
    print(f"PROCESSING SECTION: {section_name} (Length: {len(section_text)})")
    print(f"{'='*80}")
    cleaned_text = await clean_text_with_gemini_async(section_text)
    if not cleaned_text:
        print(f"Skipping empty cleaned section: {section_name}")
        return 0
    cleaned_text = ' '.join([para.strip() for para in cleaned_text.split('\n') if para.strip()])
    cleaned_text = ' '.join(cleaned_text.split())
    chunks = split_section_into_chunks(cleaned_text, max_length=2048, min_length=100)
    if not chunks:
        print(f"No chunks generated for section: {section_name}")
        return 0
    print(f"Generated {len(chunks)} chunks for section {section_name}")
    section_tags = await create_tags_with_gemini_async(cleaned_text, section_name, symbol, num_tags=1)
    section_tag = section_tags[0] if section_tags else "default_section_tag"
    chunk_tags_tasks = [create_tags_with_gemini_async(chunk, section_name, symbol, num_tags=2) for _, chunk in chunks]
    all_chunk_tags = await asyncio.gather(*chunk_tags_tasks)
    if fiscal_year is None:
        year = period[:4] if period and len(period) >= 4 else "unknown"
    else:
        year = fiscal_year
    updated_namespace = f"{symbol}-{year}"
    print(f"Using namespace: {updated_namespace} (Fiscal Year: {year})")
    index = pc.Index(INDEX_NAME, host=INDEX_HOST)
    records = []
    MAX_EMBEDDING_BATCH_SIZE = 96
    chunk_texts = [chunk for _, chunk in chunks]
    try:
        print(f"\n[EMBEDDING] Processing {len(chunk_texts)} chunks in batches of {MAX_EMBEDDING_BATCH_SIZE}")
        all_embeddings = []
        for i in range(0, len(chunk_texts), MAX_EMBEDDING_BATCH_SIZE):
            batch = chunk_texts[i:i + MAX_EMBEDDING_BATCH_SIZE]
            print(f"[EMBEDDING] Generating embeddings for batch {i // MAX_EMBEDDING_BATCH_SIZE + 1} with {len(batch)} chunks")
            try:
                batch_embeddings = pc.inference.embed(
                    model=EMBEDDING_MODEL,
                    inputs=batch,
                    parameters={"input_type": "passage", "truncate": "END"}
                )
                all_embeddings.extend(batch_embeddings)
                print(f"[EMBEDDING] Successfully generated {len(batch_embeddings)} embeddings for batch {i // MAX_EMBEDDING_BATCH_SIZE + 1}")
            except Exception as e:
                print(f"[EMBEDDING ERROR] Failed to generate embeddings for batch {i // MAX_EMBEDDING_BATCH_SIZE + 1}: {str(e)}")
                failed_embedding_requests += 1
                print(f"[FAILED REQUESTS] Embedding failures: {failed_embedding_requests}")
                continue
        print(f"[EMBEDDING] Total embeddings generated: {len(all_embeddings)}")
        for (chunk_idx, chunk), embedding, chunk_tags in zip(chunks, all_embeddings, all_chunk_tags):
            try:
                metatags = [section_tag] + chunk_tags
                raw_id = f"{symbol}-{period}-chunk_{chunk_idx}"
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
                print(f"Error processing chunk {chunk_idx}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Error during batch embedding for section {section_name}: {e}", file=sys.stderr)
        failed_embedding_requests += 1
        print(f"[FAILED REQUESTS] Embedding failures: {failed_embedding_requests}")
        return 0
    if records:
        batch_size = 100
        total_batches = (len(records) + batch_size - 1) // batch_size
        print(f"\n[PINECONE] Uploading {len(records)} vectors in {total_batches} batches")
        async def upsert_batch(batch):
            loop = asyncio.get_event_loop()
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: index.upsert(vectors=batch, namespace=updated_namespace)),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                print(f"Upsert batch timed out after 30 seconds", file=sys.stderr)
                raise
            except Exception as e:
                print(f"Error upserting batch: {e}", file=sys.stderr)
                raise
        tasks = []
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            tasks.append(upsert_batch(batch))
        await asyncio.gather(*tasks)
        print(f"[PINECONE] Successfully uploaded {len(records)} vectors for section {section_name}")
    return len(records)

def read_from_gcs(bucket_name, file_path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    return blob.download_as_string()

@router.post("/process-10k")
async def process_10k_gcs():
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
    global failed_cleaning_requests, failed_tagging_requests, failed_embedding_requests
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
                cik = await get_cik_for_ticker(ticker)
                if not cik:
                    results.append({
                        "ticker": ticker,
                        "success": False,
                        "error": "CIK not found for ticker"
                    })
                    continue
                filing_data = await fetch_10k_filing(cik, ticker, request.fiscal_year)
                if not filing_data:
                    results.append({
                        "ticker": ticker,
                        "success": False,
                        "error": "Failed to fetch 10-K filing"
                    })
                    continue
                if request.skip_embedding:
                    results.append({
                        "ticker": ticker,
                        "success": True,
                        "message": "10-K filing fetched successfully, embedding skipped"
                    })
                    continue
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
    try:
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
        ten_k_indices = [i for i, form in enumerate(forms) if form == "10-K"]
        if not ten_k_indices:
            print(f"No 10-K filing found for {symbol}")
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
            print(f"No 10-K filing found for {symbol} fiscal year {fiscal_year}")
            return None
        accession = accession_numbers[index]
        primary_doc = primary_documents[index]
        accession_no = accession.replace("-", "")
        base_url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession_no}"
        html_url = f"{base_url}/{primary_doc}"
        html_response = requests.get(html_url, headers=SEC_HEADERS)
        if html_response.status_code != 200:
            print(f"Error downloading HTML content for {symbol}: {html_response.status_code}")
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
        print(f"Error processing {symbol}: {str(e)}")
        return None

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

async def process_10k_data(pc, data, source_path=None, namespace=None):
    global failed_cleaning_requests, failed_tagging_requests, failed_embedding_requests
    meta = data.get("metadata", {})
    symbol = meta.get("symbol", "unknown")
    period = meta.get("period_of_report", "unknown")
    fiscal_year = meta.get("fiscal_year", None)
    content_sections = data.get("content", {})
    print(f"Processing data for symbol: {symbol}, period: {period}, fiscal year: {fiscal_year}")
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