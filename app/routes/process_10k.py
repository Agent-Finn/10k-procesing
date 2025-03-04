from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
import json
import time
import unicodedata
import sys
from typing import Optional, Dict, Any, List, Union
import asyncio
from google.cloud import storage
from google import genai
from pinecone import Pinecone, ServerlessSpec
import requests
from bs4 import BeautifulSoup
import re
from pydantic import BaseModel
from datetime import datetime, timedelta

router = APIRouter()

# Global counters
failed_cleaning_requests = 0
failed_tagging_requests = 0
failed_embedding_requests = 0
total_cleaning_requests = 0
successful_cleaning_requests = 0
total_tagging_requests = 0
successful_tagging_requests = 0

# Rate limiter with long pauses after retries
class GeminiRateLimiter:
    def __init__(self, max_requests=30, period=30, min_interval=0.1):
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
                wait_time = (oldest + timedelta(seconds=self.period) - now).total_seconds()
                print(f"[RATE LIMITER] Queue full ({len(self.request_times)}/{self.max_requests}). Waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)

            time_since_last = now.timestamp() - self.last_request_time
            if time_since_last < self.min_interval:
                wait_time = self.min_interval - time_since_last
                print(f"[RATE LIMITER] Min interval wait: {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
            
            self.last_request_time = now.timestamp()
            self.request_times.append(now)
            print(f"[RATE LIMITER] Approved. Queue: {len(self.request_times)}/{self.max_requests}")

# Initialize rate limiter
gemini_limiter = GeminiRateLimiter(max_requests=15, period=60, min_interval=0.1)

# Google Vertex AI settings
PROJECT_ID = "nimble-chess-449208-f3"
LOCATION = "us-central1"
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

# Google Cloud Storage settings
GCS_BUCKET = "finn-cleaned-data"
GCS_FILE_PATH = "10k_files/aapl_10k.json"

# Pinecone settings
PINECONE_API_KEY = "pcsk_6rBfZw_6oEhbN34NsgDsq57Gcj6CZhmQnXBFujB33XUcEsmrgVWCvBC5Lyv3KKgBd7cweP"
INDEX_NAME = "10k"
EMBEDDING_MODEL = "llama-text-embed-v2"
EMBEDDING_DIMENSIONS = 1024
INDEX_HOST = "https://10k-gsf4yiq.svc.gcp-us-central1-4a9f.pinecone.io"

SEC_HEADERS = {"User-Agent": "YourCompanyName YourAppName (your.email@example.com)"}

class ProcessTickerRequest(BaseModel):
    tickers: Union[str, List[str]]
    skip_embedding: bool = False
    fiscal_year: Optional[str] = "2023"

def normalize_vector_id(raw_id: str) -> str:
    return unicodedata.normalize('NFKD', raw_id).encode('ascii', 'ignore').decode('ascii')

async def clean_text_with_gemini_async(text: str, max_retries=3) -> str:
    global failed_cleaning_requests, total_cleaning_requests, successful_cleaning_requests
    total_cleaning_requests += 1

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
            response = await loop.run_in_executor(None, lambda: client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            ))
            successful_cleaning_requests += 1
            return response.text.strip()
        except Exception as e:
            failed_cleaning_requests += 1
            print(f"[FAILED CLEANING] Attempt {attempt+1}/{max_retries}: {str(e)}")
            if attempt < max_retries - 1:  # Pause only if not the last attempt
                pause_duration = 30 * (attempt + 1)  # 30s, 60s, 90s
                print(f"[PAUSE] Waiting {pause_duration}s before retry")
                await asyncio.sleep(pause_duration)
            if attempt == max_retries - 1:
                print(f"[FAILED CLEANING] All retries exhausted")
                return text

async def create_tags_with_gemini_async(text: str, section_name: str, symbol: str, max_retries=3) -> list:
    global failed_tagging_requests, total_tagging_requests, successful_tagging_requests
    total_tagging_requests += 1

    prompt = (
        "You are a financial analyst. You are given a section of a 10-K report. Your job is to create a list of exactly "
        "two tags for the section. Return the tags in a list in this format: ['tag1', 'tag2']. Do not return any other "
        "text. Here is the 10k section:" + text
    )
    
    loop = asyncio.get_event_loop()
    for attempt in range(max_retries):
        try:
            await gemini_limiter.acquire()
            response = await loop.run_in_executor(None, lambda: client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            ))
            tags_text = response.text.strip()
            successful_tagging_requests += 1
            return eval(tags_text)
        except Exception as e:
            failed_tagging_requests += 1
            print(f"[FAILED TAGGING] Attempt {attempt+1}/{max_retries}: {str(e)}")
            if attempt < max_retries - 1:  # Pause only if not the last attempt
                pause_duration = 30 * (attempt + 1)  # 30s, 60s, 90s
                print(f"[PAUSE] Waiting {pause_duration}s before retry")
                await asyncio.sleep(pause_duration)
            if attempt == max_retries - 1:
                print(f"[FAILED TAGGING] All retries exhausted")
                return ["financial_report", "10k_filing"]

def init_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSIONS,
            metric="cosine",
            spec=ServerlessSpec(cloud="gcp", region="us-central1")
        )
        while not pc.describe_index(INDEX_NAME).status['ready']:
            time.sleep(1)
    return pc

def chunk_text(text, chunk_size=1000, chunk_overlap=200, min_chunk_length=100):
    if not text:
        return []
    step = chunk_size - chunk_overlap
    chunks = []
    for i in range(0, len(text), step):
        chunk = text[i:i+chunk_size].strip()
        if chunk and len(chunk) >= min_chunk_length:
            chunks.append((i, chunk))
    return chunks

async def process_section_to_pinecone(pc, section_text, section_name, symbol, period, namespace=None, fiscal_year=None):
    global failed_embedding_requests
    CHUNK_SIZE = 5000

    cleaned_chunks = []
    for i in range(0, len(section_text), CHUNK_SIZE):
        chunk_text = section_text[i:i + CHUNK_SIZE].strip()
        if len(chunk_text) < 100:
            continue
        cleaned_text = await clean_text_with_gemini_async(chunk_text)
        if cleaned_text:
            cleaned_chunks.append(cleaned_text)
    
    if not cleaned_chunks:
        return 0
    
    full_cleaned_text = ' '.join(cleaned_chunks).strip()
    if not full_cleaned_text:
        return 0
    
    tags = await create_tags_with_gemini_async(full_cleaned_text, section_name, symbol)
    chunks = chunk_text(full_cleaned_text)
    if not chunks:
        return 0
    
    index = pc.Index(INDEX_NAME, host=INDEX_HOST)
    records = []
    year = fiscal_year if fiscal_year else (period[:4] if period and len(period) >= 4 else "unknown")
    updated_namespace = f"{symbol}-{year}"
    
    try:
        chunk_texts = [chunk for _, chunk in chunks]
        embeddings = pc.inference.embed(
            model=EMBEDDING_MODEL,
            inputs=chunk_texts,
            parameters={"input_type": "passage", "truncate": "END"}
        )
        for (start, chunk), embedding in zip(chunks, embeddings):
            raw_id = f"{symbol}-{period}-{section_name}-{start}"
            vector_id = normalize_vector_id(raw_id)
            records.append({
                "id": vector_id,
                "values": embedding['values'],
                "metadata": {"chunk_start": start, "metatags": tags, "text": chunk}
            })
    except Exception as e:
        print(f"[EMBEDDING ERROR] Failed for section {section_name}: {str(e)}")
        failed_embedding_requests += 1
        return 0
    
    if records:
        batch_size = 100
        async def upsert_batch(batch):
            loop = asyncio.get_event_loop()
            await asyncio.wait_for(
                loop.run_in_executor(None, lambda: index.upsert(vectors=batch, namespace=updated_namespace)),
                timeout=30.0
            )
        tasks = [upsert_batch(records[i:i + batch_size]) for i in range(0, len(records), batch_size)]
        await asyncio.gather(*tasks)
    
    return len(records)

async def process_10k_data(pc, data, source_path=None, namespace=None):
    global failed_cleaning_requests, failed_tagging_requests, failed_embedding_requests
    meta = data.get("metadata", {})
    symbol = meta.get("symbol", "unknown")
    period = meta.get("period_of_report", "unknown")
    fiscal_year = meta.get("fiscal_year", None)
    content_sections = data.get("content", {})
    
    print(f"Processing data for symbol: {symbol}, period: {period}")
    MIN_CHARS = 200
    total_vectors = 0
    processed_sections = []
    
    async def process_section(section_name, section_text):
        if section_name.upper() == "GENERAL" or not section_text or len(section_text) < MIN_CHARS:
            return None
        vectors_added = await process_section_to_pinecone(pc, section_text, section_name, symbol, period, namespace, fiscal_year)
        return {"section_name": section_name, "vectors_added": vectors_added} if vectors_added > 0 else None
    
    tasks = [process_section(name, "\n".join(str(p) for p in paras).strip()) for name, paras in content_sections.items()]
    results = await asyncio.gather(*tasks)
    
    for result in results:
        if result:
            total_vectors += result["vectors_added"]
            processed_sections.append(result)
    
    updated_namespace = f"{symbol}-{fiscal_year or (period[:4] if period and len(period) >= 4 else 'unknown')}"
    
    print(f"\n{'='*80}")
    print(f"PROCESSING SUMMARY FOR {symbol}")
    print(f"{'='*80}")
    print(f"Total vectors added: {total_vectors}")
    print(f"Sections processed: {len(processed_sections)}")
    print(f"Cleaning API Calls - Total: {total_cleaning_requests}, Successful: {successful_cleaning_requests}, Failed: {failed_cleaning_requests}")
    print(f"Tagging API Calls - Total: {total_tagging_requests}, Successful: {successful_tagging_requests}, Failed: {failed_tagging_requests}")
    print(f"Embedding Failures: {failed_embedding_requests}")
    print(f"Total API Calls: {total_cleaning_requests + total_tagging_requests}")
    print(f"Total Failed Requests: {failed_cleaning_requests + failed_tagging_requests + failed_embedding_requests}")
    print(f"{'='*80}")
    
    response = {
        "success": True,
        "symbol": symbol,
        "period": period,
        "total_vectors_added": total_vectors,
        "processed_sections": processed_sections,
        "namespace": updated_namespace,
        "api_call_summary": {
            "cleaning": {"total": total_cleaning_requests, "successful": successful_cleaning_requests, "failed": failed_cleaning_requests},
            "tagging": {"total": total_tagging_requests, "successful": successful_tagging_requests, "failed": failed_tagging_requests},
            "embedding": {"failed": failed_embedding_requests},
            "total_calls": total_cleaning_requests + total_tagging_requests,
            "total_failed": failed_cleaning_requests + failed_tagging_requests + failed_embedding_requests
        }
    }
    if source_path:
        response["source_path"] = source_path
    return response

@router.post("/process-by-ticker")
async def process_by_ticker(request: ProcessTickerRequest):
    global failed_cleaning_requests, failed_tagging_requests, failed_embedding_requests
    global total_cleaning_requests, successful_cleaning_requests, total_tagging_requests, successful_tagging_requests
    
    failed_cleaning_requests = total_cleaning_requests = successful_cleaning_requests = 0
    failed_tagging_requests = total_tagging_requests = successful_tagging_requests = 0
    failed_embedding_requests = 0
    
    tickers = [request.tickers] if isinstance(request.tickers, str) else request.tickers
    tickers = [ticker.upper() for ticker in tickers]
    pc = init_pinecone()
    
    results = []
    for ticker in tickers:
        try:
            cik = await get_cik_for_ticker(ticker)
            if not cik:
                results.append({"ticker": ticker, "success": False, "error": "CIK not found"})
                continue
            
            filing_data = await fetch_10k_filing(cik, ticker, request.fiscal_year)
            if not filing_data:
                results.append({"ticker": ticker, "success": False, "error": "Failed to fetch 10-K"})
                continue
            
            if request.skip_embedding:
                results.append({"ticker": ticker, "success": True, "message": "10-K fetched, embedding skipped"})
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
            results.append({"ticker": ticker, "success": False, "error": str(e)})
    
    print(f"\n{'='*80}")
    print(f"PROCESSING SUMMARY - API CALLS AND FAILURES")
    print(f"{'='*80}")
    print(f"Cleaning API Calls - Total: {total_cleaning_requests}, Successful: {successful_cleaning_requests}, Failed: {failed_cleaning_requests}")
    print(f"Tagging API Calls - Total: {total_tagging_requests}, Successful: {successful_tagging_requests}, Failed: {failed_tagging_requests}")
    print(f"Embedding Failures: {failed_embedding_requests}")
    print(f"Total API Calls: {total_cleaning_requests + total_tagging_requests}")
    print(f"Total Failed Requests: {failed_cleaning_requests + failed_tagging_requests + failed_embedding_requests}")
    print(f"{'='*80}")
    
    return {
        "success": True,
        "results": results,
        "api_call_summary": {
            "cleaning": {"total": total_cleaning_requests, "successful": successful_cleaning_requests, "failed": failed_cleaning_requests},
            "tagging": {"total": total_tagging_requests, "successful": successful_tagging_requests, "failed": failed_tagging_requests},
            "embedding": {"failed": failed_embedding_requests},
            "total_calls": total_cleaning_requests + total_tagging_requests,
            "total_failed": failed_cleaning_requests + failed_tagging_requests + failed_embedding_requests
        }
    }

async def get_cik_for_ticker(ticker: str) -> Optional[str]:
    try:
        response = requests.get("https://www.sec.gov/files/company_tickers.json", headers=SEC_HEADERS)
        if response.status_code != 200:
            print(f"Error fetching CIK data: {response.status_code}")
            return None
        for entry in response.json().values():
            if entry['ticker'] == ticker:
                return str(entry['cik_str']).zfill(10)
        return None
    except Exception as e:
        print(f"Error getting CIK for {ticker}: {str(e)}")
        return None

async def fetch_10k_filing(cik: str, symbol: str, fiscal_year: str = "2023") -> Optional[dict]:
    try:
        response = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json", headers=SEC_HEADERS)
        if response.status_code != 200:
            print(f"Error fetching submission data for {symbol}: {response.status_code}")
            return None
        
        data = response.json()['filings']['recent']
        ten_k_indices = [i for i, form in enumerate(data['form']) if form == "10-K"]
        if not ten_k_indices:
            print(f"No 10-K filing found for {symbol}")
            return None
        
        expected_filing_year = str(int(fiscal_year) + 1)
        index = next((i for i in ten_k_indices if data['filingDate'][i].startswith(expected_filing_year)), None)
        if index is None:
            print(f"No 10-K filing found for {symbol} fiscal year {fiscal_year}")
            return None
        
        accession = data['accessionNumber'][index].replace("-", "")
        html_url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession}/{data['primaryDocument'][index]}"
        html_response = requests.get(html_url, headers=SEC_HEADERS)
        if html_response.status_code != 200:
            print(f"Error downloading HTML content for {symbol}: {html_response.status_code}")
            return None
        
        soup = BeautifulSoup(html_response.text, 'html.parser')
        filing_content = {
            "metadata": {
                "symbol": symbol, "cik": cik, "filing_type": "10-K",
                "filing_year": expected_filing_year, "fiscal_year": fiscal_year,
                "accession_number": data['accessionNumber'][index],
                "filing_date": data['filingDate'][index],
                "period_of_report": data['reportDate'][index],
                "html_url": html_url
            },
            "content": {}
        }
        
        current_section = "GENERAL"
        filing_content["content"][current_section] = []
        for element in soup.find_all(['p', 'div', 'span', 'table']):
            text = re.sub(r'\s+', ' ', element.get_text(strip=True)).strip()
            if not text:
                continue
            if re.match(r'^ITEM\s+\d+[A-Z]?\.?', text, re.IGNORECASE) or re.match(r'^PART\s+[IVX]+\.?', text, re.IGNORECASE):
                current_section = text
                filing_content["content"][current_section] = []
            else:
                filing_content["content"][current_section].append(text)
        return filing_content
    except Exception as e:
        print(f"Error processing {symbol}: {str(e)}")
        return None