from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import json
import time
import unicodedata
import sys
from typing import Optional
import os
from google.cloud import storage
from google import genai
from pinecone import Pinecone, ServerlessSpec
import asyncio
from concurrent.futures import ThreadPoolExecutor

router = APIRouter()

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

# Helper function to normalize vector IDs to ASCII
def normalize_vector_id(raw_id: str) -> str:
    return unicodedata.normalize('NFKD', raw_id).encode('ascii', 'ignore').decode('ascii')

async def clean_text_with_gemini_async(text: str) -> str:
    """Asynchronously clean 10-K text using Gemini API."""
    prompt = (
        "Clean the following financial report text for analysis. Exclude legalese, addresses, filing info, signatures, "
        "checkmarks, tables of contents, and any tables or sections with numerical financial data. Remove page numbers, "
        "bullet points, extraneous headings, random characters, and any formatting that isn't relevant. For sections that "
        "are only headers or titles with no content, return an empty string. Also omit the 'Exhibit and Financial Statement "
        "Schedule' section, any parts talking about the 10k document itself. Do not summarize, add commentary or analysis. "
        "Either return the cleaned, meaningful text, or nothing at all. Here is the text:" + text
    )
    
    loop = asyncio.get_event_loop()
    try:
        print(f"\n[GEMINI REQUEST - CLEANING] Processing text of length: {len(text)}")
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
        print(f"Error cleaning text with Gemini API: {e}", file=sys.stderr)
        return text

async def create_tags_with_gemini_async(text: str, section_name: str, symbol: str) -> list:
    """Asynchronously create tags for a 10-K section using Gemini API."""
    prompt = (
        "You are a financial analyst. You are given a section of a 10-K report. Your job is to create a list of exactly "
        "two tags for the section. Return the tags in a list in this format: ['tag1', 'tag2']. Do not return any other "
        "text. Here is the 10k section:" + text
    )
    
    loop = asyncio.get_event_loop()
    try:
        print(f"\n[GEMINI REQUEST - TAGGING] Creating tags for section: {section_name}")
        
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
            return ["financial_report", "10k_filing"]
    except Exception as e:
        print(f"Error generating tags with Gemini API: {e}", file=sys.stderr)
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

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """Split text into chunks with specified size and overlap."""
    if not text:
        return []
    step = chunk_size - chunk_overlap
    chunks = []
    for i in range(0, len(text), step):
        chunk = text[i:i+chunk_size].strip()
        if chunk:
            chunks.append((i, chunk))
    
    print(f"\n[CHUNKING] Generated {len(chunks)} chunks from text of length {len(text)}")
    if chunks:
        print(f"Sample chunk: {chunks[0][1][:100]}...")
    
    return chunks

async def process_section_to_pinecone(pc, section_text, section_name, symbol, period):
    """Process a section: clean, chunk, embed in batches, and upload to Pinecone."""
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
    
    # Generate tags asynchronously
    tags = await create_tags_with_gemini_async(cleaned_text, section_name, symbol)
    
    # Chunk the cleaned text
    chunks = chunk_text(cleaned_text)
    if not chunks:
        print(f"No chunks generated for section: {section_name}")
        return 0
    
    print(f"Generated {len(chunks)} chunks for section {section_name}")
    
    index = pc.Index(INDEX_NAME, host=INDEX_HOST)
    records = []
    
    # Batch embedding
    chunk_texts = [chunk for _, chunk in chunks]
    try:
        print(f"\n[EMBEDDING] Generating embeddings for {len(chunk_texts)} chunks")
        
        # Generate embeddings for all chunks in one call
        embeddings = pc.inference.embed(
            model=EMBEDDING_MODEL,
            inputs=chunk_texts,
            parameters={"input_type": "passage", "truncate": "END"}
        )
        
        print(f"[EMBEDDING] Successfully generated {len(embeddings)} embeddings")
        
        # Process embeddings and create records
        for (start, chunk), embedding in zip(chunks, embeddings):
            try:
                raw_id = f"{symbol}-{period}-{section_name}-{start}"
                vector_id = normalize_vector_id(raw_id)
                
                record = {
                    "id": vector_id,
                    "values": embedding['values'],
                    "metadata": {
                        "symbol": symbol,
                        "period_of_report": period,
                        "section": section_name,
                        "chunk_start": start,
                        "metatags": tags,
                        "text": chunk  # Store the chunk text here
                    }
                }
                records.append(record)
            except Exception as e:
                print(f"Error processing chunk at start {start}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Error during batch embedding for section {section_name}: {e}", file=sys.stderr)
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
                    loop.run_in_executor(None, lambda: index.upsert(vectors=batch, namespace=NAMESPACE)),
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
        
        meta = data.get("metadata", {})
        symbol = meta.get("symbol", "unknown")
        period = meta.get("period_of_report", "unknown")
        content_sections = data.get("content", {})
        
        MIN_CHARS = 200
        total_vectors = 0
        processed_sections = []
        
        # Process sections in parallel
        async def process_section(section_name, section_text):
            if section_name.upper() == "GENERAL" or not section_text or len(section_text) < MIN_CHARS:
                return None
            vectors_added = await process_section_to_pinecone(pc, section_text, section_name, symbol, period)
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
        
        return {
            "success": True,
            "gcs_path": gcs_path,
            "symbol": symbol,
            "period": period,
            "total_vectors_added": total_vectors,
            "processed_sections": processed_sections
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )