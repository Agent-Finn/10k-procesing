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

def clean_text_with_gemini(text: str) -> str:
    """Clean 10-K text using Gemini API."""
    prompt = (
        "Clean the following financial report text for analysis. Exclude legalese, addresses, filing info, signatures, "
        "checkmarks, tables of contents, and any tables or sections with numerical financial data. Remove page numbers, "
        "bullet points, extraneous headings, random characters, and any formatting that isn't relevant. For sections that "
        "are only headers or titles with no content, return an empty string. Also omit the 'Exhibit and Financial Statement "
        "Schedule' section, any parts talking about the 10k document itself. Do not summarize, add commentary or analysis. "
        "Either return the cleaned, meaningful text, or nothing at all. Here is the text:" + text
    )
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error cleaning text with Gemini API: {e}", file=sys.stderr)
        return text
    
def create_tags_with_gemini(text: str, section_name: str, symbol: str) -> list:
    """Create tags for a 10-K section using Gemini API, then append ticker and section name."""
    prompt = (
        "You are a financial analyst. You are given a section of a 10-K report. Your job is to create a list of exactly "
        "two tags for the section. Return the tags in a list in this format: ['tag1', 'tag2']. Do not return any other "
        "text. Here is the 10k section:" + text
    )
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        # Extract the tags from the response
        tags_text = response.text.strip()
        
        # Try to parse as Python list
        try:
            # Convert the string representation of a list to an actual list
            tags_list = eval(tags_text)
            # Add ticker symbol and section name to the list
            return tags_list
        except:
            # If parsing fails, create a list with default tags and add symbol and section name
            print(f"Failed to parse tags: {tags_text}")
            return ["financial_report", "10k_filing"]
    except Exception as e:
        print(f"Error generating tags with Gemini API: {e}", file=sys.stderr)
        return ["financial_report", "10k_filing", symbol, section_name]

def init_pinecone():
    """Initialize Pinecone connection and ensure index exists."""
    print("Initializing Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index exists, create if not
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
        # Wait until index is ready
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
        if chunk:  # Only add non-empty chunks
            chunks.append((i, chunk))
    
    return chunks

def process_section_to_pinecone(pc, section_text, section_name, symbol, period):
    """Process a section: clean, chunk, embed, and upload to Pinecone."""
    # Clean the text
    cleaned_text = clean_text_with_gemini(section_text)
    if not cleaned_text:
        print(f"Skipping empty cleaned section: {section_name}")
        return 0
    
    # Postprocess text to normalize whitespace and remove newlines
    # Replace multiple newlines with a single space to preserve paragraph breaks
    cleaned_text = ' '.join([para.strip() for para in cleaned_text.split('\n') if para.strip()])
    # Replace any remaining consecutive whitespace with a single space
    cleaned_text = ' '.join(cleaned_text.split())
    
    # Generate tags
    tags = create_tags_with_gemini(cleaned_text, section_name, symbol)
    
    # Chunk the cleaned text
    chunks = chunk_text(cleaned_text)
    if not chunks:
        print(f"No chunks generated for section: {section_name}")
        return 0
    
    print(f"Generated {len(chunks)} chunks for section {section_name}")
    
    # Get Pinecone index
    index = pc.Index(INDEX_NAME, host=INDEX_HOST)
    
    # Process each chunk
    records = []
    for chunk_idx, (start, chunk) in enumerate(chunks, 1):
        try:
            # Generate embedding
            embeddings = pc.inference.embed(
                model=EMBEDDING_MODEL,
                inputs=[chunk],
                parameters={"input_type": "passage", "truncate": "END"}
            )
            embedding = embeddings[0]['values']
            
            # Create a unique ID for the vector
            raw_id = f"{symbol}-{period}-{section_name}-{start}"
            vector_id = normalize_vector_id(raw_id)
            
            # Create record
            record = {
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "symbol": symbol,
                    "period_of_report": period,
                    "section": section_name,
                    "chunk_start": start,
                    "preview": chunk[:200],
                    "metatags": tags
                }
            }
            records.append(record)
            
        except Exception as e:
            print(f"Error processing chunk {chunk_idx}: {e}", file=sys.stderr)
    
    # Upload records to Pinecone in batches
    if records:
        batch_size = 100
        total_batches = (len(records) + batch_size - 1) // batch_size
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            try:
                index.upsert(vectors=batch, namespace=NAMESPACE)
                print(f"Uploaded batch {i//batch_size + 1}/{total_batches}")
            except Exception as e:
                print(f"Error upserting batch: {e}", file=sys.stderr)
    
    return len(records)

def read_from_gcs(bucket_name, file_path):
    """Read a file from Google Cloud Storage."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    
    return blob.download_as_string()

@router.post("/process-10k")
async def process_10k_gcs():
    """
    Process a 10-K report JSON file from Google Cloud Storage and store it in Pinecone.
    Uses a hardcoded path: gs://finn-cleaned-data/10k_files/aapl_10k.json
    """
    try:
        # Initialize Pinecone
        pc = init_pinecone()
        
        # Read the JSON file from GCS
        gcs_path = f"gs://{GCS_BUCKET}/{GCS_FILE_PATH}"
        print(f"Reading 10-K report from {gcs_path}")
        
        # Load the content from GCS
        content = read_from_gcs(GCS_BUCKET, GCS_FILE_PATH)
        data = json.loads(content)
        
        # Extract metadata and content sections
        meta = data.get("metadata", {})
        symbol = meta.get("symbol", "unknown")
        period = meta.get("period_of_report", "unknown")
        content_sections = data.get("content", {})
        
        # Minimum character threshold for processing
        MIN_CHARS = 200
        total_vectors = 0
        processed_sections = []
        
        # Process each section
        for section_name, paragraphs in content_sections.items():
            # Skip the GENERAL section
            if section_name.upper() == "GENERAL":
                continue
                
            # Join all paragraphs in the section
            section_text = "\n".join([str(p) for p in paragraphs]).strip()
            if not section_text:
                continue
                
            # Skip sections with few characters
            if len(section_text) < MIN_CHARS:
                continue
            
            # Process section and upload to Pinecone
            vectors_added = process_section_to_pinecone(pc, section_text, section_name, symbol, period)
            total_vectors += vectors_added
            
            if vectors_added > 0:
                processed_sections.append({
                    "section_name": section_name,
                    "vectors_added": vectors_added
                })
        
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