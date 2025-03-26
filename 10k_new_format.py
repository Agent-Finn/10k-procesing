import time
import random
from google import genai
import re
import json
from pinecone import Pinecone, ServerlessSpec
import unicodedata
from google.cloud import storage
import os

# Configuration
PINECONE_API_KEY = "pcsk_6rBfZw_6oEhbN34NsgDsq57Gcj6CZhmQnXBFujB33XUcEsmrgVWCvBC5Lyv3KKgBd7cweP"
INDEX_NAME = "10k"
EMBEDDING_MODEL = "llama-text-embed-v2"
EMBEDDING_DIMENSIONS = 1024
INDEX_HOST = "https://10k-embeddings-gsf4yiq.svc.gcp-us-central1-4a9f.pinecone.io"
PROJECT_ID = "nimble-chess-449208-f3"
LOCATION = "us-central1"
COMPANIES_TO_SKIP = ["INTC"]

# Function Definitions
def normalize_vector_id(raw_id: str) -> str:
    """Normalize a raw ID for Pinecone compatibility."""
    return unicodedata.normalize('NFKD', raw_id).encode('ascii', 'ignore').decode('ascii')

def create_and_upsert_embeddings(pc, index, cleaned_chunks, section_name, ticker, period, namespace, embedding_model, local_file=None):
    """Create embeddings and upsert them to Pinecone, writing records to a local file if provided."""
    valid_chunks = [(idx, text, tags) for (idx, text, tags) in cleaned_chunks if len(text) > 0]
    if not valid_chunks:
        print(f"    No valid chunks to embed for section '{section_name}'")
        return 0
    
    chunk_texts = [cleaned_text for _, cleaned_text, _ in cleaned_chunks]
    
    try:
        embeddings = pc.inference.embed(
            model=embedding_model,
            inputs=chunk_texts,
            parameters={"input_type": "passage", "truncate": "END"}
        )
    except Exception as e:
        print(f"    Failed to create embeddings for section '{section_name}': {str(e)}")
        return 0

    records = []
    for (chunk_idx, cleaned_text, tags), embedding in zip(cleaned_chunks, embeddings):
        if len(cleaned_text) < 100:
            continue
        raw_id = f"{ticker}-{period}-{section_name}-chunk_{chunk_idx}"
        vector_id = normalize_vector_id(raw_id)
        
        record = {
            "id": vector_id,
            "values": embedding['values'],
            "metadata": {
                "metatags": tags,
                "text": cleaned_text,
            }
        }
        record_to_save = {
            "id": vector_id,
            "metadata": {
                "metatags": tags,
                "text": cleaned_text,
            }
        }
        if local_file:
            local_file.write(json.dumps(record_to_save) + '\n')
        records.append(record)

    batch_size = 100
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        try:
            index.upsert(vectors=batch, namespace=namespace)
            print(f"    Upserted {len(batch)} vectors to Pinecone")
        except Exception as e:
            print(f"    Failed to upsert batch to Pinecone: {str(e)}")
            return 0

    return len(records)

def init_pinecone(api_key, index_name, embedding_dimensions):
    """Initialize Pinecone and create or connect to the specified index."""
    pc = Pinecone(api_key=api_key)
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=embedding_dimensions,
            metric="cosine",
            spec=ServerlessSpec(cloud="gcp", region="us-central1")
        )
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
    return pc

def split_text_into_chunks(text, max_length=1500, min_length=100):
    """Split text into chunks based on sentence boundaries."""
    if len(text) <= max_length:
        if len(text) >= min_length:
            return [text]
        return []
    sentences = re.split(r'(?<=\.)\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return []
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        sentence_length = len(sentence) + 1
        if current_length + sentence_length > max_length and current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= min_length:
                chunks.append(chunk_text)
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence)
        current_length += sentence_length
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        if len(chunk_text) >= min_length:
            chunks.append(chunk_text)
    return chunks

def clean_text_with_gemini(text: str, max_retries=5, initial_delay=4) -> str:
    """Clean text using Gemini, removing irrelevant content."""
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    prompt = (
        "Clean the following financial report text for analysis. Exclude legalese, addresses, filing info, signatures, "
        "checkmarks, tables of contents, and any tables or sections with numerical financial data. Remove page numbers, "
        "bullet points, extraneous headings, random characters, and any formatting that isn't relevant. For sections that "
        "are only headers or titles with no content, return an empty string. Also omit the 'Exhibit and Financial Statement "
        "Schedule' section, any parts talking about the 10k document itself. Do not summarize, add commentary or analysis. "
        "Either return the cleaned, meaningful text, or nothing at all. Here is the text:\n\n" + text
    )
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            return response.text.strip() if response.text else ""
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                delay = initial_delay * (2 ** attempt) + initial_delay * 0.2 * (0.5 - random.random())
                time.sleep(delay)
                if attempt == max_retries - 1:
                    print(f"Failed to clean text after {max_retries} attempts due to rate limiting.")
                    return ""
            else:
                print(f"Failed to clean text due to error: {str(e)}")
                return ""
    return ""

def generate_tags_with_gemini(text: str, num_tags: int = 2, max_retries=5, initial_delay=4) -> list[str]:
    """Generate tags for text using Gemini."""
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    prompt = (
        f"You are a financial analyst. You are given a part of a 10-K report. Your job is to create a list of exactly "
        f"{num_tags} tags that capture the key topics or points in this text. Return the tags in a list in this format: "
        f"['tag1', 'tag2', ...], where each tag is no longer than 3-4 words. Do not return any other text. Here is the text:\n\n" + text
    )
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            tags_text = response.text.strip()
            return eval(tags_text) if tags_text else [f"default_tag{i+1}" for i in range(num_tags)]
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                delay = initial_delay * (2 ** attempt) + initial_delay * 0.2 * (0.5 - random.random())
                time.sleep(delay)
                if attempt == max_retries - 1:
                    print(f"Failed to generate tags after {max_retries} attempts due to rate limiting.")
                    return [f"default_tag{i+1}" for i in range(num_tags)]
            else:
                print(f"Failed to generate tags due to error: {str(e)}")
                return [f"default_tag{i+1}" for i in range(num_tags)]
    return [f"default_tag{i+1}" for i in range(num_tags)]
# Main Processing
folder_path = "/Users/amaleldick/Documents/GitHub/10k-embedding/edgar-crawler/datasets/EXTRACTED_FILINGS/10-K/"
json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

pc = init_pinecone(PINECONE_API_KEY, INDEX_NAME, EMBEDDING_DIMENSIONS)
index = pc.Index(INDEX_NAME, host=INDEX_HOST)

for json_file in json_files:
    file_path = os.path.join(folder_path, json_file)
    with open(file_path, 'r') as file:
        document = json.load(file)
    

    ticker = document.get("ticker", "unknown")
    period = document.get("period_of_report", "unknown")
    if ticker in COMPANIES_TO_SKIP:
        continue
    namespace = f"{ticker}-{period[:4]}" if period else ticker
    
    local_filename = f"embeddings_{ticker}_{period}.json"
    with open(local_filename, 'w') as local_file:
        process_started = False
        ordered_keys = sorted(document.keys())
        for section_name in ordered_keys:
            if section_name == "item_1":
                process_started = True
            if not process_started or section_name in [
                "cik", "company", "filing_date", "filing_type", "period_of_report", "sic",
                "state_of_inc", "state_location", "fiscal_year_end", "filing_html_index",
                "htm_filing_link", "complete_text_filing_link", "filename"
            ]:
                continue
            
            section_text = document[section_name]
            if not isinstance(section_text, str):
                continue
            
            print(f"\nProcessing {section_name} for {json_file}:")
            chunks = split_text_into_chunks(section_text, max_length=2048, min_length=100) if len(section_text) > 2048 else [section_text]
            
            cleaned_chunks = []
            for i, chunk in enumerate(chunks):
                print(f"\n  Chunk {i+1} ({len(chunk)} characters):")
                if len(chunk) < 100:
                    print("    Skipped: Chunk is shorter than 100 characters")
                    continue
                cleaned_text = clean_text_with_gemini(chunk)
                if cleaned_text and len(cleaned_text) >= 100:
                    tags = generate_tags_with_gemini(cleaned_text)
                    cleaned_chunks.append((i, cleaned_text, tags))
                    print(f"    Cleaned Text: {cleaned_text[:100]}...")
                    print(f"    Tags: {tags}")
                else:
                    print("    Skipped: Cleaned text is empty or too short")
            
            if not cleaned_chunks:
                print(f"  No valid chunks after cleaning for section {section_name}")
                continue
            
            vectors_added = create_and_upsert_embeddings(
                pc, index, cleaned_chunks, section_name, ticker, period, namespace,
                EMBEDDING_MODEL, local_file=local_file
            )
            if vectors_added > 0:
                print(f"  Successfully processed {vectors_added} vectors for section {section_name}")
            else:
                print(f"  Failed to process embeddings for section {section_name}")
    
