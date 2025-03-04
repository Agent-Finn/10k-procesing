from pinecone import Pinecone, ServerlessSpec
import asyncio
from typing import List, Dict, Any, Optional
import os
from app.utils import chunk_text, normalize_vector_id
from app.llm_processor import clean_text_with_gemini_async, create_tags_with_gemini_async
from concurrent.futures import ThreadPoolExecutor
from google import genai
import json

# Global counters for tracking failed API calls
failed_embedding_requests = 0

def init_pinecone():
    """Initialize Pinecone client."""
    try:
        # Get Pinecone API key from environment variable
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")
            
        # Initialize Pinecone client
        pc = Pinecone(api_key=api_key)
        
        # Define the index name
        index_name = "sec-10k-index"
        
        # Check if the index exists, if not create it
        if index_name not in [i["name"] for i in pc.list_indexes()]:
            # Create a new serverless index with 768 dimensions (for embedding model)
            pc.create_index(
                name=index_name,
                dimension=768,  # Adjust based on your embedding model
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-west-2"
                )
            )
            
        # Get the index
        index = pc.Index(index_name)
        return index
        
    except Exception as e:
        print(f"Error initializing Pinecone: {e}")
        return None

async def process_section_to_pinecone(pc, section_text, section_name, symbol, period, namespace=None, fiscal_year=None):
    """
    Process a section of text and upsert it to Pinecone.
    
    Args:
        pc: Pinecone index
        section_text (str): Text of the section
        section_name (str): Name of the section
        symbol (str): Company ticker symbol
        period (str): Filing period
        namespace (str, optional): Namespace for the vectors
        fiscal_year (str, optional): Fiscal year of the filing
    """
    if not section_text or len(section_text) < 100:
        print(f"Skipping empty/short section {section_name} for {symbol}")
        return
        
    if not pc:
        print("Pinecone index not initialized")
        return
        
    global failed_embedding_requests
        
    # Create a unique identifier for this section
    base_id = f"{symbol}_{section_name}_{period}"
    if fiscal_year:
        base_id = f"{base_id}_{fiscal_year}"
    base_id = normalize_vector_id(base_id)
    
    try:
        # First, clean the text using Gemini
        cleaned_text = await clean_text_with_gemini_async(section_text)
        if not cleaned_text:
            print(f"Failed to clean text for {section_name} for {symbol}")
            cleaned_text = section_text  # Use original if cleaning fails
            
        # Second, generate tags for the section
        tags = await create_tags_with_gemini_async(cleaned_text, section_name, symbol)
        
        # Chunk the text for processing
        chunks = chunk_text(cleaned_text)
        
        # Skip if no chunks
        if not chunks:
            print(f"No chunks generated for {section_name} for {symbol}")
            return
            
        # Create a batch for vector upsert
        vectors = []
        
        # Process all chunks concurrently
        async def process_chunk(idx, chunk_text):
            chunk_id = f"{base_id}_chunk_{idx}"
            
            try:
                # Generate embeddings using Gemini
                await asyncio.sleep(0.1)  # Rate limiting for Gemini
                model = genai.GenerativeModel('gemini-pro')
                prompt = f"""
                This is a section from a company's SEC 10-K filing.
                Please identify the key topics, risks, and business information in this text. Pay attention to financial metrics, business strategies, and risk factors.
                
                Company: {symbol}
                Section: {section_name}
                Text: {chunk_text}
                
                Return a JSON string with the following structure:
                {{
                  "summary": "A concise 1-2 sentence summary of this chunk",
                  "topics": ["topic1", "topic2", "topic3"]
                }}
                """
                
                # Get embedding response
                response = model.generate_content(prompt)
                
                # Parse the JSON response
                response_text = response.text.strip()
                
                # If response includes markdown code block formatting, extract just the JSON
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0].strip()
                
                # Parse the JSON to extract summary and topics
                parsed_json = json.loads(response_text)
                
                # Create metadata for this vector
                metadata = {
                    "symbol": symbol,
                    "section": section_name,
                    "period": period,
                    "chunk_idx": idx,
                    "text": chunk_text,
                    "summary": parsed_json.get("summary", ""),
                    "topics": parsed_json.get("topics", []),
                    "tags": tags
                }
                
                if fiscal_year:
                    metadata["fiscal_year"] = fiscal_year
                
                # For now, we'll use a random embedding (replace with actual embedding logic)
                import random
                embedding = [random.uniform(-1, 1) for _ in range(768)]
                
                # Add to vectors batch
                vectors.append({
                    "id": chunk_id,
                    "values": embedding,
                    "metadata": metadata
                })
                
            except Exception as e:
                failed_embedding_requests += 1
                print(f"Error processing chunk {idx} for {symbol} {section_name}: {e}")
        
        # Process chunks concurrently
        tasks = [process_chunk(i, chunk) for i, chunk in enumerate(chunks)]
        await asyncio.gather(*tasks)
        
        # Upsert vectors in batches
        if vectors:
            batch_size = 100  # Pinecone batch size limit
            
            async def upsert_batch(batch):
                try:
                    # If namespace specified, use it
                    if namespace:
                        pc.upsert(vectors=batch, namespace=namespace)
                    else:
                        pc.upsert(vectors=batch)
                    print(f"Upserted batch of {len(batch)} vectors for {symbol} {section_name}")
                except Exception as e:
                    print(f"Error upserting batch for {symbol} {section_name}: {e}")
            
            # Process in batches
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                await upsert_batch(batch)
                
    except Exception as e:
        print(f"Error processing section {section_name} for {symbol}: {e}")

async def process_10k_data(pc, data, source_path=None, namespace=None):
    """
    Process a full 10-K document and upsert to Pinecone.
    
    Args:
        pc: Pinecone index
        data (dict): 10-K data
        source_path (str, optional): Source path of the data
        namespace (str, optional): Namespace for the vectors
    """
    symbol = data.get("symbol")
    report_date = data.get("report_date", "unknown")
    sections = data.get("sections", {})
    
    print(f"Processing 10-K for {symbol} dated {report_date}")
    
    # Process each section concurrently
    async def process_section(section_name, section_text):
        await process_section_to_pinecone(
            pc=pc,
            section_text=section_text,
            section_name=section_name,
            symbol=symbol,
            period=report_date,
            namespace=namespace
        )
    
    # Process all sections concurrently
    tasks = []
    for section_name, section_text in sections.items():
        tasks.append(process_section(section_name, section_text))
    
    # Wait for all sections to be processed
    await asyncio.gather(*tasks)
    
    print(f"Completed processing 10-K for {symbol}")
    return {
        "symbol": symbol,
        "date": report_date,
        "sections_processed": len(sections),
        "source": source_path
    } 