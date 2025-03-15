#!/usr/bin/env python3
import os
import re
import json
from collections import defaultdict
from google.cloud import storage
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("chunk_concatenation")

# Hardcoded bucket and path configuration
SOURCE_BUCKET = "finn-cleaned-data"
SOURCE_PREFIX = "llm_processed_10k/"  # Path to the directory containing the chunks
DEST_BUCKET = "finn-cleaned-data"
DEST_PREFIX = "10k_merged/"  # Path where combined files will be stored

def parse_filename(filename):
    """
    Parse the filename to extract stock name and year.
    Expected format: "<stock_name>-<year>-<date>-<section title>.json"
    """
    # Remove .json extension
    base_name = os.path.splitext(filename)[0]
    
    # Split by hyphens
    parts = base_name.split('-')
    
    # Ensure we have at least 3 parts (stock, year, date)
    if len(parts) < 3:
        return None, None
    
    stock_name = parts[0]
    year = parts[1]
    
    return stock_name, year

def list_files_in_bucket(bucket_name, prefix=None):
    """
    List all files in a GCS bucket with an optional prefix.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    return [blob.name for blob in blobs]

def read_json_from_gcs(bucket_name, file_path, max_retries=3, retry_delay=5):
    """
    Read a JSON file from GCS and return its contents.
    Includes retry mechanism for handling temporary network issues.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    
    for attempt in range(max_retries):
        try:
            content = blob.download_as_string()
            return json.loads(content)
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Error reading {file_path} (attempt {attempt+1}/{max_retries}): {str(e)}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                # Increase delay for next retry (exponential backoff)
                retry_delay *= 2
            else:
                logger.error(f"Error reading {file_path} after {max_retries} attempts: {str(e)}")
                return None

def upload_to_gcs(bucket_name, file_path, data):
    """
    Upload JSON data to GCS.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    
    try:
        blob.upload_from_string(json.dumps(data, indent=2), content_type='application/json')
        logger.info(f"Successfully uploaded file to GCS: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to upload file to GCS: {file_path}. Error: {str(e)}")
        return False

def concatenate_files_by_stock_and_year():
    """
    Concatenate JSON files by stock name and year, then upload to destination bucket.
    """
    # List all files in the source bucket with the specified prefix
    all_files = list_files_in_bucket(SOURCE_BUCKET, SOURCE_PREFIX)
    
    # Group files by stock name and year
    grouped_files = defaultdict(list)
    for file_path in all_files:
        filename = os.path.basename(file_path)
        stock_name, year = parse_filename(filename)
        
        if stock_name and year:
            key = f"{stock_name}-{year}"
            grouped_files[key].append(file_path)
    
    # Process each group
    for key, file_paths in grouped_files.items():
        stock_name, year = key.split('-')
        logger.info(f"Processing {stock_name} for year {year} ({len(file_paths)} files)")
        
        # Initialize combined data
        combined_data = []
        
        # Read and concatenate all files in the group
        for file_path in file_paths:
            data = read_json_from_gcs(SOURCE_BUCKET, file_path)
            if data:
                if isinstance(data, list):
                    combined_data.extend(data)
                else:
                    combined_data.append(data)
        
        if combined_data:
            # Create destination path
            dest_file_path = f"{DEST_PREFIX}{stock_name}-{year}-combined.json"
            
            # Upload combined data
            success = upload_to_gcs(DEST_BUCKET, dest_file_path, combined_data)
            if success:
                logger.info(f"Successfully combined and uploaded {len(combined_data)} items for {stock_name}-{year}")
            else:
                logger.error(f"Failed to upload combined data for {stock_name}-{year}")

def main():
    logger.info(f"Starting concatenation process")
    logger.info(f"Source: {SOURCE_BUCKET}/{SOURCE_PREFIX}")
    logger.info(f"Destination: {DEST_BUCKET}/{DEST_PREFIX}")
    
    concatenate_files_by_stock_and_year()
    
    logger.info(f"Concatenation process completed")

if __name__ == "__main__":
    main()
