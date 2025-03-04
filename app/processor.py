import json
import os
import asyncio
import time
from tqdm import tqdm
from typing import Optional, Dict, Any, List, Union
from google.cloud import storage

from app.sec_data import get_cik_map, fetch_10k_filing, get_cik_for_ticker
from app.vector_db import init_pinecone, process_10k_data

def read_from_gcs(bucket_name, file_path):
    """Read a file from Google Cloud Storage."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    content = blob.download_as_text()
    return content

class Processor:
    """Main processor for 10-K filings."""
    
    def __init__(self, output_dir=None):
        """Initialize the processor."""
        # Define output directory
        if output_dir:
            self.output_dir = output_dir
        else:
            # Default to a directory in the project root
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.output_dir = os.path.join(project_root, "sp500_10k")
            
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    async def process_single_company(self, symbol, cik):
        """Process a single company's 10-K filing."""
        try:
            print(f"Processing 10-K for {symbol} (CIK: {cik})...")
            
            # Create output file path
            output_file = os.path.join(self.output_dir, f"{symbol}_10k.json")
            
            # Check if already processed
            if os.path.exists(output_file):
                print(f"10-K for {symbol} already processed, skipping...")
                return {"symbol": symbol, "status": "skipped", "file": output_file}
                
            # Fetch 10-K filing
            data = await fetch_10k_filing(cik, symbol)
            
            if not data:
                print(f"Failed to fetch 10-K for {symbol}")
                return {"symbol": symbol, "status": "error", "message": "Failed to fetch 10-K"}
                
            # Save to file
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            print(f"Saved 10-K for {symbol} to {output_file}")
            
            # Process directly with Pinecone
            # Initialize Pinecone
            pc = init_pinecone()
            if pc:
                result = await process_10k_data(pc, data)
                return {"symbol": symbol, "status": "success", "file": output_file, "processed": result}
            else:
                return {"symbol": symbol, "status": "error", "file": output_file, "message": "Pinecone initialization failed"}
        
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            return {"symbol": symbol, "status": "error", "message": str(e)}
    
    async def process_companies(self, symbols=None):
        """Process multiple companies' 10-K filings."""
        # Get CIK map for S&P 500 companies
        cik_map = get_cik_map()
        
        if not cik_map:
            print("Failed to retrieve CIK mapping")
            return None
        
        print(f"Found {len(cik_map)} companies in S&P 500")
        
        # Filter to specified symbols if provided
        if symbols:
            symbols = [s.upper() for s in symbols]
            cik_map = {k: v for k, v in cik_map.items() if k in symbols}
            print(f"Filtered to {len(cik_map)} specified companies")
        
        if not cik_map:
            print("No companies to process")
            return None
        
        # Process companies
        results = []
        
        with tqdm(total=len(cik_map), desc="Processing companies") as pbar:
            for symbol, cik in cik_map.items():
                # Process companies one by one to avoid rate limiting
                result = await self.process_single_company(symbol, cik)
                results.append(result)
                pbar.update(1)
                
                # Wait a bit to avoid rate limiting
                await asyncio.sleep(0.5)
        
        # Summarize results
        success_count = sum(1 for r in results if r["status"] == "success")
        error_count = sum(1 for r in results if r["status"] == "error")
        skipped_count = sum(1 for r in results if r["status"] == "skipped")
        
        print("\n=== Processing Summary ===")
        print(f"Total companies: {len(results)}")
        print(f"Successfully processed: {success_count}")
        print(f"Errors: {error_count}")
        print(f"Skipped (already processed): {skipped_count}")
        
        # Save summary to file
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        summary_file = os.path.join(project_root, "processing_summary.json")
        with open(summary_file, 'w') as f:
            json.dump({
                "total": len(results),
                "success": success_count,
                "error": error_count,
                "skipped": skipped_count,
                "results": results
            }, f, indent=2)
        
        print(f"Summary saved to {summary_file}")
        
        return results

    async def process_from_gcs(self, bucket_name="finn-cleaned-data", file_path="10k_files/aapl_10k.json"):
        """Process a 10-K report from Google Cloud Storage."""
        try:
            gcs_path = f"gs://{bucket_name}/{file_path}"
            print(f"Processing 10-K from GCS: {gcs_path}")
            
            # Read the file from GCS
            content = read_from_gcs(bucket_name, file_path)
            data = json.loads(content)
            
            # Initialize Pinecone
            pc = init_pinecone()
            if not pc:
                return {"success": False, "error": "Failed to initialize Pinecone"}
            
            # Process the data
            result = await process_10k_data(pc, data, source_path=gcs_path)
            return result
            
        except Exception as e:
            print(f"Error processing from GCS: {e}")
            return {"success": False, "error": str(e)}
    
    async def process_from_json(self, data):
        """Process a 10-K report from JSON data."""
        try:
            # Initialize Pinecone
            pc = init_pinecone()
            if not pc:
                return {"success": False, "error": "Failed to initialize Pinecone"}
            
            # Process the data
            result = await process_10k_data(pc, data)
            return result
            
        except Exception as e:
            print(f"Error processing JSON data: {e}")
            return {"success": False, "error": str(e)}
    
    async def process_by_ticker(self, tickers, skip_embedding=False, fiscal_year="2023"):
        """Process 10-K reports by ticker symbol(s)."""
        try:
            # Convert single ticker to list if needed
            ticker_list = tickers if isinstance(tickers, list) else [tickers]
            
            # Initialize Pinecone
            pc = init_pinecone()
            if not pc:
                return {"success": False, "results": [], "error": "Failed to initialize Pinecone"}
            
            results = []
            
            for ticker in ticker_list:
                try:
                    # Get CIK for ticker
                    cik = get_cik_for_ticker(ticker)
                    if not cik:
                        results.append({"ticker": ticker, "success": False, "error": "CIK not found"})
                        continue
                    
                    # Fetch 10-K filing
                    filing_data = await fetch_10k_filing(cik, ticker, fiscal_year=fiscal_year)
                    if not filing_data:
                        results.append({"ticker": ticker, "success": False, "error": "10-K filing not found"})
                        continue
                    
                    # If skip_embedding is True, just return the filing data
                    if skip_embedding:
                        results.append({
                            "ticker": ticker,
                            "success": True, 
                            "data": filing_data
                        })
                        continue
                    
                    # Process the data
                    result = await process_10k_data(pc, filing_data)
                    result["ticker"] = ticker
                    results.append(result)
                    
                except Exception as e:
                    results.append({"ticker": ticker, "success": False, "error": str(e)})
            
            return {"success": True, "results": results}
            
        except Exception as e:
            print(f"Error in process_by_ticker: {e}")
            return {"success": False, "results": [], "error": str(e)}


# Function to run from command line
async def main(symbols=None, output_dir=None):
    """Main function to process S&P 500 10-K filings."""
    processor = Processor(output_dir=output_dir)
    return await processor.process_companies(symbols=symbols) 