import requests
import json
from bs4 import BeautifulSoup
import re
import pandas as pd
import time
from tqdm import tqdm
import os
import asyncio
import argparse

# SEC requires a proper User-Agent header
headers = {
    "User-Agent": "YourCompanyName YourAppName (your.email@example.com)"
}

# Define root directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, "sp500_10k")

# API endpoint settings
API_HOST = "http://localhost:8000"  # Change this to your actual API host
API_ENDPOINT = f"{API_HOST}/process-10k-json"

def get_cik_map():
    """Get CIK numbers for S&P 500 companies."""
    # Get S&P 500 symbols from Wikipedia
    sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(sp500_url)
    sp500_df = tables[0]
    
    # Get CIK mapping from SEC
    sec_cik_url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(sec_cik_url, headers=headers)
    cik_data = response.json()
    
    # Create CIK mapping dictionary
    cik_map = {}
    for entry in cik_data.values():
        if entry['ticker'] in sp500_df['Symbol'].values:
            cik_map[entry['ticker']] = str(entry['cik_str']).zfill(10)
    
    return cik_map

def get_company_10k(cik, symbol):
    """Fetch and process 10-K filing for a single company."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Fetch submission history
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            print(f"Error fetching submission data for {symbol}: {response.status_code}")
            return False
            
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
            return False
        
        # Try to find the most recent 10-K filing
        # First try 2024 filings (fiscal year 2023)
        index = None
        fiscal_year = None
        filing_year = None
        
        for i in ten_k_indices:
            filing_date = filing_dates[i]
            report_date = report_dates[i]
            
            # Check for 2024 filings (fiscal year 2023)
            if filing_date.startswith('2024'):
                index = i
                fiscal_year = "2023"
                filing_year = "2024"
                break
            # If no 2024 filing, check for 2023 filings (fiscal year 2022)
            elif filing_date.startswith('2023'):
                index = i
                fiscal_year = "2022"
                filing_year = "2023"
                # Don't break here - keep looking for 2024 filings
        
        if index is None:
            print(f"No recent 10-K filing (2023-2024) found for {symbol}")
            return False
            
        accession = accession_numbers[index]
        primary_doc = primary_documents[index]
        
        # Build the filing URL
        accession_no = accession.replace("-", "")
        base_url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession_no}"
        html_url = f"{base_url}/{primary_doc}"
        
        # Fetch the filing
        html_response = requests.get(html_url, headers=headers)
        
        if html_response.status_code == 200:
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
            
            # Save the filing content
            output_file = os.path.join(OUTPUT_DIR, f"{symbol.lower()}_10k.json")
            with open(output_file, "w", encoding='utf-8') as f:
                json.dump(filing_content, f, indent=4, ensure_ascii=False)
            
            # Also fetch financial data
            xbrl_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
            xbrl_response = requests.get(xbrl_url, headers=headers)
            
            if xbrl_response.status_code == 200:
                xbrl_data = xbrl_response.json()
                financials_file = os.path.join(OUTPUT_DIR, f"{symbol.lower()}_financials.json")
                with open(financials_file, "w", encoding='utf-8') as f:
                    json.dump(xbrl_data, f, indent=4)
            
            return output_file  # Return the path to the saved file
            
        else:
            print(f"Error downloading HTML content for {symbol}: {html_response.status_code}")
            return False
            
    except Exception as e:
        print(f"Error processing {symbol}: {str(e)}")
        return False

def clean_text(text):
    """Clean text by removing extra whitespace and newlines."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def process_10k_file(file_path, process_api=True):
    """Process a 10-K file with the FastAPI endpoint."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
    
    try:
        # Read the JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            filing_content = json.load(f)
        
        if not process_api:
            print(f"Skipping API processing for {file_path}")
            return True
        
        # Call the API endpoint
        symbol = filing_content.get("metadata", {}).get("symbol", "unknown")
        print(f"Processing {symbol} 10-K with API...")
        
        response = requests.post(
            API_ENDPOINT,
            json=filing_content,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Successfully processed {symbol} 10-K:")
            print(f"  - Vectors added: {result.get('total_vectors_added', 0)}")
            print(f"  - Sections processed: {len(result.get('processed_sections', []))}")
            return True
        else:
            print(f"Error processing {symbol} 10-K with API: {response.status_code}")
            print(response.text)
            return False
    
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return False

async def process_single_company(symbol, cik, process_api=True):
    """Process a single company's 10-K report."""
    print(f"Processing {symbol}...")
    file_path = get_company_10k(cik, symbol)
    
    if file_path:
        success = process_10k_file(file_path, process_api)
        return success
    return False

async def main(symbols=None, process_api=True):
    """Main function to process all companies."""
    # Get CIK numbers for S&P 500 companies
    print("Fetching S&P 500 company information...")
    cik_map = get_cik_map()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Filter symbols if provided
    if symbols:
        symbols = [s.upper() for s in symbols]
        filtered_cik_map = {s: cik_map[s] for s in symbols if s in cik_map}
        if not filtered_cik_map:
            print(f"None of the provided symbols {symbols} were found in S&P 500.")
            return
        cik_map = filtered_cik_map
    
    print(f"Processing {len(cik_map)} companies...")
    successful = 0
    
    # Process each company with a progress bar
    for symbol, cik in tqdm(cik_map.items()):
        success = await process_single_company(symbol, cik, process_api)
        if success:
            successful += 1
        # SEC rate limit: max 10 requests per second
        time.sleep(0.1)
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed {successful} out of {len(cik_map)} companies")
    print(f"Results saved in the '{OUTPUT_DIR}' directory")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process SEC 10-K filings for S&P 500 companies")
    parser.add_argument("--symbols", nargs="+", help="List of company symbols to process (e.g., AAPL MSFT)")
    parser.add_argument("--api-host", help=f"API host address (default: {API_HOST})")
    parser.add_argument("--skip-api", action="store_true", help="Skip processing with API, only download files")
    
    args = parser.parse_args()
    
    if args.api_host:
        API_HOST = args.api_host
        API_ENDPOINT = f"{API_HOST}/process-10k-json"
    
    asyncio.run(main(symbols=args.symbols, process_api=not args.skip_api)) 