import requests
import pandas as pd
import re
import os
from typing import Dict, Optional, List, Tuple, Any
from bs4 import BeautifulSoup

# SEC requires a proper User-Agent header
HEADERS = {
    "User-Agent": "YourCompanyName YourAppName (your.email@example.com)"
}

def get_cik_map() -> Dict[str, str]:
    """Get CIK numbers for S&P 500 companies."""
    # Get S&P 500 symbols from Wikipedia
    sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(sp500_url)
    sp500_df = tables[0]
    
    # Get CIK mapping from SEC
    sec_cik_url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(sec_cik_url, headers=HEADERS)
    cik_data = response.json()
    
    # Create CIK mapping dictionary
    cik_map = {}
    for entry in cik_data.values():
        if entry['ticker'] in sp500_df['Symbol'].values:
            cik_map[entry['ticker']] = str(entry['cik_str']).zfill(10)
    
    return cik_map

async def get_cik_for_ticker(ticker: str) -> Optional[str]:
    """Get the CIK for a specific ticker symbol."""
    try:
        sec_cik_url = "https://www.sec.gov/files/company_tickers.json"
        response = requests.get(sec_cik_url, headers=HEADERS)
        data = response.json()
        
        for entry in data.values():
            if entry["ticker"].upper() == ticker.upper():
                return str(entry["cik_str"]).zfill(10)
        
        return None
    except Exception as e:
        print(f"Error getting CIK for {ticker}: {e}")
        return None

async def fetch_10k_filing(cik: str, symbol: str, fiscal_year: str = "2023") -> Optional[dict]:
    """Fetch 10-K filing for a company from the SEC Edgar database."""
    try:
        # First, get the company's submissions
        submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        response = requests.get(submissions_url, headers=HEADERS)
        submissions = response.json()
        
        # Find the latest 10-K filing from the specified fiscal year
        filing_found = False
        accession_number = None
        report_date = None
        
        for filing in submissions.get('filings', {}).get('recent', {}).get('accessionNumber', []):
            idx = submissions['filings']['recent']['accessionNumber'].index(filing)
            form = submissions['filings']['recent']['form'][idx]
            
            filing_date = submissions['filings']['recent']['filingDate'][idx]
            # Check if it's a 10-K and from the desired fiscal year
            if form == "10-K" and filing_date.startswith(fiscal_year):
                accession_number = filing.replace('-', '')
                report_date = filing_date
                filing_found = True
                break
        
        if not filing_found:
            print(f"No 10-K filing found for {symbol} (CIK: {cik}) for fiscal year {fiscal_year}")
            return None
        
        # Get the filing detail
        filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession_number}"
        
        # Get the index page to find the 10-K document
        index_url = f"{filing_url}/index.json"
        response = requests.get(index_url, headers=HEADERS)
        index_data = response.json()
        
        document_url = None
        for file in index_data.get('directory', {}).get('item', []):
            if file.get('name', '').endswith('.htm') and not file.get('name', '').startswith('R'):
                document_url = f"{filing_url}/{file['name']}"
                break
        
        if not document_url:
            print(f"Could not find 10-K document URL for {symbol}")
            return None
        
        # Fetch the 10-K HTML document
        response = requests.get(document_url, headers=HEADERS)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract important sections
        item1_found = False
        item1a_found = False
        item1b_found = False
        item2_found = False
        item3_found = False
        item7_found = False
        item7a_found = False
        item8_found = False
        
        item1_text = ""
        item1a_text = ""
        item1b_text = ""
        item2_text = ""
        item3_text = ""
        item7_text = ""
        item7a_text = ""
        item8_text = ""
        
        # Try different patterns for section headers
        section_patterns = {
            "item1": [r'item\s*1\.?\s*business', r'item\s*1\s*business', r'business'],
            "item1a": [r'item\s*1a\.?\s*risk\s*factors', r'item\s*1a\s*risk\s*factors', r'risk\s*factors'],
            "item1b": [r'item\s*1b\.?\s*unresolved\s*staff\s*comments', r'item\s*1b\s*unresolved'],
            "item2": [r'item\s*2\.?\s*properties', r'item\s*2\s*properties', r'properties'],
            "item3": [r'item\s*3\.?\s*legal\s*proceedings', r'item\s*3\s*legal', r'legal\s*proceedings'],
            "item7": [r'item\s*7\.?\s*management', r'item\s*7\s*management', r'management.*discussion'],
            "item7a": [r'item\s*7a\.?\s*quantitative', r'item\s*7a\s*quantitative', r'quantitative.*qualitative'],
            "item8": [r'item\s*8\.?\s*financial', r'item\s*8\s*financial', r'financial\s*statements']
        }
        
        for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'span', 'b', 'strong']):
            text = tag.get_text().lower()
            
            # Check for Item 1
            if not item1_found and any(re.search(pattern, text) for pattern in section_patterns["item1"]):
                item1_found = True
                continue
            
            # Check for Item 1A
            if item1_found and not item1a_found and any(re.search(pattern, text) for pattern in section_patterns["item1a"]):
                item1a_found = True
                continue
            
            # Check for Item 1B
            if item1a_found and not item1b_found and any(re.search(pattern, text) for pattern in section_patterns["item1b"]):
                item1b_found = True
                continue
            
            # Check for Item 2
            if (item1a_found or item1b_found) and not item2_found and any(re.search(pattern, text) for pattern in section_patterns["item2"]):
                item2_found = True
                continue
                
            # Check for Item 3
            if item2_found and not item3_found and any(re.search(pattern, text) for pattern in section_patterns["item3"]):
                item3_found = True
                continue
                
            # Check for Item 7
            if not item7_found and any(re.search(pattern, text) for pattern in section_patterns["item7"]):
                item7_found = True
                continue
                
            # Check for Item 7A
            if item7_found and not item7a_found and any(re.search(pattern, text) for pattern in section_patterns["item7a"]):
                item7a_found = True
                continue
                
            # Check for Item 8
            if (item7_found or item7a_found) and not item8_found and any(re.search(pattern, text) for pattern in section_patterns["item8"]):
                item8_found = True
                continue
                
            # Collect text for each section
            if item1_found and not item1a_found:
                item1_text += tag.get_text() + " "
            elif item1a_found and not item1b_found and not item2_found:
                item1a_text += tag.get_text() + " "
            elif item1b_found and not item2_found:
                item1b_text += tag.get_text() + " "
            elif item2_found and not item3_found:
                item2_text += tag.get_text() + " "
            elif item3_found and not item7_found:
                item3_text += tag.get_text() + " "
            elif item7_found and not item7a_found:
                item7_text += tag.get_text() + " "
            elif item7a_found and not item8_found:
                item7a_text += tag.get_text() + " "
            elif item8_found:
                item8_text += tag.get_text() + " "
                
        # Clean the collected text
        for item_text in [item1_text, item1a_text, item1b_text, item2_text, item3_text, item7_text, item7a_text, item8_text]:
            item_text = re.sub(r'\s+', ' ', item_text).strip()
            
        return {
            "symbol": symbol,
            "cik": cik,
            "report_date": report_date,
            "sections": {
                "item1_business": item1_text,
                "item1a_risk_factors": item1a_text,
                "item1b_unresolved_comments": item1b_text,
                "item2_properties": item2_text,
                "item3_legal_proceedings": item3_text,
                "item7_mda": item7_text,
                "item7a_market_risk": item7a_text,
                "item8_financial_statements": item8_text
            }
        }
        
    except Exception as e:
        print(f"Error fetching 10-K for {symbol} (CIK: {cik}): {e}")
        return None 