import asyncio
import json
import os
from app.routes.process_10k import get_cik_for_ticker, fetch_10k_filing

async def save_10k_json(ticker, fiscal_year="2023"):
    print(f"Fetching 10K data for {ticker} (fiscal year {fiscal_year})...")
    
    # Get CIK for the ticker
    cik = await get_cik_for_ticker(ticker)
    if not cik:
        print(f"Error: Could not find CIK for ticker {ticker}")
        return
    
    print(f"Found CIK for {ticker}: {cik}")
    
    # Fetch the 10K filing
    filing_data = await fetch_10k_filing(cik, ticker, fiscal_year)
    if not filing_data:
        print(f"Error: Could not fetch 10K filing for {ticker} (CIK: {cik}, fiscal year: {fiscal_year})")
        return
    
    # Save the data to a JSON file
    output_file = f"{ticker}_10k_{fiscal_year}.json"
    with open(output_file, 'w') as f:
        json.dump(filing_data, f, indent=2)
    
    print(f"Successfully saved 10K data to {os.path.abspath(output_file)}")
    print(f"File size: {os.path.getsize(output_file) / (1024 * 1024):.2f} MB")
    
    # Print some basic stats about the 10K
    content_sections = filing_data.get("content", {})
    print(f"\nBasic stats for {ticker} 10K:")
    print(f"Number of sections: {len(content_sections)}")
    
    print("\nSections and their sizes:")
    for section_name, paragraphs in content_sections.items():
        section_text = "\n".join([str(p) for p in paragraphs]).strip()
        print(f"- {section_name}: {len(section_text)} chars")

if __name__ == "__main__":
    ticker = "INTC"
    fiscal_year = "2023"
    asyncio.run(save_10k_json(ticker, fiscal_year)) 