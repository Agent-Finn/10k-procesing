#!/usr/bin/env python3
"""
SEC 10-K Filing Processor - Command Line Interface
-------------------------------------------------
This module provides a command-line interface for fetching and processing 
10-K filings from the SEC EDGAR database for S&P 500 companies.
"""

import asyncio
import argparse
import os
from app.processor import main as processor_main

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process S&P 500 10-K filings from the SEC EDGAR database."
    )
    parser.add_argument(
        "--symbols", 
        nargs="+", 
        help="Specific company symbols to process (e.g., AAPL MSFT GOOGL)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        help="Directory to save the output files (default: ./sp500_10k)"
    )
    return parser.parse_args()

def run_cli():
    """Run the command-line interface."""
    # Parse command line arguments
    args = parse_args()
    
    # Run the processor
    asyncio.run(
        processor_main(
            symbols=args.symbols,
            output_dir=args.output_dir
        )
    ) 

if __name__ == "__main__":
    run_cli() 