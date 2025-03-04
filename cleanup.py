#!/usr/bin/env python3
"""
Cleanup Script for 10-K Processing Project
-----------------------------------------
This script cleans up temporary files and directories, including __pycache__ directories.
"""

import os
import shutil
import argparse

def cleanup_pycache(directory='.'):
    """Remove all __pycache__ directories recursively."""
    count = 0
    for root, dirs, files in os.walk(directory):
        if '__pycache__' in dirs:
            pycache_path = os.path.join(root, '__pycache__')
            print(f"Removing {pycache_path}")
            shutil.rmtree(pycache_path)
            count += 1
    return count

def cleanup_pyc_files(directory='.'):
    """Remove all .pyc files recursively."""
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pyc'):
                pyc_path = os.path.join(root, file)
                print(f"Removing {pyc_path}")
                os.remove(pyc_path)
                count += 1
    return count

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Clean up temporary files and directories.")
    parser.add_argument("--all", action="store_true", help="Clean up all temporary files (including .pyc files)")
    args = parser.parse_args()
    
    print("Starting cleanup...")
    
    # Clean up __pycache__ directories
    pycache_count = cleanup_pycache()
    print(f"Removed {pycache_count} __pycache__ directories")
    
    # Clean up .pyc files if requested
    if args.all:
        pyc_count = cleanup_pyc_files()
        print(f"Removed {pyc_count} .pyc files")
    
    print("Cleanup complete!")

if __name__ == "__main__":
    main() 