#!/usr/bin/env python3
"""
Download letters from the Founders Online API.
Uses the metadata file to get document references and downloads the full content.
"""

import json
import os
import sys
from pathlib import Path
import urllib.request
import urllib.error
from typing import Optional
import time

# API base URL
API_BASE = "https://founders.archives.gov/API/docdata/"
METADATA_FILE = "founders-online-metadata.json"
OUTPUT_DIR = "letters"

def setup_output_dir():
    """Create the output directory if it doesn't exist."""
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

def extract_doc_id(permalink: str) -> str:
    """
    Extract document ID from permalink URL.
    Example: https://founders.archives.gov/documents/Adams/01-01-02-0001-0001-0001
    Returns: Adams/01-01-02-0001-0001-0001
    """
    if "/documents/" in permalink:
        return permalink.split("/documents/")[1]
    return None

def fetch_letter(doc_id: str) -> Optional[dict]:
    """
    Fetch a single letter from the API.
    
    Args:
        doc_id: Document ID in format "Project/number-number-..."
        
    Returns:
        Dictionary with letter data or None if request fails
    """
    url = f"{API_BASE}{doc_id}"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        print(f"Error fetching {doc_id}: HTTP {e.code}")
        return None
    except Exception as e:
        print(f"Error fetching {doc_id}: {e}")
        return None

def save_letter(doc_id: str, content: dict) -> str:
    """
    Save letter content to a file.
    
    Args:
        doc_id: Document ID
        content: Letter content dictionary
        
    Returns:
        Path to saved file
    """
    # Create filename from doc_id
    filename = doc_id.replace("/", "_").replace(":", "_") + ".json"
    filepath = Path(OUTPUT_DIR) / filename
    
    with open(filepath, 'w') as f:
        json.dump(content, f, indent=2)
    
    return str(filepath)

def main():
    """Main function to download all letters."""
    setup_output_dir()
    
    # Load metadata
    print(f"Loading metadata from {METADATA_FILE}...")
    try:
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print(f"Error: {METADATA_FILE} not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing {METADATA_FILE}: {e}")
        sys.exit(1)
    
    if not isinstance(metadata, list):
        print("Error: Metadata should be a JSON array")
        sys.exit(1)
    
    print(f"Found {len(metadata)} documents in metadata")
    
    # Download letters
    successful = 0
    failed = 0
    
    for idx, doc in enumerate(metadata):
        print(f"Progress: {idx}/{len(metadata)}")
        
        permalink = doc.get("permalink")
        if not permalink:
            print(f"Warning: No permalink for document {idx}")
            failed += 1
            continue
        
        doc_id = extract_doc_id(permalink)
        if not doc_id:
            print(f"Warning: Could not extract doc_id from {permalink}")
            failed += 1
            continue
        
        # Fetch from API
        letter_data = fetch_letter(doc_id)
        if letter_data:
            save_letter(doc_id, letter_data)
            successful += 1
        else:
            failed += 1
        
        # Rate limiting - be respectful to the API
        if idx % 10 == 0:
            time.sleep(0.1)
    
    print(f"\nDownload complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Letters saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
