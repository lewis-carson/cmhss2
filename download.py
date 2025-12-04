import json
import sys
from pathlib import Path
import urllib.request
import urllib.error
from typing import Optional
import time

API_BASE = "https://founders.archives.gov/API/docdata/"
METADATA_FILE = "founders-online-metadata.json"
OUTPUT_FILE = "letters.jsonl"
CHECKPOINT_FILE = "download_checkpoint.json"

def load_checkpoint() -> dict:
    if Path(CHECKPOINT_FILE).exists():
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
   
    return {"last_index": -1, "successful": 0, "failed": 0}

def save_checkpoint(index: int, successful: int, failed: int):
    checkpoint = {
        "last_index": index,
        "successful": successful,
        "failed": failed
    }
    
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f)

def extract_doc_id(permalink: str) -> str:
    if "/documents/" in permalink:
        return permalink.split("/documents/")[1]
    
    return None

def fetch_letter(doc_id: str) -> Optional[dict]:
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

def append_letter_to_file(content: dict):
    with open(OUTPUT_FILE, 'a') as f:
        f.write(json.dumps(content) + '\n')

def main():
    checkpoint = load_checkpoint()
    start_index = checkpoint["last_index"] + 1
    successful = checkpoint["successful"]
    failed = checkpoint["failed"]
    
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
    
    if start_index > 0:
        print(f"Resuming from document {start_index}")
        print(f"Previous progress: {successful} successful, {failed} failed")
    
    # download letters
    for idx in range(start_index, len(metadata)):
        if idx % 100 == 0:
            print(f"Progress: {idx}/{len(metadata)} (Successful: {successful}, Failed: {failed})")
        
        doc = metadata[idx]
        
        permalink = doc.get("permalink")
        if not permalink:
            print(f"Warning: No permalink for document {idx}")
            failed += 1
            save_checkpoint(idx, successful, failed)
            continue
        
        doc_id = extract_doc_id(permalink)
        if not doc_id:
            print(f"Warning: Could not extract doc_id from {permalink}")
            failed += 1
            save_checkpoint(idx, successful, failed)
            continue
        
        letter_data = fetch_letter(doc_id)
        if letter_data:
            append_letter_to_file(letter_data)
            successful += 1
        else:
            failed += 1
        
        if idx % 10 == 0:
            save_checkpoint(idx, successful, failed)
        
        if idx % 10 == 0:
            time.sleep(0.1)
    
    # Final save
    save_checkpoint(len(metadata) - 1, successful, failed)
    
    print(f"\nDownload complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Letters saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
