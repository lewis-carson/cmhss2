import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
import nltk

# Download necessary NLTK data
print("Checking NLTK data...")
required_nltk_resources = ['punkt', 'punkt_tab']
for resource in required_nltk_resources:
    try:
        if resource == 'punkt' or resource == 'punkt_tab':
            nltk.data.find(f'tokenizers/{resource}')
        else:
            nltk.data.find(f'corpora/{resource}')
    except LookupError:
        try:
            print(f"Downloading {resource}...")
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Warning: Could not download {resource}: {e}")

def get_period(date_str):
    """
    Determines the historical period based on the date string.
    """
    if not date_str:
        return None
    try:
        # Handle partial dates if necessary, but assuming YYYY-MM-DD based on sample
        if len(date_str) == 4:
            dt = datetime(int(date_str), 1, 1)
        elif len(date_str) == 7:
             dt = datetime(int(date_str[:4]), int(date_str[5:7]), 1)
        else:
            dt = datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        return None
    
    # Define period boundaries
    if dt < datetime(1775, 4, 19):
        return "Colonial"
    elif dt <= datetime(1783, 9, 3):
        return "Revolutionary War"
    elif dt < datetime(1789, 4, 30):
        return "Confederation"
    elif dt < datetime(1797, 3, 4):
        return "Washington Presidency"
    elif dt < datetime(1801, 3, 4):
        return "Adams Presidency"
    elif dt < datetime(1809, 3, 4):
        return "Jefferson Presidency"
    elif dt < datetime(1817, 3, 4):
        return "Madison Presidency"
    else:
        return "Post-Madison"

def get_yules_k(token_counts, total_words):
    """
    Calculates Yule's K measure of vocabulary richness.
    K = 10^4 * (S2 - S1) / (S1^2)
    where S1 = total words (N)
    S2 = sum(m^2 * Vm), where Vm is number of words appearing m times
    """
    N = total_words
    if N == 0:
        return 0
    
    # Vm: frequency of frequencies
    # How many words appear 1 time, 2 times, etc.
    m_counts = Counter(token_counts.values())
    
    S1 = N
    S2 = sum((m**2) * vm for m, vm in m_counts.items())
    
    k = 10000 * (S2 - S1) / (S1**2)
    return k

def main():
    input_file = 'letters.jsonl'
    
    # Data structures to aggregate stats per period
    period_stats = defaultdict(lambda: {
        'total_chars': 0,
        'total_words': 0,
        'total_sentences': 0,
        'word_counts': Counter(),
        'doc_count': 0
    })
    
    print(f"Reading {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i % 10000 == 0 and i > 0:
                    print(f"Processing letter {i}...")
                try:
                    data = json.loads(line)
                    date_str = data.get('date-from')
                    content = data.get('content', '')
                    
                    period = get_period(date_str)
                    if period and content:
                        # Tokenize sentences
                        sentences = nltk.sent_tokenize(content)
                        num_sentences = len(sentences)
                        
                        # Tokenize words (simple regex to extract words, ignoring punctuation)
                        # Lowercase for vocabulary counts
                        words = re.findall(r'\b[a-z]+\b', content.lower())
                        num_words = len(words)
                        
                        if num_words == 0:
                            continue
                            
                        total_chars = sum(len(w) for w in words)
                        
                        # Update period stats
                        stats = period_stats[period]
                        stats['total_sentences'] += num_sentences
                        stats['total_words'] += num_words
                        stats['total_chars'] += total_chars
                        stats['word_counts'].update(words)
                        stats['doc_count'] += 1
                        
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return

    print("Computing Stylometric Metrics...")
    
    # Order of periods for consistent output
    ordered_periods = [
        "Colonial", "Revolutionary War", "Confederation", 
        "Washington Presidency", "Adams Presidency", "Jefferson Presidency", 
        "Madison Presidency", "Post-Madison"
    ]
    
    results = {}
    
    for period in ordered_periods:
        if period not in period_stats:
            continue
            
        stats = period_stats[period]
        
        if stats['total_words'] == 0 or stats['total_sentences'] == 0:
            continue
            
        avg_word_len = stats['total_chars'] / stats['total_words']
        avg_sent_len = stats['total_words'] / stats['total_sentences']
        yules_k = get_yules_k(stats['word_counts'], stats['total_words'])
        
        results[period] = {
            "Letters": stats['doc_count'],
            "Avg Word Len": avg_word_len,
            "Avg Sent Len": avg_sent_len,
            "Yule's K": yules_k
        }
        
        print(f"\n--- {period} ---")
        print(f"Letters: {stats['doc_count']}")
        print(f"Avg Word Length: {avg_word_len:.4f}")
        print(f"Avg Sentence Length: {avg_sent_len:.4f}")
        print(f"Yule's K (Richness): {yules_k:.4f}")

    # Display results using Pandas for better formatting if available
    try:
        import pandas as pd
        print("\nSummary of Stylometric Analysis by Period:")
        
        df = pd.DataFrame.from_dict(results, orient='index')
        # Reorder rows
        df = df.reindex(ordered_periods)
        # Drop rows with NaN (periods not found)
        df = df.dropna()
        
        print(df.to_string())
    except ImportError:
        pass

if __name__ == "__main__":
    main()
