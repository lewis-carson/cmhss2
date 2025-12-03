import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams

# Download necessary NLTK data
print("Checking NLTK data...")
required_nltk_resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'punkt_tab']
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

def preprocess_text(text, lemmatizer, stop_words):
    """
    Tokenizes, removes stopwords, and lemmatizes the text.
    """
    # Lowercase
    text = text.lower()
    # Replace non-alphabetic characters with spaces to prevent merging (e.g. "Hamilton-Secy")
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Tokenize (simple split is faster and sufficient for this scale)
    tokens = text.split() 
    # Remove stopwords and lemmatize
    return [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]

def main():
    input_file = 'letters.jsonl'
    
    # Initialize NLTK tools
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    # Add some period-specific or common archaic stopwords/noise
    stop_words.update([
        'thou', 'thee', 'thy', 'hath', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'will', 
        'one', 'two', 'much', 'many', 'time', 'letter', 'sir', 'esq', 'mr', 'mrs', 'dear', 'humble', 'servant',
        'obedient', 'honor', 'favour', 'received', 'wrote', 'written', 'know', 'make', 'give', 'say', 'see',
        'think', 'take', 'like', 'well', 'upon', 'also', 'even', 'yet', 'now', 'made', 'great', 'good'
    ])

    period_docs = defaultdict(list)
    
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
                        period_docs[period].append(content)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return

    print("Generating N-grams...")
    
    # Order of periods for consistent output
    ordered_periods = [
        "Colonial", "Revolutionary War", "Confederation", 
        "Washington Presidency", "Adams Presidency", "Jefferson Presidency", 
        "Madison Presidency", "Post-Madison"
    ]
    
    results_bigrams = {}
    results_trigrams = {}
    
    for period in ordered_periods:
        if period not in period_docs:
            continue
            
        print(f"Processing period: {period} ({len(period_docs[period])} letters)")
        # Combine all text for the period
        full_text = " ".join(period_docs[period])
        
        # Preprocess
        tokens = preprocess_text(full_text, lemmatizer, stop_words)
        
        if not tokens:
            continue

        # Generate Bigrams
        bigram_counts = Counter(ngrams(tokens, 2))
        top_bigrams = bigram_counts.most_common(20)
        results_bigrams[period] = top_bigrams
        
        # Generate Trigrams
        trigram_counts = Counter(ngrams(tokens, 3))
        top_trigrams = trigram_counts.most_common(20)
        results_trigrams[period] = top_trigrams

        print(f"\n--- {period} Top Bigrams ---")
        for bg, count in top_bigrams:
            print(f"{' '.join(bg)}: {count}")
            
        print(f"\n--- {period} Top Trigrams ---")
        for tg, count in top_trigrams:
            print(f"{' '.join(tg)}: {count}")

    # Display results using Pandas for better formatting if available
    try:
        import pandas as pd
        
        print("\nSummary of Top Bigrams by Period:")
        data_bg = {}
        for period in ordered_periods:
            if period in results_bigrams:
                data_bg[period] = [" ".join(bg) for bg, count in results_bigrams[period]]
        df_bg = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in data_bg.items() ]))
        print(df_bg.to_string())

        print("\nSummary of Top Trigrams by Period:")
        data_tg = {}
        for period in ordered_periods:
            if period in results_trigrams:
                data_tg[period] = [" ".join(tg) for tg, count in results_trigrams[period]]
        df_tg = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in data_tg.items() ]))
        print(df_tg.to_string())
        
    except ImportError:
        pass

if __name__ == "__main__":
    main()
