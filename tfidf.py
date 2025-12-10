import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    if not date_str:
        return None
    try:
        if len(date_str) == 4:
            dt = datetime(int(date_str), 1, 1)
        elif len(date_str) == 7:
             dt = datetime(int(date_str[:4]), int(date_str[5:7]), 1)
        else:
            dt = datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        return None
    
    # period boundaries
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
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split() 
    return [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]

def main():
    input_file = 'letters.jsonl'
    
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # period-specific or common archaic stopwords/noise
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

    print("Preprocessing and computing TF...")
    period_tf = {}
    
    ordered_periods = [
        "Colonial", "Revolutionary War", "Confederation", 
        "Washington Presidency", "Adams Presidency", "Jefferson Presidency", 
        "Madison Presidency", "Post-Madison"
    ]
    
    all_words = set()
    
    for period in ordered_periods:
        if period not in period_docs:
            continue
            
        print(f"Processing period: {period} ({len(period_docs[period])} letters)")
    
        # combine all text
        full_text = " ".join(period_docs[period])
        
        # preprocess
        tokens = preprocess_text(full_text, lemmatizer, stop_words)
        
        # count terms
        tf_counts = Counter(tokens)
        total_words = len(tokens)
        
        if total_words == 0:
            continue

        # store TF (normalized by document length)
        # TF(t, d) = count(t, d) / total_words(d)
        period_tf[period] = {word: count / total_words for word, count in tf_counts.items()}
        all_words.update(tf_counts.keys())

    print("Computing IDF...")
    # IDF(t) = log(N / df(t))
    # N = number of periods
    # df(t) = number of periods containing term t
    
    N = len(period_tf)
    idf = {}
    
    for word in all_words:
        df = sum(1 for period in period_tf if word in period_tf[period])
        idf[word] = math.log(N / df) if df > 0 else 0

    print("Computing TF-IDF and extracting top terms...")
    
    results = {}
    
    for period in ordered_periods:
        if period not in period_tf:
            continue
            
        # calculate TF-IDF for this period
        tfidf_scores = {}
        for word, tf in period_tf[period].items():
            tfidf_scores[word] = tf * idf[word]
            
        # top 20
        top_terms = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:20]
        results[period] = top_terms
        
        print(f"\n--- {period} ---")
        for word, score in top_terms:
            print(f"{word}: {score:.6f}")

    print("\nSummary of Top Terms by Period:")
    data = {}
    for period in ordered_periods:
        if period in results:
            data[period] = [word for word, score in results[period]]
    
    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in data.items() ]))
    print(df.to_string())

    heatmap_terms = set()
    for period in ordered_periods:
        if period in results:
            # Get top 5 terms
            top_5 = [term for term, score in results[period][:5]]
            heatmap_terms.update(top_5)
            
    heatmap_terms = sorted(list(heatmap_terms))
    
    heatmap_data = []
    for term in heatmap_terms:
        row = []
        for period in ordered_periods:
            if period in period_tf:
                if term in period_tf[period]:
                    score = period_tf[period][term] * idf[term]
                else:
                    score = 0
                row.append(score)
            else:
                row.append(0)
        heatmap_data.append(row)
        
    heatmap_df = pd.DataFrame(heatmap_data, index=heatmap_terms, columns=ordered_periods)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(heatmap_df, cmap="YlGnBu", annot=False)
    plt.title("TF-IDF Scores of Top Terms Across Periods")
    plt.xlabel("Historical Period")
    plt.ylabel("Term")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig("tfidf_heatmap.png")


if __name__ == "__main__":
    main()
