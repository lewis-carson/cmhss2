import json
from collections import defaultdict
from itertools import product
from datetime import datetime

def get_period(date_str):
    """
    Determines the historical period based on the date string.
    """
    if not date_str:
        return None
    try:
        # Handle partial dates if necessary
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

def extract_network(metadata_file, output_file):
    """
    Extract a network of from:to connections between senders and recipients
    from the founders online metadata, grouped by period.
    """
    
    # Load metadata
    print(f"Loading metadata from {metadata_file}...")
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Dictionary to store connections with counts per period
    # Structure: { period: { "author -> recipient": count } }
    period_connections = defaultdict(lambda: defaultdict(int))
    
    print("Processing documents...")
    # Extract connections from each document
    for doc in metadata:
        authors = doc.get('authors', [])
        recipients = doc.get('recipients', [])
        date_str = doc.get('date-from')
        
        period = get_period(date_str)
        if not period:
            continue

        # Create connections from each author to each recipient
        for author, recipient in product(authors, recipients):
            connection = f"{author} -> {recipient}"
            period_connections[period][connection] += 1
    
    # Convert to list of dictionaries for better structure
    final_output = {}
    
    for period, connections in period_connections.items():
        network = [
            {
                "from": connection.split(' -> ')[0],
                "to": connection.split(' -> ')[1],
                "count": count
            }
            for connection, count in sorted(connections.items(), key=lambda x: x[1], reverse=True)
        ]
        final_output[period] = network
        print(f"Period {period}: {len(network)} unique connections")

    # Save network
    print(f"Saving networks to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(final_output, f, indent=2)
    
if __name__ == "__main__":
    extract_network(
        'founders-online-metadata.json',
        'network_per_period.json'
    )
