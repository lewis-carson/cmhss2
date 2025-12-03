import json
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import os

def draw_period_network(period_name, edges, output_filename):
    print(f"Drawing network for {period_name}...")
    
    # Create directed graph
    G = nx.DiGraph()

    # Add edges with weights
    for edge in edges:
        G.add_edge(edge['from'], edge['to'], weight=edge['count'])

    # Calculate statistics
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    if num_nodes > 1:
        density = nx.density(G)
        avg_degree = sum(dict(G.degree()).values()) / num_nodes
        max_degree_node = max(dict(G.degree()).items(), key=lambda x: x[1])
    else:
        density = 0
        avg_degree = 0
        max_degree_node = ("None", 0)

    print(f"  Stats for {period_name}:")
    print(f"    Nodes: {num_nodes}")
    print(f"    Edges: {num_edges}")
    print(f"    Density: {density:.5f}")
    print(f"    Avg Degree: {avg_degree:.2f}")
    print(f"    Max Degree Node: {max_degree_node[0]} ({max_degree_node[1]})")

    if len(G.nodes()) == 0:
        print("  Skipping empty network")
        return

    # Filter for top 50 nodes by degree (reduced from 100 for cleaner period plots)
    top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:50]
    top_nodes_set = {node for node, degree in top_nodes}
    G = G.subgraph(top_nodes_set)

    print(f"  Filtered: {len(G.nodes())} nodes")

    # Create the visualization
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

    fig, ax = plt.subplots(1, 1, figsize=(12, 12), facecolor='white')
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.axis('off')
    ax.set_title(f"{period_name}", fontsize=20, pad=20)

    # Calculate circular layout
    # Sort nodes by degree to group important ones
    nodes = sorted(list(G.nodes()), key=lambda n: G.degree(n), reverse=True)
    n_nodes = len(nodes)
    node_angles = {node: 2 * np.pi * i / n_nodes for i, node in enumerate(nodes)}
    radius = 1.0

    # Helper to get cartesian coordinates
    def get_coords(angle, r):
        return r * np.cos(angle), r * np.sin(angle)

    # Draw edges as Bezier curves (chords)
    from matplotlib.path import Path
    import matplotlib.patches as patches

    edges_data = G.edges(data=True)
    weights = [edge[2]['weight'] for edge in edges_data]
    max_weight = max(weights) if weights else 1

    for u, v, data in G.edges(data=True):
        if u not in node_angles or v not in node_angles:
            continue
            
        angle_u = node_angles[u]
        angle_v = node_angles[v]
        
        start = get_coords(angle_u, radius * 0.95)
        end = get_coords(angle_v, radius * 0.95)
        
        # Control point is (0,0) for a simple chord through center
        control = (0, 0)
        
        verts = [start, control, end]
        codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
        
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor='none', 
                                  lw=0.5 + (data['weight'] / max_weight * 3), 
                                  edgecolor='#CFD8DC', alpha=0.6)
        ax.add_patch(patch)

    # Draw nodes
    node_sizes = [G.degree(node) for node in nodes]
    max_degree = max(node_sizes) if node_sizes else 1

    for node in nodes:
        angle = node_angles[node]
        x, y = get_coords(angle, radius)
        
        # Size relative to degree
        size = 100 + (G.degree(node) / max_degree * 500)
        
        ax.scatter([x], [y], s=size, c='#FF6B6B', edgecolors='white', linewidth=1.5, zorder=10)
        
        # Add labels for high degree nodes
        if G.degree(node) > max_degree * 0.1: # Label top 10% roughly
            # Calculate label position slightly outside
            label_x, label_y = get_coords(angle, radius * 1.08)
            
            # Determine alignment based on position
            rotation = np.degrees(angle)
            if x > 0:
                ha = 'left'
                rotation = rotation + 180
                label_x += 0.02
            else:
                ha = 'right'
                label_x -= 0.02
            
            # Simple label placement
            ax.text(label_x, label_y, node.split(',')[0], ha=ha, va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {output_filename}")

# Load the network data
with open('network_per_period.json', 'r') as f:
    period_data = json.load(f)

# Define period order for consistent processing
periods = [
    "Colonial", "Revolutionary War", "Confederation", 
    "Washington Presidency", "Adams Presidency", 
    "Jefferson Presidency", "Madison Presidency", "Post-Madison"
]

for period in periods:
    if period in period_data:
        filename = f"network_{period.replace(' ', '_')}.png"
        draw_period_network(period, period_data[period], filename)
