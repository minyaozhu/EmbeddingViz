import json
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path

def add_clusters(json_path, n_clusters=7):
    path = Path(json_path)
    if not path.exists():
        print(f"Skipping {json_path}, does not exist.")
        return
    
    with open(path, "r") as f:
        data = json.load(f)
    
    points = data["points"]
    # Cluster based on 2D coordinates
    coords = np.array([[p["x"], p["y"]] for p in points])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(coords)
    
    for i, p in enumerate(points):
        p["cluster"] = int(labels[i])
        
    data["num_clusters"] = n_clusters
        
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Added {n_clusters} clusters to {json_path}")

if __name__ == "__main__":
    add_clusters("data/embeddings_vitb32.json")
    add_clusters("data/embeddings_vitbigg14.json")
