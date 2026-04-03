# Embedding Explorer

Interactive visualization of CLIP image embeddings projected to 2D space.

**[Live Demo →](https://minyaozhu.github.io/EmbeddingViz)**

![Screenshot](https://picsum.photos/id/10/800/400)

## Features

- 🖼 **472 images** from Lorem Picsum, embedded with CLIP
- 🤖 **Two embedding models**: ViT-B/32 (512d) and ViT-bigG/14 (1280d, ~1.8B params)
- 📐 **Three projection methods**: t-SNE, PCA, UMAP — switch with animated transitions
- 🎯 **Lasso & rectangle region selection** — select any cluster of dots
- 🖼 **Image gallery** — selected images rendered instantly
- 🔍 **Zoom & pan** — scroll to zoom, right-drag to pan
- 💡 **Hover tooltips** with image preview
- 🎨 **12-cluster KMeans color coding**

## Tech Stack

| Layer | Tech |
|---|---|
| Embeddings | [OpenCLIP](https://github.com/mlfoundations/open_clip) ViT-B/32 & ViT-bigG/14 |
| Projection | scikit-learn t-SNE & PCA, umap-learn UMAP |
| Clustering | KMeans (k=12) |
| Frontend | Vanilla HTML/CSS/JS (Canvas API) |
| Images | [Lorem Picsum](https://picsum.photos) |

## Run Locally

```bash
# Install Python deps
pip install open-clip-torch umap-learn scikit-learn pillow torch

# Generate embeddings (downloads ~3.5GB ViT-bigG/14 weights on first run)
python3 generate_embeddings.py --model all

# Serve
python3 -m http.server 8765
# Open http://localhost:8765
```

## Re-generate Embeddings

```bash
# Both models, skip re-downloading images
python3 generate_embeddings.py --model all --skip-download

# Just ViT-B/32 (faster, ~87M params)
python3 generate_embeddings.py --model ViT-B-32 --skip-download

# Just ViT-bigG/14 (~1.8B params)
python3 generate_embeddings.py --model ViT-bigG-14 --skip-download
```

## Project Structure

```
EmbeddingViz/
├── index.html                     # Web UI
├── style.css                      # Styles
├── app.js                         # Interactive visualization
├── generate_embeddings.py         # Embedding pipeline
├── data/
│   ├── embeddings_vitb32.json     # ViT-B/32 embeddings + 3 projections
│   └── embeddings_vitbigg14.json  # ViT-bigG/14 embeddings + 3 projections
└── images/
    ├── img_*.jpg                  # Full-size images (512×512)
    └── thumbs/img_*.jpg           # Thumbnails (200×200)
```

## Scaling Embedding Projections to 1M–1B Samples

### TL;DR

| Algorithm | 10K | 100K | 1M | 1B | Notes |
|---|---|---|---|---|---|
| **t-SNE** | ✅ slow | ⚠️ ~hours | ❌ infeasible | ❌ | O(n²) memory |
| **PCA** | ✅ | ✅ | ⚠️ (incremental) | ⚠️ (streaming) | O(n·d) — most scalable |
| **UMAP** | ✅ fast | ✅ | ⚠️ ~hours | ❌ | O(n log n) approx |
| **Parametric UMAP** | ✅ | ✅ | ✅ | ⚠️ GPU needed | Trains a NN, then infer |
| **RAPIDS cuML UMAP** | ✅ | ✅ | ✅ | ⚠️ multi-GPU | GPU-accelerated |

### Per-Algorithm Analysis

#### t-SNE — Does not scale
- Complexity is **O(n² d)** naive, or **O(n log n)** with the Barnes-Hut approximation
- At 1M points, even BH t-SNE takes **days** and tens of GB of RAM
- **MulticoreTSNE / openTSNE** help at 100K, but 1M+ remains impractical
- Used in the current pipeline — fine for hundreds/thousands, painful beyond ~50K

#### PCA — Best raw scalability
- Exact PCA is O(n·d·k) — feasible up to a few million with batching
- **Incremental/mini-batch PCA** (`sklearn.decomposition.IncrementalPCA`) streams data in chunks → constant memory, can process 1B given enough time
- At 1B with d=512: tractable with distributed compute (Spark PCA, Dask, etc.)
- Downside: linear projection, so cluster structure is often less visible than t-SNE/UMAP

#### UMAP — Sweet spot at 1M–10M
- Uses approximate nearest neighbors (ANN via `pynndescent`) → effectively **O(n log n)**
- At 1M: ~30 min on a modern CPU, ~5 min with GPU
- At 1B: needs distributed/GPU approaches
- **RAPIDS cuML UMAP** (GPU): handles 10M+ on a single A100; multi-GPU scales further
- **Parametric UMAP**: trains a neural net as the encoder → `transform()` is O(n), ideal for massive datasets where you infer on new points without retraining

### What Actually Works at 1B Scale

The real-world approach used by companies like Meta, Google, and Spotify is a **two-stage pipeline**:

```
Raw embeddings (1B × 512d)
     │
     ▼  Stage 1: PCA or random projection
     │  → reduces to 50–100d (fast, near-lossless)
     ▼
Reduced embeddings (1B × 50d)
     │
     ▼  Stage 2: Approximate clustering / ANN index
     │  → FAISS IVF-PQ index, or k-means hierarchy
     ▼
Cluster centroids (~1M) → run UMAP/t-SNE on centroids only
     │
     ▼
2D layout of cluster centers → assign nearby points the same color
```

This is how tools like **FAISS + t-SNE on centroids**, **Nomic Atlas**, and **Weaviate** handle billion-scale visualization.

### Practical Next Step: Parametric UMAP

The only approach that enables true **incremental / online** projection — train once, project new points instantly:

```python
from umap.parametric_umap import ParametricUMAP

# Train once (slow, but only done once)
pumap = ParametricUMAP()
pumap.fit(embeddings_train)

# Transform any future batch in O(n) — no retraining needed
coords = pumap.transform(new_embeddings)
```

This would allow the viz tool to handle live/streaming embedding sets without re-running the full projection pipeline.
