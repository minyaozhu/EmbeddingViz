# Embedding Explorer

Interactive visualization of CLIP image embeddings projected to 2D space.

**[Live Demo →](https://YOUR_USERNAME.github.io/EmbeddingViz)**

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
