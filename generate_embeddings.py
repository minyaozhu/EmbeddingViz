#!/usr/bin/env python3
"""
Embedding Visualization Pipeline
1. Download 500 diverse images from Lorem Picsum
2. Generate CLIP embeddings using a specified model
3. Project embeddings to 2D using t-SNE
4. Export results as JSON for the web UI

Usage:
    python3 generate_embeddings.py                          # ViT-B-32 (default)
    python3 generate_embeddings.py --model ViT-bigG-14      # ViT-bigG/14
    python3 generate_embeddings.py --model all              # Both models
"""

import os
import json
import argparse
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ── Configuration ──────────────────────────────────────────────
NUM_IMAGES = 500
IMAGE_DIR = Path("images")
THUMB_DIR = Path("images/thumbs")
DATA_DIR = Path("data")
THUMB_SIZE = (200, 200)

# Available model configs: (open_clip model name, pretrained tag, short label)
MODEL_CONFIGS = {
    "ViT-B-32": {
        "model":      "ViT-B-32",
        "pretrained": "laion2b_s34b_b79k",
        "label":      "ViT-B/32",
        "output":     "data/embeddings_vitb32.json",
    },
    "ViT-bigG-14": {
        "model":      "ViT-bigG-14",
        "pretrained": "laion2b_s39b_b160k",
        "label":      "ViT-bigG/14",
        "output":     "data/embeddings_vitbigg14.json",
    },
}

# 550 Lorem Picsum IDs (pool of 550 to ensure 500 successful downloads;
# Picsum occasionally removes or skips IDs so we over-provision by ~10%)
PICSUM_IDS = list(range(10, 560))


def download_images():
    """Download up to NUM_IMAGES images from Lorem Picsum (skips already-downloaded)."""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    THUMB_DIR.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    for i, pid in enumerate(PICSUM_IDS[:NUM_IMAGES]):
        img_path = IMAGE_DIR / f"img_{i:03d}.jpg"
        thumb_path = THUMB_DIR / f"img_{i:03d}.jpg"

        if img_path.exists() and thumb_path.exists():
            print(f"  [{i+1}/{NUM_IMAGES}] Already exists: {img_path.name}")
            downloaded += 1
            continue

        url = f"https://picsum.photos/id/{pid}/512/512"
        print(f"  [{i+1}/{NUM_IMAGES}] Downloading picsum id={pid}...")

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "EmbeddingViz/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()

            with open(img_path, "wb") as f:
                f.write(data)

            img = Image.open(img_path).convert("RGB")
            img.thumbnail(THUMB_SIZE, Image.LANCZOS)
            img.save(thumb_path, "JPEG", quality=85)

            downloaded += 1
            time.sleep(0.3)

        except Exception as e:
            print(f"    ⚠ Failed to download id={pid}: {e}")
            continue

    print(f"\n✓ Downloaded {downloaded} images")
    return downloaded


def generate_embeddings_for_model(cfg, batch_size=32):
    """Generate CLIP embeddings using the given model config (batched for speed)."""
    import open_clip

    print(f"\n── Loading {cfg['label']} ──")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  Using device: {device}  batch_size: {batch_size}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg["model"], pretrained=cfg["pretrained"]
    )
    model = model.to(device)
    model.eval()

    print(f"\n── Generating embeddings ({cfg['label']}) ──")
    image_files = sorted(IMAGE_DIR.glob("img_*.jpg"))
    total = len(image_files)

    all_embeddings = []
    all_image_paths = []
    all_labels = []

    # Process in batches
    for batch_start in range(0, total, batch_size):
        batch_files = image_files[batch_start:batch_start + batch_size]
        batch_tensors = []
        batch_meta = []  # (image_path_str, label)

        for img_path in batch_files:
            try:
                img = Image.open(img_path).convert("RGB")
                batch_tensors.append(preprocess(img))
                batch_meta.append((f"images/thumbs/{img_path.name}", img_path.stem))
            except Exception as e:
                print(f"  ⚠ Failed to preprocess {img_path.name}: {e}")

        if not batch_tensors:
            continue

        batch_tensor = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            embs = model.encode_image(batch_tensor)
            embs = embs / embs.norm(dim=-1, keepdim=True)

        all_embeddings.extend(embs.cpu().numpy())
        for ip, lb in batch_meta:
            all_image_paths.append(ip)
            all_labels.append(lb)

        done = min(batch_start + batch_size, total)
        print(f"  [{done}/{total}] batches done", end='\r', flush=True)

    print()  # newline after \r
    embeddings = np.array(all_embeddings)
    print(f"\n✓ Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")

    # Free GPU memory before t-SNE
    del model, batch_tensor
    if device == "mps":
        torch.mps.empty_cache()

    return embeddings, all_image_paths, all_labels


def _normalize(coords):
    """Normalize coordinates to [-1, 1] per dimension."""
    for d in range(coords.shape[1]):
        lo, hi = coords[:, d].min(), coords[:, d].max()
        if hi > lo:
            coords[:, d] = 2 * (coords[:, d] - lo) / (hi - lo) - 1
    return coords


def project_tsne(embeddings):
    """t-SNE projection to 2D."""
    n = len(embeddings)
    perplexity = min(50, max(5, n // 5))
    print(f"  t-SNE  (n={n}, perplexity={perplexity})...", flush=True)
    coords = TSNE(
        n_components=2, perplexity=perplexity, random_state=42,
        max_iter=1000, learning_rate="auto", init="pca",
    ).fit_transform(embeddings)
    return _normalize(coords)


def project_pca(embeddings):
    """PCA projection to 2D."""
    print(f"  PCA  (n={len(embeddings)})...", flush=True)
    coords = PCA(n_components=2, random_state=42).fit_transform(embeddings)
    return _normalize(coords)


def project_umap(embeddings):
    """UMAP projection to 2D."""
    from umap import UMAP
    n = len(embeddings)
    n_neighbors = min(15, max(5, n // 20))
    print(f"  UMAP  (n={n}, n_neighbors={n_neighbors})...", flush=True)
    coords = UMAP(
        n_components=2, n_neighbors=n_neighbors, min_dist=0.1,
        random_state=42, verbose=False,
    ).fit_transform(embeddings)
    return _normalize(coords)


def project_pumap(embeddings):
    """Parametric UMAP projection to 2D using a neural-net encoder.
    
    Unlike standard UMAP, the trained encoder can project new/unseen points
    in O(n) without re-running the full algorithm — ideal for streaming data.
    """
    try:
        from umap.parametric_umap import ParametricUMAP
    except ImportError:
        print("  ⚠ umap-learn >= 0.5.3 with TensorFlow required for ParametricUMAP. Skipping.")
        return None

    n = len(embeddings)
    n_neighbors = min(15, max(5, n // 20))
    print(f"  Parametric UMAP  (n={n}, n_neighbors={n_neighbors})...", flush=True)
    try:
        reducer = ParametricUMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            verbose=False,
        )
        coords = reducer.fit_transform(embeddings)
        return _normalize(coords)
    except Exception as e:
        print(f"  ⚠ ParametricUMAP failed: {e}")
        return None


def compute_all_projections(embeddings):
    """Compute t-SNE, PCA, UMAP, and Parametric UMAP projections and return as a dict."""
    print("\n── Computing projections ──")
    results = {}
    results["tsne"] = {"coords": project_tsne(embeddings).tolist(),
                       "label": "t-SNE", "description": "t-SNE (perplexity=50, PCA init)"}
    results["pca"]  = {"coords": project_pca(embeddings).tolist(),
                       "label": "PCA",  "description": "PCA (top 2 principal components)"}
    results["umap"] = {"coords": project_umap(embeddings).tolist(),
                       "label": "UMAP", "description": "UMAP (n_neighbors=15, min_dist=0.1)"}

    pumap_coords = project_pumap(embeddings)
    if pumap_coords is not None:
        results["pumap"] = {"coords": pumap_coords.tolist(),
                            "label": "P-UMAP", "description": "Parametric UMAP (neural-net encoder, O(n) inference)"}
    else:
        print("  ℹ  Parametric UMAP skipped — results will have tsne/pca/umap only")

    print("✓ All projections done")
    return results


def perform_clustering(tsne_coords, n_clusters=12):
    """KMeans clustering on the t-SNE layout (used for point colouring)."""
    import numpy as np
    print(f"\n── KMeans clustering (k={n_clusters}) ──")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    return kmeans.fit_predict(np.array(tsne_coords)).tolist()


def export_json(image_paths, labels, embeddings, projections, cluster_labels, cfg):
    """Export multi-projection data as JSON for the web UI."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Default x/y from t-SNE for backward compatibility
    tsne_coords = projections["tsne"]["coords"]
    points = [
        {
            "id": i,
            "x": tsne_coords[i][0],  # default (used if JS falls back)
            "y": tsne_coords[i][1],
            "image": image_paths[i],
            "label": labels[i],
            "cluster": int(cluster_labels[i]),
        }
        for i in range(len(tsne_coords))
    ]

    data = {
        "num_points": len(points),
        "embedding_dim": int(embeddings.shape[1]),
        "default_projection": "tsne",
        "num_clusters": len(set(cluster_labels)),
        "model": cfg["model"],
        "pretrained": cfg["pretrained"],
        "model_label": cfg["label"],
        "projections": projections,   # {tsne: {coords, label, description}, pca: ..., umap: ...}
        "points": points,
    }

    output_path = Path(cfg["output"])
    with open(output_path, "w") as f:
        json.dump(data, f)  # no indent — saves ~40% file size for 500 pts

    print(f"\n✓ Exported {len(points)} points → {output_path}")
    return output_path


def run_pipeline(cfg):
    """Run the full pipeline for a single model config."""
    print(f"\n{'=' * 60}")
    print(f"  Model: {cfg['label']}")
    print(f"  Output: {cfg['output']}")
    print(f"{'=' * 60}")

    embeddings, image_paths, labels = generate_embeddings_for_model(cfg)
    projections = compute_all_projections(embeddings)
    cluster_labels = perform_clustering(projections["tsne"]["coords"])
    export_json(image_paths, labels, embeddings, projections, cluster_labels, cfg)


def main():
    parser = argparse.ArgumentParser(description="Generate CLIP image embeddings")
    parser.add_argument(
        "--model",
        default="ViT-B-32",
        choices=list(MODEL_CONFIGS.keys()) + ["all"],
        help="CLIP model to use (default: ViT-B-32)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip image download step (images already present)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Embedding Visualization Pipeline")
    print("=" * 60)

    # Step 1: Download images (shared across models)
    if not args.skip_download:
        print("\n── Step 1: Downloading images ──")
        download_images()
    else:
        print("\n── Step 1: Skipping download (--skip-download) ──")

    # Step 2+: Generate embeddings for selected model(s)
    models_to_run = list(MODEL_CONFIGS.values()) if args.model == "all" else [MODEL_CONFIGS[args.model]]

    for cfg in models_to_run:
        run_pipeline(cfg)

    print(f"\n{'=' * 60}")
    print("  ✓ Pipeline complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
