/**
 * Embedding Explorer — Interactive Scatter Plot Visualization
 * Features: lasso/rect selection, zoom/pan, hover tooltips, image gallery,
 *           model switching, projection switching (t-SNE / PCA / UMAP)
 *           with smooth animated dot transitions.
 */

(function () {
    'use strict';

    // ── State ─────────────────────────────────────────────────────
    let data = null;           // Full loaded JSON
    let points = [];           // [{id, image, label, cluster, screenX, screenY, cx, cy}]
    let selectedIds = new Set();
    let currentDataUrl = 'data/embeddings_vitb32.json';
    let currentProjection = 'tsne';   // 'tsne' | 'pca' | 'umap'

    // Per-projection coords cache: projCoords[key] = [[x,y], ...]
    let projCoords = {};

    // Animation state for projection transitions
    let animFrom = null;  // [[x,y], ...] source
    let animTo   = null;  // [[x,y], ...] destination
    let animStart = null;
    const ANIM_DURATION = 600; // ms
    let animFrame = null;

    // View transform
    let viewX = 0, viewY = 0, viewScale = 1;
    let canvasW = 0, canvasH = 0;
    const PADDING = 60;

    // Selection
    let selectionMode = 'lasso';
    let isSelecting = false;
    let lassoPath = [];
    let rectStart = null, rectEnd = null;

    // Pan
    let isPanning = false;
    let panStart = { x: 0, y: 0 };

    // Hover
    let hoveredPoint = null;

    // Canvas refs
    let scatterCanvas, selectionCanvas;
    let scatterCtx, selectionCtx;

    // Cluster colours (12 distinct)
    const CLUSTER_COLORS = [
        '#818cf8', '#c084fc', '#f472b6', '#fb923c',
        '#34d399', '#38bdf8', '#facc15', '#a3e635',
        '#e879f9', '#2dd4bf', '#f87171', '#60a5fa',
    ];

    // ── DOM Refs ──────────────────────────────────────────────────
    const dom = {};

    function cacheDom() {
        dom.totalCount       = document.getElementById('total-count');
        dom.selectedCount    = document.getElementById('selected-count');
        dom.dimInfo          = document.getElementById('dim-info');
        dom.canvasContainer  = document.getElementById('canvas-container');
        dom.tooltip          = document.getElementById('tooltip');
        dom.tooltipImg       = document.getElementById('tooltip-img');
        dom.tooltipLabel     = document.getElementById('tooltip-label');
        dom.galleryGrid      = document.getElementById('gallery-grid');
        dom.galleryPlaceholder = document.getElementById('gallery-placeholder');
        dom.galleryCount     = document.getElementById('gallery-count');
        dom.modal            = document.getElementById('image-modal');
        dom.modalImg         = document.getElementById('modal-img');
        dom.modalLabel       = document.getElementById('modal-label');
        dom.modalCoords      = document.getElementById('modal-coords');
        dom.projSubtitle     = document.getElementById('projection-subtitle');
        dom.scatterSection   = document.getElementById('scatter-section');
    }

    // ── Init ──────────────────────────────────────────────────────
    async function init() {
        cacheDom();
        setupCanvases();
        setupToolbar();
        setupProjectionSelector();
        setupModelSelector();
        setupModal();
        await loadData(currentDataUrl);
        render();
        setupInteraction();
    }

    // ── Canvases ──────────────────────────────────────────────────
    function setupCanvases() {
        scatterCanvas   = document.getElementById('scatter-canvas');
        selectionCanvas = document.getElementById('selection-canvas');
        scatterCtx   = scatterCanvas.getContext('2d');
        selectionCtx = selectionCanvas.getContext('2d');
        resizeCanvases();
        window.addEventListener('resize', () => { resizeCanvases(); render(); });
    }

    function resizeCanvases() {
        const container = dom.canvasContainer;
        const dpr = window.devicePixelRatio || 1;
        canvasW = container.clientWidth;
        canvasH = container.clientHeight;
        for (const c of [scatterCanvas, selectionCanvas]) {
            c.width  = canvasW * dpr;
            c.height = canvasH * dpr;
            c.style.width  = canvasW + 'px';
            c.style.height = canvasH + 'px';
            c.getContext('2d').setTransform(dpr, 0, 0, dpr, 0, 0);
        }
    }

    // ── Data Loading ──────────────────────────────────────────────
    async function loadData(url) {
        const existing = dom.canvasContainer.querySelector('.loading-overlay');
        if (existing) existing.remove();

        const loading = document.createElement('div');
        loading.className = 'loading-overlay';
        loading.innerHTML = '<div class="loading-spinner"></div><div class="loading-text">Loading embeddings…</div>';
        dom.canvasContainer.appendChild(loading);

        try {
            const resp = await fetch(url);
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            data = await resp.json();

            // Cache all projection coords or fall back to inline x/y
            projCoords = {};
            if (data.projections) {
                for (const [key, val] of Object.entries(data.projections)) {
                    projCoords[key] = val.coords;
                }
            }
            // Backward compat: if only x/y inline
            if (!projCoords.tsne && data.points.length) {
                projCoords.tsne = data.points.map(p => [p.x, p.y]);
            }

            // Set active projection to default or keep current if available
            if (!projCoords[currentProjection]) {
                currentProjection = data.default_projection || 'tsne';
            }

            // Build points array
            const coords = projCoords[currentProjection];
            points = data.points.map((p, i) => ({
                ...p,
                cx: coords[i][0],
                cy: coords[i][1],
                screenX: 0,
                screenY: 0,
            }));

            dom.totalCount.textContent = data.num_points;
            dom.dimInfo.textContent    = `${data.embedding_dim}d → 2d`;

            viewX = 0; viewY = 0; viewScale = 1;

            // Update projection button availability
            updateProjectionButtons();

        } catch (e) {
            loading.innerHTML = `<div class="loading-text" style="color:#f472b6;">Failed to load data<br><small style="color:#9898b0">${e.message}</small></div>`;
            console.error(e);
            points = []; data = null;
            dom.totalCount.textContent = '—';
            dom.dimInfo.textContent    = '—';
        }
        loading.remove();
    }

    function updateProjectionButtons() {
        const btns = document.querySelectorAll('.proj-btn');
        btns.forEach(btn => {
            const key = btn.dataset.proj;
            const available = projCoords[key] != null;
            btn.disabled = !available;
            btn.style.opacity = available ? '1' : '0.35';
            btn.title = available
                ? (data?.projections?.[key]?.description || key)
                : `${key.toUpperCase()} not available in this file`;
            btn.classList.toggle('active', key === currentProjection);
        });
        updateSubtitle();
    }

    function updateSubtitle() {
        const labels = { tsne: 't-SNE', pca: 'PCA', umap: 'UMAP', pumap: 'Parametric UMAP' };
        const label = data?.projections?.[currentProjection]?.label || labels[currentProjection] || currentProjection;
        if (dom.projSubtitle) {
            dom.projSubtitle.textContent = `(${label} 2D Projection)`;
        }
    }

    // ── Projection Switching ──────────────────────────────────────
    function setupProjectionSelector() {
        document.getElementById('proj-selector').addEventListener('click', e => {
            const btn = e.target.closest('.proj-btn');
            if (!btn || btn.disabled || btn.classList.contains('active')) return;
            switchProjection(btn.dataset.proj);
        });
    }

    function switchProjection(key) {
        if (!projCoords[key] || key === currentProjection) return;

        // Cancel any running animation
        if (animFrame) { cancelAnimationFrame(animFrame); animFrame = null; }

        // Save current screen-space data positions as animation source
        const from = points.map(p => [p.cx, p.cy]);
        const to   = projCoords[key];

        currentProjection = key;

        // Update buttons
        document.querySelectorAll('.proj-btn').forEach(b =>
            b.classList.toggle('active', b.dataset.proj === key)
        );
        updateSubtitle();
        clearSelection();

        // Animate
        animFrom  = from;
        animTo    = to;
        animStart = performance.now();
        dom.scatterSection.classList.add('proj-transitioning');
        animateProjection();
    }

    function easeInOutCubic(t) {
        return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    }

    function animateProjection() {
        const now = performance.now();
        const raw = (now - animStart) / ANIM_DURATION;
        const t   = Math.min(1, raw);
        const ease = easeInOutCubic(t);

        // Interpolate
        for (let i = 0; i < points.length; i++) {
            points[i].cx = animFrom[i][0] + (animTo[i][0] - animFrom[i][0]) * ease;
            points[i].cy = animFrom[i][1] + (animTo[i][1] - animFrom[i][1]) * ease;
        }

        render();

        if (t < 1) {
            animFrame = requestAnimationFrame(animateProjection);
        } else {
            // Snap to final
            for (let i = 0; i < points.length; i++) {
                points[i].cx = animTo[i][0];
                points[i].cy = animTo[i][1];
            }
            dom.scatterSection.classList.remove('proj-transitioning');
            animFrame = null;
            render();
        }
    }

    // ── Coordinate Transform ──────────────────────────────────────
    function dataToScreen(dx, dy) {
        const cx    = canvasW / 2;
        const cy    = canvasH / 2;
        const scale = (Math.min(canvasW, canvasH) - PADDING * 2) / 2 * viewScale;
        return { x: cx + dx * scale + viewX, y: cy - dy * scale + viewY };
    }

    function updateScreenPositions() {
        for (const p of points) {
            const s = dataToScreen(p.cx, p.cy);
            p.screenX = s.x;
            p.screenY = s.y;
        }
    }

    // ── Rendering ─────────────────────────────────────────────────
    function render() {
        if (!points.length) return;
        updateScreenPositions();
        drawScatter();
    }

    function drawScatter() {
        const ctx = scatterCtx;
        ctx.clearRect(0, 0, canvasW, canvasH);

        const r      = Math.max(3.5, 5.5 / Math.sqrt(viewScale));
        const rHover = r + 3;

        // Unselected
        for (const p of points) {
            if (selectedIds.has(p.id) || p === hoveredPoint) continue;
            const color = CLUSTER_COLORS[p.cluster % CLUSTER_COLORS.length] || '#818cf8';
            ctx.globalAlpha = 0.55;
            ctx.beginPath();
            ctx.arc(p.screenX, p.screenY, r, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();
        }

        // Selected (pink glow)
        ctx.globalAlpha = 1;
        for (const p of points) {
            if (!selectedIds.has(p.id)) continue;
            ctx.beginPath();
            ctx.arc(p.screenX, p.screenY, r + 4, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(244,114,182,0.18)';
            ctx.fill();
            ctx.beginPath();
            ctx.arc(p.screenX, p.screenY, r, 0, Math.PI * 2);
            ctx.fillStyle = '#f472b6';
            ctx.fill();
        }

        // Hovered (yellow)
        if (hoveredPoint) {
            ctx.beginPath();
            ctx.arc(hoveredPoint.screenX, hoveredPoint.screenY, rHover, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(251,191,36,0.18)';
            ctx.fill();
            ctx.beginPath();
            ctx.arc(hoveredPoint.screenX, hoveredPoint.screenY, r + 1, 0, Math.PI * 2);
            ctx.fillStyle = '#fbbf24';
            ctx.strokeStyle = '#fbbf24';
            ctx.lineWidth = 1.5;
            ctx.fill();
            ctx.stroke();
        }
        ctx.globalAlpha = 1;
    }

    function drawSelection() {
        const ctx = selectionCtx;
        ctx.clearRect(0, 0, canvasW, canvasH);

        if (selectionMode === 'lasso' && lassoPath.length > 1) {
            ctx.beginPath();
            ctx.moveTo(lassoPath[0].x, lassoPath[0].y);
            for (let i = 1; i < lassoPath.length; i++) ctx.lineTo(lassoPath[i].x, lassoPath[i].y);
            if (!isSelecting) ctx.closePath();
            ctx.fillStyle = 'rgba(129,140,248,0.08)';
            ctx.fill();
            ctx.strokeStyle = 'rgba(192,132,252,0.6)';
            ctx.lineWidth = 2;
            ctx.setLineDash([6, 4]);
            ctx.stroke();
            ctx.setLineDash([]);
        }

        if (selectionMode === 'rect' && rectStart && rectEnd) {
            const x = Math.min(rectStart.x, rectEnd.x);
            const y = Math.min(rectStart.y, rectEnd.y);
            const w = Math.abs(rectEnd.x - rectStart.x);
            const h = Math.abs(rectEnd.y - rectStart.y);
            ctx.fillStyle = 'rgba(129,140,248,0.08)';
            ctx.fillRect(x, y, w, h);
            ctx.strokeStyle = 'rgba(192,132,252,0.6)';
            ctx.lineWidth = 2;
            ctx.setLineDash([6, 4]);
            ctx.strokeRect(x, y, w, h);
            ctx.setLineDash([]);
        }
    }

    // ── Selection ─────────────────────────────────────────────────
    function pointInPolygon(px, py, poly) {
        let inside = false;
        for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
            const xi = poly[i].x, yi = poly[i].y, xj = poly[j].x, yj = poly[j].y;
            if (((yi > py) !== (yj > py)) && px < (xj - xi) * (py - yi) / (yj - yi) + xi)
                inside = !inside;
        }
        return inside;
    }
    function pointInRect(px, py, r1, r2) {
        return px >= Math.min(r1.x,r2.x) && px <= Math.max(r1.x,r2.x) &&
               py >= Math.min(r1.y,r2.y) && py <= Math.max(r1.y,r2.y);
    }

    function computeSelection() {
        selectedIds.clear();
        for (const p of points) {
            let inside = false;
            if (selectionMode === 'lasso' && lassoPath.length > 2)
                inside = pointInPolygon(p.screenX, p.screenY, lassoPath);
            else if (selectionMode === 'rect' && rectStart && rectEnd)
                inside = pointInRect(p.screenX, p.screenY, rectStart, rectEnd);
            if (inside) selectedIds.add(p.id);
        }
        updateUI();
    }

    function clearSelection() {
        selectedIds.clear();
        lassoPath = [];
        rectStart = rectEnd = null;
        selectionCtx.clearRect(0, 0, canvasW, canvasH);
        updateUI();
        render();
    }

    // ── UI Updates ────────────────────────────────────────────────
    function updateUI() {
        dom.selectedCount.textContent = selectedIds.size;
        updateGallery();
        render();
    }

    function updateGallery() {
        const { galleryGrid: grid, galleryPlaceholder: ph, galleryCount } = dom;
        if (selectedIds.size === 0) {
            grid.innerHTML = '';
            grid.style.display = 'none';
            ph.style.display = 'flex';
            galleryCount.textContent = '';
            return;
        }
        ph.style.display = 'none';
        grid.style.display = 'grid';
        galleryCount.textContent = `${selectedIds.size} images`;

        const selected = points.filter(p => selectedIds.has(p.id));
        // Sort by cluster then by distance from centroid
        let cx = 0, cy = 0;
        for (const p of selected) { cx += p.cx; cy += p.cy; }
        cx /= selected.length; cy /= selected.length;
        selected.sort((a, b) => {
            if (a.cluster !== b.cluster) return a.cluster - b.cluster;
            return (a.cx-cx)**2+(a.cy-cy)**2 - ((b.cx-cx)**2+(b.cy-cy)**2);
        });

        const frag = document.createDocumentFragment();
        selected.forEach((p, i) => {
            const div = document.createElement('div');
            div.className = 'gallery-item';
            div.style.animationDelay = `${i * 0.025}s`;
            div.innerHTML = `<img src="${p.image}" alt="${p.label}" loading="lazy"><div class="item-label">${p.label}</div>`;
            div.addEventListener('click', () => openModal(p));
            frag.appendChild(div);
        });
        grid.innerHTML = '';
        grid.appendChild(frag);
    }

    // ── Interaction ───────────────────────────────────────────────
    function setupInteraction() {
        const canvas = selectionCanvas;

        canvas.addEventListener('mousedown', e => {
            const {x, y} = canvasXY(e);
            if (e.button === 2 || e.ctrlKey || e.metaKey) {
                isPanning = true;
                panStart = { x: e.clientX, y: e.clientY };
                canvas.style.cursor = 'grabbing';
                e.preventDefault(); return;
            }
            isSelecting = true;
            if (selectionMode === 'lasso') lassoPath = [{x, y}];
            else { rectStart = {x, y}; rectEnd = {x, y}; }
        });

        canvas.addEventListener('mousemove', e => {
            const {x, y} = canvasXY(e);

            if (isPanning) {
                viewX += e.clientX - panStart.x;
                viewY += e.clientY - panStart.y;
                panStart = { x: e.clientX, y: e.clientY };
                render(); return;
            }
            if (isSelecting) {
                if (selectionMode === 'lasso') lassoPath.push({x, y});
                else rectEnd = {x, y};
                drawSelection(); return;
            }

            // Hover
            const hit = 12;
            let closest = null, best = Infinity;
            for (const p of points) {
                const d = Math.hypot(p.screenX - x, p.screenY - y);
                if (d < hit && d < best) { closest = p; best = d; }
            }
            if (closest !== hoveredPoint) {
                hoveredPoint = closest;
                render();
                closest ? showTooltip(closest, x, y) : hideTooltip();
            } else if (hoveredPoint) {
                positionTooltip(x, y);
            }
        });

        canvas.addEventListener('mouseup', e => {
            if (isPanning) { isPanning = false; canvas.style.cursor = 'crosshair'; return; }
            if (isSelecting) { isSelecting = false; computeSelection(); drawSelection(); }
        });

        canvas.addEventListener('contextmenu', e => e.preventDefault());

        canvas.addEventListener('mouseleave', () => {
            if (isPanning) { isPanning = false; canvas.style.cursor = 'crosshair'; }
            hoveredPoint = null; hideTooltip(); render();
        });

        canvas.addEventListener('wheel', e => {
            e.preventDefault();
            const {x, y} = canvasXY(e);
            const factor = e.deltaY > 0 ? 0.9 : 1.1;
            const next = viewScale * factor;
            if (next < 0.15 || next > 25) return;
            viewX = x - (x - viewX) * factor;
            viewY = y - (y - viewY) * factor;
            viewScale = next;
            // Clear selection visuals on zoom
            if (lassoPath.length || (rectStart && rectEnd)) {
                lassoPath = []; rectStart = rectEnd = null;
                selectionCtx.clearRect(0, 0, canvasW, canvasH);
            }
            render();
        }, { passive: false });
    }

    function canvasXY(e) {
        const rect = selectionCanvas.getBoundingClientRect();
        return { x: e.clientX - rect.left, y: e.clientY - rect.top };
    }

    // ── Tooltip ───────────────────────────────────────────────────
    function showTooltip(p, mx, my) {
        dom.tooltipImg.src = p.image;
        dom.tooltipLabel.textContent = `${p.label}  (${p.cx.toFixed(2)}, ${p.cy.toFixed(2)})`;
        dom.tooltip.classList.remove('hidden');
        positionTooltip(mx, my);
    }
    function positionTooltip(mx, my) {
        const tw = 138, th = 158;
        let left = mx + 16, top = my - th / 2;
        if (left + tw > canvasW) left = mx - tw - 16;
        top = Math.max(8, Math.min(top, canvasH - th - 8));
        dom.tooltip.style.left = left + 'px';
        dom.tooltip.style.top  = top  + 'px';
    }
    function hideTooltip() { dom.tooltip.classList.add('hidden'); }

    // ── Toolbar ───────────────────────────────────────────────────
    function setupToolbar() {
        const btnLasso = document.getElementById('btn-lasso');
        const btnRect  = document.getElementById('btn-rect');

        btnLasso.addEventListener('click', () => {
            selectionMode = 'lasso';
            btnLasso.classList.add('active');
            btnRect.classList.remove('active');
        });
        btnRect.addEventListener('click', () => {
            selectionMode = 'rect';
            btnRect.classList.add('active');
            btnLasso.classList.remove('active');
        });
        document.getElementById('btn-clear').addEventListener('click', clearSelection);
        document.getElementById('btn-reset-zoom').addEventListener('click', () => {
            viewX = 0; viewY = 0; viewScale = 1;
            clearSelection();
        });
    }

    // ── Model Selector ────────────────────────────────────────────
    function setupModelSelector() {
        const select  = document.getElementById('model-select');
        const wrapper = select.closest('.model-selector-wrapper');

        select.addEventListener('change', async e => {
            currentDataUrl = e.target.value;
            wrapper.classList.add('model-loading');
            clearSelection();
            await loadData(currentDataUrl);
            render();
            wrapper.classList.remove('model-loading');
        });
    }

    // ── Modal ─────────────────────────────────────────────────────
    function setupModal() {
        dom.modal.querySelector('.modal-backdrop').addEventListener('click', closeModal);
        document.getElementById('modal-close').addEventListener('click', closeModal);
        document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });
    }
    function openModal(p) {
        const full = p.image.replace('/thumbs/', '/');
        dom.modalImg.src = full;
        dom.modalLabel.textContent  = p.label;
        dom.modalCoords.textContent = `(${p.cx.toFixed(3)}, ${p.cy.toFixed(3)})`;
        dom.modal.classList.remove('hidden');
    }
    function closeModal() { dom.modal.classList.add('hidden'); }

    // ── Boot ──────────────────────────────────────────────────────
    document.addEventListener('DOMContentLoaded', init);
})();
