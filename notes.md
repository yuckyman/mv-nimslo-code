# Nimslo Aligner - Development Notes

## Setup Issues & Fixes

### 1. Missing `onnxruntime` dependency

**Problem:** After installing `rembg>=2.0.50`, importing it failed with:
```
ModuleNotFoundError: No module named 'onnxruntime'
```

**Cause:** `rembg` uses ONNX models for U²-Net inference but doesn't explicitly list `onnxruntime` as a hard dependency (it's in extras).

**Fix:** Added `onnxruntime>=1.16.0` to `requirements.txt`:
```bash
pip install onnxruntime
```

---

### 2. numba caching error in sandbox

**Problem:** When running in Cursor's sandboxed terminal:
```
RuntimeError: cannot cache function '_make_tree': no locator available for file '.../pymatting/util/kdtree.py'
```

**Cause:** `pymatting` (a dependency of `rembg`) uses `numba` with JIT caching. The sandbox blocks write access to the numba cache directory.

**Fix:** Run with full permissions (not a code fix, just run outside sandbox):
```bash
# Works fine in regular terminal or with full permissions
python nimslo_cli.py ...
```

---

### 3. Dependency conflicts (non-blocking)

**Problem:** pip warned about conflicts:
```
mediapipe 0.10.21 requires numpy<2, but you have numpy 2.2.6 which is incompatible.
streamlit 1.44.1 requires protobuf<6, but you have protobuf 6.32.1 which is incompatible.
```

**Cause:** Other packages in the conda environment have stricter numpy/protobuf requirements.

**Fix:** These are unrelated to our project - we don't use mediapipe or streamlit. Safe to ignore. If they cause issues elsewhere, use a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Model Download

On first run, U²-Net model downloads to `~/.u2net/u2net.onnx` (~176MB). This is cached and reused for subsequent runs.

---

### 4. NumPy 2.x incompatibility with onnxruntime

**Problem:** `onnxruntime` (used by `rembg`) is not fully compatible with NumPy 2.x, causing import/runtime errors.

**Fix:** Constrain NumPy to < 2.0 in `requirements.txt`:
```txt
numpy>=1.24.0,<2.0  # onnxruntime/rembg not fully compatible with numpy 2.x yet
```

Then reinstall:
```bash
pip install "numpy>=1.24.0,<2.0"
```

This downgrades to numpy 1.26.4, which works perfectly with all our dependencies.

---

### 5. Black bars in output GIFs from stereoscopic alignment

**Problem:** After warping images for alignment, black borders appear around the edges due to the transformation. These black bars reduce the visual quality of the output GIFs.

**Fix:** Enhanced the `_crop_to_valid_region()` function in `gif_generator.py`:
- Increased threshold from 5 to 15 (more aggressive detection of black borders)
- Increased margin from 5 to 10 pixels (better padding around valid region)
- Added fallback to union if intersection of valid regions is empty
- Added minimum size check (100x100) to prevent over-cropping

The cropping is enabled by default (`crop_valid_region=True`) in `create_boomerang_gif()`, so GIFs automatically remove black bars while preserving the aligned content.

---

### 6. Flashing in output GIFs due to brightness differences

**Problem:** Even after alignment, different frames have different exposure/brightness levels, causing flashing when the GIF loops.

**Fix:** Added `_normalize_brightness()` function that:
- Calculates average brightness for each frame using LAB color space L channel
- Uses median brightness as target (robust to outliers)
- Adjusts each frame's lightness to match the target while preserving color
- Enabled by default (`normalize_brightness=True`) in `create_boomerang_gif()`

This ensures smooth transitions between frames without brightness flashing.

---

### 7. Kernel crashes when importing rembg in Jupyter - UNRESOLVED

**Status:** ⚠️ **STILL CRASHING** - Kernel dies even before importing rembg, suggesting deeper OpenMP/onnxruntime conflicts with Jupyter kernel process.

**Problem:** Jupyter kernel crashes when importing `rembg`/`onnxruntime`, showing OMP warnings before kernel death. Kernel now crashes even during startup, before any rembg import.

**Root Cause:** OpenMP library conflicts between rembg/onnxruntime and the Jupyter kernel environment on macOS. This appears to be a fundamental incompatibility - even module-level OpenMP configuration code in `segmentation.py` causes kernel crashes.

**Attempted Fixes (All Failed):**
1. ✅ Set `OMP_NUM_THREADS=1`, `OMP_MAX_ACTIVE_LEVELS=1` environment variables
2. ✅ Programmatic `omp_set_max_active_levels()` call
3. ✅ Removed segmentation from `__init__.py` to avoid module-level imports
4. ✅ Created subprocess wrapper (`segmentation_subprocess.py`) - not tested yet
5. ✅ Made depth-based segmentation the default

**Current Workaround:** 
- **Notebook uses depth-based segmentation only** (Intel DPT via transformers)
- **CLI works perfectly** - use it for U²-Net segmentation:
  ```bash
  python nimslo_cli.py ../nimslo_raw/01/ -o test.gif
  ```

**Recommended Approach:**
1. **For notebook demos:** Use depth-based segmentation (already default)
2. **For production/CLI:** Use CLI which supports U²-Net without issues
3. **Future:** Consider alternative segmentation libraries that don't use onnxruntime:
   - MediaPipe Selfie Segmentation (lighter, no onnxruntime)
   - DeepLabV3+ (PyTorch only, no onnxruntime)
   - Simple edge-based segmentation (OpenCV only)

**Note:** The CLI works perfectly because it runs in a regular Python process, not a Jupyter kernel. The kernel crash is specific to the Jupyter/IPython environment.

---

## Test Results (Batch 01)

```
Processing 01...
  Segmenting subjects...
    Frame 1: u2net (conf: 1.00)
    Frame 2: u2net (conf: 1.00)
    Frame 3: u2net (conf: 1.00)
    Frame 4: u2net (conf: 1.00)
  Aligning frames...
    Frame 2: 151 matches, 146 inliers, IoU: 0.98
    Frame 3: 128 matches, 112 inliers, IoU: 0.97
    Frame 4: 110 matches, 90 inliers, IoU: 0.95
  ✓ Saved: ../outputs/test_01.gif (602.1 KB)
```

Excellent segmentation confidence and high IoU scores indicate good alignment quality.

