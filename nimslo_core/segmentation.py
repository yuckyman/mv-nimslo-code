"""
Segmentation module for Nimslo images.

Uses U²-Net (via rembg) for salient object detection,
with optional depth-based fallback for difficult cases.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from PIL import Image
import io
import os
import warnings

# Configure OpenMP BEFORE importing rembg/onnxruntime
# This prevents the deprecated omp_set_nested warning
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'  # Limit threads to avoid conflicts
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_MAX_ACTIVE_LEVELS'] = '1'  # Use max_active_levels instead of nested

# Try to configure OpenMP programmatically if possible
try:
    import ctypes
    # Try to set max_active_levels directly if OpenMP is available
    try:
        libomp = ctypes.CDLL(None)
        if hasattr(libomp, 'omp_set_max_active_levels'):
            libomp.omp_set_max_active_levels(1)
    except:
        pass
except:
    pass

warnings.filterwarnings('ignore', message='.*omp_set_nested.*')
warnings.filterwarnings('ignore', message='.*omp_set_max_active_levels.*')

# Lazy imports for heavy dependencies
_rembg_session = None
_rembg_available = None


def _check_rembg_available():
    """Check if rembg can be imported without crashing."""
    global _rembg_available
    if _rembg_available is None:
        try:
            import rembg
            _rembg_available = True
        except Exception as e:
            _rembg_available = False
            print(f"Warning: rembg not available: {e}")
    return _rembg_available


def _get_rembg_session():
    """Lazy-load rembg session to avoid startup overhead."""
    global _rembg_session
    if _rembg_session is None:
        if not _check_rembg_available():
            raise RuntimeError("rembg is not available - cannot perform U²-Net segmentation")
        try:
            from rembg import new_session
            # u2net is the default, good balance of quality and speed
            _rembg_session = new_session("u2net")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize rembg session: {e}")
    return _rembg_session


def segment_subject(
    img: np.ndarray,
    method: str = "u2net",
    return_confidence: bool = False
) -> np.ndarray | Tuple[np.ndarray, float]:
    """
    Segment the main subject from the background.
    
    Args:
        img: Input BGR image
        method: Segmentation method ("u2net" or "depth")
        return_confidence: Whether to return confidence score
        
    Returns:
        Binary mask (255 for subject, 0 for background)
        If return_confidence=True, returns (mask, confidence)
    """
    if method == "u2net":
        mask, confidence = _segment_u2net(img)
    elif method == "depth":
        mask, confidence = _segment_depth(img)
    else:
        raise ValueError(f"Unknown segmentation method: {method}")
    
    if return_confidence:
        return mask, confidence
    return mask


def _segment_u2net(img: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Segment using U²-Net via rembg library.
    
    Returns:
        Tuple of (binary mask, confidence score)
    """
    try:
        from rembg import remove
    except ImportError as e:
        raise RuntimeError(f"rembg not available: {e}. Install with: pip install rembg onnxruntime")
    
    # Convert BGR to RGB for PIL
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    # Get session and remove background
    try:
        session = _get_rembg_session()
    except Exception as e:
        raise RuntimeError(f"Failed to get rembg session: {e}")
    
    try:
        # Remove returns RGBA image with alpha channel as mask
        result = remove(pil_img, session=session, only_mask=True)
    except Exception as e:
        raise RuntimeError(f"rembg segmentation failed: {e}")
    
    # Convert mask to numpy
    mask = np.array(result)
    
    # Ensure binary mask
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Calculate confidence based on mask properties
    confidence = _calculate_mask_confidence(binary_mask)
    
    return binary_mask, confidence


def _segment_depth(img: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Segment using monocular depth estimation.
    
    Fallback method when U²-Net doesn't work well.
    Uses Intel DPT model for depth estimation.
    
    Returns:
        Tuple of (binary mask, confidence score)
    """
    try:
        from transformers import DPTImageProcessor, DPTForDepthEstimation
        import torch
    except ImportError:
        raise ImportError("Depth segmentation requires transformers and torch")
    
    # Load model (cached after first call)
    processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")
    
    # Prepare image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inputs = processor(images=img_rgb, return_tensors="pt")
    
    # Get depth prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    
    # Interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    )
    
    # Convert to numpy and normalize
    depth = prediction.squeeze().cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth = (depth * 255).astype(np.uint8)
    
    # Threshold using Otsu's method (foreground is typically closer/brighter)
    _, mask = cv2.threshold(depth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Clean up mask with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    confidence = _calculate_mask_confidence(mask)
    
    return mask, confidence


def _calculate_mask_confidence(mask: np.ndarray) -> float:
    """
    Calculate confidence score for a segmentation mask.
    
    Based on:
    - Area ratio (subject should cover 10-40% of image)
    - Compactness (well-defined subjects have good area/perimeter ratio)
    
    Returns:
        Confidence score between 0 and 1
    """
    h, w = mask.shape[:2]
    total_pixels = h * w
    
    # Calculate area ratio
    mask_pixels = np.sum(mask > 0)
    area_ratio = mask_pixels / total_pixels
    
    # Ideal range is 10-40% of image
    if 0.1 <= area_ratio <= 0.4:
        area_score = 1.0
    elif area_ratio < 0.1:
        area_score = area_ratio / 0.1
    elif area_ratio > 0.4:
        area_score = max(0, 1 - (area_ratio - 0.4) / 0.3)
    else:
        area_score = 0.5
    
    # Calculate compactness
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)
        if perimeter > 0:
            compactness = 4 * np.pi * area / (perimeter ** 2)
            # Normalize compactness (circle = 1, more complex = lower)
            compactness_score = min(compactness * 2, 1.0)
        else:
            compactness_score = 0.5
    else:
        compactness_score = 0.0
    
    # Combined confidence
    confidence = 0.6 * area_score + 0.4 * compactness_score
    
    return confidence


def get_segmentation_mask(
    img: np.ndarray,
    confidence_threshold: float = 0.5,
    fallback_to_depth: bool = True,
    use_subprocess: bool = True
) -> Tuple[np.ndarray, float, str]:
    """
    Get the best segmentation mask, with automatic method selection.
    
    Tries U²-Net first (optionally via subprocess to avoid kernel crashes),
    falls back to depth-based if confidence is low or rembg fails.
    
    Args:
        img: Input BGR image
        confidence_threshold: Minimum confidence to accept U²-Net result
        fallback_to_depth: Whether to try depth method if U²-Net fails
        use_subprocess: Use subprocess isolation for rembg (prevents kernel crashes)
        
    Returns:
        Tuple of (mask, confidence, method_used)
    """
    # Try U²-Net first (with subprocess isolation if requested)
    if use_subprocess:
        try:
            from .segmentation_subprocess import segment_u2net_subprocess
            mask, confidence = segment_u2net_subprocess(img)
            if confidence >= confidence_threshold:
                return mask, confidence, "u2net_subprocess"
        except Exception as e:
            # Subprocess failed, try direct import or fallback
            pass
    
    # Try direct U²-Net (may crash kernel in Jupyter)
    try:
        mask, confidence = segment_subject(img, method="u2net", return_confidence=True)
        if confidence >= confidence_threshold:
            return mask, confidence, "u2net"
    except Exception:
        # rembg failed, will fall back to depth
        mask = None
        confidence = 0.0
    
    # Fall back to depth-based segmentation
    if fallback_to_depth or mask is None:
        try:
            depth_mask, depth_confidence = segment_subject(
                img, method="depth", return_confidence=True
            )
            
            if mask is None or depth_confidence > confidence:
                return depth_mask, depth_confidence, "depth"
        except Exception as e:
            # Depth also failed, return what we have or create fallback
            if mask is not None:
                return mask, confidence, "u2net_fallback"
            # Last resort: center mask
            h, w = img.shape[:2]
            fallback_mask = np.zeros((h, w), dtype=np.uint8)
            y1, y2 = int(h*0.2), int(h*0.8)
            x1, x2 = int(w*0.2), int(w*0.8)
            fallback_mask[y1:y2, x1:x2] = 255
            return fallback_mask, 0.3, "fallback"
    
    return mask, confidence, "u2net"


def refine_mask_grabcut(
    img: np.ndarray,
    initial_mask: np.ndarray,
    iterations: int = 5
) -> np.ndarray:
    """
    Refine a segmentation mask using GrabCut.
    
    Args:
        img: Input BGR image
        initial_mask: Initial binary mask (255=foreground, 0=background)
        iterations: Number of GrabCut iterations
        
    Returns:
        Refined binary mask
    """
    # Create GrabCut mask
    gc_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    gc_mask[initial_mask > 127] = cv2.GC_PR_FGD  # Probable foreground
    gc_mask[initial_mask <= 127] = cv2.GC_PR_BGD  # Probable background
    
    # Erode mask to get definite foreground
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    definite_fg = cv2.erode(initial_mask, kernel, iterations=2)
    gc_mask[definite_fg > 127] = cv2.GC_FGD
    
    # Dilate inverse mask to get definite background
    definite_bg = cv2.dilate(initial_mask, kernel, iterations=2)
    gc_mask[definite_bg == 0] = cv2.GC_BGD
    
    # Apply GrabCut
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    try:
        cv2.grabCut(img, gc_mask, None, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        # GrabCut can fail on certain images, return original mask
        return initial_mask
    
    # Extract foreground
    output_mask = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
        255, 0
    ).astype(np.uint8)
    
    return output_mask


def visualize_mask(
    img: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """
    Overlay segmentation mask on image for visualization.
    
    Args:
        img: Input BGR image
        mask: Binary mask
        alpha: Transparency of overlay
        color: BGR color for mask overlay
        
    Returns:
        Image with mask overlay
    """
    overlay = img.copy()
    mask_bool = mask > 127
    
    # Apply color to mask region
    overlay[mask_bool] = (
        overlay[mask_bool] * (1 - alpha) + 
        np.array(color) * alpha
    ).astype(np.uint8)
    
    return overlay

