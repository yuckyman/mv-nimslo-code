"""
Subprocess-based segmentation wrapper to isolate rembg from Jupyter kernel.

This module runs rembg in a separate process to avoid kernel crashes.
"""

import cv2
import numpy as np
import subprocess
import tempfile
import json
from pathlib import Path
from typing import Tuple
import sys


def segment_u2net_subprocess(img: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Segment using U²-Net via rembg in a separate process.
    
    This avoids kernel crashes by isolating rembg from the Jupyter kernel.
    
    Args:
        img: Input BGR image
        
    Returns:
        Tuple of (binary mask, confidence score)
    """
    # Save image to temp file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_in:
        cv2.imwrite(tmp_in.name, img)
        tmp_in_path = tmp_in.name
    
    tmp_out_path = tmp_in_path.replace('.jpg', '_mask.png')
    
    try:
        # Run rembg in separate process
        script = f"""
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OMP_MAX_ACTIVE_LEVELS'] = '1'

from rembg import remove
from PIL import Image
import cv2
import numpy as np

img = cv2.imread('{tmp_in_path}')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(img_rgb)

result = remove(pil_img, only_mask=True)
mask = np.array(result)
_, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

cv2.imwrite('{tmp_out_path}', binary_mask)

# Calculate confidence
h, w = binary_mask.shape
total = h * w
mask_pixels = np.sum(binary_mask > 0)
area_ratio = mask_pixels / total
confidence = min(area_ratio / 0.1, 1.0) if area_ratio < 0.1 else (1.0 if area_ratio <= 0.4 else max(0, 1 - (area_ratio - 0.4) / 0.3))

import json
print(json.dumps({{'confidence': float(confidence)}}))
"""
        
        # Write script to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as script_file:
            script_file.write(script)
            script_path = script_file.name
        
        # Run script
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"rembg subprocess failed: {result.stderr}")
        
        # Load mask
        mask = cv2.imread(tmp_out_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError("Failed to load mask from subprocess")
        
        # Parse confidence from output
        try:
            # Find JSON line in output
            for line in result.stdout.strip().split('\n'):
                if line.strip().startswith('{'):
                    conf_data = json.loads(line)
                    confidence = conf_data['confidence']
                    break
            else:
                raise ValueError("No JSON found")
        except:
            # Fallback confidence calculation
            h, w = mask.shape
            area_ratio = np.sum(mask > 127) / (h * w)
            confidence = min(area_ratio / 0.1, 1.0) if area_ratio < 0.1 else (1.0 if area_ratio <= 0.4 else max(0, 1 - (area_ratio - 0.4) / 0.3))
        
        return mask, confidence
        
    finally:
        # Cleanup
        for path in [tmp_in_path, tmp_out_path, script_path]:
            try:
                Path(path).unlink(missing_ok=True)
            except:
                pass

