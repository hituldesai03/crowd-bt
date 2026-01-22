# Holdout Set Visualization Guide

This guide shows how to use the `visualize_holdout.py` script to score and visualize the holdout dataset.

## What it Does

The script:
1. Loads the holdout images from `holdout.txt`
2. Runs the trained model to predict quality scores
3. Creates annotated images with scores overlaid
4. Generates summary visualizations and statistics

## Quick Start

### Basic Usage

```bash
python visualize_holdout.py \
  --checkpoint results/best_model.pt \
  --backbone efficientnet_b0 \
  --output-dir holdout_visualizations
```

### With All Options

```bash
python visualize_holdout.py \
  --checkpoint results/best_model.pt \
  --backbone efficientnet_b0 \
  --input-size 448 \
  --device cuda \
  --holdout-file /home/hitul/Desktop/quality-comparison-toolkit/data/holdout.txt \
  --image-dir /home/hitul/Desktop/quality-comparison-toolkit/data/iter_0 \
  --output-dir holdout_visualizations \
  --summary-grid holdout_summary.png \
  --distribution-plot score_distribution.png \
  --max-grid-images 100 \
  --font-size 40
```

## Outputs

After running, you'll get:

### 1. Annotated Images Directory (`holdout_visualizations/`)
- Each image from holdout set with quality score overlaid
- Color-coded scores (green=high, yellow=medium, red=low)
- Original filenames preserved

### 2. Scores JSON (`holdout_visualizations/scores.json`)
```json
[
  {
    "path": "/path/to/img_00005_patch_8.png",
    "filename": "img_00005_patch_8.png",
    "score": 1.2345,
    "output_path": "holdout_visualizations/img_00005_patch_8.png"
  },
  ...
]
```

### 3. Summary Grid (`holdout_visualizations/holdout_summary.png`)
- Grid showing top images ranked by quality
- Useful for quick visual inspection
- Default: 100 images, 10 per row

### 4. Score Distribution Plot (`holdout_visualizations/score_distribution.png`)
- Histogram of all predicted scores
- Box plot showing distribution
- Statistics (mean, median, std, min, max)

## Example Output

```
============================================================
Holdout Set Visualization
============================================================

Loading model from results/best_model.pt
Model loaded: efficientnet_b0, input size: 448

Loading holdout images from holdout.txt
Found 184 images

Creating annotated images in holdout_visualizations
Scoring 184 holdout images...
100%|████████████████████████████████| 184/184 [00:15<00:00, 11.89it/s]

Scores saved to holdout_visualizations/scores.json

============================================================
Score Statistics
============================================================
Total images: 184
Successfully scored: 184
Mean score: 0.2341
Median score: 0.1876
Std score: 0.8923
Min score: -1.4532
Max score: 2.1876

Top 5 highest quality:
  img_00345_patch_7.png: 2.1876
  img_00199_patch_5.png: 1.9234
  img_00384_patch_6.png: 1.8765
  ...

Bottom 5 lowest quality:
  img_00028_patch_8.png: -1.4532
  img_00496_patch_0.png: -1.3421
  ...
```

## Customization

### Change Font Size
```bash
--font-size 60  # Larger text
```

### Show More Images in Grid
```bash
--max-grid-images 200  # Show up to 200 images
```

### Use CPU Instead of GPU
```bash
--device cpu
```

### Different Output Location
```bash
--output-dir my_custom_output
```

## Score Interpretation

- **Positive scores**: Higher quality than average
- **Negative scores**: Lower quality than average
- **Score range**: Typically between -2 and +2
- **Color coding**:
  - 🟢 Green: High quality (score > 0.5)
  - 🟡 Yellow: Medium quality (-0.5 to 0.5)
  - 🔴 Red: Low quality (score < -0.5)

## Troubleshooting

### "No module named matplotlib"
```bash
pip install matplotlib
```

### "CUDA out of memory"
```bash
# Use CPU instead
--device cpu
```

### "Font not found" warning
The script will fall back to default font automatically. To use a better font:
```bash
# Install fonts
sudo apt-get install fonts-dejavu
```

## Integration with Other Scripts

### Get scores programmatically:
```python
import json

with open('holdout_visualizations/scores.json', 'r') as f:
    results = json.load(f)

# Get all scores
scores = [r['score'] for r in results if r['score'] is not None]

# Get top 10 images
sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
top_10 = sorted_results[:10]
```

### Use as module:
```python
from visualize_holdout import score_and_visualize_holdout, load_holdout_images
from model import load_model

model = load_model('results/best_model.pt', 'efficientnet_b0')
holdout_images = load_holdout_images('holdout.txt', 'images/')
results = score_and_visualize_holdout(model, holdout_images, 'output/')
```
