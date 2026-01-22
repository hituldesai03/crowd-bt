# Score Calibration: 0-100 Range

## Why Calibration?

The trained model outputs **unbounded raw scores** (any real number) because:
1. The model uses a simple linear head without activation constraints
2. Training uses **Bradley-Terry ranking loss** which only cares about relative ordering, not absolute values

Raw scores are meaningful for **comparison** (higher = better) but don't have intuitive bounds.

**Calibration** maps raw scores to a **0-100 range** for easier interpretation.

## How It Works

### Calibration Methods

#### 1. Percentile-Based (Recommended) ✓
Maps score distribution percentiles to 0-100:
- **5th percentile → 0**
- **95th percentile → 100**
- Linear interpolation between

**Advantages:**
- Robust to outliers
- Uses full 0-100 range
- Based on actual data distribution

**When to use:** Default choice for most cases

#### 2. Min-Max Scaling
Maps observed min/max to 0-100:
- **Min score → 0**
- **Max score → 100**
- Linear scaling

**Advantages:**
- Simple and intuitive
- Preserves exact ordering

**When to use:** When you want strict bounds based on observed extremes

### Calibration Process

```
Raw Scores (training set):     [-23.5, ..., +29.8]
                                        ↓
                    Fit Calibrator (percentile method)
                                        ↓
Calibration Function:          5th percentile (-18.4) → 0
                               95th percentile (24.7) → 100
                                        ↓
Calibrated Scores:             [0, ..., 100]
```

## Usage

### Step 1: Fit Calibrator

Fit the calibrator on your training/validation images:

```bash
python calibrate_scores.py \
  --checkpoint results/best_model.pt \
  --backbone efficientnet_b0 \
  --image-dir /path/to/training/images \
  --method percentile \
  --output score_calibrator.pkl \
  --max-images 1000
```

**Output:** `score_calibrator.pkl` (reusable calibration function)

### Step 2: Use Calibrated Scores

#### Option A: Visualize Holdout Set

```bash
python visualize_holdout_calibrated.py \
  --checkpoint results/best_model.pt \
  --calibrator score_calibrator.pkl \
  --output-dir holdout_visualizations_calibrated
```

#### Option B: Score Single Image

```bash
python example_score_single_image.py image.png
```

Output:
```
Scoring image: image.png
------------------------------------------------------------
Raw Score:        12.3456
Calibrated Score: 72.45 / 100
Quality Rating:   Good Quality
```

#### Option C: Use in Python Code

```python
from model import load_model
from infer import score_single_image
from calibrate_scores import ScoreCalibrator

# Load model and calibrator
model = load_model('results/best_model.pt', 'efficientnet_b0')
calibrator = ScoreCalibrator.load('score_calibrator.pkl')

# Score image
raw_score = score_single_image(model, 'image.png')
calibrated_score = calibrator.transform(raw_score)

print(f"Quality Score: {calibrated_score:.1f}/100")
```

### Full Pipeline (One Command)

```bash
./run_calibrated_pipeline.sh
```

This will:
1. Fit calibrator on 1000 training images
2. Visualize all 184 holdout images with 0-100 scores
3. Generate summary statistics and plots

## Interpreting Scores

| Score Range | Quality Rating | Color |
|-------------|---------------|-------|
| 75-100      | Excellent     | 🟢 Green |
| 50-75       | Good          | 🟡 Yellow |
| 25-50       | Fair          | 🟠 Orange |
| 0-25        | Poor          | 🔴 Red |

## Results from Holdout Set

After calibration on 1000 training images:

```
Calibrated Score Statistics (0-100)
====================================
Total images: 184
Mean score:   51.93
Median score: 52.39
Std score:    32.34
Min score:    0.00
Max score:    100.00

Quality Distribution:
  Excellent (75-100): 56 (30.4%)
  Good (50-75):       37 (20.1%)
  Fair (25-50):       40 (21.7%)
  Poor (0-25):        51 (27.7%)
```

## Files Generated

```
crowd-bt/
├── score_calibrator.pkl                              # Fitted calibrator (reusable)
├── holdout_visualizations_calibrated/                # Annotated images with 0-100 scores
│   ├── img_00005_patch_8.png                        # Each patch with score overlay
│   ├── ...
│   ├── holdout_summary.png                          # Grid of all patches ranked
│   ├── score_distribution.png                       # Histogram and box plot
│   └── scores_calibrated.json                       # All scores in JSON format
└── calibration_output.log                           # Full pipeline log
```

## Customization

### Change Percentile Range

```bash
python calibrate_scores.py \
  --lower-percentile 2.5 \   # More aggressive lower bound
  --upper-percentile 97.5    # More aggressive upper bound
```

### Use Min-Max Instead

```bash
python calibrate_scores.py \
  --method minmax
```

### Fit on Different Dataset

```bash
python calibrate_scores.py \
  --image-dir /path/to/different/images \
  --max-images 5000
```

## Technical Details

### Calibrator Class

```python
class ScoreCalibrator:
    """Maps raw scores to 0-100 range."""

    def fit(self, raw_scores: List[float]):
        """Learn mapping from observed score distribution."""
        # Compute percentiles from training data

    def transform(self, raw_score: float) -> float:
        """Transform single score to 0-100."""
        # Apply learned mapping

    def save(self, path: str):
        """Save for reuse."""

    @staticmethod
    def load(path: str):
        """Load fitted calibrator."""
```

### Why Percentile-Based Works Best

1. **Robustness**: Not affected by extreme outliers
2. **Distribution-aware**: Adapts to actual score distribution
3. **Full range utilization**: Ensures 0 and 100 are meaningful
4. **Consistent interpretation**: 50 = median quality in your dataset

### Calibration vs. Retraining

**Calibration (what we did):**
- ✅ No retraining needed
- ✅ Fast (seconds)
- ✅ Post-processing only
- ⚠️ Relative to training distribution

**Retraining with constrained output:**
- Modify model to output `sigmoid(x) * 100`
- Requires full retraining
- Changes model architecture
- May affect ranking performance

## FAQ

**Q: Do calibrated scores change model predictions?**
A: No! Calibration is a monotonic transformation. Relative ordering is preserved. If image A scores higher than B before calibration, it still does after.

**Q: Can I use one calibrator for different models?**
A: No. Each model needs its own calibrator fitted to its score distribution.

**Q: What if new images score outside 0-100?**
A: The calibrator clips scores to [0, 100]. If a new image scores below the 5th percentile, it gets 0. Above 95th percentile gets 100.

**Q: Should I refit calibrator after retraining?**
A: Yes. Score distributions can change with retraining, so refit the calibrator.

## Advanced: Batch Scoring with Calibration

```python
import json
from model import load_model
from infer import score_batch
from calibrate_scores import ScoreCalibrator

model = load_model('results/best_model.pt', 'efficientnet_b0')
calibrator = ScoreCalibrator.load('score_calibrator.pkl')

# Score many images
image_paths = ['img1.png', 'img2.png', ...]
results = score_batch(model, image_paths)

# Calibrate all scores
for result in results:
    result['calibrated_score'] = calibrator.transform(result['score'])

# Save
with open('batch_scores_calibrated.json', 'w') as f:
    json.dump(results, f, indent=2)
```
