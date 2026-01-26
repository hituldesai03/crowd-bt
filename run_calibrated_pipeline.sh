#!/bin/bash
# Complete pipeline: Fit calibrator and visualize with 0-100 scores

echo "=========================================="
echo "Calibrated Score Pipeline (0-100)"
echo "=========================================="
echo ""

source quality/bin/activate

# Step 1: Fit calibrator on training images
echo "Step 1: Fitting score calibrator..."
echo "------------------------------------"
python calibrate_scores.py \
  --checkpoint results_golden_overfit/best_model.pt \
  --backbone efficientnet_b0 \
  --image-dir /home/hitul/Desktop/quality-comparison-toolkit/data/gold_ranking_patches \
  --method minmax \
  --lower-percentile 5 \
  --upper-percentile 95 \
  --output score_calibrator_gold.pkl \
  --max-images 1000

if [ $? -ne 0 ]; then
    echo "Error: Calibrator fitting failed!"
    exit 1
fi

echo ""
echo "Step 2: Visualizing holdout set with calibrated scores..."
echo "-----------------------------------------------------------"
python visualize_holdout_calibrated.py \
  --checkpoint results_golden_overfit/best_model.pt \
  --calibrator score_calibrator.pkl \
  --backbone efficientnet_b4 \
  --output-dir holdout_visualizations_calibrated_gold \
  --max-grid-images 184

if [ $? -ne 0 ]; then
    echo "Error: Visualization failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Outputs:"
echo "  - Calibrator: score_calibrator.pkl"
echo "  - Annotated images: holdout_visualizations_calibrated/"
echo "  - Summary grid: holdout_visualizations_calibrated/holdout_summary.png"
echo "  - Distribution: holdout_visualizations_calibrated/score_distribution.png"
echo "  - Scores JSON: holdout_visualizations_calibrated/scores_calibrated.json"
echo ""
echo "All scores are now in 0-100 range!"
