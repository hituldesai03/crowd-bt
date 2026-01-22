#!/bin/bash
# Quick script to run holdout visualization

echo "Running Holdout Visualization..."
echo "================================"

source quality/bin/activate

python visualize_holdout.py \
  --checkpoint results/best_model.pt \
  --backbone efficientnet_b0 \
  --input-size 448 \
  --device cuda \
  --output-dir holdout_visualizations \
  --max-grid-images 184 \
  --font-size 40

echo ""
echo "Done! Check the output:"
echo "  - Annotated images: holdout_visualizations/"
echo "  - Summary grid: holdout_visualizations/holdout_summary.png"
echo "  - Distribution: holdout_visualizations/score_distribution.png"
echo "  - Scores JSON: holdout_visualizations/scores.json"
