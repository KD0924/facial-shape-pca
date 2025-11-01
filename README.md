# Facial Shape Analysis using PCA

This repository performs **statistical shape analysis** on 2D facial landmarks.
It aligns shapes using a **similarity transform** and applies **PCA** to
visualize deformation modes, compare **neutral vs smiling**, and evaluate a
simple classifier.

## Pipeline
1. Load landmarks from `data/face_points45.txt` (each row: x0,y0,x1,y1,...)
2. Align all shapes (`tc_shape.sim_transform`)
3. Apply PCA (`tc_shape.apply_pca`)
4. Visualize:
   - Mean shape
   - ±3σ for the top 3 PCA modes
   - Mean neutral vs mean smiling
   - Histogram of projections on the “smile direction”

## Structure
