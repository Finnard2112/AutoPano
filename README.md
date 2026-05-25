# AutoPano

An automatic panorama stitching pipeline that detects keypoints, computes homographies, and blends overlapping images into a seamless panorama.

## Overview

AutoPano takes a set of overlapping images and automatically stitches them into a wide-angle panorama. It uses classical computer vision techniques — feature detection, descriptor matching, RANSAC for robust homography estimation, and image blending — to produce a seamless result.

## Features

- Keypoint detection using Harris Corner Detector or similar
- Feature descriptor computation and matching
- Robust homography estimation with RANSAC
- Image warping and blending for seamless panorama output
- Includes a written report (TeX/PDF) documenting the approach

## Getting Started

### Prerequisites

- Python 3.x
- OpenCV:
  ```bash
  pip install opencv-python numpy
  ```

### Run

```bash
git clone https://github.com/Finnard2112/AutoPano.git
cd AutoPano
python pano.py --images <path_to_image_folder>
```

## Tech Stack

- **Language:** Python / TeX (report)
- **Libraries:** OpenCV, NumPy
- **Domain:** Computer Vision

## Concepts Covered

- Harris Corner Detection
- Feature matching and descriptor computation
- RANSAC homography estimation
- Image warping (perspective transform)
- Alpha blending and image compositing
