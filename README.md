# Computer Vision: Algorithms & 3D Reconstruction

[![GitHub Repo](https://img.shields.io/badge/GitHub-sekikoudesu--source%2Fcomputer__vision-black?logo=github)](https://github.com/sekikoudesu-source/computer_vision)
[![Language](https://img.shields.io/badge/Language-Python-blue.svg)](https://www.python.org/)
[![Library](https://img.shields.io/badge/Library-OpenCV-green.svg)](https://opencv.org/)

## 📖 Overview
This repository contains core algorithm implementations and theoretical explorations in Computer Vision, completed during my graduate studies at the **Graduate School of Information Science and Electrical Engineering, Kyushu University**.

The project primarily focuses on **Camera Geometry**, **Multiple View Geometry**, and **3D Reconstruction**. Through mathematical derivations and Python/OpenCV coding practices, this repository covers classic computer vision pipelines—from low-level image feature extraction to high-level Structure from Motion (SfM) and Visual Odometry.

## ✨ Core Research Topics

### 1. Camera Geometry & Calibration
*   **Camera Models**: Pinhole Camera Model and Intrinsic/Extrinsic parameter matrix mapping.
*   **Self-Calibration**: Camera self-calibration theory based on the Kruppa Equation.
*   **Projection Models**: Analysis of linear models including Weak-perspective and Orthogonal projection.

### 2. Epipolar Geometry
*   **Matrix Computation**: Theory and calculation of the Fundamental Matrix and Essential Matrix.
*   **Algorithm Implementation**: The classic 8-point Algorithm and depth estimation using Epipolar Plane Images (EPI).

### 3. Feature Detection & Tracking
*   **Feature Extraction**: OpenCV-based implementations of ORB, SIFT, and Harris Corner detection.
*   **Feature Tracking**: Continuous video frame tracking using the KLT (Kanade-Lucas-Tomasi) Optical Flow method.

### 4. 3D Reconstruction & Structure from Motion (SfM)
*   **Factorization Method**: The Tomasi-Kanade Factorization Method for separating the Shape Matrix and Motion Matrix.
*   **Non-linear Optimization**: Bundle Adjustment minimizing reprojection error using the Levenberg-Marquardt algorithm.
*   **System Comparison**: Theoretical equivalence and engineering differences between V-SLAM (Simultaneous Localization and Mapping) and SfM.

## 🚀 Getting Started

### Prerequisites
It is recommended to use Python 3.8+ and install the following dependencies:
```bash
pip install -r requirements.txt
