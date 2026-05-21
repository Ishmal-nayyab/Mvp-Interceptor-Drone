# Autonomous Drone Interceptor MVP 🛸🎯

An ultra-lightweight, high-performance Minimum Viable Product (MVP) for tracking rogue drones and predicting interception vectors. Built purely using **OpenCV** and **Kalman Filtering**, eliminating the overhead of heavy deep learning frameworks.

## ✨ Core Features
* **Motion-Based Detection:** Uses MOG2 Background Subtraction with morphological transformations to detect dynamic objects without PyTorch/YOLO overhead.
* **State Estimation:** Implements a 4D Linear Kalman Filter (State vector: $[x, y, v_x, v_y]$) to calculate continuous velocities.
* **Trajectory Prediction:** Calculates dynamic interception coordinates based on user-defined lookahead frames ($seconds \times FPS$).
* **Visual Debugging:** Renders track histories, scaled velocity vectors, and future intercept markers in real-time.

## 🚀 Setup & Execution

### 1. Installation
```bash
git clone [https://github.com/YOUR_USERNAME/drone-interceptor-mvp.git](https://github.com/YOUR_USERNAME/drone-interceptor-mvp.git)
cd drone-interceptor-mvp
pip install -r requirements.txt
