# Motor Alertness Detection (DARPA Triage Challenge)

<img width="1000" height="750" alt="UGV LiDAR Setup" src="https://github.com/user-attachments/assets/0e01f5f8-78ac-445b-af6d-2dd32073da7d" />

**DTU with Blickfeld LiDAR at DARPA Triage Challenge**  
UGV platform with a 3D LiDAR sensor and depth camera used for non-invasive patient motor alertness detection.

This system detects subtle patient movements (e.g., leg twitches) using:
- LiDAR Point Clouds  
- Depth Camera Streams  
- YOLOv8 Segmentation  
- AllTracker Motion Tracking  
- Temporal Feature Extraction  
- Deep Learning Classification  

The pipeline integrates sensor fusion, segmentation, tracking, feature engineering, and neural inference for autonomous triage scenarios.

---

# 🛠️ Setup & Requirements

## 1. Clone the Repository
```bash
git clone https://github.com/SaarthakG-Dtu/DARPA_TRIAGE_CHALLENGE.git
cd DARPA_TRIAGE_CHALLENGE

2. Install Dependencies

Python 3.8+ is required.
pip install -r requirements.txt

Key Libraries

PyTorch

OpenCV

NumPy

Open3D

ROS (for depth bag processing)

Ultralytics YOLOv8

Hardware Requirements

Linux PC (Ubuntu 20.04 / 22.04 recommended)

NVIDIA GPU (Jetson / RTX for real-time inference)

3D LiDAR (Blickfeld / Ouster)

Depth Camera (Intel RealSense / Kinect)

ROS Noetic (for processing .bag files)

