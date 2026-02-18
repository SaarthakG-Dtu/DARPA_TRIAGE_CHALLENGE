
---

## 🧠 System Architecture

Sensors (LiDAR + Depth + RGB)
↓
YOLOv8 Segmentation (Human Mask)
↓
LiDAR Cropping via Angular Mapping
↓
Motion Tracking (Temporal)
↓
Feature Extraction (Stacked Temporal Features)
↓
Deep Learning Inference (Motor Alertness Classification)


---

## 📂 Repository Structure

DARPA_TRIAGE_CHALLENGE/
│
├── Lidar_Motor_alertness/
│ ├── pcd_image_extract.py
│ ├── run_alltracker.py
│ ├── final_features.py
│ ├── inference_final.py
│ └── Lidar_inference.sh
│
├── Depthcam_motor_alertness/
│ ├── process_bag.py
│ ├── run_alltracker.py
│ ├── final_features.py
│ ├── inference_final.py
│ └── motor_alertness.sh
│
├── Training_logic/
│ └── Twitching.ipynb
│
├── best_model_checkpoint.pth
└── requirements.txt


---

## ⚙️ Setup

### 1. Clone the Repository
```bash
git clone https://github.com/SaarthakG-Dtu/DARPA_TRIAGE_CHALLENGE.git
cd DARPA_TRIAGE_CHALLENGE

2. Install Dependencies

pip install -r requirements.txt

Requirements

    Python 3.8+

    Ubuntu 20.04 / 22.04 (Recommended)

    NVIDIA GPU (Jetson / RTX for real-time inference)

    ROS Noetic (for depth bag processing)

Key Libraries:

    PyTorch

    OpenCV

    NumPy

    Open3D

    ROS

🚀 LiDAR Motor Alertness Pipeline

Directory: Lidar_Motor_alertness/
Pipeline Stages
1. Preprocessing

pcd_image_extract.py converts raw .pcd LiDAR scans into 2D depth images by projecting 3D point clouds onto an image plane.
2. Human Segmentation & Cropping

    YOLOv8 instance segmentation isolates the human subject

    Mask centroid is converted to angular bearing

    LiDAR points within the azimuth window are selected

    Clustering extracts the nearest human point cluster

3. Motion Tracking

run_alltracker.py tracks temporal movement using clustering and optical flow across frames.
4. Feature Extraction

final_features.py computes stacked temporal features:

    Displacement

    Velocity

    Variance

    Motion history

5. Classification

inference_final.py loads feature stacks and applies the trained PyTorch model (best_model_checkpoint.pth) to classify:

    Left Leg Twitch

    Right Leg Twitch

    No Motion

Run LiDAR Pipeline

cd Lidar_Motor_alertness
bash Lidar_inference.sh

🎥 Depth Camera Motor Alertness Pipeline

Directory: Depthcam_motor_alertness/
Pipeline Stages
1. ROS Bag Processing

process_bag.py extracts RGB and depth frames from .bag files.
2. Motion Tracking

run_alltracker.py performs tracking using pose estimation and optical flow on depth frames.
3. Feature Extraction

Temporal motion features are computed and stacked over time to form model input tensors.
4. Inference

inference_final.py applies the trained model for alertness classification using depth-based features.
Run Depth Pipeline

cd Depthcam_motor_alertness
bash motor_alertness.sh

🧪 Model & Training
Model

    Architecture: 1D CNN (PyTorch)

    Input: Stacked temporal motion features

    Output: Motor alertness classification

Training Notebook

Training_logic/Twitching.ipynb

Training Process

    Load feature .npy stacks

    Load labels.npy

    Train model on twitch vs non-twitch data

    Save best weights as:

best_model_checkpoint.pth

Clustering utilities (cluster.py, cluster_var.py) improve robustness by grouping motion patterns and computing variance statistics.
🔬 Sensor Fusion Methodology

    YOLOv8 segmentation provides human mask in RGB frame

    Mask centroid → angular bearing conversion

    LiDAR points filtered by azimuth alignment

    Clustered points yield precise 3D human localization

    Temporal tracking extracts motion signatures for classification

This tight integration of vision and LiDAR enables accurate non-contact motor assessment.
🖥️ Environment Notes

    Tested on Ubuntu 22.04

    ROS Noetic required for depth bag processing

    GPU recommended for real-time inference

    Proper camera–LiDAR calibration is mandatory for accurate mapping
