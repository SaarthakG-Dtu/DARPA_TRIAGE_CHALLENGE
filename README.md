Copy this EXACTLY into README.md:

# Motor Alertness Detection (DARPA Triage Challenge)

<img width="1000" height="750" alt="UGV LiDAR Setup" src="https://github.com/user-attachments/assets/0e01f5f8-78ac-445b-af6d-2dd32073da7d" />

DTU UGV platform with a 3D LiDAR and depth camera for non-invasive patient motor alertness detection.  
The system uses LiDAR point clouds, depth streams, YOLOv8 segmentation, AllTracker motion tracking, and deep learning to detect subtle movements such as leg twitches in autonomous triage scenarios.

## Overview
This pipeline combines:
- LiDAR + Depth Camera sensor fusion  
- YOLO-based human segmentation  
- AllTracker temporal motion tracking  
- Temporal feature extraction  
- Neural network classification  

Key stages: data acquisition → segmentation → tracking → feature extraction → inference.

## Setup & Requirements

### Clone the Repository
```bash
git clone https://github.com/SaarthakG-Dtu/DARPA_TRIAGE_CHALLENGE.git
cd DARPA_TRIAGE_CHALLENGE

Install Dependencies
pip install -r requirements.txt


Requirements:

Python 3.8+

Ubuntu 20.04 / 22.04 (recommended)

NVIDIA GPU (Jetson / RTX for real-time inference)

ROS Noetic (for depth bag processing)

Key libraries: PyTorch, OpenCV, NumPy, Open3D, Ultralytics YOLO.

LiDAR Pipeline (Lidar_Motor_alertness)
Preprocessing

pcd_image_extract.py converts raw .pcd LiDAR scans into 2D depth images by projecting 3D point clouds onto an image plane, generating depth frames around the casualty.

YOLO Segmentation & LiDAR Cropping

YOLOv8 instance segmentation detects the human in RGB frames.
The mask centroid is converted to an angular bearing, and LiDAR points within the corresponding azimuth window are selected.
Clustering is then applied to isolate the human, and the nearest cluster centroid gives the 3D position.

Motion Tracking (AllTracker)

run_alltracker.py tracks movement across frames using clustering and optical flow to follow limb motion and detect subtle twitches.

Feature Extraction

final_features.py computes temporal features such as displacement, velocity, and variance.
Features are stacked over a short time window to capture motion history for model input.

Classification

inference_final.py loads stacked features and applies the trained model (best_model_checkpoint.pth) to classify:

Left leg twitch

Right leg twitch

No motion

Run LiDAR Pipeline
cd Lidar_Motor_alertness
bash Lidar_inference.sh


This script runs: preprocessing → YOLO segmentation → tracking → feature extraction → inference.

Depth Camera Pipeline (Depthcam_motor_alertness)
Bag Processing

process_bag.py reads ROS .bag files and extracts RGB and depth frames for motion analysis.

YOLO-Based Human Segmentation

YOLOv8 is used to detect and segment the human subject in RGB frames, ensuring tracking focuses only on the casualty and reduces background noise.

Motion Tracking (AllTracker)

run_alltracker.py performs temporal tracking on depth frames using optical flow, pose/pixel tracking, and motion clustering to generate tracked motion sequences.

Feature Extraction

final_features.py computes temporal motion features including pixel displacement, joint velocities, variance, and stacked temporal feature vectors representing motion dynamics.

Classification

inference_final.py uses the same trained neural network to classify motor alertness from depth-based feature stacks.

Run Depth Pipeline
cd Depthcam_motor_alertness
bash motor_alertness.sh


Pipeline: bag processing → YOLO segmentation → AllTracker → feature extraction → inference.

Features & Model Training
Feature Engineering

The model uses stacked temporal feature tensors capturing motion across consecutive frames.
Clustering utilities (cluster.py, cluster_var.py) group similar motion patterns and compute variance for robustness.

Model

Architecture: 1D CNN (PyTorch)

Input: stacked temporal feature tensors (.npy)

Output: motor alertness classification

Training is implemented in:
Training_logic/Twitching.ipynb

Training steps:

Load feature .npy stacks and labels.npy

Train the network on twitch vs non-twitch data

Save best weights as best_model_checkpoint.pth

Running the System
LiDAR Flow
cd Lidar_Motor_alertness
bash Lidar_inference.sh


Processes .pcd scans and outputs detected alerts (e.g., “Right leg twitch at t=12.3s”).

Depth Flow
cd Depthcam_motor_alertness
bash motor_alertness.sh


Processes ROS bag files and prints movement detections.

Retraining (Optional)

Open Training_logic/Twitching.ipynb, prepare feature .npy files and labels.npy, retrain, and replace best_model_checkpoint.pth for updated inference.

System Notes

Tested on Ubuntu 22.04 with ROS Noetic and Python 3.8+.
An NVIDIA GPU is recommended for real-time inference.

Camera intrinsics and extrinsics must be calibrated correctly for accurate centroid-to-angle conversion and LiDAR cropping.

For each segmented person, the 2D YOLO mask centroid is mapped to LiDAR azimuth, filtered point clouds are clustered, and the resulting centroid provides precise 3D localization. This tight integration of vision (YOLO) and LiDAR segmentation is central to the pipeline.

Application

Designed for autonomous robotic triage scenarios (DARPA Triage Challenge), where UGV platforms detect patient motor responses non-invasively using multimodal perception and deep learning inference.


### Why this fixes your spacing issue
- Only commands are in code blocks (not paragraphs)
- No giant boxed sections
- No excessive blank lines
- Proper GitHub-native Markdown flow
- No chat-style formatting artifacts

One last check:  
Make sure your file name is **exactly**:


README.md

and not `README.MD`, `readme.md`, or `README.txt` — GitHub formatting breaks otherwise.
