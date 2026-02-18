# Motor Alertness Detection (DARPA Triage Challenge)

![UGV with LiDAR and Depth Camera](https://github.com/user-attachments/assets/0e01f5f8-78ac-445b-af6d-2dd32073da7d)

**DTU with Blickfeld LiDAR at DARPA Triage Challenge**  
*UGV platform with a 3D LiDAR sensor and depth camera used for patient monitoring.*

This system detects patient motor alertness (e.g., leg twitches) non-invasively using:
- LiDAR point clouds
- Depth camera streams
- YOLOv8 segmentation
- Temporal feature extraction
- Deep learning classification

The pipeline combines sensor fusion, tracking, and neural inference to classify alertness in real-time triage scenarios.

---

## 📦 Setup & Requirements

### 1. Clone the Repository
```bash
git clone https://github.com/SaarthakG-Dtu/DARPA_TRIAGE_CHALLENGE.git
cd DARPA_TRIAGE_CHALLENGE

2. Install Dependencies

    Python 3.8+ is required

pip install -r requirements.txt

Key Packages

    PyTorch

    OpenCV

    NumPy

    Open3D

    ROS (for depth bag processing)

Hardware Requirements

    Linux PC (Ubuntu 20.04 / 22.04 recommended)

    NVIDIA GPU (Jetson / RTX preferred for real-time inference)

    3D LiDAR (Blickfeld / Ouster)

    Depth Camera (Kinect / Intel RealSense)

    ROS Noetic (for .bag file processing)

🚀 LiDAR Pipeline (Lidar_Motor_alertness)
1. Preprocessing

pcd_image_extract.py converts raw LiDAR .pcd files into 2D depth images by projecting 3D scans onto an image plane.
This generates depth frames representing the surrounding geometry.
2. Segmentation & Cropping

    YOLOv8 instance segmentation is applied on the RGB camera feed.

    The human mask centroid is computed.

    The centroid is converted into an angular bearing.

    LiDAR points within a narrow azimuth window are selected.

    Clustering is applied to isolate the casualty.

    The nearest cluster centroid is treated as the person’s 3D position.

3. Motion Tracking

run_alltracker.py tracks movement frame-by-frame using:

    Clustering

    Optical flow techniques
    This allows detection of subtle limb movements like twitches.

4. Feature Extraction

final_features.py computes temporal motion features:

    Displacements

    Velocities

    Variance

    Motion history (stacked frame features)

Feature stacks over short time windows form the model input tensor.
5. Classification

inference_final.py loads:

    Feature stacks

    best_model_checkpoint.pth

The neural network classifies movements such as:

    Left leg twitch

    Right leg twitch

    No motion

Run Full LiDAR Pipeline

cd Lidar_Motor_alertness
bash Lidar_inference.sh

This script automates:

Preprocess → Segment → Track → Feature Extraction → Inference

🎥 Depth Camera Pipeline (Depthcam_motor_alertness)
1. Bag Processing

process_bag.py extracts:

    RGB frames

    Depth frames
    from ROS .bag files.

2. Motion Tracking

run_alltracker.py tracks the person using:

    Pose estimation

    Optical flow on depth frames

3. Feature Extraction

final_features.py computes temporal features including:

    Pixel displacement

    Joint velocities

    Motion variance

Frames are stacked over time to generate feature vectors.
4. Classification

inference_final.py applies the trained model to classify motor alertness using depth-based feature stacks.
Run Full Depth Pipeline

cd Depthcam_motor_alertness
bash motor_alertness.sh

Pipeline flow:

Bag Processing → Tracking → Feature Extraction → Inference

🧠 Features & Model Training
Feature Engineering

    Stacked temporal feature tensors

    Captures motion dynamics across consecutive frames

    Robust clustering via:

        cluster.py

        cluster_var.py

These utilities group similar motion patterns and compute variance for improved robustness.
Model Architecture

    Supervised Neural Network (1D-CNN based)

    Implemented in PyTorch

    Trained on temporal motion feature stacks

Training notebook:

Training_logic/Twitching.ipynb

Training Workflow

    Load .npy feature stacks

    Load labels.npy

    Train model

    Save best weights as:

best_model_checkpoint.pth

Dataset includes:

    Twitch scenarios

    Non-twitch scenarios
    allowing the model to learn subtle movement distinctions.

🧪 Running the System
LiDAR Flow

cd Lidar_Motor_alertness
bash Lidar_inference.sh

Outputs example:

Right leg twitch detected at t = 12.3s

Depth Camera Flow

cd Depthcam_motor_alertness
bash motor_alertness.sh

Processes ROS bag files and prints detected movements.
Retraining (Optional)

    Open:

Training_logic/Twitching.ipynb

    Prepare:

        labels.npy

        Feature .npy files

    Train and replace:

best_model_checkpoint.pth

⚙️ System Notes
Environment

    Tested on Ubuntu 22.04

    ROS Noetic (for depth pipeline)

    Python 3.8+

    NVIDIA GPU recommended for real-time inference

Sensor Calibration

    Camera intrinsics & extrinsics must be calibrated

    Required for accurate centroid-to-angle conversion

    Ensures correct LiDAR cropping using YOLO masks

Vision–LiDAR Mapping

For each segmented human:

    Convert 2D mask centroid → angular bearing

    Select LiDAR points within that azimuth window

    Apply clustering

    Extract precise 3D person location

This fusion of YOLO segmentation and LiDAR spatial filtering is the core innovation of the pipeline.
