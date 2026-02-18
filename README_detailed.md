# Motor Alertness Detection using LiDAR & Depth Camera (AllTracker Pipeline)

This repository implements a dual-pipeline motor alertness detection
system designed for mass casualty and triage scenarios. The system
detects subtle motor activity (twitch / non-twitch) of a casualty using
LiDAR point clouds, depth-camera streams, dense pixel tracking
(AllTracker), and temporal feature-based deep learning inference.

The pipeline is designed to be practical, modular, and reproducible,
with shell scripts triggering the complete workflow end-to-end.

Tested Environment: Ubuntu 22.04 + GPU + CUDA

------------------------------------------------------------------------

## System Overview

The core objective is to automatically monitor a casualty's motor
response in a cluttered scene. Instead of tracking the entire frame, the
pipeline first identifies and focuses on the **nearest / priority
person** and then performs motion analysis only on that subject. This
significantly improves robustness in multi-person or noisy environments.

High-level flow: Nearest Person Selection → Segmentation → Dense
Tracking → Feature Extraction → Classification

Sensors Supported: - Blickfeld LiDAR (PCD based depth) - Depth Camera
(RGB-D / bag streams) - RGB input for segmentation guidance

------------------------------------------------------------------------

## Setup & Requirements

### Clone the Repository

``` bash
git clone <repo-url>
cd Final_twitch_assess
```

### Install Dependencies

Python 3.9+ recommended.

``` bash
pip install -r requirements.txt
```

Key packages: - PyTorch - OpenCV - NumPy - Open3D (PCD processing) -
Ultralytics YOLOv8 (segmentation) - SciPy - Matplotlib

### Hardware

-   Ubuntu 22.04 Linux machine
-   NVIDIA GPU (recommended for tracking & inference)
-   3D LiDAR (e.g., Blickfeld)
-   Depth Camera (e.g., RealSense or RGB-D sensor)
-   (Optional) ROS bag support for depth pipeline

------------------------------------------------------------------------

## Nearest Person Focusing (Priority Casualty Selection)

A critical design choice in this system is that the pipeline does not
track the full scene. Instead, it automatically selects the nearest
casualty before tracking begins.

This is done by: - Running YOLO segmentation to detect human regions -
Estimating depth/position of detected subjects - For LiDAR: selecting
the nearest point cloud cluster aligned with the detected mask - For
Depth: selecting the closest depth region within the segmentation mask

Only this nearest ROI is passed to the tracking stage.\
This reduces background motion, prevents false twitch signals, and
ensures stable single-casualty monitoring.

------------------------------------------------------------------------

## LiDAR Pipeline (Lidar_Motor_alertness)

Run:

``` bash
cd Lidar_Motor_alertness
bash Lidar_inference.sh
```

This shell script executes the full LiDAR-based motor alertness pipeline
in sequence.

### 1. PCD Preprocessing

Script: `pcd_image_extract.py`\
- Loads raw `.pcd` scans from LiDAR - Projects 3D point cloud onto
image/depth plane - Generates frame-wise depth representations - Filters
noise and outliers for stable motion analysis

### 2. Depth Frame Preparation

Script: `depth_final.py`\
- Aligns projected depth frames - Normalizes and formats sequences -
Prepares inputs for tracking and feature stages

### 3. Segmentation & Nearest Person Cropping

-   YOLOv8 segmentation isolates the human subject
-   Mask centroid and spatial alignment guide LiDAR cropping
-   Nearest point cluster corresponding to the person is selected
-   Background point cloud is discarded

### 4. Dense Motion Tracking (AllTracker)

Script: `run_alltracker.py`\
- Tracks selected pixels across temporal frames - Uses query frame
initialization and pixel subsampling - Produces trajectory tensor of
shape (T, N, 2) - Focuses only on the cropped nearest-person region

### 5. Temporal Feature Extraction

Script: `final_features.py`\
Extracted features include: - Pixel displacement across frames - Motion
magnitude and variance - Short-term temporal stability -
Trajectory-based motion patterns

Feature stacks capture motion history instead of single-frame signals.

### 6. Inference

Script: `inference_final.py`\
- Loads `best_model_checkpoint.pth` - Consumes stacked temporal
features - Outputs motor alertness classification (twitch / non-twitch)

Optional visualization: `visualize_final.py` for qualitative motion
inspection.

------------------------------------------------------------------------

## Depth Camera Pipeline (Depthcam_motor_alertness)

Run:

``` bash
cd Depthcam_motor_alertness
bash motor_alertness.sh
```

This pipeline processes RGB-D or bag-based depth sequences.

### 1. Bag / Frame Processing

Script: `process_bag.py`\
- Reads ROS bag or depth sequences - Extracts synchronized RGB and depth
frames - Aligns depth with image coordinates

### 2. Nearest Person Isolation

-   YOLO segmentation masks the casualty
-   Closest depth region is selected as priority subject
-   Non-relevant regions are ignored before tracking

### 3. Dense Pixel Tracking

Script: `run_alltracker.py`\
- Applies AllTracker on segmented depth/RGB frames - Tracks dense pixel
trajectories over time - Generates motion trajectories focused on the
subject only

### 4. Feature Generation

Script: `final_features.py`\
- Computes displacement and motion consistency - Builds stacked temporal
feature tensors - Encodes subtle micro-movements and twitch patterns

### 5. Inference

Script: `inference_final.py`\
- Uses the trained checkpoint - Classifies motor activity based on
temporal motion features - Same model is shared with LiDAR pipeline for
consistency

------------------------------------------------------------------------

## Model & Training

Training resources are located in: `Training_logic/`

Main notebook: `Twitching.ipynb` - Loads stacked `.npy` feature
datasets - Applies clustering and variance preprocessing - Trains
temporal neural model (PyTorch) - Saves best weights as
`best_model_checkpoint.pth`

Supporting modules: - `cluster.py` -- motion clustering -
`cluster_var.py` -- variance-based preprocessing - `input_dataset.py` --
dataset loader - `labels.py` -- label handling

The model learns temporal motion patterns distinguishing twitch from
non-twitch sequences.

------------------------------------------------------------------------

## Feature Design

Each model input is a stacked temporal tensor derived from tracked
trajectories: - Frame-wise pixel displacement - Motion magnitude
evolution - Temporal variance - Short window motion history

This temporal stacking is essential for detecting low-amplitude twitch
movements that are not visible in single frames.

------------------------------------------------------------------------

## Running the System

### LiDAR Flow

``` bash
cd Lidar_Motor_alertness
bash Lidar_inference.sh
```

### Depth Camera Flow

``` bash
cd Depthcam_motor_alertness
bash motor_alertness.sh
```

Both scripts automate: Preprocessing → Nearest Person Selection →
Tracking → Feature Extraction → Inference

------------------------------------------------------------------------

## System Notes

-   Designed for single priority casualty monitoring
-   GPU strongly recommended for dense tracking and inference
-   Accurate sensor calibration (camera--LiDAR alignment) improves
    performance
-   Pretrained YOLOv8 segmentation is used (not custom trained)
-   Dense pixel tracking (AllTracker) is computationally intensive but
    enables fine motion detection

This repository is structured for research-grade triage, robotics
perception, and non-contact motor monitoring applications.
