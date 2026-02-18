# Motor Alertness Detection (DARPA Triage Challenge)

【<img width="1000" height="750" alt="image" src="https://github.com/user-attachments/assets/74612e8c-e5f6-4049-8448-962abed3484d" />
】 *Figure: UGV platform equipped with 3D LiDAR and depth camera used to capture human motion.* This repository provides an end-to-end pipeline for detecting patient movements (like leg twitches) using non-contact sensors. We use **LiDAR point clouds** and **depth-camera video** streams from a mobile robot to assess motor alertness. By stacking temporal features and classifying with a trained model, the system flags subtle motions while preserving privacy【1†L126-L133】【22†L199-L202】.

## Setup & Installation

- **Clone the repository:**  
  ```bash
  git clone https://github.com/SaarthakG-Dtu/DARPA_TRIAGE_CHALLENGE.git
  cd DARPA_TRIAGE_CHALLENGE
  ```  
- **Install dependencies:**  
  The project requires Python 3.x. Install Python packages:  
  ```bash
  pip install -r requirements.txt
  ```  
  *Requirements include PyTorch, OpenCV, NumPy, Open3D, etc.*  
- **Environment:** We recommend Ubuntu 20.04 with ROS Noetic (for bag file processing) and an NVIDIA GPU (e.g., Jetson or RTX) for acceleration【30†L165-L170】.  
- **Hardware:** A mobile robot (UGV) carrying a 3D LiDAR sensor (e.g., Blickfeld or Ouster) and a depth camera (e.g., Kinect/RealSense). For example, DTU’s UGV used a Blickfeld LiDAR and cameras for triage【1†L126-L133】. Ensure proper sensor calibration.  

## Data Acquisition

Sensors record two data streams:

- **LiDAR Stream (3D point clouds):** The LiDAR continually scans the scene, producing point cloud files (`.pcd`). This data captures the geometry of the environment and the person.  
- **Depth-Camera Stream:** A ROS bag or video file contains synchronized color+depth frames. The depth images provide per-pixel distance (used for skeleton tracking).  

Each pipeline folder assumes the raw data is placed in a subdirectory:
- Put LiDAR scans into `Lidar_Motor_alertness/input/pcd_files/`.  
- Put depth bag files into `Depthcam_motor_alertness/input/`.  

The code then converts and processes these raw inputs.

## LiDAR Pipeline (Lidar_Motor_alertness)

This pipeline processes point clouds end-to-end:

1. **Preprocessing:** `Lidar_Motor_alertness/pcd_image_extract.py` reads `.pcd` files and projects them into 2D depth images (e.g. top-down view of the scene).  
2. **Tracking:** `Lidar_Motor_alertness/run_alltracker.py` detects and tracks the person’s body or limbs in each depth-image frame. It applies clustering/flow algorithms on the images.  
3. **Feature Extraction:** `Lidar_Motor_alertness/final_features.py` computes temporal features from the tracked motion (such as displacement, velocity, and variance over time windows). These features are stacked into a NumPy array.  
4. **Inference:** `Lidar_Motor_alertness/inference_final.py` loads the stacked features and runs the pre-trained model (`best_model_checkpoint.pth`) to predict motion classes (e.g. left-leg-twitch, right-leg-twitch, no-motion).

A helper shell script is provided:  
- `Lidar_Motor_alertness/Lidar_inference.sh` (or similar) runs the above steps sequentially. For example:  
  ```bash
  cd Lidar_Motor_alertness
  bash Lidar_inference.sh
  ```  
  This automates preprocessing, feature extraction, and inference with one command.

## Depth-Camera Pipeline (Depthcam_motor_alertness)

Parallel to LiDAR, this pipeline handles depth video:

1. **Bag Processing:** `Depthcam_motor_alertness/process_bag.py` extracts frames from a ROS bag. It splits the bag into depth and (optional) color image sequences.  
2. **Tracking:** `Depthcam_motor_alertness/run_alltracker.py` runs a pose or motion tracker on each frame sequence to follow the person.  
3. **Feature Extraction:** `Depthcam_motor_alertness/final_features.py` computes the same kinds of features (e.g. temporal depth changes) from the tracked points.  
4. **Inference:** `Depthcam_motor_alertness/inference_final.py` feeds these features into the model for classification.

A shell script `Depthcam_motor_alertness/motor_alertness.sh` is provided to launch this flow:  
  ```bash
  cd Depthcam_motor_alertness
  bash motor_alertness.sh
  ```

## Feature Engineering

Both pipelines produce stacked feature arrays (`.npy`). Key points:

- We use **temporal stacking:** concatenate a sequence of successive features into one input tensor. This gives the model context over time to spot brief twitches.  
- Feature scripts (`cluster.py`, `cluster_var.py`) (in `Training_logic/`) perform clustering on training data and compute motion variances. These help create robust feature sets.  
- Typical features include: depth differences between frames, limb joint velocities, and variance of motion over a time window. Stacked temporal features are effective in capturing subtle movements【30†L123-L130】.  

The stacked feature datasets are included (`stacked_features.zip`) for training.

## Model Training (Training_logic)

The `Training_logic/` folder contains the training setup:

- **Data Preparation:** `input_dataset.py` loads the `.npy` feature stacks and corresponding labels. `labels.py` maps integer classes to motion descriptions.  
- **Training Notebook:** `Twitching.ipynb` orchestrates training. It shows steps to load data, apply any preprocessing, define the neural network, and train. The model is a supervised classifier (e.g. 1D CNN) on temporal feature stacks.  
- **Model Checkpoint:** After training, the best model weights are saved as `best_model_checkpoint.pth`. This file is used by `inference_final.py` in both pipelines.  

The training script reports accuracy on detecting twitch events. In line with DARPA Challenge requirements, the focus is on high recall for critical movements【28†L113-L122】.

## Running the System

1. **LiDAR Pipeline Example:**  
   ```bash
   cd Lidar_Motor_alertness
   bash Lidar_inference.sh
   ```  
   This processes all point clouds in the input folder and outputs detected alerts.

2. **Depth Pipeline Example:**  
   ```bash
   cd Depthcam_motor_alertness
   bash motor_alertness.sh
   ```  
   This processes the ROS bag files and outputs alerts from depth data.

3. **Re-Training Model (Optional):**  
   - Open `Training_logic/Twitching.ipynb` in Jupyter.  
   - Follow the cells to load feature data and train. A new `best_model_checkpoint.pth` will be saved upon completion.

**Hardware/Software Requirements:** A Linux PC with GPU is recommended. Install ROS (for processing bag files) and ensure Python 3.8+ with libraries in `requirements.txt`. For example, DARPA teams used NVIDIA Jetson and RTX GPUs for similar workloads【30†L165-L170】.

【45†embed_image】 *Figure: Autonomous UGVs (Spot robots) assisting a medic during the DARPA Triage Challenge【2†L84-L92】.* This system is designed to operate in such scenarios, providing rapid patient assessment. 

## Visualization & Outputs

- **Visualization:** Use `visualize_final.py` (in the LiDAR folder) to plot motion paths and highlight detections. This helps verify the system.  
- **Results:** The scripts print or save a list of detections (e.g. “Left-leg Twitch at 12.3s”). These can be sent to a UI or first-responder device.  

All code comments and the above structure are derived from the repository files. The approach follows published triage systems that emphasize rapid casualty detection and non-contact sensing【1†L126-L133】【2†L84-L92】.

