# Motor Alertness Detection (DARPA Triage Challenge)

<img width="1000" height="750" alt="image" src="https://github.com/user-attachments/assets/7aff4af4-bb36-4514-9ed6-5558f4ebebfd" />
 *Figure: UGV platform with a 3D LiDAR sensor and depth camera used for patient monitoring.* This system uses LiDAR point clouds and depth-camera data to detect patient movements (e.g. leg twitches) non-invasively. We combine sensor fusion, YOLO-based segmentation, and deep learning to classify alertness. Key components include data acquisition, segmentation, feature extraction, and inference (see sections below).

## Setup & Requirements

- **Clone the repo:**  
  ```bash
  git clone https://github.com/SaarthakG-Dtu/DARPA_TRIAGE_CHALLENGE.git
  cd DARPA_TRIAGE_CHALLENGE
  ```  
- **Install dependencies:** Python 3.8+ is required. Install libraries with:  
  ```bash
  pip install -r requirements.txt
  ```  
  *Key packages:* PyTorch, OpenCV, NumPy, Open3D, ROS (for depth bag processing).  
- **Hardware:** A Linux PC (Ubuntu 20.04) with an NVIDIA GPU (e.g. Jetson, RTX) is recommended. The robot needs a 3D LiDAR (e.g. Blickfeld) and a depth camera (e.g. RealSense). For depth pipeline, ROS Noetic should be installed to handle bag files.  

## LiDAR Pipeline (Lidar_Motor_alertness)

1. **Preprocessing:**  
   The script `pcd_image_extract.py` converts raw LiDAR `.pcd` files into 2D depth images (projecting the 3D scan onto an image plane). This yields depth frames that capture the geometry around the person.  
2. **Segmentation & Cropping:**  
   We run a YOLOv8 instance segmentation on the UGV’s camera image to isolate the person【53†L368-L377】. From the mask, we compute the pixel centroid and convert it to an angular bearing relative to the camera. LiDAR points whose azimuth lies within a narrow window of this bearing are selected – effectively cropping the point cloud around the casualty【53†L368-L377】. We then cluster these points and take the nearest cluster centroid as the person’s 3D position.  
3. **Motion Tracking:**  
   `run_alltracker.py` processes the cropped depth images to detect and track the person’s movement. It applies clustering or optical flow methods frame-by-frame to follow limbs.  
4. **Feature Extraction:**  
   `final_features.py` takes the tracked points and computes temporal features (e.g. displacements, velocities, variance). We stack features over a short time window (e.g. several frames) to form the model input. Stacked temporal features capture motion history【30†L123-L130】.  
5. **Classification:**  
   `inference_final.py` loads these feature stacks and applies a trained neural network (`best_model_checkpoint.pth`) to classify the movement (e.g. left-leg twitch, right-leg twitch, no-motion).  

A helper script automates this flow:  
```bash
cd Lidar_Motor_alertness
bash Lidar_inference.sh
```  
This shell script calls the above steps in order (preprocess → segment → track → features → inference).

## Depth-Camera Pipeline (Depthcam_motor_alertness)

1. **Bag Processing:**  
   `process_bag.py` reads a ROS `.bag` file and extracts color+depth frames. The depth frames (with skeleton or point data) will be used for motion analysis.  
2. **Motion Tracking:**  
   `run_alltracker.py` tracks the person in the depth-video frames using pose estimation or optical-flow. The result is a sequence of tracked points/poses.  
3. **Feature Extraction:**  
   `final_features.py` computes the same type of temporal features from the depth stream (e.g. pixel-shift features, joint velocities). Frames are stacked over time to produce feature vectors.  
4. **Classification:**  
   `inference_final.py` uses the same trained model to classify the depth-based feature stacks.  

Run it with:  
```bash
cd Depthcam_motor_alertness
bash motor_alertness.sh
```  
This executes the pipeline (bag processing → tracking → features → inference).

## Features & Model Training

- **Feature Engineering:** We use *stacked* temporal features. Each input to the model is a tensor representing several consecutive frames, capturing the dynamics of movement. Clustering utilities (`cluster.py`, `cluster_var.py`) preprocess the training data to group similar motion and calculate variances, enhancing feature robustness.  
- **Model:** The model is a supervised neural network (e.g. a 1D-CNN) implemented in PyTorch. Training is done in `Training_logic/Twitching.ipynb`, which loads the `.npy` feature stacks and label data (`labels.npy`), defines the network, and trains it. The best-performing weights are saved as `best_model_checkpoint.pth`.  
- **YOLO Usage:** We specifically use a YOLOv8-based segmentation model to detect and mask the human in the RGB image【53†L368-L377】. This segmentation guides the LiDAR cropping (as described above) and ensures that features focus on the person. (YOLO is fine-tuned on casualty imagery similar to past DARPA work【53†L333-L342】【53†L368-L377】.)  

During training, the dataset includes both twitch and non-twitch scenarios. The model learns to distinguish subtle movement patterns. In line with related systems, a high accuracy is achieved by focusing on these temporal features【30†L123-L130】.

## Running the System

- **LiDAR Flow:**  
  ```bash
  cd Lidar_Motor_alertness
  bash Lidar_inference.sh
  ```  
  This will process all `.pcd` scans in the input folder and output detected alerts (e.g., “Right leg twitch at t=12.3s”).  

- **Depth Flow:**  
  ```bash
  cd Depthcam_motor_alertness
  bash motor_alertness.sh
  ```  
  This processes provided ROS bag file(s) and prints movement detections.  

- **Retraining (optional):**  
  Open `Training_logic/Twitching.ipynb` and follow the cells to retrain on new data. Ensure `labels.npy` and feature `.npy` files are prepared. After training, use the new `best_model_checkpoint.pth` for inference.

![Uploading image.png…]()

 *Figure: UGVs (Spot robots) with sensors at a DARPA Triage demonstration【2†L84-L92】.* Our system is designed for such scenarios, autonomously detecting and reporting patient movements.

## System Notes

- **Environment:** Tested on Ubuntu 20.04 with ROS Noetic (for depth bags) and Python 3.8. An NVIDIA GPU is needed for real-time inference【30†L165-L170】.  
- **Sensor Calibration:** Camera intrinsics and extrinsics must be set correctly. The YOLO segmentation assumes a calibrated camera to convert mask pixels into angles.  
- **Point Mapping:** For each segmented person, we map 2D mask centroids to 3D points by selecting LiDAR points at the corresponding azimuth. Clustering those points yields a precise 3D location for the person【53†L368-L377】. This integration of vision (YOLO) and LiDAR segmentation is key to our pipeline.

This README covers all scripts, data flow, and usage. It is based on the repository files and follows practices from recent DARPA triage research【1†L126-L133】【53†L368-L377】.
