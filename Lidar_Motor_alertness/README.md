# 📸 Motor Alertness Assessment of a Casualty using LiDAR Point Cloud Data and a SOTA frame Pixel tracker model - AllTracker 

Motor Alertness assessment of a casualty with Twitch/non-twitch interpretation of a joint of a casualty in a mass casualty scenario using depth cam and LiDAR(Light Detection and Ranging) with the use of AllTracker, a SOTA model to track each and every pixel in a image frame. Various subsampling methods such as Query Frame(frame to track over all the frames of a video sequence) and Pixel Track Rate( How many select pixels to track). The results are obtained via a sequence of python scripts assessing depth, tracking the priority casualty(only 1 at a time) using depth obtained from PyrealSense D455 camera and Blickfeld LiDAR to extract depth using Point Cloud Data. Inference on motor alertness is run using LSTM( Long Short Term Memory) to predict twitch/non twitch points using input features of pixel displacement, Angle changed over T frames for a Nth point where T,N are obtained by a trajectory tuple given by AllTracker with the shape of T,N,2

---

<!-- ### 🎬 Demo
*(This is where you would put a GIF of your final video output!)*
![Demo GIF of LiDAR Projection](path/to/your/demo.gif) -->

---

## 🔧 Setup and Installation

Follow these steps to set up your environment and run the pipeline.

### 1. Clone the Repository

If you haven't already, clone the main project repository and navigate into this project's directory.

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name/Lidar_Motion-Assessment/
```

### 2. Create Conda Environment

This project uses a Conda environment named `alltracker` to manage its dependencies.

```bash
conda create --name alltracker python=3.10
conda activate alltracker
```

### 3. Install Dependencies

Install required dependencies.

```bash
conda activate alltracker
pip install -r requirements.txt 
```

### 4. Hardware and Network Setup

*   **Camera:** We used Iriun Webcam for video feed. Capture the required video frames using your preferred webcam.
*   **LiDAR:** Connect your Blickfeld LiDAR to the same network as your computer and ensure it is reachable via its IP address.

### 5. Critical: Configure Calibration

The accuracy of this pipeline depends entirely on a correct calibration file.

➡️ **Edit the `config.yaml` file:**
*   **`camera`:** Update the intrinsic parameters (`fx`, `fy`, `cx`, `cy`) to match your specific camera.
*   **`extrinsic_matrix`:**  Update the 4x4 etrinsic matrix inside config.yaml with the precise transformation that takes points from the LiDAR's coordinate system to the camera's coordinate system (`# LiDAR → Camera transform`).

---

## 🚀 Running the Pipeline

The entire workflow is managed by the `Lidar_inference.sh` script. This is the recommended way to use the project.

### Full End-to-End Run

This shell script runs and outputs model inference on raw trajectories by Alltracker pixel tracker along with efficient nearest single casualty tracking along with twitch/non-twitch plots.

```bash
chmod +x Lidar_inference.sh
./Lidar_inference.sh
```
