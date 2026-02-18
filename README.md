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
