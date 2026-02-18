import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope

# Extra methods (requires pyod for HBOS & LODA)
from pyod.models.hbos import HBOS
from pyod.models.loda import LODA

# ----------------- STEP 1: Load and preprocess data -----------------
data = np.load("/home/saarthak/Motor_alertness_final/trajectories.npy")
data = np.transpose(data, (1, 0, 2))
print("Data shape:", data.shape)

# ----------------- STEP 2: Compute motion features (variance) -----------------
pixel_variances = []
for pixel in range(data.shape[0]):
    traj = data[pixel]
    displacements = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    pixel_variances.append(np.var(displacements))  # Variance of motion

pixel_variances = np.array(pixel_variances).reshape(-1, 1)
pixel_positions = data[:, 0, :]  # First frame positions

# ========================= ALGORITHM FUNCTIONS ========================= #
def detect_dbscan(features):
    dbscan = DBSCAN(eps=0.4, min_samples=3)
    labels = dbscan.fit_predict(features)
    if len(set(labels)) <= 2 and -1 not in set(labels):
        print("DBSCAN: No twitch points detected.")
        twitch_indices = []
    else:
        twitch_indices = np.where(labels == -1)[0]
    return labels, twitch_indices

def detect_hbos(features):
    hbos = HBOS()
    hbos.fit(features)
    scores = hbos.decision_scores_
    threshold = np.percentile(scores, 90)
    labels = np.where(scores > threshold, -1, 0)
    twitch_indices = np.where(labels == -1)[0]
    print(f"HBOS: {len(twitch_indices)} outliers detected.")
    return labels, twitch_indices

def detect_loda(features):
    loda = LODA()
    loda.fit(features)
    scores = loda.decision_scores_
    threshold = np.percentile(scores, 90)
    labels = np.where(scores > threshold, -1, 0)
    twitch_indices = np.where(labels == -1)[0]
    print(f"LODA: {len(twitch_indices)} outliers detected.")
    return labels, twitch_indices

# ========================= VISUALIZATION FUNCTION ========================= #
def visualize_results(method_name, labels, twitch_indices):
    pixel_labels = np.zeros(len(pixel_variances), dtype=int)
    pixel_labels[twitch_indices] = 1

    # 1D Variance Plot
    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(pixel_variances)), pixel_variances,
                c=pixel_labels, cmap='coolwarm', s=10)
    plt.xlabel("Pixel Index")
    plt.ylabel("Motion Variance")
    plt.title(f"{method_name}: Pixel Motion Variance")
    plt.colorbar(label="0 = Non-Twitch, 1 = Twitch")
    plt.show()

    # 2D Spatial Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(pixel_positions[pixel_labels == 0, 0],
                pixel_positions[pixel_labels == 0, 1],
                color='blue', s=10, label="Non-Twitch")
    plt.scatter(pixel_positions[pixel_labels == 1, 0],
                pixel_positions[pixel_labels == 1, 1],
                color='red', s=10, label="Twitch")
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.title(f"{method_name}: Twitch Detection in Image Space")
    plt.legend()
    plt.show()

# ========================= RUN ALL METHODS ========================= #
methods = [
    ("DBSCAN", detect_dbscan),
    ("LOF", detect_lof),
    ("Isolation Forest", detect_isolation),
    ("One-Class SVM", detect_ocsvm),
    ("Elliptic Envelope", detect_elliptic),
    ("HBOS", detect_hbos),
    ("LODA", detect_loda)
]

for name, method in methods:
    labels, twitch_indices = method(pixel_variances)
    visualize_results(name, labels, twitch_indices)
