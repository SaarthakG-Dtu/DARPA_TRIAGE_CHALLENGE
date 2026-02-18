import os
import numpy as np

def transform_trajectories(trajs):
   
    T, N, _ = trajs.shape
    features = np.zeros((N, T, 4), dtype=np.float32)

    # x, y
    features[:, :, 0] = trajs[:, :, 0].T
    features[:, :, 1] = trajs[:, :, 1].T

    displacements = np.zeros((T, N))
    angles = np.zeros((T, N))
    diffs = np.diff(trajs, axis=0)  # [T-1, N, 2]

    for t in range(1, T):
        dxdy = diffs[t-1]
        displacements[t] = np.linalg.norm(dxdy, axis=1)

        prev_vec = diffs[t-2] if t > 1 else np.zeros_like(dxdy)
        dot = np.sum(dxdy * prev_vec, axis=1)
        norm = (np.linalg.norm(dxdy, axis=1) * np.linalg.norm(prev_vec, axis=1)) + 1e-8
        cos_theta = np.clip(dot / norm, -1.0, 1.0)
        angles[t] = np.arccos(cos_theta)

    features[:, :, 2] = displacements.T
    features[:, :, 3] = angles.T
    return features


def process_npy_folder(input_dir, output_path):
    all_features = []

    for file in sorted(os.listdir(input_dir)):
        if file.endswith(".npy"):
            file_path = os.path.join(input_dir, file)
            print(f"Processing {file_path}...")

            trajs = np.load(file_path)  
            features = transform_trajectories(trajs)  

            all_features.append(features)

    final_data = np.vstack(all_features)
    print(f"Final data shape: {final_data.shape}")

    np.save(output_path, final_data)
    print(f"Saved stacked data to {output_path}")

process_npy_folder(
    "/home/uas-dtu/Motor_alertness_final/npy_folder_dataset",
    "/home/uas-dtu/Motor_alertness_final/stacked_features.npy"
)

data=np.load("/home/uas-dtu/Motor_alertness_final/stacked_features.npy")
print(data)