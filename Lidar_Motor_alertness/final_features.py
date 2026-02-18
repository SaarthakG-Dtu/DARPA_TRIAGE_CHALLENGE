import numpy as np
import os
import cv2
import argparse

def transform_trajectories(trajs):
    """Computes features (x, y, displacement, angle) from trajectories."""
    T, N, _ = trajs.shape
    if N == 0:
        return np.zeros((0, T, 4), dtype=np.float32)

    features = np.zeros((N, T, 4), dtype=np.float32)
    features[:, :, 0] = trajs[:, :, 0].T  # x
    features[:, :, 1] = trajs[:, :, 1].T  # y

    # Calculate displacement and angle
    diffs = np.diff(trajs, axis=0)  # Shape: [T-1, N, 2]
    displacements = np.linalg.norm(diffs, axis=2) # Shape: [T-1, N]
    
    features[:, 1:, 2] = displacements.T

    # Angle change (curvature)
    angles = np.zeros((N, T))
    for t in range(2, T):
        v1 = diffs[t-2] # Vector from t-2 to t-1
        v2 = diffs[t-1] # Vector from t-1 to t
        
        dot_product = np.einsum('ij,ij->i', v1, v2)
        norm_product = (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)) + 1e-7
        
        cosine_angle = np.clip(dot_product / norm_product, -1.0, 1.0)
        angles[:, t] = np.arccos(cosine_angle)
        
    features[:, :, 3] = angles

    return features

def main(args):
    print(f"[INFO] Loading raw trajectories from: {args.trajs_file}")
    raw_trajs = np.load(args.trajs_file) # Shape: [T, N, 2]
    
    print(f"[INFO] Loading person pixel data from: {args.pixels_file}")
    pixel_data = np.load(args.pixels_file)
    
    query_frame_key = f"frame_{args.query_frame}"
    if query_frame_key not in pixel_data:
        print(f"[ERROR] Query frame {args.query_frame} not found in pixel data file.")
        return
        
    person_pixels_orig = pixel_data[query_frame_key] # Shape: [num_pixels, 2] with (y, x)

    # This assumes the input video for All-Tracker is the one from process_bag
    cap = cv2.VideoCapture(args.video_path)
    H_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    HH = 384
    scale = min(HH / H_orig, HH / W_orig)
    H_resized, W_resized = int(H_orig * scale) // 8 * 8, int(W_orig * scale) // 8 * 8
    
    # Create a resized mask of the person
    person_mask_resized = np.zeros((H_resized, W_resized), dtype=np.uint8)
    
    # Scale the person's pixel coordinates
    scaled_y = (person_pixels_orig[:, 0] * scale).astype(int)
    scaled_x = (person_pixels_orig[:, 1] * scale).astype(int)
    
    # Filter coordinates to be within the new bounds
    valid_indices = (scaled_y < H_resized) & (scaled_x < W_resized)
    person_mask_resized[scaled_y[valid_indices], scaled_x[valid_indices]] = 1

    # Identify which trajectories to keep
    T, N_total, _ = raw_trajs.shape
    rate = args.rate
    grid_W = W_resized // rate
    
    # Get the coordinates of the first point of each trajectory in the resized frame
    trajs_at_query_frame = raw_trajs[args.query_frame, :, :] # Shape [N_total, 2]
    
    keep_indices = []
    for i in range(N_total):
        # The trajectory points are on a grid. Find the grid point's location.
        grid_y = (i // grid_W) * rate
        grid_x = (i % grid_W) * rate

        # Check if this grid point is inside the person's resized mask
        if person_mask_resized[grid_y, grid_x] == 1:
            keep_indices.append(i)

    filtered_trajs = raw_trajs[:, keep_indices, :]
    print(f"[INFO] Filtered trajectories. Kept {len(keep_indices)} out of {N_total} trajectories.")
    
    filtered_trajs_output_path = args.output_file.replace("stacked_features", "filtered_trajs")
    np.save(filtered_trajs_output_path, filtered_trajs)
    print(f"[INFO] Saved filtered trajectories to {filtered_trajs_output_path}")

    print("[INFO] Transforming filtered trajectories into features...")
    features = transform_trajectories(filtered_trajs)
    
    print(f"[INFO] Final feature data shape: {features.shape}")
    np.save(args.output_file, features)
    print(f"[COMPLETE] Saved final features to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter trajectories and create features for inference.")
    parser.add_argument("--trajs_file", type=str, required=True, help="Path to raw trajectories .npy file from All-Tracker.")
    parser.add_argument("--pixels_file", type=str, required=True, help="Path to the primary person pixels .npz file.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the original video fed to All-Tracker to get dimensions.")
    parser.add_argument("--query_frame", type=int, default=16, help="The frame used for filtering.")
    parser.add_argument("--rate", type=int, default=2, help="The subsampling rate used in All-Tracker.")
    parser.add_argument("--output_file", type=str, default="stacked_features.npy", help="Path to save the final features.")
    args = parser.parse_args()
    main(args)