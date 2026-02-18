from __future__ import print_function
import argparse
import numpy as np
import time
import blickfeld_scanner
from blickfeld_scanner.protocol.config import scan_pattern_pb2
import cv2
import os
import shutil 
import open3d as o3d

TARGET_FRAME_COUNT = 150
CAPTURE_INTERVAL_SEC = 0.1 

OUTPUT_ROOT_DIR = "Synced_image_and_pcd"
CAMERA_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "images")
LIDAR_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "pcd")

def capture_and_save_pcd(target):
    if os.path.exists(OUTPUT_ROOT_DIR):
        print(f"'{OUTPUT_ROOT_DIR}' already exists. Removing it to start a clean run.")
        shutil.rmtree(OUTPUT_ROOT_DIR)

    os.makedirs(CAMERA_OUTPUT_DIR)
    os.makedirs(LIDAR_OUTPUT_DIR)
    print(f"Saving images to: {CAMERA_OUTPUT_DIR}")
    print(f"Saving LiDAR .pcd files to: {LIDAR_OUTPUT_DIR}")

    cap = cv2.VideoCapture('/dev/video0')
    if not cap.isOpened():
        print("Error: Could not open video device /dev/video0. Exiting.")
        exit()
    print("Camera initialized successfully.")

    print(f"Connecting to Blickfeld LiDAR at {target}...")
    try:
        device = blickfeld_scanner.scanner(target)
    except Exception as e:
        print(f"Error connecting to LiDAR: {e}")
        cap.release()
        exit()
    print("LiDAR connected successfully.")

    point_filter = scan_pattern_pb2.ScanPattern().Filter()
    point_filter.max_number_of_returns_per_point = 2
    point_filter.delete_points_without_returns = True
    stream = device.get_point_cloud_stream(point_filter=point_filter, as_numpy=True)
    print("LiDAR stream started. Beginning capture...")
    print("-" * 30)

    for i in range(TARGET_FRAME_COUNT):
        loop_start_time = time.monotonic()

        ret, frame = cap.read()
        try:
            lidar_frame, lidar_data = stream.recv_frame_as_numpy()
            lidar_capture_successful = True
        except Exception as e:
            print(f"Error receiving LiDAR frame for iteration {i+1}: {e}")
            lidar_capture_successful = False

        if not ret:
            print(f"Warning: Could not read camera frame for iteration {i+1}. Skipping save.")
            image_filename = "NO_IMAGE"
        else:
            image_filename = f"frame_{i+1}.jpg"
            image_filepath = os.path.join(CAMERA_OUTPUT_DIR, image_filename)
            cv2.imwrite(image_filepath, frame)

        if lidar_capture_successful:
            lidar_filename = f"pointcloud_{i+1}.pcd"
            lidar_filepath = os.path.join(LIDAR_OUTPUT_DIR, lidar_filename)

            x = lidar_data['cartesian']['x']
            y = lidar_data['cartesian']['y']
            z = lidar_data['cartesian']['z']
            xyz_points = np.vstack((x, y, z)).transpose()

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz_points)
            
            o3d.io.write_point_cloud(lidar_filepath, pcd)
            
            print(f"Synced {i+1}/{TARGET_FRAME_COUNT}: Saved {image_filename} and {lidar_filename}")
        else:
            print(f"Synced {i+1}/{TARGET_FRAME_COUNT}: Saved {image_filename} but LiDAR capture FAILED.")

        elapsed_time = time.monotonic() - loop_start_time
        time_to_sleep = CAPTURE_INTERVAL_SEC - elapsed_time
        
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)
        else:
            print(f"  -> Warning: Loop took {elapsed_time:.4f}s (> {CAPTURE_INTERVAL_SEC}s). Cannot maintain target Hz.")

    print("Finished capturing. Cleaning up...")
    stream.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("Successfully saved all data.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="hostname or IP of scanner to connect to")
    args = parser.parse_args()
    capture_and_save_pcd(args.target)