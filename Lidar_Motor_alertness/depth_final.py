import os
import argparse
import cv2
import numpy as np
import open3d as o3d
from ultralytics import YOLO
import yaml
from collections import defaultdict
import shutil
import re 

class DepthEstimator:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        self.fx, self.fy = cfg['camera']['fx'], cfg['camera']['fy']
        self.cx, self.cy = cfg['camera']['cx'], cfg['camera']['cy']
        self.T_cam_lidar = np.array(cfg['extrinsic_matrix'])

    def project_pcd_to_image(self, pcd, img_shape):
        points_lidar = np.asarray(pcd.points)
        points_lidar_hom = np.hstack([points_lidar, np.ones((points_lidar.shape[0], 1))])
        
        points_cam_hom = (self.T_cam_lidar @ points_lidar_hom.T).T
        points_cam = points_cam_hom[:, :3]
        
        in_front_mask = points_cam[:, 2] > 0
        if not np.any(in_front_mask):
            return None, None, None
        
        points_cam_valid = points_cam[in_front_mask]
        
        x, y, z = points_cam_valid.T
        u = (self.fx * x / z + self.cx).astype(int)
        v = (self.fy * y / z + self.cy).astype(int)

        h, w = img_shape[:2]
        bounds_mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        
        return u[bounds_mask], v[bounds_mask], points_cam_valid[bounds_mask]

def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    masks_dir = os.path.join(args.output_dir, "masks")
    annotated_frames_dir = os.path.join(args.output_dir, "annotated_frames")
    
    for d in [masks_dir, annotated_frames_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)
        
    print(f"[INFO] Loading YOLO model: {args.yolo_model}")
    model = YOLO(args.yolo_model)
    
    unsorted_files = [f for f in os.listdir(args.image_dir) if f.endswith(('.jpg', '.png'))]
    image_files = sorted(unsorted_files, key=lambda f: int(re.findall(r'\d+', f)[-1]))
    
    all_detections = []
    print("[INFO] Running person tracking and saving annotated frames...")
    
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(args.image_dir, img_file)
        frame = cv2.imread(img_path)
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, classes=[0])
        result = results[0]
        
        annotated_frame = result.plot()
        frame_filename = os.path.join(annotated_frames_dir, f"frame_{i:05d}.png")
        cv2.imwrite(frame_filename, annotated_frame)
        
        frame_data = {"frame_id": i, "persons": []}
        if result.boxes.id is not None and result.masks is not None:
            for track_id, mask_data in zip(result.boxes.id.int().cpu().tolist(), result.masks.data.cpu().numpy()):
                mask = cv2.resize(mask_data, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask_path = os.path.join(masks_dir, f"mask_f{i:05d}_t{track_id}.png")
                cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
                frame_data["persons"].append({"track_id": track_id, "mask_file": mask_path})
        all_detections.append(frame_data)
        print(f"  > Processed frame {i+1}/{len(image_files)}", end='\r')
    print(f"\n[INFO] All annotated frames saved to: '{annotated_frames_dir}'")
    
    tracks = defaultdict(list)
    for frame_data in all_detections:
        for person in frame_data["persons"]:
            tracks[person["track_id"]].append({"frame_id": frame_data["frame_id"], "mask_file": person["mask_file"]})

    print("\n[INFO] Calculating median depth for each person track...")
    depth_processor = DepthEstimator(args.config)
    track_median_depths = {}
    for track_id, track_data in tracks.items():
        all_depth_values = []
        for detection in track_data:
            frame_number = detection['frame_id'] + 1
            pcd_path = os.path.join(args.pcd_dir, f"pointcloud_{frame_number}.pcd")
            if not os.path.exists(pcd_path): continue
            
            pcd = o3d.io.read_point_cloud(pcd_path)
            mask = cv2.imread(detection['mask_file'], cv2.IMREAD_GRAYSCALE)
            if mask is None: continue
            
            u, v, points_in_camera_frame = depth_processor.project_pcd_to_image(pcd, mask.shape)
            if u is None: continue

            person_points = points_in_camera_frame[mask[v, u] > 0]
            if len(person_points) > 0:
                all_depth_values.extend(person_points[:, 2])

        if all_depth_values:
            track_median_depths[track_id] = np.median(all_depth_values)
            print(f"  > Track {track_id}: Median Depth = {track_median_depths[track_id]:.2f}m")

    if track_median_depths:
        primary_track_id = min(track_median_depths, key=track_median_depths.get)
        print(f"\n[RESULT] Nearest person identified: Track ID {primary_track_id} (Depth: {track_median_depths[primary_track_id]:.2f}m)")
        output_pixel_data = {}
        for detection in tracks[primary_track_id]:
            mask = cv2.imread(detection["mask_file"], cv2.IMREAD_GRAYSCALE)
            pixel_coords = np.argwhere(mask > 0) 
            output_pixel_data[f"frame_{detection['frame_id']}"] = pixel_coords
        output_path = os.path.join(args.output_dir, "nearest_person_pixels.npz")
        np.savez_compressed(output_path, **output_pixel_data)
        print(f"[SUCCESS] Pixel data for nearest person saved to '{output_path}'")
    else:
        print("\n[ERROR] Could not determine depth for any tracks. Check calibration and coordinate systems.")
        
    # Create videos at the end
    create_video_from_frames(annotated_frames_dir, args.output_dir, "annotated_video.mp4", framerate=args.framerate)
    create_source_rgb_video(args.image_dir, image_files, args.output_dir, framerate=args.framerate)

# --- MODIFIED VIDEO CREATION FUNCTION ----
def create_video_from_frames(frames_dir, output_dir, output_filename, framerate=10):
    """Creates a video from a directory of frames using OpenCV."""
    print(f"\n[INFO] Creating video '{output_filename}' from frames...")
    output_video_path = os.path.join(output_dir, output_filename)
    
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    if not frame_files:
        print(f"[WARNING] No frames found in '{frames_dir}' to create a video.")
        return

    first_frame_path = os.path.join(frames_dir, frame_files[0])
    frame = cv2.imread(first_frame_path)
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or 'XVID'
    video_writer = cv2.VideoWriter(output_video_path, fourcc, framerate, (width, height))

    for filename in frame_files:
        img_path = os.path.join(frames_dir, filename)
        img = cv2.imread(img_path)
        video_writer.write(img)

    video_writer.release()
    print(f"[SUCCESS] Video created successfully at: '{output_video_path}'")
    
def create_source_rgb_video(image_dir, sorted_image_files, output_dir, framerate=10):
    """Creates a video from the original source frames using OpenCV."""
    print(f"\n[INFO] Creating source RGB video...")
    output_video_path = os.path.join(output_dir, "source_rgb_video.mp4")

    if not sorted_image_files:
        print(f"[WARNING] No source images found in '{image_dir}' to create a video.")
        return

    first_frame_path = os.path.join(image_dir, sorted_image_files[0])
    frame = cv2.imread(first_frame_path)
    height, width, layers = frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, framerate, (width, height))

    for filename in sorted_image_files:
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path)
        video_writer.write(img)

    video_writer.release()
    print(f"[SUCCESS] Source RGB video created successfully at: '{output_video_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process frames to find the nearest person and create a video.")
    parser.add_argument("--image_dir", required=True, help="Path to images.")
    parser.add_argument("--pcd_dir", required=True, help="Path to point clouds.")
    parser.add_argument("--output_dir", default="processing_output", help="Directory to save results.")
    parser.add_argument("--config", default="config.yaml", help="Path to calibration YAML file.")
    parser.add_argument("--yolo_model", default="yolov8x-seg.pt", help="Path to YOLO model.")
    parser.add_argument("--framerate", type=int, default=10, help="Framerate for the output videos.")
    args = parser.parse_args()
    main(args)