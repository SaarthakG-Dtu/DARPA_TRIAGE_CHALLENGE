import pyrealsense2 as rs
import numpy as np
import cv2
import os
import shutil
import json
from ultralytics import YOLO
import argparse
from collections import defaultdict

# --- Constants ---
YOLO_MODEL = 'Yolo_seg_models/yolov8x-seg.pt'
PERSON_CLASS_ID = 0
MIN_TRACK_LENGTH = 5
MERGE_TIME_GAP = 15
MERGE_DIST_THRESHOLD = 75

def setup_directories(output_folder):
    """Clears old data and sets up new directories."""
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(os.path.join(output_folder, "yolo_frames"))
    os.makedirs(os.path.join(output_folder, "masks"))
    os.makedirs(os.path.join(output_folder, "depth_frames_raw"))
    print("[INFO] Directories cleared and set up.")

def get_centroid(bbox):
    """Calculates the centroid of a bounding box."""
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2

def main(args):
    # Unpack arguments
    bag_file = args.bag_file
    output_folder = args.output_dir
    video_output_path = args.video_output_path

    # Define paths based on output folder
    yolo_frames_folder = os.path.join(output_folder, "yolo_frames")
    masks_folder = os.path.join(output_folder, "masks")
    depth_frames_folder = os.path.join(output_folder, "depth_frames_raw")
    tracking_data_file = os.path.join(output_folder, "tracking_data.json")
    primary_person_pixels_file = os.path.join(output_folder, "primary_person_pixels.npz")

    setup_directories(output_folder)

    # --- Part 1: YOLO Segmentation, Tracking, and Data Extraction ---
    print(f"[INFO] Loading YOLO segmentation model: {YOLO_MODEL}")
    model = YOLO(YOLO_MODEL)
    
    if not os.path.exists(bag_file):
        print(f"[ERROR] Bag file not found: {bag_file}")
        return
        
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device_from_file(bag_file, repeat_playback=False)
    
    profile = pipe.start(cfg)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)
    
    align = rs.align(rs.stream.color)
    all_tracked_data = []
    
    # Video writer setup
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    width = int(color_stream.width())
    height = int(color_stream.height())
    fps = int(color_stream.fps())
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

    print("[INFO] Starting segmentation, tracking, and data extraction...")
    frame_idx = 0
    try:
        while frame_idx<=150:
            try:
                frames = pipe.wait_for_frames(timeout_ms=5000)
            except RuntimeError:
                print("\n[INFO] End of file stream reached.")
                break

            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # Rotate both color and depth frames 180 degrees because the camera is upside down.
            color_image = cv2.rotate(color_image, cv2.ROTATE_180)
            depth_image = cv2.rotate(depth_image, cv2.ROTATE_180)

            video_writer.write(cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))

            results = model.track(color_image, persist=True, tracker='bytetrack.yaml', verbose=False, classes=[PERSON_CLASS_ID], conf=0.5)
            result = results[0]

            annotated_frame = result.plot()
            cv2.imwrite(os.path.join(yolo_frames_folder, f"tracked_frame_{frame_idx:04d}.png"), annotated_frame)
            np.save(os.path.join(depth_frames_folder, f"depth_raw_{frame_idx:04d}.npy"), depth_image)
            
            frame_data = {"frame_id": frame_idx, "persons": []}
            if result.boxes.id is not None and result.masks is not None:
                track_ids = result.boxes.id.int().cpu().tolist()
                boxes = result.boxes.xyxy.cpu().numpy().astype(int)
                all_low_res_masks = result.masks.data.cpu().numpy()

                for i, track_id in enumerate(track_ids):
                    full_res_mask = cv2.resize(all_low_res_masks[i], (width, height), interpolation=cv2.INTER_NEAREST)
                    binary_mask_image = (full_res_mask > 0.8).astype(np.uint8) * 255
                    mask_filename = f"mask_f{frame_idx:04d}_t{track_id}.png"
                    cv2.imwrite(os.path.join(masks_folder, mask_filename), binary_mask_image)
                    
                    frame_data["persons"].append({
                        "track_id": int(track_id), "bbox": boxes[i].tolist(), "mask_file": mask_filename
                    })
            
            all_tracked_data.append(frame_data)
            
            if frame_idx % 50 == 0: print(f"  > Processed frame {frame_idx}")
            frame_idx += 1
    finally:
        pipe.stop()
        video_writer.release()
        print(f"[INFO] RGB video saved to {video_output_path}")

    with open(tracking_data_file, 'w') as f: json.dump(all_tracked_data, f, indent=2)
    print("[INFO] Segmentation & Extraction finished.")

    # --- Part 2: Track Merging and Primary Person Identification ---
    print("[INFO] Grouping detections by initial track ID...")
    initial_tracks = defaultdict(list)
    for frame_data in all_tracked_data:
        for person in frame_data["persons"]:
            initial_tracks[person["track_id"]].append({
                "frame_id": frame_data["frame_id"], "bbox": person["bbox"], "mask_file": person["mask_file"]
            })

    print("[INFO] Merging broken tracks...")
    sorted_tracks = sorted(initial_tracks.values(), key=lambda t: t[0]['frame_id'])
    
    merged_tracks = []
    if sorted_tracks:
        merged_tracks.append(sorted_tracks[0])
        for track in sorted_tracks[1:]:
            merged = False
            last_frame_of_new_track = track[-1]
            last_centroid = get_centroid(last_frame_of_new_track['bbox'])

            for merged_track in merged_tracks:
                end_of_merged_track = merged_track[-1]
                end_centroid = get_centroid(end_of_merged_track['bbox'])
                
                time_gap = track[0]['frame_id'] - end_of_merged_track['frame_id']
                dist = np.linalg.norm(np.array(last_centroid) - np.array(end_centroid))

                if 0 < time_gap < MERGE_TIME_GAP and dist < MERGE_DIST_THRESHOLD:
                    merged_track.extend(track)
                    merged = True
                    break
            if not merged:
                merged_tracks.append(track)

    final_tracks = {i: track for i, track in enumerate(merged_tracks)}
    print(f"[INFO] Merged {len(initial_tracks)} tracks into {len(final_tracks)}.")

    long_tracks = {tid: track for tid, track in final_tracks.items() if len(track) >= MIN_TRACK_LENGTH}
    if not long_tracks:
        print("[ERROR] No valid long tracks found after merging."); return
    
    long_tracks = {tid: track for tid, track in final_tracks.items() if len(track) >= MIN_TRACK_LENGTH}
    
    # # --- START: MODIFIED SECTION ---
    # # If the primary tracking method fails to find a long track...
    # if not long_tracks:
    #     print("\n[WARNING] No valid long tracks found. Tracking likely failed.")
    #     print("[INFO] Falling back to per-frame 'closest person' detection method.")
        
    #     output_pixel_data = {}
    #     # Iterate through each frame's stored detection data
    #     for frame_data in all_tracked_data:
    #         frame_id = frame_data["frame_id"]
            
    #         # Skip frames where no people were detected
    #         if not frame_data["persons"]:
    #             continue

    #         persons_in_frame_with_depth = []
    #         depth_map = np.load(os.path.join(depth_frames_folder, f"depth_raw_{frame_id:04d}.npy"))

    #         # Calculate the median depth for each person detected in this single frame
    #         for person in frame_data["persons"]:
    #             mask = cv2.imread(os.path.join(masks_folder, person["mask_file"]), cv2.IMREAD_GRAYSCALE)
    #             if mask is not None:
    #                 valid_depths = depth_map[mask > 0]
    #                 valid_depths = valid_depths[valid_depths > 0] # Filter out zero-depth pixels
    #                 if valid_depths.size > 0:
    #                     persons_in_frame_with_depth.append({
    #                         "mask_file": person["mask_file"],
    #                         "median_depth": np.median(valid_depths)
    #                     })
            
    #         # If any people with valid depths were found in this frame...
    #         if persons_in_frame_with_depth:
    #             # Find the one with the smallest median depth (the closest)
    #             closest_person = min(persons_in_frame_with_depth, key=lambda p: p["median_depth"])
                
    #             # Save the pixel coordinates from the closest person's mask
    #             mask = cv2.imread(os.path.join(masks_folder, closest_person["mask_file"]), cv2.IMREAD_GRAYSCALE)
    #             if mask is not None:
    #                 pixel_coords = np.argwhere(mask > 0)
    #                 output_pixel_data[f"frame_{frame_id}"] = pixel_coords

    #     print(f"\n[INFO] Fallback complete. Exporting pixels for closest person in each frame to: '{primary_person_pixels_file}'")
    #     np.savez_compressed(primary_person_pixels_file, **output_pixel_data)
    #     print(f"\n[COMPLETE] Script finished using fallback method.")
    #     return # Exit cleanly after the fallback logic is done
    # # --- END: MODIFIED SECTION ---

    print(f"[INFO] Found {len(long_tracks)} significant tracks to analyze.")
    
    track_median_depths = {}
    for track_id, track_data in long_tracks.items():
        all_depth_values = []
        for detection in track_data:
            depth_map = np.load(os.path.join(depth_frames_folder, f"depth_raw_{detection['frame_id']:04d}.npy"))
            mask = cv2.imread(os.path.join(masks_folder, detection["mask_file"]), cv2.IMREAD_GRAYSCALE)
            if mask is not None and mask.shape == depth_map.shape:
                valid_depths = depth_map[mask > 0]
                if valid_depths.size > 0:
                    all_depth_values.extend(valid_depths[valid_depths > 0])
        
        if all_depth_values: track_median_depths[track_id] = np.median(all_depth_values)
    
    if not track_median_depths:
        print("[ERROR] Could not determine median depth for any track."); return

    primary_person_track_id = min(track_median_depths, key=track_median_depths.get)
    primary_person_track_data = long_tracks[primary_person_track_id]
    print(f"\n[RESULT] Primary person selected: Merged Track ID {primary_person_track_id}")
    print(f"\n[RESULT] Depths : {track_median_depths}")


    print(f"[INFO] Exporting pixel coordinates for the primary person to: '{primary_person_pixels_file}'")
    
    output_pixel_data = {}
    for detection in primary_person_track_data:
        frame_id = detection["frame_id"]
        mask = cv2.imread(os.path.join(masks_folder, detection["mask_file"]), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            pixel_coords = np.argwhere(mask > 0) # Returns [[y1,x1], [y2,x2], ...]
            output_pixel_data[f"frame_{frame_id}"] = pixel_coords

    np.savez_compressed(primary_person_pixels_file, **output_pixel_data)
    
    print(f"\n[COMPLETE] Script 1 finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a RealSense .bag file for person segmentation and tracking.")
    parser.add_argument("--bag_file", type=str, required=True, help="Path to the input .bag file.")
    parser.add_argument("--output_dir", type=str, default="processed_data", help="Directory to save all processed data.")
    parser.add_argument("--video_output_path", type=str, default="processed_data/output.mp4", help="Path to save the extracted RGB video.")
    args = parser.parse_args()
    main(args)