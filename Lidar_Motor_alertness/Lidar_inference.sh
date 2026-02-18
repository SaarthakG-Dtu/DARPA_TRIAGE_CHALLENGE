set -e

CAPTURE_ROOT_DIR="Synced_image_and_pcd"

IMAGE_CAPTURE_DIR="${CAPTURE_ROOT_DIR}/images"
PCD_CAPTURE_DIR="${CAPTURE_ROOT_DIR}/pcd"

QUERY_FRAME=2
ALLTRACKER_RATE=8
MAX_FRAMES=150
MODEL_CHECKPOINT="best_model_checkpoint.pth"

PROCESS_DIR="processed_data"
ALLTRACKER_DIR="alltracker_output"

SOURCE_VIDEO_PATH="${PROCESS_DIR}/source_rgb_video.mp4"      
ANNOTATED_VIDEO_PATH="${PROCESS_DIR}/annotated_video.mp4"   
PERSON_PIXELS_NPZ_PATH="${PROCESS_DIR}/nearest_person_pixels.npz"
RAW_TRAJS_NPY_PATH="${ALLTRACKER_DIR}/trajectories.npy"
FILTERED_TRAJS_NPY_PATH="filtered_trajs.npy"
FINAL_FEATURES_NPY_PATH="stacked_features.npy"
TRAJ_PREDICTIONS_NPY_PATH="trajectory_predictions.npy"
HIGHLIGHTED_VIDEO_PATH="highlighted_video.mp4"

echo "Starting Twitch Assessment Pipeline"

echo -e "\n[STEP 1/7] Activating 'alltracker' env..."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate alltracker

echo -e "\n[STEP 2/7] Capturing live PCD and Image frames..."

python3 pcd_image_extract.py \
    --target 192.168.26.26 
    
echo "Image and PCD data saved to '${IMAGE_CAPTURE_DIR}' and '${PCD_CAPTURE_DIR}'."

echo -e "\n[STEP 3/7] Capturing Priority Casualty Pixels Using Depth -->"
python3 depth_final.py \
    --image_dir "$IMAGE_CAPTURE_DIR" \
    --pcd_dir "$PCD_CAPTURE_DIR" \
    --output_dir "$PROCESS_DIR" \
    --config "config.yaml" \
    --yolo_model "yolov8x-seg.pt"

echo -e "\n[STEP 4/7] Running All-Tracker to generate raw trajectories -->"
python3 run_alltracker.py \
    --mp4_path "$SOURCE_VIDEO_PATH" \
    --output_dir "$ALLTRACKER_DIR" \
    --query_frame "$QUERY_FRAME" \
    --rate "$ALLTRACKER_RATE" \
    --max_frames "$MAX_FRAMES"
    
echo -e "\n[STEP 5/7] Preparing input for twitch inference -->"
python3 final_features.py \
    --trajs_file "$RAW_TRAJS_NPY_PATH" \
    --pixels_file "$PERSON_PIXELS_NPZ_PATH" \
    --video_path "$SOURCE_VIDEO_PATH" \
    --query_frame "$QUERY_FRAME" \
    --rate "$ALLTRACKER_RATE" \
    --output_file "$FINAL_FEATURES_NPY_PATH"

echo -e "\n[STEP 6/7] Model Inference -->"
python3 inference_final.py \
    --input_file "$FINAL_FEATURES_NPY_PATH" \
    --checkpoint "$MODEL_CHECKPOINT" \
    --output_file "$TRAJ_PREDICTIONS_NPY_PATH"

echo -e "\n[STEP 7/7] Plotting Model Inference on pixel trajectories -->"
python3 visualize_final.py \
    --mp4_path "$SOURCE_VIDEO_PATH" \
    --trajs_file "$FILTERED_TRAJS_NPY_PATH" \
    --predictions_file "$TRAJ_PREDICTIONS_NPY_PATH" \
    --output_video_path "$HIGHLIGHTED_VIDEO_PATH" \
    --rate "$ALLTRACKER_RATE"

echo "Pipeline Finished Successfully!"
echo "Final highlighted video is at: ${HIGHLIGHTED_VIDEO_PATH}"
