set -e

if [ -z "$1" ]; then
    echo "Usage: ./run_pipeline.sh /path/to/your/file.bag"
    exit 1
fi
BAG_FILE="$1"

QUERY_FRAME=8

ALLTRACKER_RATE=8

MODEL_CHECKPOINT="best_model_checkpoint.pth"

BAGNAME=$(basename "$BAG_FILE" .bag)
BAG_PROCESS_DIR="processed_data_${BASENAME}"
ALLTRACKER_DIR="alltracker_output_${BASENAME}"
VIDEO_OUTPUT_PATH="${PROCESS_DIR}/${BASENAME}.mp4"
PIXELS_NPZ_PATH="${PROCESS_DIR}/primary_person_pixels.npz"
RAW_TRAJS_NPY_PATH="${ALLTRACKER_DIR}/trajectories.npy"
FINAL_FEATURES_NPY_PATH="stacked_features_${BASENAME}.npy"

echo "/n"
echo "Starting Twitch Detection Pipeline for: $BAG_FILE"
echo "/n"

echo -e "\n[STEP 1/4] Preprocessing .bag file for DEPTH,TRACKING OF NEAREST CASUALTY"
python3 process_bag.py \
    --bag_file "$BAG_FILE" \
    --output_dir "$BAG_PROCESS_DIR" \
    --video_output_path "$VIDEO_OUTPUT_PATH"

echo -e "\n[STEP 2/4] Running All-Tracker to generate raw trajectories..."
python3 run_alltracker.py \
    --mp4_path "$VIDEO_OUTPUT_PATH" \
    --output_dir "$ALLTRACKER_DIR" \
    --query_frame "$QUERY_FRAME" \
    --rate "$ALLTRACKER_RATE"

echo -e "\n[STEP 3/4] Filtering trajectories and creating features..."
python3 final_features.py \
    --trajs_file "$RAW_TRAJS_NPY_PATH" \
    --pixels_file "$PIXELS_NPZ_PATH" \
    --video_path "$VIDEO_OUTPUT_PATH" \
    --query_frame "$QUERY_FRAME" \
    --rate "$ALLTRACKER_RATE" \
    --output_file "$FINAL_FEATURES_NPY_PATH"

echo -e "\n[STEP 4/4] Running final inference..."
python3 inference_final.py \
    --input_file "$FINAL_FEATURES_NPY_PATH" \
    --checkpoint "$MODEL_CHECKPOINT"

echo -e "\n"
echo "Success!"
echo "/n"