# in this example, your images should be "$DATASET_ROOT/images-cropped/$IDENTITY_NAME/*.jpg"
DATASET_ROOT="/home/lichnost/programming/work/ml/head/latent-pose/dataset"
IDENTITY_NAME="AriDan"
MAX_BATCH_SIZE=4             # pick the largest possible, start with 8 and decrease until it fits in VRAM
CHECKPOINT_PATH="/home/lichnost/programming/work/ml/head/latent-pose/latent-pose-release.pth"
OUTPUT_PATH="outputs/"       # a directory for outputs, will be created
RUN_NAME="AriDan_1"  # give your run a name if you want

# Important. See the note below
TARGET_NUM_ITERATIONS=360

# Don't change these
NUM_IMAGES=`ls -1 "$DATASET_ROOT/images-cropped/$IDENTITY_NAME" | wc -l`
BATCH_SIZE=$((NUM_IMAGES<MAX_BATCH_SIZE ? NUM_IMAGES : MAX_BATCH_SIZE))
ITERATIONS_IN_EPOCH=$(( NUM_IMAGES / BATCH_SIZE ))

mkdir -p $OUTPUT_PATH

python3 train.py \
    --config finetuning-base                 \
    --checkpoint_path "$CHECKPOINT_PATH"     \
    --data_root "$DATASET_ROOT"              \
    --train_split_path "$IDENTITY_NAME"      \
    --batch_size $BATCH_SIZE                 \
    --num_epochs $(( (TARGET_NUM_ITERATIONS + ITERATIONS_IN_EPOCH - 1) / ITERATIONS_IN_EPOCH )) \
    --experiments_dir "$OUTPUT_PATH"         \
    --experiment_name "$RUN_NAME"