"""Training configuration for the odom-to-GT RNN model."""

INPUT_PATH = "data/Odom_temporal_aligned.jsonl"
TARGET_PATH = "data/pose_GT_by_mocap_temporal_aligned.jsonl"

FEATURE_KEYS = ("vx", "vy", "vtheta")
TARGET_KEYS = ("vx", "vy", "vtheta")

DEVICE = "cpu"
SEED = 42

RNN_TYPE = "rnn"
HIDDEN_SIZE = 32
NUM_LAYERS = 2
DROPOUT = 0.0

SEQ_LEN = 128
STRIDE = 16
TRAIN_RATIO = 0.8
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
GRAD_CLIP_NORM = 1.0

LOSS_TYPE = "berhu"
BERHU_C = 0.2
THETA_LOSS_WEIGHT = 1.0
WRAP_THETA_RESIDUAL = False
THETA_INDEX = 2

MODEL_DIR = "results/models"
TRAINING_DIR = "results/training"
CHECKPOINT_PATH = "results/models/rnn_odom_to_gt.pt"
CONFIG_SNAPSHOT_PATH = "results/models/rnn_odom_to_gt_config.json"
HISTORY_PATH = "results/training/train_history.json"
LOSS_CURVE_PATH = "results/training/loss_curve.png"
PREDICTION_PREVIEW_PATH = "results/training/prediction_preview.png"
INFERENCE_INPUT_PATH = "data/Odom_temporal_aligned.jsonl"
INFERENCE_OUTPUT_PATH = "data/Odom_model_prediction.jsonl"
