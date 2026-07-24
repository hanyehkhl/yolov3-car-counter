"""Configuration for YOLOv3 traffic vehicle detection and counting."""
from pathlib import Path

ROOT_DIR = Path(__file__).parent.resolve()

MODEL_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"

YOLO_CFG = MODEL_DIR / "yolov3.cfg"
YOLO_WEIGHTS = MODEL_DIR / "yolov3.weights"
COCO_NAMES = MODEL_DIR / "coco.names"

DEFAULT_INPUT_VIDEO = INPUT_DIR / "traffic.mp4"
DEFAULT_OUTPUT_VIDEO = OUTPUT_DIR / "output.mp4"
DEFAULT_CSV = OUTPUT_DIR / "car_count.csv"

# Model input size (multiple of 32): 320, 416, 608
INPUT_WIDTH = 416
INPUT_HEIGHT = 416

CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

VEHICLE_CLASSES = {"car", "truck", "bus"}

BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 0, 255)
FONT_SCALE = 0.6
THICKNESS = 2
