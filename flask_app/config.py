from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "mobilenetv2_finetuned_final_model.keras"

UPLOAD_FOLDER = Path(__file__).resolve().parent / "static" / "uploads"
OUTPUT_FOLDER = Path(__file__).resolve().parent / "static" / "outputs"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

MAX_CONTENT_LENGTH = 8 * 1024 * 1024

LOW_CONFIDENCE_THRESHOLD = 0.45
VERY_LOW_CONFIDENCE_THRESHOLD = 0.36
LOW_MARGIN_THRESHOLD = 0.08