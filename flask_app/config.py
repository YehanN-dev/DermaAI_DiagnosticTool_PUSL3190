from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "mobilenetv2_finetuned_final_model.keras"

UPLOAD_FOLDER = Path(__file__).resolve().parent / "static" / "uploads"
OUTPUT_FOLDER = Path(__file__).resolve().parent / "static" / "outputs"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

MAX_CONTENT_LENGTH = 8 * 1024 * 1024
