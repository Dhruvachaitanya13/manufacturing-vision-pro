import sys
import torch
from pathlib import Path
from typing import Union, Optional
from pydantic_settings import BaseSettings
from loguru import logger

# =============================================================================
# LOGGING SETUP
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(
    sys.stderr, 
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(LOGS_DIR / "system.log", rotation="50 MB", retention="30 days", level="DEBUG", compression="zip")

# =============================================================================
# SETTINGS CLASS
# =============================================================================
class Settings(BaseSettings):
    """
    Application Configuration using Pydantic.
    Reads from Environment Variables or defaults.
    """
    # Project Info
    PROJECT_NAME: str = "manufacturing_vision_pro"
    VERSION: str = "2.1.0"
    DEBUG_MODE: bool = True

    # Paths
    BASE_PATH: Path = BASE_DIR
    DATA_PATH: Path = BASE_DIR / "data"
    MODEL_PATH: Path = BASE_DIR / "models"
    
    # Model Hyperparameters
    MODEL_TYPE: str = "yolov8n.pt" # Nano model for Edge Devices
    IMG_SIZE: int = 640
    BATCH_SIZE: int = 16
    EPOCHS: int = 25 # Increased for better convergence
    LEARNING_RATE: float = 0.01
    PATIENCE: int = 10 # Early stopping
    
    # Inference Thresholds
    CONF_THRESHOLD: float = 0.55
    IOU_THRESHOLD: float = 0.45
    
    # Hardware
    DEVICE: str = "cpu" # Will be updated dynamically

    def configure_device(self) -> str:
        """Auto-detects the best available accelerator (MPS/CUDA/CPU)."""
        if torch.backends.mps.is_available():
            logger.success("🚀 Hardware Acceleration: Apple Silicon (MPS) Activated")
            return "mps"
        elif torch.cuda.is_available():
            logger.success("🚀 Hardware Acceleration: NVIDIA CUDA Activated")
            return "cuda"
        else:
            logger.warning("⚠️ No Accelerator Detected. Running on CPU (High Latency).")
            return "cpu"

    class Config:
        env_file = ".env"

# Initialize
settings = Settings()
settings.DEVICE = settings.configure_device()

# Ensure directories exist
for path in [settings.DATA_PATH, settings.MODEL_PATH]:
    path.mkdir(parents=True, exist_ok=True)