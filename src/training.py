import shutil
from ultralytics import YOLO
from loguru import logger
from .config import settings
from .utils import plot_training_results

class TrainingPipeline:
    """
    End-to-end MLOps pipeline for training the defect detector.
    Handles:
    1. Pre-checks (Data validation).
    2. Training (Transfer Learning).
    3. Evaluation & Metrics logging.
    4. Model Versioning & Export.
    """
    
    def __init__(self):
        self.yaml_path = settings.DATA_PATH / "data.yaml"
        self.save_dir = settings.MODEL_PATH / settings.PROJECT_NAME
        
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"Missing data configuration: {self.yaml_path}")

    def clean_previous_runs(self):
        """Archives or deletes old training runs to keep workspace clean."""
        if self.save_dir.exists():
            logger.warning(f"Found existing run at {self.save_dir}. Archiving...")
            shutil.move(str(self.save_dir), str(self.save_dir.parent / f"{settings.PROJECT_NAME}_backup"))

    def run(self):
        self.clean_previous_runs()
        
        logger.info(f"🧠 Loading Base Model: {settings.MODEL_TYPE}")
        model = YOLO(settings.MODEL_TYPE)
        
        logger.info(f"⚙️  Starting Training on {settings.DEVICE.upper()}")
        try:
            results = model.train(
                data=str(self.yaml_path),
                epochs=settings.EPOCHS,
                imgsz=settings.IMG_SIZE,
                batch=settings.BATCH_SIZE,
                device=settings.DEVICE,
                project=str(settings.MODEL_PATH),
                name=settings.PROJECT_NAME,
                patience=settings.PATIENCE,
                save=True,
                save_period=5, # Save checkpoint every 5 epochs
                plots=True,    # Generate standard YOLO plots
                verbose=True
            )
            
            self._post_training_hooks(model)
            
        except Exception as e:
            logger.critical(f"❌ Training Failed: {e}")
            raise e

    def _post_training_hooks(self, model):
        """Runs after training completes."""
        logger.info("📈 Processing Training Metrics...")
        
        # 1. Export to ONNX (Optimized for Production)
        logger.info("📦 Exporting to ONNX...")
        export_path = model.export(format="onnx", dynamic=False)
        logger.success(f"Model exported to: {export_path}")
        
        # 2. Validation
        metrics = model.val()
        map50 = metrics.box.map50
        logger.info(f"🏆 Final mAP@50: {map50:.4f}")
        
        if map50 < 0.5:
            logger.warning("⚠️ High model error rate detected. Consider more data.")