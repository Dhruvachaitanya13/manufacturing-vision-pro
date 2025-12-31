import cv2
import time
import numpy as np
from ultralytics import YOLO
from loguru import logger
from .config import settings

class InferenceEngine:
    """
    Singleton Inference Engine.
    Ensures the model is loaded only once in memory.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(InferenceEngine, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.weights_path = settings.MODEL_PATH / settings.PROJECT_NAME / "weights" / "best.pt"
        self.model = None
        
        if not self.weights_path.exists():
            logger.warning(f"⚠️  No trained model found at {self.weights_path}. Inference will fail.")
            return

        try:
            logger.info(f"Loading weights from {self.weights_path}...")
            self.model = YOLO(self.weights_path)
            
            # Warmup Inference
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.predict(dummy, device=settings.DEVICE, verbose=False)
            logger.success("✅ Model Loaded & Warmed Up.")
            
        except Exception as e:
            logger.critical(f"Failed to load model: {e}")

    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        """
        Standardizes input images.
        1. Decode bytes -> BGR
        2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
           This handles variable lighting in factories.
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Could not decode image.")

        # CLAHE applied to Lightness channel of LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        return final

    def predict(self, image_bytes: bytes, filename: str) -> dict:
        if not self.model:
            raise RuntimeError("Model is not initialized.")

        start_ts = time.perf_counter()
        
        try:
            img = self.preprocess(image_bytes)
            
            # Run Inference
            results = self.model.predict(
                img, 
                conf=settings.CONF_THRESHOLD, 
                iou=settings.IOU_THRESHOLD,
                device=settings.DEVICE, 
                verbose=False
            )
            
            result = results[0]
            detections = []
            
            for box in result.boxes:
                # Extract coordinates and cast to int
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf.cpu().numpy()[0])
                cls = int(box.cls.cpu().numpy()[0])
                label = result.names[cls]
                
                detections.append({
                    "label": label,
                    "confidence": round(conf, 4),
                    "bbox": {
                        "x_min": int(x1), "y_min": int(y1), 
                        "x_max": int(x2), "y_max": int(y2)
                    }
                })

            latency = (time.perf_counter() - start_ts) * 1000
            
            return {
                "filename": filename,
                "status": "REJECT" if detections else "PASS",
                "inference_time_ms": round(latency, 2),
                "count": len(detections),
                "detections": detections
            }
            
        except Exception as e:
            logger.error(f"Inference Error on {filename}: {e}")
            raise e