import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
from typing import List

def plot_training_results(results_csv: Path, save_dir: Path):
    """
    Generates training loss and accuracy curves from YOLO logs.
    Essential for analyzing model convergence.
    """
    try:
        if not results_csv.exists():
            logger.warning(f"Results file not found at {results_csv}")
            return
            
        # Parse CSV manually to avoid pandas dependency for just plotting
        import pandas as pd
        df = pd.read_csv(results_csv)
        df.columns = [x.strip() for x in df.columns]
        
        plt.figure(figsize=(12, 6))
        
        # Plot Loss
        plt.subplot(1, 2, 1)
        # Check for correct column names (YOLOv8 standard)
        if 'train/box_loss' in df.columns:
            plt.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
        if 'val/box_loss' in df.columns:
            plt.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
        plt.title('Loss Convergence')
        plt.xlabel('Epochs')
        plt.legend()
        
        # Plot mAP
        plt.subplot(1, 2, 2)
        if 'metrics/mAP50(B)' in df.columns:
            plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@50')
        plt.title('Mean Average Precision (mAP)')
        plt.xlabel('Epochs')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_dir / "training_metrics.png")
        plt.close()
        logger.info(f"Training plots saved to {save_dir}")
        
    except Exception as e:
        logger.error(f"Failed to plot training metrics: {e}")

def draw_detections(img: np.ndarray, detections: List[dict]) -> np.ndarray:
    """
    Overlays bounding boxes and labels onto the image.
    Used for debug/visualization endpoints.
    """
    annotated = img.copy()
    for det in detections:
        bbox = det['bbox']
        # Handle bbox structure (could be dict or object depending on schema)
        if isinstance(bbox, dict):
            x1, y1 = int(bbox['x_min']), int(bbox['y_min'])
            x2, y2 = int(bbox['x_max']), int(bbox['y_max'])
        else:
            x1, y1 = int(bbox.x_min), int(bbox.y_min)
            x2, y2 = int(bbox.x_max), int(bbox.y_max)
        
        # Draw Box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Draw Label
        label = f"{det['label']} {det['confidence']:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - 20), (x1 + w, y1), (0, 0, 255), -1)
        cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    return annotated