import cv2
import numpy as np
import yaml
import random
import albumentations as A
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from .config import settings

class AdvancedDataFactory:
    """
    Generates synthetic industrial datasets.
    Simulates:
    1. Base Material: Brushed Metal, Plastic, Ceramic textures.
    2. Defects: Linear Scratches, Circular Pits, Surface Stains.
    3. Noise: Sensor noise, lighting variations.
    """
    
    def __init__(self, num_train: int = 500, num_val: int = 100):
        self.num_train = num_train
        self.num_val = num_val
        self.dataset_dir = settings.DATA_PATH / "dataset"
        
        # Initialize Albumentations pipeline for realism
        self.augmentor = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.MotionBlur(p=0.1) # Simulate conveyor belt movement
        ])
        
        self._setup_directories()

    def _setup_directories(self):
        for split in ['train', 'val']:
            (self.dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    def _generate_texture(self) -> np.ndarray:
        """Creates a brushed metal texture."""
        img = np.zeros((settings.IMG_SIZE, settings.IMG_SIZE, 3), dtype=np.uint8)
        img[:] = (180, 180, 180) # Grey base
        
        # Add grain
        noise = np.random.randn(settings.IMG_SIZE, settings.IMG_SIZE, 3) * 20
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        
        # Add simulated lighting gradient
        for i in range(settings.IMG_SIZE):
            img[i, :, :] = np.clip(img[i, :, :] - (i / settings.IMG_SIZE) * 50, 0, 255)
            
        return img

    def _add_defect(self, img: np.ndarray) -> tuple[np.ndarray, str]:
        """Adds a random defect (Scratch or Stain) and returns label."""
        defect_type = random.choice(['scratch', 'stain'])
        h, w = settings.IMG_SIZE, settings.IMG_SIZE
        
        x_c, y_c, bbox_w, bbox_h = 0, 0, 0, 0
        
        if defect_type == 'scratch':
            # Draw a line
            x1, y1 = random.randint(50, w-50), random.randint(50, h-50)
            x2 = x1 + random.randint(-100, 100)
            y2 = y1 + random.randint(-100, 100)
            cv2.line(img, (x1, y1), (x2, y2), (50, 50, 50), 2)
            
            # BBox logic
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            bbox_w = (x_max - x_min + 10) / w
            bbox_h = (y_max - y_min + 10) / h
            x_c = ((x_min + x_max) / 2) / w
            y_c = ((y_min + y_max) / 2) / h

        elif defect_type == 'stain':
            # Draw a circle
            center = (random.randint(50, w-50), random.randint(50, h-50))
            radius = random.randint(10, 30)
            cv2.circle(img, center, radius, (100, 100, 100), -1)
            
            # BBox logic
            x_c = center[0] / w
            y_c = center[1] / h
            bbox_w = (radius * 2 + 5) / w
            bbox_h = (radius * 2 + 5) / h

        # Clamp values
        x_c, y_c = np.clip([x_c, y_c], 0, 1)
        bbox_w, bbox_h = np.clip([bbox_w, bbox_h], 0, 1)

        # Class 0 = Defect
        return img, f"0 {x_c:.6f} {y_c:.6f} {bbox_w:.6f} {bbox_h:.6f}"

    def generate(self):
        logger.info("🏭 initializing Factory Data Simulation...")
        
        splits = {
            'train': self.num_train, 
            'val': self.num_val
        }
        
        for split, count in splits.items():
            img_dir = self.dataset_dir / "images" / split
            lbl_dir = self.dataset_dir / "labels" / split
            
            for i in tqdm(range(count), desc=f"Generating {split}"):
                img = self._generate_texture()
                
                # Apply Augmentation
                augmented = self.augmentor(image=img)['image']
                
                # 40% chance of defect
                if random.random() < 0.4:
                    augmented, annotation = self._add_defect(augmented)
                    with open(lbl_dir / f"{split}_{i}.txt", "w") as f:
                        f.write(annotation)
                else:
                    # Negative sample (Empty file)
                    (lbl_dir / f"{split}_{i}.txt").touch()

                cv2.imwrite(str(img_dir / f"{split}_{i}.jpg"), augmented)

        self._write_yaml()
        logger.success("✅ Dataset Generation Complete.")

    def _write_yaml(self):
        config = {
            "path": str(self.dataset_dir.absolute()),
            "train": "images/train",
            "val": "images/val",
            "names": {0: "defect"}
        }
        with open(settings.DATA_PATH / "data.yaml", "w") as f:
            yaml.dump(config, f)