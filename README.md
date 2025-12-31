# 🏭 Manufacturing Vision Pro

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.103-green.svg)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**An Enterprise-Grade Automated Visual Inspection System.** Designed to detect manufacturing defects (scratches, stains, pits) in real-time using Deep Learning.

---

## 📋 Overview

Manufacturing Vision Pro is an end-to-end MLOps pipeline designed to automate quality control on high-speed conveyor belts. It leverages **YOLOv8** for object detection and **FastAPI** for serving predictions via a microservice architecture.

Unlike standard tutorials, this project features:
* **Synthetic Data Factory:** Generates realistic training data with simulated lighting, textures, and defects.
* **Hardware Acceleration:** Auto-detects **Apple Silicon (MPS)**, NVIDIA CUDA, or CPU.
* **Advanced Preprocessing:** Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) to handle variable factory lighting.
* **Production Standards:** Includes Pydantic validation, structured logging, and Docker containerization.

---

## 🚀 Key Features

* **Real-Time Inference:** <15ms latency on GPU.
* **Defect Classification:** Distinguishes between scratches, stains, and pits.
* **RESTful API:** Swagger UI documentation included.
* **CI/CD Ready:** Type-safe code with `pydantic` and automated testing suites.
* **Explainable AI:** Generates heatmaps and bounding boxes for visual verification.

---

## 🛠️ Tech Stack

* **Core:** Python 3.10, OpenCV, NumPy
* **ML Framework:** PyTorch, Ultralytics YOLOv8
* **API:** FastAPI, Uvicorn, Pydantic
* **Data Augmentation:** Albumentations
* **Deployment:** Docker
* **Logging:** Loguru

---

## ⚡ Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/yourusername/manufacturing-vision-pro.git](https://github.com/yourusername/manufacturing-vision-pro.git)
    cd manufacturing-vision-pro
    ```

2.  **Install Dependencies**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

---

## 🏃‍♂️ How to Run

### 1. Generate Synthetic Data
Simulate a factory environment by creating 500 realistic training images.
```bash
python main.py gen --train 500 --val 100