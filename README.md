# XenSense 🚦 – Intelligent Video Analytics for Road Safety

**XenSense** is a real-time video analytics system for **traffic monitoring, depth-aware motion analysis, and road-hazard detection**, designed for autonomous-driving and smart-city applications.  
It integrates **state-of-the-art deep learning models** into a modular, efficient processing pipeline.

---

## Key Features

- **Multi-object detection and segmentation** using **YOLOv11n**
- **DeepSort tracking** with stable ID assignment and re-identification
- **Speed, distance, and direction estimation** using **MiDaS depth inference**
  - Pixel-to-meter scaling via class-based height approximation
  - Rolling history smoothing for stable outputs
- **Custom smoke and fog detection module** for low-visibility handling
- **Road hazard detection** with a **semi-functional pothole and speed-bump model**
- **Prompt-based detection** via **OwlViT**
- **Interactive helper menu**
  - Pause/resume
  - Enable/disable modules
  - Debug and visualization toggles

---

## ⚡ Demo

*(Demo coming soon — XenSense performing detection, tracking, depth-based speed estimation, and hazard recognition)*



## ⚡ Demo

*(Demo coming soon — sample run of XenSense performing detection, tracking, depth-based speed estimation, and hazard recognition)*  
You may place your GIF here:









---


---

## 📊 Results

- **Throughput:** ~30 FPS on 1080p video (GPU-accelerated)
- **Robustness:** Stable performance under smoke and fog
- **Motion Estimation:** Smooth and physically consistent using depth + temporal tracking

---

## 🛠 Tech Stack

- **Python** – Core language  
- **PyTorch** – Deep learning framework  
- **YOLOv11n (Ultralytics)** – Detection and segmentation  
- **DeepSort** – Multi-object tracking  
- **MiDaS** – Depth estimation  
- **OwlViT** – Prompt-based recognition  
- **OpenCV** – Video processing and visualization  
- **CUDA / ROCm** – GPU acceleration  
- **NumPy, Pandas** – Metrics and data handling  

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/poenitens-42/Xensense_v1.git
cd Xensense_v1

# Install dependencies
pip install -r requirements.txt

# Create data folder to store inputs(video/cam) 
mkdir -p data



