# XenSense 🚦 – Intelligent Video Analytics

XenSense is a real-time video analytics system for **traffic and hazard detection**, built with **YOLOv8, DeepSort, and custom ML models**.

## ✨ Features
- 🚘 Multi-object detection & segmentation (YOLOv8)
- 🎯 Real-time tracking (DeepSort)
- 🌫️ Smoke & fog detection for low-visibility monitoring
- ⚡ Speed, direction & distance estimation
- 🛣️ Road hazard detection (potholes, speed bumps)

## ⚡ Demo
![Demo](results/demo.gif)

## 📊 Results
- mAP@0.5: **0.54** (custom pothole dataset)
- 30 FPS on 1080p streams (GPU-accelerated)
- Robust detection under low-visibility conditions

## 🛠 Tech Stack
Python · PyTorch · YOLOv8 · OpenCV · DeepSort · CUDA/ROCm

## 🚀 Quick Start
```bash
git clone https://github.com/poenitens-42/Xensense_v1.git
cd Xensense_v1
pip install -r requirements.txt
python src/main.py --video sample.mp4
