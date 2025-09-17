# XenSense 🚦 – Intelligent Video Analytics

**XenSense** is a real-time video analytics system for **traffic monitoring and hazard detection**, combining **state-of-the-art deep learning models** with efficient video processing pipelines.

---

## ✨ Key Features
- 🚘 **Multi-object detection & segmentation** using **YOLO11**
- 🎯 **Robust tracking** with **DeepSort**  
- 🌫️ **Custom smoke & fog filter** for low-visibility environments  
- ⚡ **Speed, direction, and distance estimation** powered by **MiDaS depth estimation**  
- 🛣️ **Road hazard detection** for **potholes and speed bumps**

---

## ⚡ Demo
![Demo](results/demo.gif)  
*(Sample run of XenSense in action — real-time detection, tracking, and hazard identification)*

---

## 📊 Results
- **mAP@0.5:** `0.54` on custom pothole dataset  
- **Throughput:** ~30 FPS on 1080p video (GPU-accelerated)  
- **Robustness:** Handles smoke/fog and other low-visibility scenarios  

---

## 🛠 Tech Stack
- **Python** – Core language  
- **PyTorch** – Deep learning framework  
- **YOLO11 (Ultralytics)** – Object detection & segmentation  
- **DeepSort** – Multi-object tracking  
- **MiDaS** – Depth estimation (for speed & distance)  
- **OpenCV** – Video processing & visualization  
- **Custom Smoke/FogNet Filter** – Adverse weather handling  
- **CUDA / ROCm** – GPU acceleration  
- **NumPy, Pandas** – Data handling and metrics  

---

## 🚀 Quick Start
```bash
# Clone repository
git clone https://github.com/poenitens-42/Xensense_v1.git
cd Xensense_v1

# Install dependencies
pip install -r requirements.txt

# Run inference on sample video
python src/main.py --video sample.mp4
