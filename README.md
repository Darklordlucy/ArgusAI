# 🚨 ArgusAI — The Third Eye for Every Rider

> **World's first offline AI road hazard detection and automatic crash response system for two-wheelers.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![YOLOv10n](https://img.shields.io/badge/Model-YOLOv10n-red.svg)](https://github.com/THU-MIG/yolov10)
[![AMD Radeon 780M](https://img.shields.io/badge/AMD-Radeon%20780M%20DirectML-ED1C24.svg)](https://www.amd.com)
[![ONNX Runtime](https://img.shields.io/badge/Inference-ONNX%20Runtime-lightgrey.svg)](https://onnxruntime.ai/)

---

## 📌 Problem Statement

India records **1.78 lakh road deaths annually**. 44% are two-wheeler riders — **78,320 deaths per year**.

- **0.8 seconds** — the human reaction window at 60 km/h. Not enough to brake safely.
- **43%** of riders die before reaching hospital due to 45–90 min rural response times.
- **<0.5%** of Indian two-wheeler riders have access to any safety device.
- Existing solutions (Ajjas, GeoRide) detect crashes **after** they happen. None see the road.

**ArgusAI solves both sides — detects before impact, responds after crash.**

---

## 💡 What is ArgusAI?

ArgusAI is a **₹6,000 edge-AI device** that mounts on any two-wheeler and runs entirely offline.

| Capability | Detail |
|---|---|
| Real-time hazard detection | Potholes, pedestrians, obstacles — 88.8% mAP |
| Pre-impact warning | 40 meters ahead, 2 seconds before impact |
| Alert system | 3-tier buzzer — 1 beep/2 beeps/3 beeps per class |
| Crash detection | G-force + speed drop via MPU6050 at 50Hz |
| Emergency SOS | GPS coordinates via 2G SMS in 33 seconds |
| False alarm protection | 30-second cancel window — zero false dispatches |
| Internet required | ❌ Zero — fully offline, 2G SMS only |
| Smartphone required | ❌ Zero — standalone device |
| Monthly subscription | ❌ Zero — one-time hardware cost |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    POWER ON                                 │
│                        ↓                                    │
│          Hardware Init (Camera + GPS + IMU + GSM)           │
│                        ↓                                    │
│          Load AI Model + Emergency Contacts                 │
│              ↙                        ↘                    │
│    DETECTION LOOP (5 FPS)      CRASH LOOP (50Hz)            │
│         ↓                            ↓                      │
│  Camera Captures Frame        IMU Detects Impact            │
│         ↓                            ↓                      │
│  AI Identifies Hazard         Speed Drop Confirmed          │
│         ↓                            ↓                      │
│   Buzzer Alert                 CRASH CONFIRMED              │
│  (1/2/3 beeps)                       ↓                      │
│                            15s Cancel Window                │
│                           ↙            ↘                   │
│                      CANCELLED      NO RESPONSE             │
│                           ↓            ↓                    │
│                        Re-arm      GPS SMS Sent             │
└─────────────────────────────────────────────────────────────┘
```

## 🤖 AI Model

| Parameter | Value |
|---|---|
| Architecture | YOLOv10n (Nano) |
| Parameters | 2.7 million |
| Export format | ONNX (`best.onnx`) |
| Input shape | `[1, 3, 640, 640]` float32 |
| Training images | 40,715 |
| Validation images | 1,372 |
| Overall mAP@0.5 | **0.888** |
| Precision | 0.886 |
| Recall | 0.794 |

### Per-Class Accuracy

| Class | mAP@0.5 | Precision | Recall |
|---|---|---|---|
| Pothole (Class 0) | 0.803 | 0.813 | 0.712 |
| Pedestrian (Class 1) | 0.897 | 0.918 | 0.769 |
| Obstacle (Class 2) | 0.964 | 0.926 | 0.904 |

### Dataset Augmentations
Rain overlay · Fog/haze · Night brightness (45-65%) · Motion blur · Horizontal flip · Mosaic

---

## 💻 Software Stack

### Edge Device (Raspberry Pi 4)
```
Raspberry Pi OS Lite 64-bit
ONNX Runtime 1.17 (CPU inference)
OpenCV 4.8 (camera capture + preprocessing)
gpsd (GPS NMEA parsing)
smbus2 (MPU6050 I2C communication)
pyserial (SIM800L AT commands)
SQLite3 (local event logging)
asyncio (concurrent process management)
```

### AI Training
```
PyTorch + CUDA (current) → PyTorch + ROCm / AMD RX Series (roadmap)
Ultralytics YOLOv10
ONNX Simplifier
MLflow (experiment tracking)
Label Studio (dataset annotation)
```

### Fleet Dashboard (AMD Radeon 780M)
```
ONNX Runtime DirectML (260.94 FPS inference)
FastAPI (REST API backend)
PostgreSQL + TimescaleDB (time-series data)
Grafana (real-time fleet monitoring)
Redis (hazard alert caching)
React + Leaflet.js (web dashboard)
```

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.9+
Raspberry Pi OS Lite 64-bit
ONNX Runtime 1.17+
OpenCV 4.8+
```

### Installation

```bash
# Clone the repository
git clone https://github.com/Darklordlucy/ArgusAI.git
cd ArgusAI

# Install dependencies
pip install -r requirements.txt

# Configure emergency contacts and system settings
cp config/config.example.json config/config.json
nano config/config.json
```

### Running on Raspberry Pi 4

```bash
# Normal boot — starts all three daemon threads
python main.py

# Config mode — opens WiFi hotspot for setup
# Hold cancel button for 5 seconds at boot
python main.py --config
```

### Running AMD Benchmark (Windows/Linux)

```bash
# AMD Radeon 780M / any DirectML-compatible GPU
pip install onnxruntime-directml

# Run benchmark
python benchmark/amd_igpu_bench.py

# Intel comparison
python benchmark/intel_igpu_bench.py
```

---

## 📁 Project Structure

```
ArgusAI/
├── main.py                    # Main entry point — spawns all daemon threads
├── inference/
│   ├── inference_loop.py      # YOLOv10n ONNX inference at 5 FPS
│   └── best.onnx              # Trained model (ONNX export)
├── crash/
│   └── crash_detector.py      # MPU6050 crash detection at 50Hz
├── gps/
│   └── gps_reader.py          # NEO-6M GPS NMEA parsing
├── comms/
│   └── sms_dispatch.py        # SIM800L AT command SMS sender
├── config/
│   └── config.json            # Emergency contacts + system config
├── benchmark/
│   ├── amd_igpu_bench.py      # AMD DirectML benchmark script
│   └── intel_igpu_bench.py    # Intel DirectML benchmark script
├── training/
│   └── train.py               # YOLOv10n training script
├── dataset/
│   └── data.yaml              # Dataset configuration
└── requirements.txt
```

---

## 🗺️ Roadmap

### Phase 1 — Complete ✅
- YOLOv10n trained at 88.8% mAP on 40,715 Indian road images
- ONNX export and edge deployment on Raspberry Pi 4
- Crash detection via MPU6050 + GPS + SIM800L SMS dispatch
- AMD Radeon 780M DirectML benchmark — 260.94 FPS verified

### Phase 2 — In Progress 🔄
- Real-world rider testing on Indian roads
- SOS parameter hardening — false alarm reduction
- Model retraining on expanded dataset for improved pothole recall
- INT8 quantization — targeting 15-18 FPS on Pi 4

### Phase 3 — Roadmap 🔮
- Crowd-sourced hazard map via fleet API
- Insurance telematics scoring API
- AMD Ryzen Embedded V2 hardware (pending AMD provision)
- AMD EPYC cloud fleet intelligence server
- AMD Instinct continuous retraining pipeline
- OTA model updates via GSM

---

## 👥 Team Arise

| Member | Role | LinkedIn |
|---|---|---|
| K Sai Uallash Reddy | AI & Backend Lead — YOLOv10n training, ONNX export, inference pipeline, FastAPI, AMD benchmark | [LinkedIn](https://www.linkedin.com/in/uallasreddy) |
| Saideep Paladi | Hardware & Embedded Lead — Circuit design, MPU6050, GPS, SIM800L, GPIO/I2C/UART | [LinkedIn](https://www.linkedin.com/in/saideep-paladi) |
| Umang Pawar | Data & Backend Engineer — 40,715 image dataset, augmentation pipeline, FastAPI, SQL | [LinkedIn](https://www.linkedin.com/in/umangpawar) |

---

## 🏆 AMD Slingshot Hackathon 2024

This project was built for the AMD Slingshot Hackathon.

> *"We don't use AMD — we build on AMD. Every layer of ArgusAI, from edge inference to cloud retraining, is architected around the AMD ecosystem."*

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## ⭐ Star this repo if ArgusAI could save a life on Indian roads.
