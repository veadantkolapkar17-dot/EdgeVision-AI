# EdgeVision AI

> **EDGE AI-BASED REAL-TIME OBJECT DETECTION FOR SMART MEDIA APPLICATIONS**  
> *2nd-Year B.Tech Computer Science & Engineering Project*

---

## 📌 Project Overview

**EdgeVision AI** is a client-side computer vision application demonstrating on-device Edge Artificial Intelligence. It runs the state-of-the-art **YOLO11n** neural network directly within the browser using **ONNX Runtime Web** with hardware acceleration via **WebGPU** and **WebAssembly (WASM SIMD)**.

Unlike conventional cloud-based AI systems, **no video feeds, camera streams, or uploaded media are ever sent to an external server**. All preprocessing, tensor inference, Non-Maximum Suppression (NMS), and analytics tracking execute 100% locally on the user's edge device.

---

## 🚀 Key Modules & Capabilities

1. **Dashboard & Workflow Overview**
   - Live session telemetry: Total detections, unique classes, real-time FPS, inference latency, engine status, and snapshot counter.
   - Visual workflow pipeline showing the path from media input to local analytics.
2. **Unified Detection Workspace**
   - **Live Webcam Mode:** Real-time camera feed with unmirrored, pixel-aligned bounding boxes and dynamic aspect ratio matching (4:3, 16:9, etc.).
   - **Static Image Detection Mode:** Local drag-and-drop or browse image upload with instant YOLO11n inference, object summary, and detection stats.
   - **Video File Detection Mode:** Local video playback with frame-by-frame throttled detection, play/pause controls, and live telemetry without UI freezing.
3. **Session Analytics & Insights**
   - Real-time tracking of detection events, unique classes, and most frequently detected object.
   - Temporal cooldown filter (requires 3 consecutive frames and a 2-second cooldown per class) to avoid raw frame-rate spam.
   - Live frequency bar chart and reverse-chronological detection stream with timestamps.
   - Export detection history to CSV and clear history controls.
4. **Edge AI Architecture & Privacy**
   - Step-by-step architectural breakdown comparing Edge AI with Cloud AI.
   - Clear privacy guarantee: Media processing is entirely on-device, preserving user data privacy and requiring zero cloud bandwidth.
5. **Universal Model & Settings Center**
   - Pretrained YOLO11n default model with automatic hardware acceleration (WebGPU $\to$ WASM fallback).
   - Universal YOLO output adapter supporting both transposed `[1, 4+C, N]` and non-transposed `[1, N, 4+C]` formats.
   - Optional custom ONNX model loader with dry-run validation, custom class labels parsing, and safe one-click reset to YOLO11n.
   - Two-way synchronized confidence slider and persistent local configuration via `localStorage`.
6. **Universal Snapshot Capture**
   - Captures active detection frame (camera, image, or video) with overlaid bounding boxes, confidence tags, and timestamp watermark into a downloadable PNG.

---

## 🛠️ Technology Stack

| Component | Technology |
| :--- | :--- |
| **Frontend UI** | HTML5, CSS3 (Modern Flexbox & CSS Grid), Responsive Dark Theme |
| **Logic & Pipeline** | Vanilla JavaScript (ES6+ Modules, Zero-allocation Buffers) |
| **Neural Network** | Ultralytics YOLO11n (Pretrained COCO 80 classes) |
| **Inference Engine** | ONNX Runtime Web (`ort.all.min.mjs` 1.21.0) |
| **Hardware Accel** | WebGPU API (Primary) with WebAssembly SIMD (Fallback) |
| **Camera Feed** | WebRTC `navigator.mediaDevices.getUserMedia` |

---

## 💻 How to Run Locally

1. **Extract or Clone Repository:**
   Ensure `index.html`, `app.js`, and `style.css` are in the project folder.
2. **Launch with Local HTTP Server:**
   Because browser security policies require a secure context (`localhost` or `HTTPS`) for camera access and WebGPU:
   - **Using VS Code:** Right-click `index.html` and select **Open with Live Server**.
   - **Using Python:** Run in terminal:
     ```bash
     python -m http.server 8080
     ```
     Then open `http://localhost:8080/index.html` in Chrome or Edge.
3. **Grant Camera Permission:** Click **Start Camera** in the Detection Workspace and allow access.

---

## 🎯 Recommended Demonstration (Viva Demo Flow)

1. **Dashboard & Concept:**
   - Open the app, highlight the project title and academic badge.
   - Explain the 6-step Edge AI processing pipeline and zero-cloud architecture.
2. **Live Webcam Detection:**
   - Click **Start Detection** to open the workspace.
   - Click **Start Camera**, show real-time detection boxes around objects (person, phone, cup, laptop, etc.).
   - Point out the real-time FPS counter, inference latency in milliseconds, and active engine (`WebGPU` or `WASM CPU`).
   - Adjust the confidence slider from 55% to 75% to show live thresholding.
   - Click **Snapshot** to download a high-resolution framed PNG.
3. **Smart Media Modes:**
   - Switch to **Image Detection**, drag and drop a test picture, click **Run Detection**, and show the bounding boxes and detection time.
   - Switch to **Video Detection**, upload a sample video clip, and demonstrate frame-by-frame on-device inference.
4. **Session Analytics:**
   - Scroll to **Analytics**, review the Total Events counter, Most Detected Object, and Top Classes bar chart.
   - Click **Export CSV** to demonstrate local data reporting.
5. **Architecture & Settings:**
   - Walk through the **Architecture** section explaining on-device privacy, zero bandwidth, and WebGPU $\to$ WASM fallback.
   - In **Settings**, demonstrate the Model Specifications Panel displaying input shape `[1, 3, 640, 640]`, output shape `[1, 84, 8400]`, and active classes.

---

## 📄 License & Attribution

Developed for academic presentation as a 2nd-Year B.Tech Computer Science & Engineering Project.
Powered by Ultralytics YOLO11 and ONNX Runtime Web.
