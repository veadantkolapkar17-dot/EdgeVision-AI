# EdgeVision AI V5

Edge AI-based real-time object detection for Smart Media Applications.

## V5 features
- Live webcam object detection with YOLO11n + ONNX Runtime Web
- Browser-side Edge AI inference using WebGPU/WASM
- Bounding boxes, confidence, object count, FPS and latency
- Detection analytics and history with CSV export
- Settings for confidence, inference engine and camera selection
- Local settings saved in browser storage
- **Custom YOLO ONNX model loader** for future trained-model integration
- Custom class-name mapping so the app can support a project-specific label set

## Run locally
1. Extract the ZIP.
2. Open the folder in VS Code.
3. Use Live Server on `index.html`.
4. Allow camera permission.
5. Open Settings to configure the detector.

## Custom model workflow
When your custom model is ready, export it as a compatible YOLO detection ONNX model. In Settings:
1. Stop the camera.
2. Choose the `.onnx` file.
3. Enter one class name per line in the exact training order.
4. Click **Load Custom Model**.
5. Start the camera again.

The current decoder supports standard YOLO detection output shaped like `[1, 4 + classCount, anchors]`. A custom model with a different architecture/output format may require a decoder update.

## Important
The pretrained YOLO11n model can still confuse visually similar classes. The custom-model feature is intended for the project's later training phase rather than artificially changing labels.
