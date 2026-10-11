import * as ort from "./vendor/ort.all.min.mjs";

// Set local WASM binaries path for offline capability
if (ort?.env?.wasm) {
  ort.env.wasm.wasmPaths = "./vendor/";
}

// Local bundled model with remote CDN fallback
const LOCAL_MODEL_URL = "./models/yolo11n.onnx";
const REMOTE_MODEL_URL =
  "https://huggingface.co/webnn/yolo11n/resolve/main/onnx/yolo11n.onnx?download=true";

const DEFAULT_MODEL_URL = LOCAL_MODEL_URL;

let activeModelUrl = DEFAULT_MODEL_URL;
let activeClassNames = null;
let activeModelName = "YOLO11n";
let activeOutputDims = [1, 84, 8400];
let activeLayoutInfo = { layout: "A", classCount: 80, anchors: 8400, channels: 84, dims: [1, 84, 8400] };
let activeEngineName = "WebGPU";

const COCO = [
  "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
  "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
  "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
  "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
  "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
  "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
  "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
  "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
  "hair drier","toothbrush"
];

// ---------------- DOM References ----------------
// Navigation & PWA Controls
const navStatusEl = document.querySelector("#navStatus");
const statusEl = document.querySelector("#engineStatus");
const netStatusEl = document.querySelector("#networkStatus");
const pwaInstallBtn = document.querySelector("#pwaInstallBtn");
const heroInstallBtn = document.querySelector("#heroInstallBtn");
const menuToggleBtn = document.querySelector("#menuToggleBtn");
const mainNav = document.querySelector("#mainNav");
const workspaceAlert = document.querySelector("#workspaceAlert");
const alertMessage = document.querySelector("#alertMessage");
const dismissAlertBtn = document.querySelector("#dismissAlertBtn");

// PWA Modals & Banners
const pwaUpdateBanner = document.querySelector("#pwaUpdateBanner");
const pwaReloadBtn = document.querySelector("#pwaReloadBtn");
const pwaDismissBannerBtn = document.querySelector("#pwaDismissBannerBtn");
const iosInstallModal = document.querySelector("#iosInstallModal");
const closeIosModalBtn = document.querySelector("#closeIosModalBtn");
const ackIosModalBtn = document.querySelector("#ackIosModalBtn");
const toastContainer = document.querySelector("#toastContainer");
const pwaCacheStatus = document.querySelector("#pwaCacheStatus");
const storageQuotaStatus = document.querySelector("#storageQuotaStatus");

// Dashboard Live Metrics
const dashTotal = document.querySelector("#dashTotal");
const dashClasses = document.querySelector("#dashClasses");
const dashFps = document.querySelector("#dashFps");
const dashLatency = document.querySelector("#dashLatency");
const dashEngine = document.querySelector("#dashEngine");
const dashSnapshots = document.querySelector("#dashSnapshots");

// Workspace Tabs
const tabCameraBtn = document.querySelector("#tabCameraBtn");
const tabImageBtn = document.querySelector("#tabImageBtn");
const tabVideoBtn = document.querySelector("#tabVideoBtn");
const viewCamera = document.querySelector("#viewCamera");
const viewImage = document.querySelector("#viewImage");
const viewVideo = document.querySelector("#viewVideo");

// Camera Mode
const video = document.querySelector("#video");
const canvas = document.querySelector("#overlay");
const ctx = canvas.getContext("2d");
const cameraWrap = document.querySelector("#cameraWrap");
const placeholder = document.querySelector("#placeholder");
const startBtn = document.querySelector("#startBtn");
const stopBtn = document.querySelector("#stopBtn");
const flipCamBtn = document.querySelector("#flipCamBtn");
const snapshotBtn = document.querySelector("#snapshotBtn");
const confSlider = document.querySelector("#confidence");
const confValue = document.querySelector("#confidenceValue");
const gpuToggle = document.querySelector("#gpuToggle");

// Image Mode
const imageDropzone = document.querySelector("#imageDropzone");
const imageFileInput = document.querySelector("#imageFileInput");
const imageWrap = document.querySelector("#imageWrap");
const imagePreview = document.querySelector("#imagePreview");
const imageCanvas = document.querySelector("#imageCanvas");
const imageCtx = imageCanvas.getContext("2d");
const imageControls = document.querySelector("#imageControls");
const runImageBtn = document.querySelector("#runImageBtn");
const clearImageBtn = document.querySelector("#clearImageBtn");
const snapshotImageBtn = document.querySelector("#snapshotImageBtn");
const imageStats = document.querySelector("#imageStats");

// Video Mode
const videoDropzone = document.querySelector("#videoDropzone");
const videoFileInput = document.querySelector("#videoFileInput");
const videoWrap = document.querySelector("#videoWrap");
const fileVideo = document.querySelector("#fileVideo");
const videoCanvas = document.querySelector("#videoCanvas");
const videoCtx = videoCanvas.getContext("2d");
const videoControls = document.querySelector("#videoControls");
const startVideoDetectBtn = document.querySelector("#startVideoDetectBtn");
const stopVideoDetectBtn = document.querySelector("#stopVideoDetectBtn");
const clearVideoBtn = document.querySelector("#clearVideoBtn");
const snapshotVideoBtn = document.querySelector("#snapshotVideoBtn");
const videoStats = document.querySelector("#videoStats");

// Telemetry & Results
const latencyEl = document.querySelector("#latency");
const fpsEl = document.querySelector("#fps");
const objectCountEl = document.querySelector("#objectCount");
const engineEl = document.querySelector("#engine");
const detectionsEl = document.querySelector("#detections");

// Analytics
const totalEl = document.querySelector("#analyticsTotal");
const classesEl = document.querySelector("#analyticsClasses");
const snapshotsEl = document.querySelector("#analyticsSnapshots");
const topObjectEl = document.querySelector("#analyticsTopObject");
const barsEl = document.querySelector("#analyticsBars");
const historyEl = document.querySelector("#history");
const exportBtn = document.querySelector("#exportHistory");
const clearBtn = document.querySelector("#clearHistory");

// Settings
const settingsConfidence = document.querySelector("#settingsConfidence");
const settingsConfidenceValue = document.querySelector("#settingsConfidenceValue");
const engineSelect = document.querySelector("#engineSelect");
const cameraSelect = document.querySelector("#cameraSelect");
const saveSettingsBtn = document.querySelector("#saveSettings");
const resetSettingsBtn = document.querySelector("#resetSettingsBtn");
const settingsClearHistoryBtn = document.querySelector("#settingsClearHistoryBtn");
const settingsSaved = document.querySelector("#settingsSaved");

// AI Model & Custom Loader
const customModelInput = document.querySelector("#customModelInput");
const chooseModelLabel = document.querySelector("#chooseModelLabel");
const customClassesInput = document.querySelector("#customClasses");
const loadCustomModelBtn = document.querySelector("#loadCustomModel");
const resetModelBtn = document.querySelector("#resetModelBtn");
const customModelStatus = document.querySelector("#customModelStatus");
const modelState = document.querySelector("#modelState");
const defaultModelCard = document.querySelector("#defaultModelCard");
const customModelCard = document.querySelector("#customModelCard");
const defaultModelBadge = document.querySelector("#defaultModelBadge");

// Model Information Panel
const infoModelName = document.querySelector("#infoModelName");
const infoEngine = document.querySelector("#infoEngine");
const infoInputShape = document.querySelector("#infoInputShape");
const infoOutputShape = document.querySelector("#infoOutputShape");
const infoLayout = document.querySelector("#infoLayout");
const infoClassesCount = document.querySelector("#infoClassesCount");
const infoClassesList = document.querySelector("#infoClassesList");
const infoStatusPill = document.querySelector("#infoStatusPill");

// ---------------- State ----------------
let session = null;
let stream = null;
let cameraRunning = false;
let videoDetecting = false;
let busy = false;
let fpsFrames = 0;
let fpsLast = performance.now();
let currentDetections = [];
let activeMode = "camera"; // "camera" | "image" | "video"
let loadedImageDetections = [];
let videoFileUrl = null;

// Preprocessing persistent buffers (Zero-allocation)
const PREPROCESS_SIZE = 640;
const preprocessCanvas = typeof OffscreenCanvas !== "undefined"
  ? new OffscreenCanvas(PREPROCESS_SIZE, PREPROCESS_SIZE)
  : document.createElement("canvas");
preprocessCanvas.width = PREPROCESS_SIZE;
preprocessCanvas.height = PREPROCESS_SIZE;
const preprocessCtx = preprocessCanvas.getContext("2d", { willReadFrequently: true });
const inputTensorBuffer = new Float32Array(1 * 3 * PREPROCESS_SIZE * PREPROCESS_SIZE);

// Analytics State
const evState = {
  total: 0,
  classes: {},
  history: [],
  snapshots: 0,
  lastSignature: "",
  stableFrames: 0,
  lastRecordedAt: {}
};

// ---------------- Alerts & Status ----------------
function showAlert(message, isDanger = true) {
  if (!workspaceAlert) return;
  workspaceAlert.className = `alert-box ${isDanger ? 'danger' : 'info'}`;
  alertMessage.textContent = message;
  workspaceAlert.style.display = "flex";
}

function hideAlert() {
  if (workspaceAlert) workspaceAlert.style.display = "none";
}

dismissAlertBtn?.addEventListener("click", hideAlert);

function setStatus(text, good = true) {
  if (statusEl) {
    statusEl.textContent = `● ${text}`;
    statusEl.style.color = good ? "var(--accent)" : "var(--danger)";
  }
  if (navStatusEl) {
    navStatusEl.textContent = text;
    navStatusEl.style.color = good ? "var(--accent)" : "var(--danger)";
  }
}

function updateTelemetry(ms, fps, count, engineName) {
  if (latencyEl && ms !== undefined) latencyEl.textContent = `${Math.round(ms)} ms`;
  if (fpsEl && fps !== undefined) fpsEl.textContent = String(fps);
  if (objectCountEl && count !== undefined) objectCountEl.textContent = String(count);
  if (engineEl && engineName) engineEl.textContent = engineName;

  // Sync with Dashboard Metrics
  if (dashFps && fps !== undefined) dashFps.textContent = String(fps);
  if (dashLatency && ms !== undefined) dashLatency.textContent = `${Math.round(ms)} ms`;
  if (dashEngine && engineName) dashEngine.textContent = engineName;
  if (dashTotal) dashTotal.textContent = String(evState.total);
  if (dashClasses) dashClasses.textContent = String(Object.keys(evState.classes).length);
  if (dashSnapshots) dashSnapshots.textContent = String(evState.snapshots);
}

// ---------------- Two-way Confidence Synchronization ----------------
function syncConfidence(value) {
  const v = Number(value);
  if (confSlider) confSlider.value = String(v);
  if (confValue) confValue.textContent = `${Math.round(v * 100)}%`;
  if (settingsConfidence) settingsConfidence.value = String(v);
  if (settingsConfidenceValue) settingsConfidenceValue.textContent = `${Math.round(v * 100)}%`;
}

confSlider?.addEventListener("input", (e) => syncConfidence(e.target.value));
settingsConfidence?.addEventListener("input", (e) => syncConfidence(e.target.value));

// ---------------- Model Loading & Provider Fallback ----------------
async function createSessionWithFallback(modelSource, preferredEngine) {
  const hasWebGpu = typeof navigator !== "undefined" && Boolean(navigator.gpu);
  const wantWebGpu = preferredEngine !== "wasm" && hasWebGpu;

  async function attemptCreate(source, ep) {
    try {
      return await ort.InferenceSession.create(source, {
        executionProviders: [ep],
        graphOptimizationLevel: "all"
      });
    } catch (err) {
      if (typeof source === "string" && source === LOCAL_MODEL_URL) {
        console.warn(`Local model (${LOCAL_MODEL_URL}) failed, attempting remote fallback:`, err);
        return await ort.InferenceSession.create(REMOTE_MODEL_URL, {
          executionProviders: [ep],
          graphOptimizationLevel: "all"
        });
      }
      throw err;
    }
  }

  if (wantWebGpu) {
    try {
      const sess = await attemptCreate(modelSource, "webgpu");
      return { session: sess, provider: "webgpu" };
    } catch (gpuErr) {
      console.warn("WebGPU initialization failed, falling back to WASM CPU:", gpuErr);
      setStatus("WebGPU unavailable, using CPU", true);
      showToast("WebGPU unavailable; falling back to CPU / WASM SIMD", false);
    }
  }

  // CPU / WASM fallback
  const sess = await attemptCreate(modelSource, "wasm");
  return { session: sess, provider: "wasm" };
}

// ---------------- Universal YOLO Output Tensor Adapter ----------------
function inspectYoloTensor(dims) {
  if (!dims || dims.length < 2) return null;
  const d0 = dims[dims.length - 2];
  const d1 = dims[dims.length - 1];

  // Layout A: [..., 4 + C, N]
  // Layout B: [..., N, 4 + C]
  if (d0 < d1 && d0 > 4) {
    const classCount = d0 - 4;
    const anchors = d1;
    return { layout: "A", classCount, anchors, channels: d0, dims };
  } else if (d1 < d0 && d1 > 4) {
    const classCount = d1 - 4;
    const anchors = d0;
    return { layout: "B", classCount, anchors, channels: d1, dims };
  }
  return null;
}

function parseClassNames(text) {
  if (!text) return [];
  const trimmed = text.trim();
  if (trimmed.startsWith("[") && trimmed.endsWith("]")) {
    try {
      const arr = JSON.parse(trimmed);
      if (Array.isArray(arr) && arr.length > 0) {
        return arr.map(x => String(x).trim()).filter(Boolean);
      }
    } catch {}
  }
  const lines = trimmed.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
  const result = [];
  for (const line of lines) {
    if (line.includes(",") && !/^\d+\s*[:=]/.test(line)) {
      const parts = line.split(",").map(p => p.trim().replace(/^['"]|['"]$/g, "")).filter(Boolean);
      result.push(...parts);
    } else {
      const cleaned = line.replace(/^(\d+\s*[:=\-.]\s*|#\d+\s*)/, "").trim().replace(/^['"]|['"]$/g, "");
      if (cleaned) result.push(cleaned);
    }
  }
  return result;
}

function renderModelInfo() {
  const isCustom = Boolean(activeClassNames && activeClassNames.length > 0 && activeModelUrl !== DEFAULT_MODEL_URL);
  const names = isCustom ? activeClassNames : COCO;
  const layoutDesc = activeLayoutInfo
    ? (activeLayoutInfo.layout === "A" ? "Layout A [1, 4+C, N]" : "Layout B [1, N, 4+C]")
    : "Layout A [1, 4+C, N]";

  if (infoModelName) infoModelName.textContent = activeModelName;
  if (infoEngine) infoEngine.textContent = activeEngineName;
  if (infoInputShape) infoInputShape.textContent = "1 × 3 × 640 × 640";
  if (infoOutputShape) {
    infoOutputShape.textContent = activeOutputDims ? `[${activeOutputDims.join(", ")}]` : `[1, ${4 + names.length}, 8400]`;
  }
  if (infoLayout) infoLayout.textContent = layoutDesc;
  if (infoClassesCount) infoClassesCount.textContent = `${names.length} classes`;
  if (infoStatusPill) {
    infoStatusPill.textContent = isCustom ? "Custom Active" : "Pretrained Active";
    infoStatusPill.style.color = "var(--accent)";
  }

  if (infoClassesList) {
    infoClassesList.innerHTML = names.slice(0, 40).map((name, i) => `
      <span class="class-tag" title="Index ${i}: ${escapeHtml(name)}">${i}: ${escapeHtml(name)}</span>
    `).join("") + (names.length > 40 ? `<span class="class-tag" style="opacity:0.7">+${names.length - 40} more</span>` : "");
  }
}

async function loadModel() {
  if (!activeModelUrl) {
    activeModelUrl = DEFAULT_MODEL_URL;
    activeClassNames = null;
    activeModelName = "YOLO11n";
    if (modelState) modelState.textContent = "YOLO11n ACTIVE";
  }
  setStatus(`Loading ${activeModelName}…`);
  if (engineEl) engineEl.textContent = "Loading";
  if (pwaCacheStatus) {
    pwaCacheStatus.textContent = "Loading model into memory…";
    pwaCacheStatus.style.color = "var(--muted)";
  }

  const requestedEngine = engineSelect?.value || (gpuToggle?.checked ? "auto" : "wasm");

  try {
    const result = await createSessionWithFallback(activeModelUrl, requestedEngine);
    session = result.session;
    const isGpu = result.provider === "webgpu";
    activeEngineName = isGpu ? "WebGPU" : "WASM CPU";
    updateTelemetry(undefined, undefined, undefined, activeEngineName);
    setStatus(isGpu ? "WebGPU Edge AI Ready" : "CPU Edge AI Ready");
    if (pwaCacheStatus) {
      pwaCacheStatus.textContent = "✓ YOLO11n Cached (100% Offline Ready)";
      pwaCacheStatus.style.color = "var(--accent)";
    }

    try {
      const dummyTensor = new ort.Tensor("float32", inputTensorBuffer, [1, 3, PREPROCESS_SIZE, PREPROCESS_SIZE]);
      const inputName = session.inputNames[0];
      const outputMap = await session.run({ [inputName]: dummyTensor });
      const output = outputMap[session.outputNames[0]];
      if (output && output.dims) {
        activeOutputDims = output.dims;
        activeLayoutInfo = inspectYoloTensor(output.dims) || activeLayoutInfo;
      }
    } catch (dryErr) {
      console.warn("Dry run inspection:", dryErr);
    }
    renderModelInfo();
  } catch (err) {
    console.error("Failed to load model:", err);
    setStatus("Model load failed", false);
    if (engineEl) engineEl.textContent = "Error";
    if (pwaCacheStatus) {
      pwaCacheStatus.textContent = "⚠️ Model load failed";
      pwaCacheStatus.style.color = "var(--danger)";
    }
    showAlert("Model failed to load. Check that model and vendor assets are available or cached.", true);
    throw err;
  }
}

// ---------------- Generic Preprocessing & Universal Decoder ----------------
function letterbox(sourceW, sourceH, target = PREPROCESS_SIZE) {
  const scale = Math.min(target / sourceW, target / sourceH);
  const w = Math.round(sourceW * scale);
  const h = Math.round(sourceH * scale);
  return { scale, w, h, dx: (target - w) / 2, dy: (target - h) / 2 };
}

function preprocessMedia(sourceElement, srcW, srcH) {
  const lb = letterbox(srcW, srcH, PREPROCESS_SIZE);
  preprocessCtx.fillStyle = "#000";
  preprocessCtx.fillRect(0, 0, PREPROCESS_SIZE, PREPROCESS_SIZE);
  preprocessCtx.drawImage(sourceElement, lb.dx, lb.dy, lb.w, lb.h);

  const imgData = preprocessCtx.getImageData(0, 0, PREPROCESS_SIZE, PREPROCESS_SIZE).data;
  const planeSize = PREPROCESS_SIZE * PREPROCESS_SIZE;

  for (let i = 0, p = 0; p < planeSize; i += 4, p++) {
    inputTensorBuffer[p] = imgData[i] / 255;
    inputTensorBuffer[planeSize + p] = imgData[i + 1] / 255;
    inputTensorBuffer[2 * planeSize + p] = imgData[i + 2] / 255;
  }
  return { input: inputTensorBuffer, lb };
}

function iou(a, b) {
  const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
  const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
  return inter / Math.max(1e-6, areaA + areaB - inter);
}

function nms(boxes, threshold = 0.45) {
  const sorted = [...boxes].sort((a, b) => b.score - a.score);
  const keep = [];
  while (sorted.length) {
    const best = sorted.shift();
    keep.push(best);
    for (let i = sorted.length - 1; i >= 0; i--) {
      if (sorted[i].label === best.label && iou(best, sorted[i]) > threshold) {
        sorted.splice(i, 1);
      }
    }
  }
  return keep;
}

function decode(output, lb, sourceW, sourceH) {
  const data = output.data;
  const dims = output.dims || [1, 84, 8400];
  const info = inspectYoloTensor(dims);

  const layout = info ? info.layout : "A";
  const classCount = info ? info.classCount : (dims[dims.length - 2] - 4);
  const anchors = info ? info.anchors : dims[dims.length - 1];

  const names = (activeClassNames && activeClassNames.length > 0)
    ? activeClassNames
    : (classCount === 80 ? COCO : []);

  const candidates = [];
  const threshold = Number(confSlider.value);

  if (layout === "A") {
    for (let a = 0; a < anchors; a++) {
      let bestClass = -1, bestScore = 0;
      for (let c = 0; c < classCount; c++) {
        const score = data[(4 + c) * anchors + a];
        if (score > bestScore) {
          bestScore = score;
          bestClass = c;
        }
      }
      if (bestScore < threshold) continue;

      const cx = data[a];
      const cy = data[anchors + a];
      const w = data[2 * anchors + a];
      const h = data[3 * anchors + a];

      let x1 = (cx - w / 2 - lb.dx) / lb.scale;
      let y1 = (cy - h / 2 - lb.dy) / lb.scale;
      let x2 = (cx + w / 2 - lb.dx) / lb.scale;
      let y2 = (cy + h / 2 - lb.dy) / lb.scale;

      x1 = Math.max(0, Math.min(sourceW, x1));
      y1 = Math.max(0, Math.min(sourceH, y1));
      x2 = Math.max(0, Math.min(sourceW, x2));
      y2 = Math.max(0, Math.min(sourceH, y2));

      const label = names[bestClass] || `class_${bestClass}`;
      candidates.push({ x1, y1, x2, y2, score: bestScore, label });
    }
  } else {
    const stride = 4 + classCount;
    for (let a = 0; a < anchors; a++) {
      const offset = a * stride;
      let bestClass = -1, bestScore = 0;
      for (let c = 0; c < classCount; c++) {
        const score = data[offset + 4 + c];
        if (score > bestScore) {
          bestScore = score;
          bestClass = c;
        }
      }
      if (bestScore < threshold) continue;

      const cx = data[offset];
      const cy = data[offset + 1];
      const w = data[offset + 2];
      const h = data[offset + 3];

      let x1 = (cx - w / 2 - lb.dx) / lb.scale;
      let y1 = (cy - h / 2 - lb.dy) / lb.scale;
      let x2 = (cx + w / 2 - lb.dx) / lb.scale;
      let y2 = (cy + h / 2 - lb.dy) / lb.scale;

      x1 = Math.max(0, Math.min(sourceW, x1));
      y1 = Math.max(0, Math.min(sourceH, y1));
      x2 = Math.max(0, Math.min(sourceW, x2));
      y2 = Math.max(0, Math.min(sourceH, y2));

      const label = names[bestClass] || `class_${bestClass}`;
      candidates.push({ x1, y1, x2, y2, score: bestScore, label });
    }
  }

  candidates.sort((a, b) => b.score - a.score);
  return nms(candidates.slice(0, 300), 0.45).slice(0, 50);
}

// ---------------- Drawing & UI Updates ----------------
function renderBoundingBoxes(targetCanvas, targetCtx, detections, srcW, srcH) {
  if (!srcW || !srcH) return;
  if (targetCanvas.width !== srcW || targetCanvas.height !== srcH) {
    targetCanvas.width = srcW;
    targetCanvas.height = srcH;
  }
  targetCtx.clearRect(0, 0, srcW, srcH);
  targetCtx.lineWidth = Math.max(2, srcW / 500);
  targetCtx.font = `${Math.max(12, srcW / 75)}px system-ui, sans-serif`;

  for (const d of detections) {
    const color = "#50e6d0";
    targetCtx.strokeStyle = color;
    targetCtx.fillStyle = "rgba(5, 15, 28, 0.82)";
    targetCtx.strokeRect(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1);

    const text = `${d.label} ${Math.round(d.score * 100)}%`;
    const tw = targetCtx.measureText(text).width + 12;
    const th = 24;
    targetCtx.fillRect(d.x1, Math.max(0, d.y1 - th), tw, th);
    targetCtx.fillStyle = color;
    targetCtx.fillText(text, d.x1 + 6, Math.max(16, d.y1 - 6));
  }
}

function updateDetectionList(detections) {
  if (!detectionsEl) return;
  if (!detections.length) {
    detectionsEl.innerHTML = '<div class="empty">No objects above the confidence threshold.</div>';
    return;
  }
  const counts = {};
  for (const d of detections) counts[d.label] = (counts[d.label] || 0) + 1;
  detectionsEl.innerHTML = Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .map(([name, count]) => {
      const best = Math.max(...detections.filter(d => d.label === name).map(d => d.score));
      return `<div class="detection-row"><span class="name">${escapeHtml(name)} × ${count}</span><span class="score">${Math.round(best * 100)}%</span></div>`;
    }).join("");
}

// ---------------- MODULE 2: Live Camera Detection ----------------
function updateCameraAspect() {
  if (!cameraWrap || !video.videoWidth || !video.videoHeight) return;
  cameraWrap.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
}

async function inferCameraFrame() {
  if (!session || !cameraRunning || busy || video.readyState < 2) return;
  const srcW = video.videoWidth;
  const srcH = video.videoHeight;
  if (!srcW || !srcH) return;

  busy = true;
  const t0 = performance.now();

  try {
    const { input, lb } = preprocessMedia(video, srcW, srcH);
    const tensor = new ort.Tensor("float32", input, [1, 3, PREPROCESS_SIZE, PREPROCESS_SIZE]);
    const inputName = session.inputNames[0];
    const outputMap = await session.run({ [inputName]: tensor });
    const output = outputMap[session.outputNames[0]];

    currentDetections = decode(output, lb, srcW, srcH);
    renderBoundingBoxes(canvas, ctx, currentDetections, srcW, srcH);
    updateDetectionList(currentDetections);

    if (cameraRunning) recordDetectionFrame(currentDetections);

    const ms = performance.now() - t0;
    fpsFrames++;
    const now = performance.now();
    let currentFps = fpsEl ? fpsEl.textContent : "--";
    if (now - fpsLast >= 1000) {
      currentFps = String(fpsFrames);
      fpsFrames = 0;
      fpsLast = now;
    }
    updateTelemetry(ms, currentFps, currentDetections.length, activeEngineName);
  } catch (err) {
    console.error("Camera inference error:", err);
    setStatus("Inference error", false);
  } finally {
    busy = false;
  }
}

function cameraLoop() {
  if (!cameraRunning) return;
  inferCameraFrame().finally(() => {
    if (cameraRunning) requestAnimationFrame(cameraLoop);
  });
}

async function startCamera() {
  hideAlert();
  if (!session) await loadModel();

  try {
    const deviceId = cameraSelect?.value;
    const constraints = {
      video: deviceId && deviceId !== "user" && deviceId !== "environment"
        ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
        : { facingMode: deviceId || "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false
    };

    stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = stream;
    await video.play();

    updateCameraAspect();
    video.addEventListener("loadedmetadata", updateCameraAspect, { once: true });

    placeholder.style.display = "none";
    startBtn.disabled = true;
    stopBtn.disabled = false;
    snapshotBtn.disabled = false;
    cameraRunning = true;
    setStatus("Live Edge AI Running");
    requestAnimationFrame(cameraLoop);
  } catch (err) {
    console.error("Camera error:", err);
    let msg = "Could not start camera. Please allow camera permissions and use HTTPS or localhost.";
    if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
      msg = "Camera permission denied. Please allow camera access in browser site settings.";
    } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
      msg = "No camera hardware detected on this device.";
    } else if (err.name === "NotReadableError" || err.name === "TrackStartError") {
      msg = "Camera is already in use by another application.";
    }
    showAlert(msg, true);
    setStatus("Camera unavailable", false);
  }
}

function stopCamera() {
  cameraRunning = false;
  if (stream) stream.getTracks().forEach(t => t.stop());
  stream = null;
  video.srcObject = null;

  if (cameraWrap) cameraWrap.style.aspectRatio = "16 / 9";
  placeholder.style.display = "grid";
  startBtn.disabled = false;
  stopBtn.disabled = true;
  snapshotBtn.disabled = true;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  currentDetections = [];
  updateTelemetry(undefined, "--", 0, activeEngineName);
  if (detectionsEl) detectionsEl.innerHTML = '<div class="empty">Camera stopped.</div>';
  setStatus("AI Ready");
}

async function flipCamera() {
  if (!cameraSelect) return;
  if (cameraSelect.options.length > 1) {
    const nextIndex = (cameraSelect.selectedIndex + 1) % cameraSelect.options.length;
    cameraSelect.selectedIndex = nextIndex;
  } else {
    cameraSelect.value = cameraSelect.value === "user" ? "environment" : "user";
  }
  const label = cameraSelect.options[cameraSelect.selectedIndex]?.text || cameraSelect.value;
  showToast(`Switched camera: ${label}`, false);

  if (cameraRunning) {
    if (stream) stream.getTracks().forEach(t => t.stop());
    stream = null;
    video.srcObject = null;
    await startCamera();
  }
}

startBtn?.addEventListener("click", startCamera);
stopBtn?.addEventListener("click", stopCamera);
flipCamBtn?.addEventListener("click", flipCamera);

// Dynamic Camera Enumeration
async function populateCameraDevices() {
  if (!navigator.mediaDevices?.enumerateDevices || !cameraSelect) return;
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoInputs = devices.filter(d => d.kind === "videoinput");
    if (videoInputs.length > 0) {
      cameraSelect.innerHTML = videoInputs.map((d, i) => `
        <option value="${d.deviceId}">${d.label || `Camera ${i + 1}`}</option>
      `).join("") + '<option value="user">Front / User</option><option value="environment">Back / Environment</option>';
    }
  } catch (e) {
    console.warn("Could not enumerate camera devices:", e);
  }
}

// ---------------- MODULE 3: Static Image Detection ----------------
function handleImageFile(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    showAlert("Invalid file format. Please upload an image (.jpg, .png, .webp).", true);
    return;
  }
  if (file.size > 25 * 1024 * 1024) {
    showAlert("Image file is very large (>25MB). Processing may take a few seconds.", false);
  } else {
    hideAlert();
  }

  const reader = new FileReader();
  reader.onload = (e) => {
    imagePreview.onload = () => {
      imageDropzone.style.display = "none";
      imageWrap.style.display = "flex";
      imageControls.style.display = "flex";
      imageCanvas.width = imagePreview.naturalWidth;
      imageCanvas.height = imagePreview.naturalHeight;
      imageCtx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
      imageStats.textContent = `Ready: ${imagePreview.naturalWidth}×${imagePreview.naturalHeight}px`;
      runImageDetection();
    };
    imagePreview.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

async function runImageDetection() {
  if (!session) await loadModel();
  const srcW = imagePreview.naturalWidth;
  const srcH = imagePreview.naturalHeight;
  if (!srcW || !srcH) return;

  setStatus("Analyzing image locally…");
  const t0 = performance.now();

  try {
    const { input, lb } = preprocessMedia(imagePreview, srcW, srcH);
    const tensor = new ort.Tensor("float32", input, [1, 3, PREPROCESS_SIZE, PREPROCESS_SIZE]);
    const inputName = session.inputNames[0];
    const outputMap = await session.run({ [inputName]: tensor });
    const output = outputMap[session.outputNames[0]];

    loadedImageDetections = decode(output, lb, srcW, srcH);
    renderBoundingBoxes(imageCanvas, imageCtx, loadedImageDetections, srcW, srcH);
    updateDetectionList(loadedImageDetections);

    // Record detections into analytics
    for (const d of loadedImageDetections) {
      evState.total += 1;
      evState.classes[d.label] = (evState.classes[d.label] || 0) + 1;
      evState.history.unshift({
        label: d.label,
        conf: Math.round(d.score * 100),
        time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })
      });
    }
    evState.history = evState.history.slice(0, 100);
    renderAnalytics();

    const ms = performance.now() - t0;
    imageStats.textContent = `✓ Detected ${loadedImageDetections.length} objects in ${Math.round(ms)} ms`;
    updateTelemetry(ms, "--", loadedImageDetections.length, activeEngineName);
    setStatus("Image detection complete");
  } catch (err) {
    console.error("Image detection error:", err);
    showAlert("Failed to process image with local YOLO model.", true);
    setStatus("Image error", false);
  }
}

function clearImage() {
  imagePreview.src = "";
  imageCtx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
  imageWrap.style.display = "none";
  imageControls.style.display = "none";
  imageDropzone.style.display = "block";
  loadedImageDetections = [];
  imageStats.textContent = "";
  if (imageFileInput) imageFileInput.value = "";
  updateDetectionList([]);
  updateTelemetry(undefined, "--", 0, activeEngineName);
}

imageDropzone?.addEventListener("click", () => imageFileInput?.click());
imageFileInput?.addEventListener("change", (e) => handleImageFile(e.target.files?.[0]));
runImageBtn?.addEventListener("click", runImageDetection);
clearImageBtn?.addEventListener("click", clearImage);

// Drag & Drop for Image
imageDropzone?.addEventListener("dragover", (e) => {
  e.preventDefault();
  imageDropzone.classList.add("dragover");
});
imageDropzone?.addEventListener("dragleave", () => imageDropzone.classList.remove("dragover"));
imageDropzone?.addEventListener("drop", (e) => {
  e.preventDefault();
  imageDropzone.classList.remove("dragover");
  if (e.dataTransfer?.files?.[0]) handleImageFile(e.dataTransfer.files[0]);
});

// ---------------- MODULE 4: Local Video Detection ----------------
function handleVideoFile(file) {
  if (!file) return;
  if (!file.type.startsWith("video/")) {
    showAlert("Invalid file format. Please upload a video (.mp4, .webm, .mov).", true);
    return;
  }
  if (file.size > 200 * 1024 * 1024) {
    showAlert("Video file is large (>200MB). Video playback may use high browser memory.", false);
  } else {
    hideAlert();
  }

  if (videoFileUrl) URL.revokeObjectURL(videoFileUrl);
  videoFileUrl = URL.createObjectURL(file);
  fileVideo.src = videoFileUrl;

  fileVideo.onloadedmetadata = () => {
    videoDropzone.style.display = "none";
    videoWrap.style.display = "flex";
    videoControls.style.display = "flex";
    videoCanvas.width = fileVideo.videoWidth;
    videoCanvas.height = fileVideo.videoHeight;
    videoWrap.style.aspectRatio = `${fileVideo.videoWidth} / ${fileVideo.videoHeight}`;
    videoStats.textContent = `Ready: ${fileVideo.videoWidth}×${fileVideo.videoHeight}px (${Math.round(fileVideo.duration)}s)`;
  };
}

async function inferVideoFrame() {
  if (!session || !videoDetecting || busy || fileVideo.paused || fileVideo.ended || fileVideo.readyState < 2) {
    if (videoDetecting && !fileVideo.paused && !fileVideo.ended) {
      requestAnimationFrame(inferVideoFrame);
    }
    return;
  }

  busy = true;
  const t0 = performance.now();

  try {
    const srcW = fileVideo.videoWidth;
    const srcH = fileVideo.videoHeight;
    const { input, lb } = preprocessMedia(fileVideo, srcW, srcH);
    const tensor = new ort.Tensor("float32", input, [1, 3, PREPROCESS_SIZE, PREPROCESS_SIZE]);
    const inputName = session.inputNames[0];
    const outputMap = await session.run({ [inputName]: tensor });
    const output = outputMap[session.outputNames[0]];

    const detections = decode(output, lb, srcW, srcH);
    renderBoundingBoxes(videoCanvas, videoCtx, detections, srcW, srcH);
    updateDetectionList(detections);
    recordDetectionFrame(detections);

    const ms = performance.now() - t0;
    fpsFrames++;
    const now = performance.now();
    let currentFps = fpsEl ? fpsEl.textContent : "--";
    if (now - fpsLast >= 1000) {
      currentFps = String(fpsFrames);
      fpsFrames = 0;
      fpsLast = now;
    }
    videoStats.textContent = `Live: ${detections.length} objects (${Math.round(ms)} ms/frame)`;
    updateTelemetry(ms, currentFps, detections.length, activeEngineName);
  } catch (err) {
    console.error("Video frame inference error:", err);
  } finally {
    busy = false;
    if (videoDetecting && !fileVideo.paused && !fileVideo.ended) {
      requestAnimationFrame(inferVideoFrame);
    }
  }
}

async function startVideoDetection() {
  if (!session) await loadModel();
  if (fileVideo.paused) await fileVideo.play();
  videoDetecting = true;
  startVideoDetectBtn.disabled = true;
  stopVideoDetectBtn.disabled = false;
  setStatus("Video Edge AI Running");
  requestAnimationFrame(inferVideoFrame);
}

function stopVideoDetection() {
  videoDetecting = false;
  startVideoDetectBtn.disabled = false;
  stopVideoDetectBtn.disabled = true;
  videoCtx.clearRect(0, 0, videoCanvas.width, videoCanvas.height);
  setStatus("Video Detection Paused");
}

function clearVideo() {
  stopVideoDetection();
  if (videoFileUrl) {
    URL.revokeObjectURL(videoFileUrl);
    videoFileUrl = null;
  }
  fileVideo.src = "";
  videoCtx.clearRect(0, 0, videoCanvas.width, videoCanvas.height);
  videoWrap.style.display = "none";
  videoControls.style.display = "none";
  videoDropzone.style.display = "block";
  videoStats.textContent = "";
  if (videoFileInput) videoFileInput.value = "";
  updateDetectionList([]);
  updateTelemetry(undefined, "--", 0, activeEngineName);
  setStatus("AI Ready");
}

videoDropzone?.addEventListener("click", () => videoFileInput?.click());
videoFileInput?.addEventListener("change", (e) => handleVideoFile(e.target.files?.[0]));
startVideoDetectBtn?.addEventListener("click", startVideoDetection);
stopVideoDetectBtn?.addEventListener("click", stopVideoDetection);
clearVideoBtn?.addEventListener("click", clearVideo);

// Drag & Drop for Video
videoDropzone?.addEventListener("dragover", (e) => {
  e.preventDefault();
  videoDropzone.classList.add("dragover");
});
videoDropzone?.addEventListener("dragleave", () => videoDropzone.classList.remove("dragover"));
videoDropzone?.addEventListener("drop", (e) => {
  e.preventDefault();
  videoDropzone.classList.remove("dragover");
  if (e.dataTransfer?.files?.[0]) handleVideoFile(e.dataTransfer.files[0]);
});

// ---------------- Workspace Tab Switcher ----------------
function switchWorkspaceTab(mode) {
  activeMode = mode;
  tabCameraBtn?.classList.toggle("active", mode === "camera");
  tabImageBtn?.classList.toggle("active", mode === "image");
  tabVideoBtn?.classList.toggle("active", mode === "video");

  if (viewCamera) viewCamera.style.display = mode === "camera" ? "block" : "none";
  if (viewImage) viewImage.style.display = mode === "image" ? "block" : "none";
  if (viewVideo) viewVideo.style.display = mode === "video" ? "block" : "none";

  // Stop camera if navigating away from camera
  if (mode !== "camera" && cameraRunning) stopCamera();
  // Pause video detection if navigating away from video
  if (mode !== "video" && videoDetecting) stopVideoDetection();
}

tabCameraBtn?.addEventListener("click", () => switchWorkspaceTab("camera"));
tabImageBtn?.addEventListener("click", () => switchWorkspaceTab("image"));
tabVideoBtn?.addEventListener("click", () => switchWorkspaceTab("video"));

// ---------------- MODULE 13: Universal Snapshot ----------------
function captureSnapshot() {
  let sourceEl, targetDetections;

  if (activeMode === "camera") {
    sourceEl = video;
    targetDetections = currentDetections;
  } else if (activeMode === "image") {
    sourceEl = imagePreview;
    targetDetections = loadedImageDetections;
  } else {
    sourceEl = fileVideo;
    targetDetections = currentDetections;
  }

  const w = sourceEl.videoWidth || sourceEl.naturalWidth;
  const h = sourceEl.videoHeight || sourceEl.naturalHeight;
  if (!w || !h) {
    showAlert("No active media frame available to capture.", true);
    return;
  }

  const shot = document.createElement("canvas");
  shot.width = w;
  shot.height = h;
  const sctx = shot.getContext("2d");
  sctx.drawImage(sourceEl, 0, 0);

  const strokeW = Math.max(2, w / 500);
  const fontSize = Math.max(14, w / 60);

  for (const d of targetDetections) {
    sctx.strokeStyle = "#50e6d0";
    sctx.lineWidth = strokeW;
    sctx.strokeRect(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1);
    sctx.fillStyle = "rgba(5, 15, 28, 0.85)";
    sctx.font = `${fontSize}px system-ui, sans-serif`;
    const label = `${d.label} ${Math.round(d.score * 100)}%`;
    const tw = sctx.measureText(label).width + 12;
    const th = fontSize + 8;
    sctx.fillRect(d.x1, Math.max(0, d.y1 - th), tw, th);
    sctx.fillStyle = "#50e6d0";
    sctx.fillText(label, d.x1 + 6, Math.max(fontSize, d.y1 - 6));
  }

  // Academic watermark
  sctx.fillStyle = "rgba(4, 13, 24, 0.75)";
  sctx.fillRect(10, h - 30, 320, 24);
  sctx.fillStyle = "#50e6d0";
  sctx.font = "11px system-ui, sans-serif";
  sctx.fillText("EdgeVision AI • Local Edge Inference", 18, h - 14);

  const a = document.createElement("a");
  a.download = `edgevision-${activeMode}-${Date.now()}.png`;
  a.href = shot.toDataURL("image/png");
  a.click();

  evState.snapshots += 1;
  renderAnalytics();
  updateTelemetry(undefined, undefined, undefined, activeEngineName);
}

snapshotBtn?.addEventListener("click", captureSnapshot);
snapshotImageBtn?.addEventListener("click", captureSnapshot);
snapshotVideoBtn?.addEventListener("click", captureSnapshot);

// ---------------- MODULE 5: Analytics & Detection History ----------------
function renderAnalytics() {
  if (totalEl) totalEl.textContent = String(evState.total);
  if (classesEl) classesEl.textContent = String(Object.keys(evState.classes).length);
  if (snapshotsEl) snapshotsEl.textContent = String(evState.snapshots);

  // Determine Most Detected Object
  if (topObjectEl) {
    const entries = Object.entries(evState.classes).sort((a, b) => b[1] - a[1]);
    if (entries.length > 0) {
      topObjectEl.textContent = `${entries[0][0]} (${entries[0][1]})`;
    } else {
      topObjectEl.textContent = "None";
    }
  }

  // Object Frequency Chart
  if (barsEl) {
    const items = Object.entries(evState.classes).sort((a, b) => b[1] - a[1]).slice(0, 8);
    if (!items.length) {
      barsEl.innerHTML = '<div class="empty">Start detection to build analytics.</div>';
    } else {
      const max = items[0][1];
      barsEl.innerHTML = items.map(([name, count]) => `
        <div class="bar-row">
          <span>${escapeHtml(name)}</span>
          <div class="bar-track"><div class="bar-fill" style="width:${Math.max(5, (count / max) * 100)}%"></div></div>
          <b>${count}</b>
        </div>`).join("");
    }
  }

  // Detection History Stream
  if (historyEl) {
    if (!evState.history.length) {
      historyEl.innerHTML = '<div class="empty">No detection history yet.</div>';
    } else {
      historyEl.innerHTML = evState.history.slice(0, 40).map(x => `
        <div class="history-row">
          <b>${escapeHtml(x.label)}</b>
          <span>${x.conf}%</span>
          <span>${x.time}</span>
        </div>`).join("");
    }
  }

  // Update Dashboard Metrics
  if (dashTotal) dashTotal.textContent = String(evState.total);
  if (dashClasses) dashClasses.textContent = String(Object.keys(evState.classes).length);
  if (dashSnapshots) dashSnapshots.textContent = String(evState.snapshots);
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, ch => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
  }[ch]));
}

function recordDetectionFrame(detections) {
  if (!detections.length) {
    evState.lastSignature = "";
    evState.stableFrames = 0;
    return;
  }

  const analyticsDetections = detections.filter(d => d.score >= 0.55);
  if (!analyticsDetections.length) {
    evState.stableFrames = 0;
    return;
  }

  const signature = analyticsDetections
    .map(d => `${d.label}:${Math.round(d.score * 10)}`)
    .sort()
    .join("|");

  if (signature === evState.lastSignature) {
    evState.stableFrames += 1;
  } else {
    evState.lastSignature = signature;
    evState.stableFrames = 1;
  }

  // Require same scene to be confirmed across 3 consecutive processed frames
  if (evState.stableFrames < 3) return;

  const now = Date.now();
  for (const d of analyticsDetections) {
    // 2-second cooldown per class to log discrete detection events instead of raw frame count
    if (now - (evState.lastRecordedAt[d.label] || 0) < 2000) continue;

    evState.lastRecordedAt[d.label] = now;
    evState.total += 1;
    evState.classes[d.label] = (evState.classes[d.label] || 0) + 1;
    evState.history.unshift({
      label: d.label,
      conf: Math.round(d.score * 100),
      time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })
    });
  }

  evState.history = evState.history.slice(0, 100);
  renderAnalytics();
}

function clearAnalyticsHistory() {
  evState.total = 0;
  evState.classes = {};
  evState.history = [];
  evState.snapshots = 0;
  evState.lastSignature = "";
  evState.stableFrames = 0;
  evState.lastRecordedAt = {};
  renderAnalytics();
}

clearBtn?.addEventListener("click", clearAnalyticsHistory);
settingsClearHistoryBtn?.addEventListener("click", clearAnalyticsHistory);

exportBtn?.addEventListener("click", () => {
  if (!evState.history.length) {
    alert("There is no detection history to export yet.");
    return;
  }
  const rows = [["Time", "Object", "Confidence"]];
  for (const item of [...evState.history].reverse()) {
    rows.push([item.time, item.label, `${item.conf}%`]);
  }
  const csv = rows.map(row => row.map(v => `"${String(v).replace(/"/g, '""')}"`).join(",")).join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `edgevision-detection-history-${Date.now()}.csv`;
  a.click();
  URL.revokeObjectURL(url);
});

// ---------------- MODULE 6: Settings Persistence & Reset ----------------
function loadSavedSettings() {
  try {
    const saved = JSON.parse(localStorage.getItem("edgevision-settings") || "{}");
    if (saved.confidence) syncConfidence(saved.confidence);
    if (saved.engine && engineSelect) engineSelect.value = saved.engine;
    if (saved.camera && cameraSelect) cameraSelect.value = saved.camera;
    if (saved.classes && customClassesInput) customClassesInput.value = saved.classes;
  } catch {}
}

saveSettingsBtn?.addEventListener("click", async () => {
  const data = {
    confidence: Number(settingsConfidence ? settingsConfidence.value : confSlider.value),
    engine: engineSelect ? engineSelect.value : "auto",
    camera: cameraSelect ? cameraSelect.value : "environment",
    classes: customClassesInput ? customClassesInput.value : ""
  };
  localStorage.setItem("edgevision-settings", JSON.stringify(data));
  syncConfidence(data.confidence);
  if (settingsSaved) {
    settingsSaved.textContent = "✓ Settings saved on this device";
    setTimeout(() => { settingsSaved.textContent = ""; }, 2500);
  }
  if (!cameraRunning && !videoDetecting && activeModelUrl) {
    session = null;
    try { await loadModel(); } catch {}
  }
});

resetSettingsBtn?.addEventListener("click", () => {
  localStorage.removeItem("edgevision-settings");
  syncConfidence(0.55);
  if (engineSelect) engineSelect.value = "auto";
  if (cameraSelect) cameraSelect.value = "environment";
  if (gpuToggle) gpuToggle.checked = true;
  if (settingsSaved) {
    settingsSaved.textContent = "✓ Settings reset to default";
    setTimeout(() => { settingsSaved.textContent = ""; }, 2500);
  }
});

// ---------------- MODULE 12: Custom ONNX Model Loader ----------------
customModelInput?.addEventListener("change", (e) => {
  const file = e.target.files?.[0];
  if (file && chooseModelLabel) {
    const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
    chooseModelLabel.textContent = `${file.name} (${sizeMb} MB)`;
    if (customModelStatus) {
      customModelStatus.textContent = `Selected "${file.name}" (${sizeMb} MB). Ready to validate & load.`;
      customModelStatus.style.color = "var(--text)";
    }
  }
});

async function loadCustomModel() {
  const file = customModelInput?.files?.[0];
  if (!file) {
    if (customModelStatus) {
      customModelStatus.textContent = "⚠️ Please choose a .onnx model file first.";
      customModelStatus.style.color = "var(--danger)";
    }
    return alert("Choose a .onnx model file first.");
  }

  const lowerName = file.name.toLowerCase();
  if (!lowerName.endsWith(".onnx") && !lowerName.endsWith(".bin")) {
    const msg = "Invalid file type: Please select a valid .onnx model file.";
    if (customModelStatus) {
      customModelStatus.textContent = `❌ ${msg}`;
      customModelStatus.style.color = "var(--danger)";
    }
    return alert(msg);
  }

  if (file.size < 1024) {
    const msg = "Invalid file: The file is too small (< 1 KB) to be a valid ONNX model.";
    if (customModelStatus) {
      customModelStatus.textContent = `❌ ${msg}`;
      customModelStatus.style.color = "var(--danger)";
    }
    return alert(msg);
  }

  const parsedNames = parseClassNames(customClassesInput.value);
  if (!parsedNames.length) {
    const msg = "Custom class names required: Enter one class name per line (or '0: name') matching your training labels.";
    if (customModelStatus) {
      customModelStatus.textContent = `⚠️ ${msg}`;
      customModelStatus.style.color = "var(--danger)";
    }
    return alert(msg);
  }

  setStatus("Validating custom model…");
  if (customModelStatus) {
    customModelStatus.textContent = "⏳ Reading file and initializing ONNX Runtime…";
    customModelStatus.style.color = "var(--accent)";
  }

  try {
    const buffer = await file.arrayBuffer();
    const requestedEngine = engineSelect?.value || (gpuToggle?.checked ? "auto" : "wasm");

    const result = await createSessionWithFallback(new Uint8Array(buffer), requestedEngine);
    const newSession = result.session;
    const provider = result.provider;

    const dummyTensor = new ort.Tensor("float32", inputTensorBuffer, [1, 3, PREPROCESS_SIZE, PREPROCESS_SIZE]);
    const inputName = newSession.inputNames[0];
    const outputMap = await newSession.run({ [inputName]: dummyTensor });
    const outputName = newSession.outputNames[0];
    const output = outputMap[outputName];

    if (!output || !output.dims) {
      throw new Error("Model ran successfully but produced no output tensor with dimensions.");
    }

    const info = inspectYoloTensor(output.dims);
    if (!info) {
      throw new Error(`Incompatible output tensor shape: [${output.dims.join(", ")}]. EdgeVision requires a YOLO detection output formatted as [1, 4+C, N] or [1, N, 4+C].`);
    }

    let finalNames = [...parsedNames];
    let noteText = "";
    if (parsedNames.length !== info.classCount) {
      if (parsedNames.length < info.classCount) {
        for (let i = parsedNames.length; i < info.classCount; i++) finalNames.push(`class_${i}`);
        noteText = ` ⚠️ Note: Model has ${info.classCount} classes, but ${parsedNames.length} names were entered. Missing classes padded as class_*.`;
      } else {
        finalNames = parsedNames.slice(0, info.classCount);
        noteText = ` ⚠️ Note: Model has ${info.classCount} classes, but ${parsedNames.length} names were entered. Using first ${info.classCount} classes.`;
      }
    }

    session = newSession;
    activeModelUrl = null;
    activeClassNames = finalNames;
    activeModelName = file.name;
    activeOutputDims = output.dims;
    activeLayoutInfo = info;
    activeEngineName = provider === "webgpu" ? "WebGPU" : "WASM CPU";

    if (defaultModelCard) defaultModelCard.classList.remove("active-model");
    if (customModelCard) customModelCard.classList.add("active-model");
    if (defaultModelBadge) defaultModelBadge.textContent = "INACTIVE";
    const customBadge = customModelCard?.querySelector(".model-badge");
    if (customBadge) customBadge.textContent = "ACTIVE";

    if (modelState) modelState.textContent = "CUSTOM MODEL ACTIVE";
    if (engineEl) engineEl.textContent = activeEngineName;

    if (customModelStatus) {
      customModelStatus.textContent = `✓ Successfully loaded "${file.name}"! Output: [${output.dims.join(", ")}] (${info.layout === 'A' ? 'Layout A [1, 4+C, N]' : 'Layout B [1, N, 4+C]'}), ${info.classCount} classes.${noteText}`;
      customModelStatus.style.color = "var(--accent)";
    }

    renderModelInfo();
    setStatus("Custom Edge AI Ready");
  } catch (err) {
    console.error("Custom model validation failed:", err);
    if (customModelStatus) {
      customModelStatus.textContent = `❌ Validation Failed: ${err.message || err}`;
      customModelStatus.style.color = "var(--danger)";
    }
    setStatus("Custom model failed", false);
    alert(`Could not load custom model:\n\n${err.message || err}`);
  }
}

async function resetToDefaultModel() {
  if (activeModelUrl === DEFAULT_MODEL_URL && !activeClassNames) {
    if (customModelStatus) {
      customModelStatus.textContent = "Default YOLO11n model is already active.";
      customModelStatus.style.color = "var(--muted)";
    }
    return;
  }

  setStatus("Resetting to YOLO11n…");
  if (customModelStatus) {
    customModelStatus.textContent = "⏳ Loading default pretrained YOLO11n model…";
    customModelStatus.style.color = "var(--accent)";
  }

  try {
    activeModelUrl = DEFAULT_MODEL_URL;
    activeClassNames = null;
    activeModelName = "YOLO11n";
    activeOutputDims = [1, 84, 8400];
    activeLayoutInfo = { layout: "A", classCount: 80, anchors: 8400, channels: 84, dims: [1, 84, 8400] };

    await loadModel();

    if (customModelInput) customModelInput.value = "";
    if (chooseModelLabel) chooseModelLabel.textContent = "Choose .onnx";

    if (defaultModelCard) defaultModelCard.classList.add("active-model");
    if (customModelCard) customModelCard.classList.remove("active-model");
    if (defaultModelBadge) defaultModelBadge.textContent = "ACTIVE";
    const customBadge = customModelCard?.querySelector(".model-badge");
    if (customBadge) customBadge.textContent = "INACTIVE";

    if (modelState) modelState.textContent = "YOLO11n ACTIVE";
    if (customModelStatus) {
      customModelStatus.textContent = "✓ Reset to default pretrained YOLO11n model. COCO 80 classes active.";
      customModelStatus.style.color = "var(--accent)";
    }

    renderModelInfo();
  } catch (err) {
    console.error("Failed to reset model:", err);
    if (customModelStatus) {
      customModelStatus.textContent = `❌ Failed to load default model: ${err.message || err}`;
      customModelStatus.style.color = "var(--danger)";
    }
  }
}

loadCustomModelBtn?.addEventListener("click", loadCustomModel);
resetModelBtn?.addEventListener("click", resetToDefaultModel);
defaultModelCard?.addEventListener("click", () => {
  if (activeModelUrl !== DEFAULT_MODEL_URL) resetToDefaultModel();
});

gpuToggle?.addEventListener("change", async () => {
  if (cameraRunning || videoDetecting) return;
  session = null;
  try { await loadModel(); } catch {}
});

// ---------------- PWA, Toasts, Offline & Mobile Navigation ----------------
let deferredInstallPrompt = null;

function showToast(message, isDanger = false, duration = 3500) {
  if (!toastContainer) return;
  const toast = document.createElement("div");
  toast.className = `toast ${isDanger ? 'toast-danger' : 'toast-success'}`;
  toast.innerHTML = `<span>${isDanger ? '⚠️' : '✓'}</span><span>${escapeHtml(message)}</span>`;
  toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateY(10px)";
    toast.style.transition = "all 0.3s ease";
    setTimeout(() => toast.remove(), 320);
  }, duration);
}

function setupMobileNavigation() {
  if (!menuToggleBtn || !mainNav) return;
  menuToggleBtn.addEventListener("click", () => {
    mainNav.classList.toggle("mobile-open");
    menuToggleBtn.textContent = mainNav.classList.contains("mobile-open") ? "✕" : "☰";
  });

  mainNav.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", () => {
      mainNav.classList.remove("mobile-open");
      if (menuToggleBtn) menuToggleBtn.textContent = "☰";
    });
  });
}

function updateNetworkStatus() {
  const isOnline = navigator.onLine;
  if (netStatusEl) {
    netStatusEl.className = `net-pill ${isOnline ? 'online' : 'offline'}`;
    netStatusEl.textContent = isOnline ? "🟢 Online" : "🟠 Offline (PWA)";
  }
}

function setupNetworkStatus() {
  updateNetworkStatus();
  window.addEventListener("online", () => {
    updateNetworkStatus();
    showToast("🟢 Connection restored. Online mode active.", false);
  });
  window.addEventListener("offline", () => {
    updateNetworkStatus();
    showToast("📶 Offline mode active. Running on-device Edge AI.", false);
  });
}

function registerServiceWorker() {
  if (!("serviceWorker" in navigator)) {
    if (pwaCacheStatus) {
      pwaCacheStatus.textContent = "Service Worker not supported";
      pwaCacheStatus.style.color = "var(--muted)";
    }
    return;
  }

  window.addEventListener("load", () => {
    navigator.serviceWorker
      .register("./sw.js", { scope: "./" })
      .then((reg) => {
        console.log("[PWA] Service Worker registered with scope:", reg.scope);
        if (pwaCacheStatus && !pwaCacheStatus.textContent.includes("✓")) {
          pwaCacheStatus.textContent = "✓ Service Worker Active";
          pwaCacheStatus.style.color = "var(--accent)";
        }

        reg.onupdatefound = () => {
          const installingWorker = reg.installing;
          if (installingWorker) {
            installingWorker.onstatechange = () => {
              if (installingWorker.state === "installed" && navigator.serviceWorker.controller) {
                if (pwaUpdateBanner) pwaUpdateBanner.style.display = "block";
              }
            };
          }
        };
      })
      .catch((err) => {
        console.warn("[PWA] Service Worker registration error:", err);
        if (pwaCacheStatus) {
          pwaCacheStatus.textContent = "SW registration failed";
          pwaCacheStatus.style.color = "var(--danger)";
        }
      });
  });
}

async function promptPwaInstall() {
  if (deferredInstallPrompt) {
    deferredInstallPrompt.prompt();
    const { outcome } = await deferredInstallPrompt.userChoice;
    console.log("[PWA] Install prompt outcome:", outcome);
    deferredInstallPrompt = null;
    if (pwaInstallBtn) pwaInstallBtn.style.display = "none";
    if (heroInstallBtn) heroInstallBtn.style.display = "none";
  } else {
    // Show iOS / manual install modal
    if (iosInstallModal) iosInstallModal.style.display = "grid";
  }
}

function setupPwaInstall() {
  window.addEventListener("beforeinstallprompt", (e) => {
    e.preventDefault();
    deferredInstallPrompt = e;
    if (pwaInstallBtn) pwaInstallBtn.style.display = "inline-flex";
    if (heroInstallBtn) heroInstallBtn.style.display = "inline-flex";
  });

  window.addEventListener("appinstalled", () => {
    console.log("[PWA] EdgeVision AI was successfully installed.");
    if (pwaInstallBtn) pwaInstallBtn.style.display = "none";
    if (heroInstallBtn) heroInstallBtn.style.display = "none";
    showToast("🎉 EdgeVision AI installed to your home screen / desktop!", false);
  });

  pwaInstallBtn?.addEventListener("click", promptPwaInstall);
  heroInstallBtn?.addEventListener("click", promptPwaInstall);

  pwaReloadBtn?.addEventListener("click", () => {
    if ("serviceWorker" in navigator) {
      navigator.serviceWorker.getRegistration().then((reg) => {
        reg?.waiting?.postMessage({ type: "SKIP_WAITING" });
        window.location.reload();
      });
    } else {
      window.location.reload();
    }
  });

  pwaDismissBannerBtn?.addEventListener("click", () => {
    if (pwaUpdateBanner) pwaUpdateBanner.style.display = "none";
  });

  closeIosModalBtn?.addEventListener("click", () => {
    if (iosInstallModal) iosInstallModal.style.display = "none";
  });

  ackIosModalBtn?.addEventListener("click", () => {
    if (iosInstallModal) iosInstallModal.style.display = "none";
  });
}

async function updateStorageQuota() {
  if ("storage" in navigator && "estimate" in navigator.storage) {
    try {
      const { quota, usage } = await navigator.storage.estimate();
      const usedMb = (usage / (1024 * 1024)).toFixed(1);
      const totalMb = (quota / (1024 * 1024)).toFixed(0);
      if (storageQuotaStatus) {
        storageQuotaStatus.textContent = `${usedMb} MB used (of ~${totalMb} MB storage)`;
        storageQuotaStatus.style.color = "var(--text)";
      }
    } catch (e) {
      if (storageQuotaStatus) storageQuotaStatus.textContent = "Storage API available";
    }
  } else if (storageQuotaStatus) {
    storageQuotaStatus.textContent = "Storage quota API unsupported";
  }
}

// ---------------- Initialize Application ----------------
if (!window.isSecureContext) {
  setStatus("Use HTTPS / localhost for Camera & PWA", false);
  if (engineEl) engineEl.textContent = "Browser";
  showAlert("Secure Context (HTTPS or localhost) required for Camera access and PWA installation.", false);
}

setupMobileNavigation();
setupNetworkStatus();
registerServiceWorker();
setupPwaInstall();
updateStorageQuota();
populateCameraDevices();
loadSavedSettings();
renderAnalytics();
renderModelInfo();
updateTelemetry(undefined, "--", 0, "WebGPU / CPU");

// Expose state and controls for test automation and presentation
window.edgeVisionLoaded = true;
window.evState = evState;
window.switchWorkspaceTab = switchWorkspaceTab;
window.renderAnalytics = renderAnalytics;
window.syncConfidence = syncConfidence;
window.promptPwaInstall = promptPwaInstall;
window.updateStorageQuota = updateStorageQuota;
window.showToast = showToast;

