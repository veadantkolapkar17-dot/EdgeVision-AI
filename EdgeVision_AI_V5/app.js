import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.all.min.mjs";

const DEFAULT_MODEL_URL =
  "https://huggingface.co/webnn/yolo11n/resolve/main/onnx/yolo11n.onnx?download=true";

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

// DOM references - Live Detection & Camera
const video = document.querySelector("#video");
const canvas = document.querySelector("#overlay");
const ctx = canvas.getContext("2d");
const cameraWrap = document.querySelector(".camera-wrap");
const placeholder = document.querySelector("#placeholder");
const startBtn = document.querySelector("#startBtn");
const stopBtn = document.querySelector("#stopBtn");
const snapshotBtn = document.querySelector("#snapshotBtn");
const confSlider = document.querySelector("#confidence");
const confValue = document.querySelector("#confidenceValue");
const gpuToggle = document.querySelector("#gpuToggle");

// DOM references - Settings & Navigation
const settingsConfidence = document.querySelector("#settingsConfidence");
const settingsConfidenceValue = document.querySelector("#settingsConfidenceValue");
const engineSelect = document.querySelector("#engineSelect");
const cameraSelect = document.querySelector("#cameraSelect");
const saveSettingsBtn = document.querySelector("#saveSettings");
const settingsSaved = document.querySelector("#settingsSaved");
const statusEl = document.querySelector("#engineStatus");
const navStatusEl = document.querySelector("#navStatus");
const latencyEl = document.querySelector("#latency");
const fpsEl = document.querySelector("#fps");
const objectCountEl = document.querySelector("#objectCount");
const engineEl = document.querySelector("#engine");
const detectionsEl = document.querySelector("#detections");

// DOM references - AI Model & Custom Loader (Phase 3)
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

// DOM references - Model Information Panel (Phase 3)
const infoModelName = document.querySelector("#infoModelName");
const infoEngine = document.querySelector("#infoEngine");
const infoInputShape = document.querySelector("#infoInputShape");
const infoOutputShape = document.querySelector("#infoOutputShape");
const infoLayout = document.querySelector("#infoLayout");
const infoClassesCount = document.querySelector("#infoClassesCount");
const infoClassesList = document.querySelector("#infoClassesList");
const infoStatusPill = document.querySelector("#infoStatusPill");

// Inference state
let session = null;
let stream = null;
let running = false;
let busy = false;
let fpsFrames = 0;
let fpsLast = performance.now();
let currentDetections = [];

// Preprocessing persistent buffers (Zero memory churn per frame)
const PREPROCESS_SIZE = 640;
const preprocessCanvas = typeof OffscreenCanvas !== "undefined"
  ? new OffscreenCanvas(PREPROCESS_SIZE, PREPROCESS_SIZE)
  : document.createElement("canvas");
preprocessCanvas.width = PREPROCESS_SIZE;
preprocessCanvas.height = PREPROCESS_SIZE;
const preprocessCtx = preprocessCanvas.getContext("2d", { willReadFrequently: true });
const inputTensorBuffer = new Float32Array(1 * 3 * PREPROCESS_SIZE * PREPROCESS_SIZE);

// Status helper: keeps Live workspace status and Nav status synchronized
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

// Two-way confidence synchronization
function syncConfidence(value) {
  const v = Number(value);
  if (confSlider) confSlider.value = String(v);
  if (confValue) confValue.textContent = `${Math.round(v * 100)}%`;
  if (settingsConfidence) settingsConfidence.value = String(v);
  if (settingsConfidenceValue) settingsConfidenceValue.textContent = `${Math.round(v * 100)}%`;
}

confSlider?.addEventListener("input", (e) => {
  syncConfidence(e.target.value);
});

settingsConfidence?.addEventListener("input", (e) => {
  syncConfidence(e.target.value);
});

// Robust model loader with WebGPU -> WASM fallback
async function createSessionWithFallback(modelSource, preferredEngine) {
  const wantWebGpu = preferredEngine !== "wasm" && typeof navigator !== "undefined" && Boolean(navigator.gpu);

  if (wantWebGpu) {
    try {
      const sess = await ort.InferenceSession.create(modelSource, {
        executionProviders: ["webgpu"],
        graphOptimizationLevel: "all"
      });
      return { session: sess, provider: "webgpu" };
    } catch (gpuErr) {
      console.warn("WebGPU initialization failed, falling back to WASM CPU:", gpuErr);
      setStatus("WebGPU unavailable, using CPU", true);
    }
  }

  // CPU / WASM fallback
  const sess = await ort.InferenceSession.create(modelSource, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all"
  });
  return { session: sess, provider: "wasm" };
}

// ---------------- Universal YOLO Output Tensor Adapter ----------------
function inspectYoloTensor(dims) {
  if (!dims || dims.length < 2) return null;
  const d0 = dims[dims.length - 2];
  const d1 = dims[dims.length - 1];

  // Layout A: [..., 4 + C, N] (e.g. [1, 84, 8400] for 80 classes, [1, 9, 8400] for 5 classes)
  // Layout B: [..., N, 4 + C] (e.g. [1, 8400, 84] for 80 classes, [1, 8400, 9] for 5 classes)
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

// Helper to parse custom class names formatted as lines, indexed ("0: person", "0 = person"), or lists
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
      // Strip index prefixes like "0 = person", "0: person", "0 - person", "#0 person"
      const cleaned = line.replace(/^(\d+\s*[:=\-.]\s*|#\d+\s*)/, "").trim().replace(/^['"]|['"]$/g, "");
      if (cleaned) result.push(cleaned);
    }
  }
  return result;
}

// Render Model Information Panel
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

  const requestedEngine = engineSelect?.value || (gpuToggle?.checked ? "auto" : "wasm");

  try {
    const result = await createSessionWithFallback(activeModelUrl, requestedEngine);
    session = result.session;
    const isGpu = result.provider === "webgpu";
    activeEngineName = isGpu ? "WebGPU" : "WASM CPU";
    if (engineEl) engineEl.textContent = activeEngineName;
    setStatus(isGpu ? "WebGPU Edge AI Ready" : "CPU Edge AI Ready");

    // Perform dry-run inspection on initialization to determine real tensor dimensions
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
      console.warn("Dry run tensor inspection:", dryErr);
    }

    renderModelInfo();
  } catch (err) {
    console.error("Failed to load model:", err);
    setStatus("Model load failed", false);
    if (engineEl) engineEl.textContent = "Error";
    throw err;
  }
}

function letterbox(sourceW, sourceH, target = PREPROCESS_SIZE) {
  const scale = Math.min(target / sourceW, target / sourceH);
  const w = Math.round(sourceW * scale);
  const h = Math.round(sourceH * scale);
  return { scale, w, h, dx: (target - w) / 2, dy: (target - h) / 2 };
}

// Optimized zero-allocation preprocessing
function preprocess() {
  const srcW = video.videoWidth;
  const srcH = video.videoHeight;
  const lb = letterbox(srcW, srcH, PREPROCESS_SIZE);

  preprocessCtx.fillStyle = "#000";
  preprocessCtx.fillRect(0, 0, PREPROCESS_SIZE, PREPROCESS_SIZE);
  preprocessCtx.drawImage(video, lb.dx, lb.dy, lb.w, lb.h);

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

// Universal YOLO decoder supporting Layout A [1, 4+C, N] and Layout B [1, N, 4+C]
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
    // Layout A: [1, 4 + classCount, anchors]
    for (let a = 0; a < anchors; a++) {
      let bestClass = -1;
      let bestScore = 0;
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
    // Layout B: [1, anchors, 4 + classCount]
    const stride = 4 + classCount;
    for (let a = 0; a < anchors; a++) {
      const offset = a * stride;
      let bestClass = -1;
      let bestScore = 0;
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

function drawDetections(detections) {
  const w = video.videoWidth, h = video.videoHeight;
  if (!w || !h) return;

  // Only resize canvas backing store if dimensions changed
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }

  ctx.clearRect(0, 0, w, h);
  ctx.lineWidth = Math.max(2, w / 500);
  ctx.font = `${Math.max(12, w / 75)}px system-ui`;

  for (const d of detections) {
    const color = "#5eead4";
    ctx.strokeStyle = color;
    ctx.fillStyle = "rgba(5,15,28,.78)";
    ctx.strokeRect(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1);
    const text = `${d.label} ${Math.round(d.score * 100)}%`;
    const tw = ctx.measureText(text).width + 12;
    const th = 25;
    ctx.fillRect(d.x1, Math.max(0, d.y1 - th), tw, th);
    ctx.fillStyle = color;
    ctx.fillText(text, d.x1 + 6, Math.max(17, d.y1 - 7));
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

// Clean, modular inference pipeline
async function inferFrame() {
  if (!session || !running || busy || video.readyState < 2) return;
  const srcW = video.videoWidth;
  const srcH = video.videoHeight;
  if (!srcW || !srcH) return;

  busy = true;
  const t0 = performance.now();

  try {
    const { input, lb } = preprocess();
    const tensor = new ort.Tensor("float32", input, [1, 3, PREPROCESS_SIZE, PREPROCESS_SIZE]);
    const inputName = session.inputNames[0];
    const outputMap = await session.run({ [inputName]: tensor });
    const output = outputMap[session.outputNames[0]];

    currentDetections = decode(output, lb, srcW, srcH);
    drawDetections(currentDetections);
    updateDetectionList(currentDetections);
    if (objectCountEl) objectCountEl.textContent = String(currentDetections.length);

    // Feed detections directly to analytics using the active model's class names
    if (running) {
      recordDetectionFrame(currentDetections);
    }

    const ms = performance.now() - t0;
    if (latencyEl) latencyEl.textContent = `${Math.round(ms)} ms`;
    fpsFrames++;
    const now = performance.now();
    if (now - fpsLast >= 1000) {
      if (fpsEl) fpsEl.textContent = String(fpsFrames);
      fpsFrames = 0;
      fpsLast = now;
    }
  } catch (err) {
    console.error("Inference error:", err);
    setStatus("Inference error", false);
  } finally {
    busy = false;
  }
}

function loop() {
  if (!running) return;
  inferFrame().finally(() => {
    if (running) requestAnimationFrame(loop);
  });
}

function updateCameraAspect() {
  if (!cameraWrap || !video.videoWidth || !video.videoHeight) return;
  cameraWrap.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
}

async function startCamera() {
  if (!session) await loadModel();
  stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: cameraSelect?.value || "environment" },
    audio: false
  });
  video.srcObject = stream;
  await video.play();

  updateCameraAspect();
  video.addEventListener("loadedmetadata", updateCameraAspect, { once: true });

  placeholder.style.display = "none";
  startBtn.disabled = true;
  stopBtn.disabled = false;
  snapshotBtn.disabled = false;
  running = true;
  setStatus("Live Edge AI Running");
  requestAnimationFrame(loop);
}

function stopCamera() {
  running = false;
  if (stream) stream.getTracks().forEach(t => t.stop());
  stream = null;
  video.srcObject = null;

  if (cameraWrap) {
    cameraWrap.style.aspectRatio = "16 / 9";
  }

  placeholder.style.display = "grid";
  startBtn.disabled = false;
  stopBtn.disabled = true;
  snapshotBtn.disabled = true;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  currentDetections = [];
  if (objectCountEl) objectCountEl.textContent = "0";
  if (detectionsEl) detectionsEl.innerHTML = '<div class="empty">Camera stopped.</div>';
  setStatus("AI Ready");
}

snapshotBtn?.addEventListener("click", () => {
  if (!video.videoWidth || !video.videoHeight) return;
  const shot = document.createElement("canvas");
  shot.width = video.videoWidth;
  shot.height = video.videoHeight;
  const sctx = shot.getContext("2d");
  sctx.drawImage(video, 0, 0);

  const strokeW = Math.max(2, video.videoWidth / 500);
  const fontSize = Math.max(14, video.videoWidth / 60);

  for (const d of currentDetections) {
    sctx.strokeStyle = "#5eead4";
    sctx.lineWidth = strokeW;
    sctx.strokeRect(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1);
    sctx.fillStyle = "rgba(5,15,28,.8)";
    sctx.font = `${fontSize}px system-ui`;
    const label = `${d.label} ${Math.round(d.score * 100)}%`;
    const tw = sctx.measureText(label).width + 12;
    const th = fontSize + 8;
    sctx.fillRect(d.x1, Math.max(0, d.y1 - th), tw, th);
    sctx.fillStyle = "#5eead4";
    sctx.fillText(label, d.x1 + 6, Math.max(fontSize, d.y1 - 6));
  }
  const a = document.createElement("a");
  a.download = `edgevision-${Date.now()}.png`;
  a.href = shot.toDataURL("image/png");
  a.click();
  evState.snapshots += 1;
  renderAnalytics();
});

startBtn?.addEventListener("click", async () => {
  try {
    await startCamera();
  } catch (err) {
    console.error(err);
    alert("Camera could not start. Allow camera permission and open the site through HTTPS or localhost.");
    setStatus("Camera permission needed", false);
  }
});

stopBtn?.addEventListener("click", stopCamera);

if (!window.isSecureContext) {
  setStatus("Use HTTPS / localhost", false);
  if (engineEl) engineEl.textContent = "Browser";
}

// ---------------- EdgeVision Analytics & Detection History ----------------
const evState = {
  total: 0,
  classes: {},
  history: [],
  snapshots: 0,
  lastSignature: "",
  stableFrames: 0,
  lastRecordedAt: {}
};

const totalEl = document.querySelector("#analyticsTotal");
const classesEl = document.querySelector("#analyticsClasses");
const snapshotsEl = document.querySelector("#analyticsSnapshots");
const barsEl = document.querySelector("#analyticsBars");
const historyEl = document.querySelector("#history");
const exportBtn = document.querySelector("#exportHistory");
const clearBtn = document.querySelector("#clearHistory");

function renderAnalytics() {
  if (totalEl) totalEl.textContent = String(evState.total);
  if (classesEl) classesEl.textContent = String(Object.keys(evState.classes).length);
  if (snapshotsEl) snapshotsEl.textContent = String(evState.snapshots);

  if (barsEl) {
    const items = Object.entries(evState.classes)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 8);
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

  // Analytics use a stricter threshold and temporal confirmation
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

  // Require same scene to be observed in 3 processed frames
  if (evState.stableFrames < 3) return;

  const now = Date.now();
  for (const d of analyticsDetections) {
    if (now - (evState.lastRecordedAt[d.label] || 0) < 2000) continue;

    evState.lastRecordedAt[d.label] = now;
    evState.total += 1;
    evState.classes[d.label] = (evState.classes[d.label] || 0) + 1;
    evState.history.unshift({
      label: d.label,
      conf: Math.round(d.score * 100),
      time: new Date().toLocaleTimeString([], {
        hour: "2-digit", minute: "2-digit", second: "2-digit"
      })
    });
  }

  evState.history = evState.history.slice(0, 100);
  renderAnalytics();
}

clearBtn?.addEventListener("click", () => {
  evState.total = 0;
  evState.classes = {};
  evState.history = [];
  evState.snapshots = 0;
  evState.lastSignature = "";
  evState.stableFrames = 0;
  evState.lastRecordedAt = {};
  renderAnalytics();
});

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

// ---------------- Settings & Custom Model Support (Phase 3) ----------------
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
  if (!running && activeModelUrl) {
    session = null;
    try { await loadModel(); } catch {}
  }
});

// Update Choose Model file button text upon selection
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

// Custom ONNX Model Validation & Loading
async function loadCustomModel() {
  const file = customModelInput?.files?.[0];
  if (!file) {
    if (customModelStatus) {
      customModelStatus.textContent = "⚠️ Please choose a .onnx model file first.";
      customModelStatus.style.color = "var(--danger)";
    }
    return alert("Choose a .onnx model file first.");
  }

  // File type and size validation
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

  // Parse custom class names
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

    // Create session with WebGPU/WASM fallback
    const result = await createSessionWithFallback(new Uint8Array(buffer), requestedEngine);
    const newSession = result.session;
    const provider = result.provider;

    // Perform dry-run validation to inspect runtime tensor shapes
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
      throw new Error(`Incompatible output tensor shape: [${output.dims.join(", ")}]. EdgeVision requires a YOLO detection output tensor formatted as [1, 4+C, N] or [1, N, 4+C].`);
    }

    // Class count reconciliation
    let finalNames = [...parsedNames];
    let noteText = "";
    if (parsedNames.length !== info.classCount) {
      if (parsedNames.length < info.classCount) {
        for (let i = parsedNames.length; i < info.classCount; i++) {
          finalNames.push(`class_${i}`);
        }
        noteText = ` ⚠️ Note: Model has ${info.classCount} classes, but ${parsedNames.length} names were entered. Missing classes padded as class_*.`;
      } else {
        finalNames = parsedNames.slice(0, info.classCount);
        noteText = ` ⚠️ Note: Model has ${info.classCount} classes, but ${parsedNames.length} names were entered. Using first ${info.classCount} classes.`;
      }
    }

    // Success: activate custom model
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

// Reset to default pretrained YOLO11n model
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
  if (activeModelUrl !== DEFAULT_MODEL_URL) {
    resetToDefaultModel();
  }
});

gpuToggle?.addEventListener("change", async () => {
  if (running) return;
  session = null;
  try { await loadModel(); } catch {}
});

loadSavedSettings();
renderAnalytics();
renderModelInfo();
