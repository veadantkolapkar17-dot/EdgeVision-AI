import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.all.min.mjs";

const DEFAULT_MODEL_URL =
  "https://huggingface.co/webnn/yolo11n/resolve/main/onnx/yolo11n.onnx?download=true";
let activeModelUrl = DEFAULT_MODEL_URL;
let activeClassNames = null;
let activeModelName = "YOLO11n";

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

const video = document.querySelector("#video");
const canvas = document.querySelector("#overlay");
const ctx = canvas.getContext("2d");
const placeholder = document.querySelector("#placeholder");
const startBtn = document.querySelector("#startBtn");
const stopBtn = document.querySelector("#stopBtn");
const snapshotBtn = document.querySelector("#snapshotBtn");
const confSlider = document.querySelector("#confidence");
const confValue = document.querySelector("#confidenceValue");
const gpuToggle = document.querySelector("#gpuToggle");
const settingsConfidence = document.querySelector("#settingsConfidence");
const settingsConfidenceValue = document.querySelector("#settingsConfidenceValue");
const engineSelect = document.querySelector("#engineSelect");
const cameraSelect = document.querySelector("#cameraSelect");
const saveSettingsBtn = document.querySelector("#saveSettings");
const settingsSaved = document.querySelector("#settingsSaved");
const customModelInput = document.querySelector("#customModelInput");
const customClassesInput = document.querySelector("#customClasses");
const loadCustomModelBtn = document.querySelector("#loadCustomModel");
const customModelStatus = document.querySelector("#customModelStatus");
const modelState = document.querySelector("#modelState");
const statusEl = document.querySelector("#engineStatus");
const latencyEl = document.querySelector("#latency");
const fpsEl = document.querySelector("#fps");
const objectCountEl = document.querySelector("#objectCount");
const engineEl = document.querySelector("#engine");
const detectionsEl = document.querySelector("#detections");

let session = null;
let stream = null;
let running = false;
let busy = false;
let lastInference = 0;
let fpsFrames = 0;
let fpsLast = performance.now();
let currentDetections = [];

confSlider.addEventListener("input", () => {
  confValue.textContent = `${Math.round(Number(confSlider.value) * 100)}%`;
});

function setStatus(text, good = true) {
  statusEl.textContent = `● ${text}`;
  statusEl.style.color = good ? "var(--accent)" : "var(--danger)";
}

async function loadModel() {
  if (!activeModelUrl) {
    activeModelUrl = DEFAULT_MODEL_URL; activeClassNames = null; activeModelName = "YOLO11n";
    if (modelState) modelState.textContent = "YOLO11n ACTIVE";
  }
  setStatus(`Loading ${activeModelName}…`);
  engineEl.textContent = "Loading";
  const providers = [];
  const requestedEngine = engineSelect?.value || (gpuToggle.checked ? "auto" : "wasm");
  if (requestedEngine !== "wasm" && navigator.gpu) providers.push("webgpu");
  providers.push("wasm");

  try {
    session = await ort.InferenceSession.create(activeModelUrl, {
      executionProviders: providers,
      graphOptimizationLevel: "all"
    });
    const active = providers[0];
    engineEl.textContent = active === "webgpu" ? "WebGPU" : "WASM CPU";
    setStatus(active === "webgpu" ? "WebGPU Edge AI Ready" : "CPU Edge AI Ready");
  } catch (err) {
    console.error(err);
    setStatus("Model load failed", false);
    engineEl.textContent = "Error";
    throw err;
  }
}

function letterbox(sourceW, sourceH, target=640) {
  const scale = Math.min(target / sourceW, target / sourceH);
  const w = Math.round(sourceW * scale);
  const h = Math.round(sourceH * scale);
  return { scale, w, h, dx: (target - w) / 2, dy: (target - h) / 2 };
}

function preprocess() {
  const tmp = document.createElement("canvas");
  tmp.width = 640; tmp.height = 640;
  const tctx = tmp.getContext("2d", {willReadFrequently:true});
  const lb = letterbox(video.videoWidth, video.videoHeight);
  tctx.fillStyle = "#000";
  tctx.fillRect(0,0,640,640);
  tctx.drawImage(video, lb.dx, lb.dy, lb.w, lb.h);
  const image = tctx.getImageData(0,0,640,640).data;
  const input = new Float32Array(1 * 3 * 640 * 640);
  for (let i=0, p=0; i<image.length; i+=4, p++) {
    input[p] = image[i] / 255;
    input[640*640+p] = image[i+1] / 255;
    input[2*640*640+p] = image[i+2] / 255;
  }
  return { input, lb };
}

function iou(a,b) {
  const x1 = Math.max(a.x1,b.x1), y1 = Math.max(a.y1,b.y1);
  const x2 = Math.min(a.x2,b.x2), y2 = Math.min(a.y2,b.y2);
  const inter = Math.max(0,x2-x1) * Math.max(0,y2-y1);
  const areaA = Math.max(0,a.x2-a.x1)*Math.max(0,a.y2-a.y1);
  const areaB = Math.max(0,b.x2-b.x1)*Math.max(0,b.y2-b.y1);
  return inter / Math.max(1e-6, areaA+areaB-inter);
}

function nms(boxes, threshold=0.45) {
  const sorted = [...boxes].sort((a,b)=>b.score-a.score);
  const keep = [];
  while (sorted.length) {
    const best = sorted.shift();
    keep.push(best);
    for (let i=sorted.length-1;i>=0;i--) {
      if (sorted[i].label === best.label && iou(best,sorted[i]) > threshold) sorted.splice(i,1);
    }
  }
  return keep;
}

function decode(output, lb, sourceW, sourceH) {
  // Standard YOLO detection output: [1, 4 + classCount, anchors].
  const data = output.data;
  const dims = output.dims || [1, 84, 8400];
  const channels = dims[dims.length - 2];
  const anchors = dims[dims.length - 1];
  const classCount = channels - 4;
  const names = activeClassNames && activeClassNames.length === classCount ? activeClassNames : COCO;
  const candidates = [];
  const threshold = Number(confSlider.value);

  for (let a=0; a<anchors; a++) {
    let bestClass = -1, bestScore = 0;
    for (let c=0; c<classCount; c++) {
      const score = data[(4+c)*anchors+a];
      if (score > bestScore) { bestScore = score; bestClass = c; }
    }
    if (bestScore < threshold) continue;

    const cx = data[a], cy = data[anchors+a], w = data[2*anchors+a], h = data[3*anchors+a];
    let x1 = (cx-w/2-lb.dx)/lb.scale;
    let y1 = (cy-h/2-lb.dy)/lb.scale;
    let x2 = (cx+w/2-lb.dx)/lb.scale;
    let y2 = (cy+h/2-lb.dy)/lb.scale;

    x1 = Math.max(0,Math.min(sourceW,x1)); y1 = Math.max(0,Math.min(sourceH,y1));
    x2 = Math.max(0,Math.min(sourceW,x2)); y2 = Math.max(0,Math.min(sourceH,y2));

    candidates.push({x1,y1,x2,y2,score:bestScore,label:names[bestClass] ?? `class_${bestClass}`});
  }
  candidates.sort((a, b) => b.score - a.score);
  return nms(candidates.slice(0, 300), 0.45).slice(0, 50);
}

function drawDetections(detections) {
  const w = video.videoWidth, h = video.videoHeight;
  if (!w || !h) return;
  canvas.width = w; canvas.height = h;
  ctx.clearRect(0,0,w,h);
  ctx.lineWidth = Math.max(2, w/500);
  ctx.font = `${Math.max(12,w/75)}px system-ui`;

  for (const d of detections) {
    const color = "#5eead4";
    ctx.strokeStyle = color;
    ctx.fillStyle = "rgba(5,15,28,.78)";
    ctx.strokeRect(d.x1,d.y1,d.x2-d.x1,d.y2-d.y1);
    const text = `${d.label} ${Math.round(d.score*100)}%`;
    const tw = ctx.measureText(text).width + 12;
    const th = 25;
    ctx.fillRect(d.x1, Math.max(0,d.y1-th), tw, th);
    ctx.fillStyle = color;
    ctx.fillText(text, d.x1+6, Math.max(17,d.y1-7));
  }
}

function updateDetectionList(detections) {
  if (!detections.length) {
    detectionsEl.innerHTML = '<div class="empty">No objects above the confidence threshold.</div>';
    return;
  }
  const counts = {};
  for (const d of detections) counts[d.label] = (counts[d.label] || 0) + 1;
  detectionsEl.innerHTML = Object.entries(counts)
    .sort((a,b)=>b[1]-a[1])
    .map(([name,count]) => {
      const best = Math.max(...detections.filter(d=>d.label===name).map(d=>d.score));
      return `<div class="detection-row"><span class="name">${name} × ${count}</span><span class="score">${Math.round(best*100)}%</span></div>`;
    }).join("");
}

async function inferFrame() {
  if (!session || !running || busy || video.readyState < 2) return;
  busy = true;
  const t0 = performance.now();

  try {
    const {input, lb} = preprocess();
    const tensor = new ort.Tensor("float32", input, [1,3,640,640]);
    const inputName = session.inputNames[0];
    const outputMap = await session.run({[inputName]: tensor});
    const output = outputMap[session.outputNames[0]];
    currentDetections = decode(output, lb, video.videoWidth, video.videoHeight);
    drawDetections(currentDetections);
    updateDetectionList(currentDetections);
    objectCountEl.textContent = currentDetections.length;
    const ms = performance.now() - t0;
    latencyEl.textContent = `${Math.round(ms)} ms`;
    fpsFrames++;
    const now = performance.now();
    if (now - fpsLast >= 1000) {
      fpsEl.textContent = String(fpsFrames);
      fpsFrames = 0; fpsLast = now;
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
  inferFrame().finally(() => requestAnimationFrame(loop));
}

async function startCamera() {
  if (!session) await loadModel();
  stream = await navigator.mediaDevices.getUserMedia({
    video: { width: {ideal:1280}, height:{ideal:720}, facingMode: cameraSelect?.value || "environment" },
    audio: false
  });
  video.srcObject = stream;
  await video.play();
  placeholder.style.display = "none";
  startBtn.disabled = true; stopBtn.disabled = false; snapshotBtn.disabled = false;
  running = true;
  setStatus("Live Edge AI Running");
  requestAnimationFrame(loop);
}

function stopCamera() {
  running = false;
  if (stream) stream.getTracks().forEach(t=>t.stop());
  stream = null;
  video.srcObject = null;
  placeholder.style.display = "grid";
  startBtn.disabled = false; stopBtn.disabled = true; snapshotBtn.disabled = true;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  currentDetections = [];
  objectCountEl.textContent = "0";
  detectionsEl.innerHTML = '<div class="empty">Camera stopped.</div>';
  setStatus("AI Ready");
}

snapshotBtn.addEventListener("click", () => {
  if (!video.videoWidth) return;
  const shot = document.createElement("canvas");
  shot.width = video.videoWidth; shot.height = video.videoHeight;
  const sctx = shot.getContext("2d");
  sctx.drawImage(video,0,0);
  for (const d of currentDetections) {
    sctx.strokeStyle="#5eead4"; sctx.lineWidth=4;
    sctx.strokeRect(d.x1,d.y1,d.x2-d.x1,d.y2-d.y1);
    sctx.fillStyle="rgba(5,15,28,.8)";
    sctx.font="20px system-ui";
    sctx.fillText(`${d.label} ${Math.round(d.score*100)}%`,d.x1,d.y1-8);
  }
  const a=document.createElement("a");
  a.download=`edgevision-${Date.now()}.png`;
  a.href=shot.toDataURL("image/png");
  a.click();
});

startBtn.addEventListener("click", async () => {
  try { await startCamera(); }
  catch (err) {
    console.error(err);
    alert("Camera could not start. Allow camera permission and open the site through HTTPS or localhost.");
    setStatus("Camera permission needed", false);
  }
});
stopBtn.addEventListener("click", stopCamera);

if (!window.isSecureContext) {
  setStatus("Use HTTPS / localhost", false);
  engineEl.textContent = "Browser";
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
          <div class="bar-track"><div class="bar-fill" style="width:${Math.max(5, count / max * 100)}%"></div></div>
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

  // Analytics use a stricter threshold and temporal confirmation so brief
  // low-confidence false positives do not flood the history.
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

  // Require the same scene to be observed in 3 processed frames.
  if (evState.stableFrames < 3) return;

  const now = Date.now();
  for (const d of analyticsDetections) {
    // Record a class at most once every 2 seconds. This makes the analytics
    // represent detection events rather than every video frame.
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

// Wrap the inference function after its original definition so every processed frame
// contributes to the local analytics without changing the detector itself.
const detectorInferFrame = inferFrame;
inferFrame = async function () {
  await detectorInferFrame();
  if (running) recordDetectionFrame(currentDetections);
};

snapshotBtn.addEventListener("click", () => {
  evState.snapshots += 1;
  renderAnalytics();
});

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
  const blob = new Blob([csv], {type: "text/csv;charset=utf-8"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `edgevision-detection-history-${Date.now()}.csv`;
  a.click();
  URL.revokeObjectURL(url);
});

// ---------------- Settings & Custom Model Support ----------------
function syncConfidence(value) {
  const v = Number(value);
  confSlider.value = String(v);
  confValue.textContent = `${Math.round(v * 100)}%`;
  if (settingsConfidence) settingsConfidence.value = String(v);
  if (settingsConfidenceValue) settingsConfidenceValue.textContent = `${Math.round(v * 100)}%`;
}
function loadSavedSettings() {
  try {
    const saved = JSON.parse(localStorage.getItem("edgevision-settings") || "{}");
    if (saved.confidence) syncConfidence(saved.confidence);
    if (saved.engine) engineSelect.value = saved.engine;
    if (saved.camera) cameraSelect.value = saved.camera;
    if (saved.classes) customClassesInput.value = saved.classes;
  } catch {}
}
settingsConfidence?.addEventListener("input", e => syncConfidence(e.target.value));
saveSettingsBtn?.addEventListener("click", async () => {
  const data = { confidence:Number(settingsConfidence.value), engine:engineSelect.value, camera:cameraSelect.value, classes:customClassesInput.value };
  localStorage.setItem("edgevision-settings", JSON.stringify(data));
  syncConfidence(data.confidence);
  settingsSaved.textContent = "✓ Settings saved on this device";
  setTimeout(() => settingsSaved.textContent = "", 2500);
  if (!running) { session = null; try { await loadModel(); } catch {} }
});

async function loadCustomModel() {
  const file = customModelInput?.files?.[0];
  if (!file) return alert("Choose a .onnx model file first.");
  if (running) return alert("Stop the camera before loading a different model.");
  const names = customClassesInput.value.split(/\r?\n/).map(x=>x.trim()).filter(Boolean);
  if (!names.length) return alert("Enter the custom model class names, one class per line, in training order.");
  try {
    const buffer = await file.arrayBuffer();
    const providers = [];
    if (engineSelect.value !== "wasm" && navigator.gpu) providers.push("webgpu");
    providers.push("wasm");
    setStatus("Loading custom model…");
    session = await ort.InferenceSession.create(new Uint8Array(buffer), {executionProviders:providers, graphOptimizationLevel:"all"});
    activeModelUrl = null; activeClassNames = names; activeModelName = file.name;
    modelState.textContent = "CUSTOM MODEL ACTIVE";
    customModelStatus.textContent = `✓ Loaded ${file.name} with ${names.length} classes. Local inference is ready.`;
    engineEl.textContent = providers[0] === "webgpu" ? "WebGPU" : "WASM CPU";
    setStatus("Custom Edge AI Ready");
  } catch (err) {
    console.error(err);
    customModelStatus.textContent = "Could not load this ONNX model. Make sure it is a compatible YOLO detection export.";
    setStatus("Custom model failed", false);
  }
}
loadCustomModelBtn?.addEventListener("click", loadCustomModel);
gpuToggle.addEventListener("change", async () => {
  if (running) return;
  session = null;
  try { await loadModel(); } catch {}
});
loadSavedSettings();

renderAnalytics();
