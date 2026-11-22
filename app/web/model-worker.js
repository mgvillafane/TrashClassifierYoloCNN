import * as ort from "./vendor/ort.webgpu.min.mjs";

ort.env.wasm.wasmPaths = new URL("./vendor/", import.meta.url).href;
// GitHub Pages does not supply the isolation headers required for WASM threads.
ort.env.wasm.numThreads = 1;
ort.env.logLevel = "error";

let session;
let classes;
let size;
const canvas = new OffscreenCanvas(1, 1);
const context = canvas.getContext("2d", { willReadFrequently: true });

function progress(message) { self.postMessage({ type: "progress", message }); }

async function loadModel(message) {
  if (session) await session.release();
  session = null;
  classes = message.classes;
  size = message.imageSize;
  const response = await fetch(message.url);
  if (!response.ok) throw new Error(`Model download failed (${response.status}).`);
  const total = Number(response.headers.get("content-length"));
  const reader = response.body.getReader();
  const chunks = [];
  let length = 0;
  let lastPercent = -1;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    length += value.length;
    const percent = total ? Math.floor(length / total * 100) : -1;
    if (percent !== lastPercent) {
      progress(`Downloading ${message.name}${percent >= 0 ? ` · ${percent}%` : "…"}`);
      lastPercent = percent;
    }
  }
  const buffer = new Uint8Array(length);
  let offset = 0;
  for (const chunk of chunks) { buffer.set(chunk, offset); offset += chunk.length; }
  chunks.length = 0;
  progress("Preparing your model on this device. First load can take a moment.");
  let backend = "wasm";
  if (message.backend !== "wasm" && navigator.gpu) {
    try {
      session = await ort.InferenceSession.create(buffer, { executionProviders: ["webgpu", "wasm"], graphOptimizationLevel: "all" });
      backend = "webgpu";
    } catch { /* Fall back to the portable CPU runtime. */ }
  }
  if (!session) session = await ort.InferenceSession.create(buffer, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  return { backend };
}

function intersectionOverUnion(a, b) {
  const area = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])) * Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - area;
  return union > 0 ? area / union : 0;
}

function decode(output, confidence, width, height, scale, padX, padY) {
  const data = output.data;
  const [, rows, columns] = output.dims;
  const candidates = [];
  const raw = rows === classes.length + 4 || columns === classes.length + 4;
  const count = raw && rows === classes.length + 4 ? columns : rows;
  const at = (row, field) => raw && rows === classes.length + 4 ? data[field * count + row] : data[row * columns + field];
  if (!raw && columns !== 6) throw new Error(`Unsupported model output shape: ${output.dims.join("×")}`);
  for (let row = 0; row < count; row++) {
    let score;
    let identifier;
    let coordinates;
    if (raw) {
      score = -Infinity;
      for (let id = 0; id < classes.length; id++) {
        const value = at(row, 4 + id);
        if (value > score) { score = value; identifier = id; }
      }
      if (score < confidence) continue;
      const [x, y, w, h] = [0, 1, 2, 3].map((field) => at(row, field));
      coordinates = [x - w / 2, y - h / 2, x + w / 2, y + h / 2];
    } else {
      score = at(row, 4);
      identifier = Math.round(at(row, 5));
      if (score < confidence) continue;
      coordinates = [0, 1, 2, 3].map((field) => at(row, field));
    }
    if (!Number.isFinite(score) || !coordinates.every(Number.isFinite) || !classes[identifier]) continue;
    candidates.push({ score, identifier, coordinates });
  }
  candidates.sort((a, b) => b.score - a.score);
  const selected = [];
  for (const candidate of candidates) {
    if (raw && selected.some((previous) => previous.identifier === candidate.identifier && intersectionOverUnion(previous.coordinates, candidate.coordinates) > 0.7)) continue;
    selected.push(candidate);
    if (selected.length === 30) break;
  }
  return selected.map(({ score, identifier, coordinates }) => ({
    class_id: identifier, class_name: classes[identifier], confidence: score,
    box: coordinates.map((coordinate, index) => Math.max(0, Math.min(1, (coordinate - (index % 2 ? padY : padX)) / scale / (index % 2 ? height : width)))),
  }));
}

async function predict({ blob, confidence }) {
  if (!session) throw new Error("The model has not finished loading.");
  const bitmap = await createImageBitmap(blob);
  const width = bitmap.width;
  const height = bitmap.height;
  const started = performance.now();
  const scale = Math.min(size / width, size / height);
  const resizedWidth = Math.round(width * scale);
  const resizedHeight = Math.round(height * scale);
  const padX = Math.round((size - resizedWidth) / 2 - 0.1);
  const padY = Math.round((size - resizedHeight) / 2 - 0.1);
  canvas.width = canvas.height = size;
  context.fillStyle = "rgb(114,114,114)";
  context.fillRect(0, 0, size, size);
  context.drawImage(bitmap, padX, padY, resizedWidth, resizedHeight);
  bitmap.close();
  const pixels = context.getImageData(0, 0, size, size).data;
  const plane = size * size;
  const input = new Float32Array(3 * plane);
  for (let index = 0; index < plane; index++) {
    input[index] = pixels[index * 4] / 255;
    input[plane + index] = pixels[index * 4 + 1] / 255;
    input[2 * plane + index] = pixels[index * 4 + 2] / 255;
  }
  const tensor = new ort.Tensor("float32", input, [1, 3, size, size]);
  const outputs = await session.run({ [session.inputNames[0]]: tensor });
  try {
    const detections = decode(outputs[session.outputNames[0]], confidence, width, height, scale, padX, padY);
    return { detections, width, height, inference_ms: performance.now() - started };
  } finally {
    tensor.dispose();
    for (const value of Object.values(outputs)) value.dispose();
  }
}

self.onmessage = async ({ data }) => {
  try {
    const result = data.type === "load" ? await loadModel(data) : await predict(data);
    self.postMessage({ id: data.id, result });
  } catch (error) {
    self.postMessage({ id: data.id, error: error.message || String(error) });
  }
};
