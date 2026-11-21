"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const ui = {
    video: $("camera"), photo: $("photo-preview"), overlay: $("overlay"),
    viewfinder: $("viewfinder"), idle: $("viewfinder-idle"), live: $("live-badge"),
    toggle: $("toggle-camera"), toggleLabel: $("camera-button-label"),
    toggleIcon: $("camera-button-icon"), flip: $("flip-camera"),
    choosePhoto: $("choose-photo"), photoInput: $("photo-input"),
    status: $("status-text"), indicator: $("status-indicator"),
    caption: $("scene-caption"), name: $("result-name"),
    description: $("result-description"), score: $("result-confidence"),
    detections: $("detections"), timing: $("timing"),
    classes: $("class-chips"), classesTitle: $("classes-title"),
    classCount: $("class-count"), classNote: $("class-note"),
    settings: $("model-settings"), modelPath: $("model-path"),
    modelSummary: $("model-summary"), modelMessage: $("model-message"),
    reload: $("reload-model"), confidence: $("confidence"),
    confidenceValue: $("confidence-value"),
  };
  const capture = document.createElement("canvas");
  const captureContext = capture.getContext("2d");
  const overlayContext = ui.overlay.getContext("2d");
  const colors = ["#b9df96", "#91d8d0", "#f2c779", "#b5bef2", "#edb1b3", "#d1bc9c"];
  const state = {
    model: null, stream: null, session: 0, request: null, timer: null,
    mode: "idle", opening: false, reloading: false, facing: "environment",
    photoUrl: null, detections: [], sourceWidth: 0, sourceHeight: 0,
    confidence: 0.35, networkFailures: 0,
    hasConfidenceOverride: false, configuredConfidenceApplied: false,
  };

  try {
    const saved = localStorage.getItem("sort-confidence");
    const stored = saved === null || saved.trim() === "" ? NaN : Number(saved);
    if (Number.isFinite(stored) && stored >= 0 && stored <= 1) {
      state.confidence = stored;
      state.hasConfidenceOverride = true;
    }
  } catch { /* Browser storage is optional. */ }
  ui.confidence.value = String(Math.round(state.confidence * 100));
  ui.confidenceValue.textContent = `${Math.round(state.confidence * 100)}%`;

  function status(message, kind = "ready") {
    if (ui.status.textContent !== message) ui.status.textContent = message;
    ui.indicator.dataset.state = kind;
  }

  function label(name) {
    const text = String(name).replaceAll("_", " ");
    return text ? text[0].toUpperCase() + text.slice(1) : "Unknown";
  }

  function color(classId) {
    return colors[Math.abs(Number(classId) || 0) % colors.length];
  }

  function updateControls() {
    const running = state.mode === "camera";
    ui.toggle.disabled = state.opening || state.reloading;
    ui.toggleLabel.textContent = state.opening ? "Opening camera…" : running ? "Stop camera" : "Start camera";
    ui.toggleIcon.setAttribute("href", running ? "#i-stop" : "#i-camera");
    ui.flip.disabled = !running || state.opening || state.reloading;
    ui.choosePhoto.disabled = state.opening || state.reloading;
    ui.reload.disabled = state.reloading;
    ui.reload.textContent = state.reloading ? "Loading model…" : "Reload model";
    ui.live.hidden = !running;
  }

  function clearResults(message = "Nothing scanned yet", detail = "A label will appear when your model sees an object.") {
    state.detections = [];
    ui.name.textContent = message;
    ui.description.textContent = detail;
    ui.score.textContent = "—";
    ui.detections.replaceChildren();
    ui.detections.hidden = true;
    ui.timing.textContent = "Awaiting a scan";
    for (const chip of ui.classes.children) chip.classList.remove("detected");
    drawOverlay();
  }

  function renderModel(model) {
    state.model = model;
    if (!state.configuredConfidenceApplied && !state.hasConfidenceOverride && Number.isFinite(model.confidence) && model.confidence >= 0 && model.confidence <= 1) {
      state.confidence = model.confidence;
      ui.confidence.value = String(model.confidence * 100);
      ui.confidenceValue.textContent = `${Math.round(model.confidence * 100)}%`;
      state.configuredConfidenceApplied = true;
    }
    const actualClasses = Array.isArray(model.classes) ? model.classes : [];
    const classes = model.ready ? actualClasses : (model.expected_classes || []).map((name, id) => ({ id, name }));
    ui.classes.replaceChildren();
    for (const item of classes) {
      const chip = document.createElement("span");
      chip.className = "class-chip";
      chip.dataset.classId = String(item.id);
      chip.textContent = label(item.name);
      ui.classes.append(chip);
    }
    ui.classesTitle.textContent = model.ready ? "Model classes" : "Configured classes";
    ui.classCount.textContent = classes.length ? `${classes.length} classes` : "";
    ui.classNote.textContent = model.ready
      ? "Labels read directly from your active checkpoint."
      : "Available after your trained checkpoint is connected.";
    const modelPath = model.path || "models/best.pt";
    ui.modelPath.textContent = /^(?:[A-Za-z]:|\/|app[\\/])/.test(modelPath) ? modelPath : `app/${modelPath}`;
    ui.modelSummary.textContent = model.ready
      ? `${model.filename || "best.pt"} · ${actualClasses.length} classes`
      : "Checkpoint needed · tap to set up";
    ui.modelMessage.textContent = model.message || "";
    if (!model.ready) {
      ui.settings.open = true;
      status("Add your trained checkpoint to connect the model.", "warning");
      ui.caption.textContent = "Waiting for your model";
    } else if (state.mode === "idle") {
      status("Model connected. Ready to scan.");
      ui.caption.textContent = "Ready when you are";
    }
    updateControls();
  }

  async function readResponse(response) {
    let data;
    try { data = await response.json(); } catch { data = {}; }
    if (!response.ok) {
      const detail = typeof data.detail === "string" ? data.detail : typeof data.message === "string" ? data.message : "The server could not process this frame.";
      const error = new Error(detail);
      error.code = response.status;
      throw error;
    }
    return data;
  }

  async function loadModel(reload = false) {
    if (state.reloading) return;
    if (reload) stopScene();
    state.reloading = true;
    updateControls();
    if (reload) status("Loading your checkpoint…", "loading");
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 60000);
    try {
      const response = await fetch(reload ? "/api/model/reload" : "/api/model", {
        method: reload ? "POST" : "GET", cache: "no-store", signal: controller.signal,
      });
      renderModel(await readResponse(response));
    } catch (error) {
      status(error.name === "AbortError" ? "Model loading timed out. Try reloading." : "Server unavailable. Keep the Python server running, then reload.", "error");
      ui.modelMessage.textContent = error.name === "AbortError" ? "The server took too long to respond." : "Could not reach the model server.";
      if (!state.model) {
        ui.modelSummary.textContent = "Server unavailable";
        ui.settings.open = true;
      }
    } finally {
      clearTimeout(timeout);
      state.reloading = false;
      updateControls();
    }
  }

  function ensureModel() {
    if (state.model?.ready) return true;
    ui.settings.open = true;
    status("Connect your trained checkpoint in Model & settings first.", "warning");
    return false;
  }

  function stopScene(message) {
    state.session += 1;
    clearTimeout(state.timer);
    state.timer = null;
    state.request?.controller.abort();
    state.request = null;
    if (state.stream) state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
    ui.video.srcObject = null;
    ui.video.hidden = true;
    ui.photo.hidden = true;
    ui.photo.removeAttribute("src");
    if (state.photoUrl) URL.revokeObjectURL(state.photoUrl);
    state.photoUrl = null;
    state.mode = "idle";
    state.opening = false;
    state.sourceWidth = 0;
    state.sourceHeight = 0;
    state.networkFailures = 0;
    ui.idle.hidden = false;
    ui.caption.textContent = "Ready when you are";
    clearResults();
    updateControls();
    if (message) status(message);
  }

  function cameraError(error) {
    if (error.name === "NotAllowedError" || error.name === "PermissionDeniedError") {
      return "Camera permission was denied. Allow camera access in your browser settings, or choose a photo.";
    }
    if (error.name === "NotFoundError" || error.name === "DevicesNotFoundError") {
      return "No camera found. You can still choose a photo.";
    }
    if (error.name === "NotReadableError" || error.name === "TrackStartError") {
      return "Your camera is busy. Close other camera apps, then try again.";
    }
    return "Could not open the camera. Try again, or choose a photo.";
  }

  async function startCamera() {
    if (state.opening || state.reloading || !ensureModel()) return;
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      status("Live camera needs HTTPS on your phone. Open the secure app URL, or choose a photo.", "warning");
      return;
    }
    stopScene();
    const session = state.session;
    state.opening = true;
    updateControls();
    status("Waiting for camera access…", "loading");
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: { ideal: state.facing }, width: { ideal: 1280 }, height: { ideal: 960 } },
      });
      if (session !== state.session) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      state.stream = stream;
      ui.video.srcObject = stream;
      await ui.video.play();
      if (session !== state.session) return;
      state.mode = "camera";
      state.opening = false;
      ui.video.hidden = false;
      ui.idle.hidden = true;
      ui.caption.textContent = "Point at an object";
      clearResults("Looking for an object…", "Keep the object in view while your model takes a look.");
      updateControls();
      status("Camera live. Looking for objects.", "live");
      for (const track of stream.getVideoTracks()) {
        track.addEventListener("ended", () => {
          if (stream === state.stream) stopScene("Camera disconnected. Tap Start camera to reconnect.");
        }, { once: true });
      }
      scheduleFrame(session, 0);
    } catch (error) {
      if (session !== state.session) return;
      if (stream) stream.getTracks().forEach((track) => track.stop());
      stopScene();
      status(cameraError(error), "error");
    }
  }

  function scheduleFrame(session, delay) {
    if (session !== state.session || state.mode !== "camera") return;
    clearTimeout(state.timer);
    state.timer = setTimeout(() => scanFrame(session), delay);
  }

  async function makeFrame(source, width, height) {
    const scale = Math.min(1, 960 / Math.max(width, height));
    capture.width = Math.max(1, Math.round(width * scale));
    capture.height = Math.max(1, Math.round(height * scale));
    captureContext.drawImage(source, 0, 0, capture.width, capture.height);
    return new Promise((resolve, reject) => capture.toBlob((blob) => blob ? resolve(blob) : reject(new Error("Could not read this image.")), "image/jpeg", 0.82));
  }

  async function predict(blob, session) {
    if (session !== state.session) return null;
    const controller = new AbortController();
    const request = { controller, session };
    state.request = request;
    let timedOut = false;
    const timeout = setTimeout(() => { timedOut = true; controller.abort(); }, 30000);
    try {
      const response = await fetch(`/api/predict?confidence=${state.confidence}`, {
        method: "POST", headers: { "Content-Type": "image/jpeg" }, body: blob,
        signal: controller.signal, cache: "no-store",
      });
      const result = await readResponse(response);
      return session === state.session ? result : null;
    } catch (error) {
      if (timedOut) {
        const timeoutError = new Error("Inference timed out. The model may still be warming up; try again.");
        timeoutError.code = 408;
        throw timeoutError;
      }
      throw error;
    } finally {
      clearTimeout(timeout);
      if (state.request === request) state.request = null;
    }
  }

  async function scanFrame(session) {
    if (session !== state.session || state.mode !== "camera") return;
    if (!ui.video.videoWidth || ui.video.readyState < 2) {
      scheduleFrame(session, 150);
      return;
    }
    const started = performance.now();
    let delay = 250;
    try {
      const blob = await makeFrame(ui.video, ui.video.videoWidth, ui.video.videoHeight);
      if (session !== state.session) return;
      const result = await predict(blob, session);
      if (!result || session !== state.session) return;
      state.networkFailures = 0;
      renderResults(result);
      status("Camera live. Predictions update automatically.", "live");
      delay = Math.max(25, 250 - (performance.now() - started));
    } catch (error) {
      if (session !== state.session || error.name === "AbortError") return;
      state.networkFailures += 1;
      clearResults("Scan interrupted", "Results will return when the server responds.");
      if (error.code === 503) {
        stopScene();
        status("Model unavailable. Check your checkpoint and reload it.", "error");
        ui.settings.open = true;
        return;
      }
      status(error.code === 429 ? "Server is busy. Trying the next frame…" : error.code === 408 ? error.message : "Connection interrupted. Retrying…", error.code === 429 ? "loading" : "warning");
      delay = Math.min(4000, 750 * state.networkFailures);
    }
    scheduleFrame(session, delay);
  }

  async function scanPhoto(file) {
    if (!file || !ensureModel()) return;
    if (file.size > 30 * 1024 * 1024) {
      status("Choose a photo smaller than 30 MB.", "warning");
      return;
    }
    stopScene();
    const session = state.session;
    state.mode = "photo";
    state.photoUrl = URL.createObjectURL(file);
    ui.photo.src = state.photoUrl;
    status("Opening your photo…", "loading");
    try {
      await ui.photo.decode();
      if (session !== state.session) return;
      ui.photo.hidden = false;
      ui.idle.hidden = true;
      ui.caption.textContent = "Your photo";
      state.sourceWidth = ui.photo.naturalWidth;
      state.sourceHeight = ui.photo.naturalHeight;
      clearResults("Taking a look…", "Your model is analyzing this photo.");
      await analyzePhoto(session);
    } catch (error) {
      if (session !== state.session || error.name === "AbortError") return;
      clearResults("Photo could not be analyzed", "Choose another photo or try again.");
      status(error.code === 429 ? "Server is busy. Choose the photo again in a moment." : error.code === 408 ? error.message : error.code ? error.message : "Could not read this photo or reach the server. Try a JPEG or PNG.", "error");
    }
    updateControls();
  }

  async function analyzePhoto(session) {
    if (session !== state.session || state.mode !== "photo") return;
    status("Analyzing photo…", "loading");
    const blob = await makeFrame(ui.photo, ui.photo.naturalWidth, ui.photo.naturalHeight);
    if (session !== state.session) return;
    const result = await predict(blob, session);
    if (!result || session !== state.session) return;
    renderResults(result);
    status("Photo analyzed. Start the camera for a live view.");
  }

  function renderResults(result) {
    state.detections = (Array.isArray(result.detections) ? result.detections : [])
      .filter((item) => Number.isFinite(item.confidence) && Array.isArray(item.box) && item.box.length === 4 && item.box.every(Number.isFinite))
      .sort((a, b) => b.confidence - a.confidence);
    state.sourceWidth = Number(result.width) || 1;
    state.sourceHeight = Number(result.height) || 1;
    ui.timing.textContent = `${Math.round(Number(result.inference_ms) || 0)} ms / frame`;
    const count = state.detections.length;
    if (!count) {
      ui.name.textContent = "No object detected";
      ui.description.textContent = "Try moving closer or lowering the minimum confidence.";
      ui.score.textContent = "—";
      ui.caption.textContent = state.mode === "camera" ? "Looking for an object" : "No detections in this photo";
    } else {
      const top = state.detections[0];
      ui.name.textContent = label(top.class_name);
      ui.score.textContent = `${Math.round(top.confidence * 100)}%`;
      ui.description.textContent = count === 1 ? "One object detected in this frame." : `${count} objects detected. Highest confidence shown above.`;
      ui.caption.textContent = `${label(top.class_name)} · ${Math.round(top.confidence * 100)}%${count > 1 ? ` · ${count} objects` : ""}`;
    }
    const grouped = new Map();
    for (const item of state.detections) {
      const key = String(item.class_id);
      if (grouped.has(key)) grouped.get(key).count += 1;
      else grouped.set(key, { ...item, count: 1 });
    }
    ui.detections.replaceChildren();
    if (count > 1) {
      for (const item of grouped.values()) {
        const row = document.createElement("div");
        row.className = "detection-row";
        const title = document.createElement("span");
        const dot = document.createElement("span");
        dot.className = "detection-color";
        dot.style.backgroundColor = color(item.class_id);
        title.append(dot, document.createTextNode(label(item.class_name)));
        if (item.count > 1) {
          const copies = document.createElement("small");
          copies.textContent = `×${item.count}`;
          title.append(copies);
        }
        const score = document.createElement("span");
        score.textContent = `${Math.round(item.confidence * 100)}%`;
        row.append(title, score);
        ui.detections.append(row);
      }
    }
    ui.detections.hidden = count <= 1;
    for (const chip of ui.classes.children) chip.classList.toggle("detected", grouped.has(chip.dataset.classId));
    drawOverlay();
  }

  function drawOverlay() {
    const width = ui.viewfinder.clientWidth;
    const height = ui.viewfinder.clientHeight;
    if (!width || !height) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 3);
    ui.overlay.width = Math.round(width * ratio);
    ui.overlay.height = Math.round(height * ratio);
    overlayContext.setTransform(ratio, 0, 0, ratio, 0, 0);
    overlayContext.clearRect(0, 0, width, height);
    if (!state.sourceWidth || !state.sourceHeight) return;
    const scale = Math.min(width / state.sourceWidth, height / state.sourceHeight);
    const contentWidth = state.sourceWidth * scale;
    const contentHeight = state.sourceHeight * scale;
    const offsetX = (width - contentWidth) / 2;
    const offsetY = (height - contentHeight) / 2;
    overlayContext.font = "500 12px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
    overlayContext.lineWidth = 2;
    for (const item of state.detections) {
      const [x1, y1, x2, y2] = item.box.map((value) => Math.max(0, Math.min(1, value)));
      const x = offsetX + x1 * contentWidth;
      const y = offsetY + y1 * contentHeight;
      const boxWidth = (x2 - x1) * contentWidth;
      const boxHeight = (y2 - y1) * contentHeight;
      if (boxWidth <= 0 || boxHeight <= 0) continue;
      const ink = color(item.class_id);
      overlayContext.strokeStyle = ink;
      overlayContext.strokeRect(x, y, boxWidth, boxHeight);
      let title = `${label(item.class_name)} ${Math.round(item.confidence * 100)}%`;
      const maxWidth = Math.max(40, width - 24);
      while (overlayContext.measureText(title).width + 16 > maxWidth && title.length > 8) title = title.slice(0, -2) + "…";
      const titleWidth = Math.min(maxWidth, overlayContext.measureText(title).width + 16);
      const titleX = Math.max(0, Math.min(x, width - titleWidth));
      const titleY = Math.max(0, y - 25);
      overlayContext.fillStyle = ink;
      overlayContext.fillRect(titleX, titleY, titleWidth, 25);
      overlayContext.fillStyle = "#17271e";
      overlayContext.fillText(title, titleX + 8, titleY + 17, titleWidth - 12);
    }
  }

  ui.toggle.addEventListener("click", () => state.mode === "camera" ? stopScene("Camera stopped. Ready for your next scan.") : startCamera());
  ui.flip.addEventListener("click", () => {
    state.facing = state.facing === "environment" ? "user" : "environment";
    startCamera();
  });
  ui.choosePhoto.addEventListener("click", () => { if (ensureModel()) ui.photoInput.click(); });
  ui.photoInput.addEventListener("change", () => {
    const file = ui.photoInput.files?.[0];
    ui.photoInput.value = "";
    scanPhoto(file);
  });
  ui.reload.addEventListener("click", () => loadModel(true));
  ui.confidence.addEventListener("input", () => {
    state.confidence = Number(ui.confidence.value) / 100;
    state.hasConfidenceOverride = true;
    ui.confidenceValue.textContent = `${ui.confidence.value}%`;
    try { localStorage.setItem("sort-confidence", String(state.confidence)); } catch { /* Optional preference only. */ }
    if (state.mode === "camera") {
      state.session += 1;
      state.request?.controller.abort();
      state.request = null;
      clearResults("Updating confidence…", "The next frame will use your new threshold.");
      scheduleFrame(state.session, 250);
    } else if (state.mode === "photo" && state.photoUrl && !ui.photo.hidden && ui.photo.complete && ui.photo.naturalWidth) {
      state.session += 1;
      state.request?.controller.abort();
      state.request = null;
      clearResults("Updating confidence…", "Your photo will be analyzed with the new threshold.");
      clearTimeout(state.timer);
      const session = state.session;
      state.timer = setTimeout(() => analyzePhoto(session).catch((error) => {
        if (session !== state.session || error.name === "AbortError") return;
        clearResults("Photo could not be analyzed", "Try changing confidence again, or choose another photo.");
        status(error.code === 429 ? "Server is busy. Change confidence again in a moment to retry." : error.code === 408 ? error.message : "Could not reach the server. Try again.", "error");
      }), 400);
    }
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden && (state.mode === "camera" || state.opening)) stopScene("Camera paused while away. Tap Start camera to resume.");
  });
  window.addEventListener("pagehide", () => stopScene());
  window.addEventListener("offline", () => status("You’re offline. Reconnect to your model server to scan.", "warning"));
  window.addEventListener("online", () => {
    if (state.mode === "idle") loadModel();
  });
  if ("ResizeObserver" in window) new ResizeObserver(drawOverlay).observe(ui.viewfinder);
  else window.addEventListener("resize", drawOverlay);
  loadModel();
})();
