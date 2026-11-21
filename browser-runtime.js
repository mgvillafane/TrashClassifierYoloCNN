"use strict";

// Used only by the GitHub Pages build. Camera frames stay inside this browser.
(() => {
  let worker;
  let loading;
  let metadata;
  let busy = false;
  let sequence = 0;
  let progress;
  const requests = new Map();

  function request(type, payload = {}) {
    return new Promise((resolve, reject) => {
      const id = ++sequence;
      requests.set(id, { resolve, reject });
      worker.postMessage({ id, type, ...payload });
    });
  }

  function createWorker() {
    worker = new Worker(new URL("./model-worker.js", document.baseURI), { type: "module" });
    worker.onmessage = ({ data }) => {
      if (data.type === "progress") {
        progress?.(data.message);
        return;
      }
      const pending = requests.get(data.id);
      if (!pending) return;
      requests.delete(data.id);
      if (data.error) pending.reject(new Error(data.error));
      else pending.resolve(data.result);
    };
    worker.onerror = () => {
      for (const pending of requests.values()) pending.reject(new Error("Your browser could not start the model. Try the latest Chrome."));
      requests.clear();
      loading = null;
      metadata = null;
      worker.terminate();
      worker = null;
    };
  }

  window.sortBrowserRuntime = {
    async load(onProgress) {
      progress = onProgress;
      if (metadata) return metadata;
      if (loading) return loading;
      if (!worker) createWorker();
      loading = (async () => {
        onProgress?.("Downloading your model. Camera images stay on this device.");
        const response = await fetch(new URL("./model.json", document.baseURI));
        if (!response.ok) throw new Error("The model manifest is unavailable.");
        const manifest = await response.json();
        const result = await request("load", {
          url: new URL(manifest.model_url, document.baseURI).href,
          classes: manifest.classes,
          imageSize: manifest.image_size,
          backend: new URLSearchParams(location.search).get("backend") || "auto",
        });
        metadata = {
          ready: true, filename: "YOLO_full_TACO.onnx", path: "models/YOLO_full_TACO.onnx",
          classes: manifest.classes.map((name, id) => ({ id, name })),
          expected_classes: manifest.classes, confidence: manifest.confidence,
          message: `YOLO_full_TACO runs on this device (${result.backend === "webgpu" ? "GPU" : "CPU"}). Camera images are not uploaded.`,
        };
        return metadata;
      })().catch((error) => {
        loading = null;
        throw error;
      });
      return loading;
    },
    async predict(blob, confidence) {
      if (!metadata) throw new Error("Load the model first.");
      if (busy) {
        const error = new Error("The model is still processing a frame.");
        error.code = 429;
        throw error;
      }
      busy = true;
      try { return await request("predict", { blob, confidence }); }
      finally { busy = false; }
    },
  };
})();
