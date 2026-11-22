"use strict";

// Used only by the GitHub Pages build. Camera frames stay inside this browser.
(() => {
  let worker;
  let loading;
  let metadata;
  let manifest;
  let busy = false;
  let sequence = 0;
  let progress;
  const requests = new Map();

  function resetWorker() {
    worker?.terminate();
    worker = null;
    for (const pending of requests.values()) pending.reject(new Error("Model changed. Start a new scan."));
    requests.clear();
    metadata = null;
    loading = null;
  }

  async function catalog() {
    if (manifest) return manifest;
    const response = await fetch(new URL("./model.json", document.baseURI), { cache: "no-store" });
    if (!response.ok) throw new Error("The model manifest is unavailable.");
    manifest = await response.json();
    return manifest;
  }

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
    catalog,
    async load(onProgress, modelId = null, force = false) {
      progress = onProgress;
      const manifest = await catalog();
      const selected = manifest.models.find((model) => model.id === (modelId || manifest.default_model));
      if (!selected) throw new Error("Unknown model. Choose one of the listed models.");
      if (metadata?.selected_model === selected.id && !force) return metadata;
      if (loading) return loading;
      resetWorker();
      createWorker();
      loading = (async () => {
        onProgress?.("Downloading your model. Camera images stay on this device.");
        const result = await request("load", {
          url: new URL(selected.model_url, document.baseURI).href,
          name: selected.name,
          classes: selected.classes,
          imageSize: selected.image_size,
          backend: new URLSearchParams(location.search).get("backend") || "auto",
        });
        metadata = {
          ready: true, filename: selected.model_url.split("/").pop(), path: selected.model_url,
          name: selected.name, selected_model: selected.id, available_models: manifest.models,
          classes: selected.classes.map((name, id) => ({ id, name })),
          expected_classes: selected.classes, confidence: selected.confidence,
          message: `${selected.name} runs on this device (${result.backend === "webgpu" ? "GPU" : "CPU"}). Camera images are not uploaded.`,
        };
        return metadata;
      })().then((result) => {
        loading = null;
        return result;
      }).catch((error) => {
        resetWorker();
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
