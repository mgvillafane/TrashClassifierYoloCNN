"""Serve the phone UI and run bounded, sequential YOLO inference.

Select a checkpoint registered in models.json, or set config.json's model_path.
Checkpoints are local trusted files; this server never downloads weights.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from io import BytesIO
import json
import logging
import math
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Callable
import warnings

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

from app.classifier import CropClassifier


APP_DIR = Path(__file__).resolve().parent
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGE_PIXELS = 12_000_000
LOGGER = logging.getLogger(__name__)


def expected_classes() -> list[str]:
    """Training labels are a hint only; the loaded checkpoint owns its labels."""
    mapping = APP_DIR.parent / "taco_trashnet_mapping.json"
    try:
        return json.loads(mapping.read_text(encoding="utf-8"))["class_names"]
    except (OSError, ValueError, KeyError):
        return ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


@dataclass(frozen=True)
class Settings:
    model_path: str = "models/best.pt"
    confidence: float = 0.35
    image_size: int = 640
    device: str = "cpu"

    @classmethod
    def read(cls, path: Path) -> "Settings":
        data = json.loads(path.read_text(encoding="utf-8"))
        settings = cls(**data)
        if not isinstance(settings.model_path, str) or not settings.model_path.strip():
            raise ValueError("model_path must name a local trained .pt checkpoint.")
        if not isinstance(settings.confidence, (int, float)) or not 0 <= settings.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1.")
        if isinstance(settings.image_size, bool) or not isinstance(settings.image_size, int):
            raise ValueError("image_size must be an integer.")
        if not 160 <= settings.image_size <= 1280 or settings.image_size % 32:
            raise ValueError("image_size must be a multiple of 32 between 160 and 1280.")
        if not isinstance(settings.device, str) or not settings.device:
            raise ValueError("device must be a nonempty string, such as cpu or 0.")
        return settings


def load_yolo(path: Path):
    # Import only when a local checkpoint exists. Missing-model UI and tests stay light.
    from ultralytics import YOLO

    return YOLO(str(path))


class ModelRuntime:
    def __init__(self, config_path: Path, loader: Callable = load_yolo, classifier_loader: Callable = CropClassifier):
        self.config_path = Path(config_path).resolve()
        self.loader = loader
        self.classifier_loader = classifier_loader
        self.lock = Lock()
        self.model = None
        self.classifier = None
        self.selected_model_id = None
        self.settings = Settings()
        self.status = self._status(False, "Model has not been loaded yet.")

    def _status(self, ready: bool, message: str, names: dict | None = None) -> dict:
        choices = self.model_choices()
        selected = next((item for item in choices if item["id"] == self.selected_model_id), None)
        if selected is None:
            selected = next((item for item in choices if item["path"] == self.settings.model_path), None)
        return {
            "ready": ready,
            "filename": Path(self.settings.model_path).name,
            "path": self.settings.model_path,
            "classes": [{"id": int(key), "name": str(value)} for key, value in (names or {}).items()],
            "expected_classes": selected["classes"] if selected else expected_classes(),
            "available_models": choices,
            "selected_model": selected["id"] if selected else None,
            "name": selected["name"] if selected else Path(self.settings.model_path).name,
            "description": selected.get("description", "") if selected else "",
            "confidence": self.settings.confidence,
            "message": message,
        }

    def metadata(self) -> dict:
        return dict(self.status)

    def model_choices(self) -> list[dict]:
        catalog = self.config_path.parent / "models.json"
        if not catalog.is_file():
            return []
        choices = json.loads(catalog.read_text(encoding="utf-8"))["models"]
        return [{**item, "available": (self.config_path.parent / item["path"]).is_file()
                 and (not item.get("detector_path") or (self.config_path.parent / item["detector_path"]).is_file())}
                for item in choices]

    def reload(self, model_id: str | None = None) -> dict:
        if not self.lock.acquire(blocking=False):
            raise HTTPException(429, "Model is busy. Stop scanning and try again.")
        try:
            selected = None
            if model_id is not None:
                selected = next((item for item in self.model_choices() if item["id"] == model_id), None)
                if selected is None:
                    raise HTTPException(400, "Unknown model. Choose one of the listed models.")
            self.model = None
            self.classifier = None
            self.status = self._status(False, "Loading model…")
            self.settings = Settings.read(self.config_path)
            if selected:
                self.settings = replace(self.settings, model_path=selected["path"])
            else:
                selected = next((item for item in self.model_choices() if item["path"] == self.settings.model_path), None)
            self.selected_model_id = selected["id"] if selected else None
            configured = Path(self.settings.model_path).expanduser()
            path = configured if configured.is_absolute() else self.config_path.parent / configured
            path = path.resolve()
            if path.suffix.lower() != ".pt":
                raise ValueError("Choose a trained Ultralytics .pt checkpoint in config.json.")
            if not path.is_file():
                self.status = self._status(
                    False,
                    f"No model at {self.settings.model_path}. Copy your trained best.pt there, "
                    "or edit model_path in app/config.json, then reload.",
                )
                return self.metadata()
            classifier = None
            if selected and selected.get("type") == "yolo_resnet":
                detector_path = (self.config_path.parent / selected["detector_path"]).resolve()
                if not detector_path.is_file():
                    raise ValueError("The selected pipeline's YOLO checkpoint is missing.")
                model = self.loader(detector_path)
                classifier = self.classifier_loader(path)
                if list(classifier.classes) != selected["classes"]:
                    raise ValueError("The classifier labels do not match the selected pipeline.")
            else:
                model = self.loader(path)
            if model.task not in {"detect", "segment"}:
                raise ValueError("Use an object detection or segmentation checkpoint, not a classification model.")
            raw_names = classifier.classes if classifier else model.names
            names = dict(enumerate(raw_names)) if isinstance(raw_names, (list, tuple)) else dict(raw_names)
            names = dict(sorted((int(key), str(value)) for key, value in names.items()))
            if not names:
                raise ValueError("This checkpoint does not contain class names.")
            self.model = model
            self.classifier = classifier
            message = selected["description"] if classifier else f"{len(names)} classes loaded from {path.name}."
            self.status = self._status(True, message, names)
        except HTTPException:
            raise
        except Exception as exc:
            LOGGER.warning("Could not load configured model: %s", exc)
            self.status = self._status(False, f"Could not load model: {exc}")
        finally:
            self.lock.release()
        return self.metadata()

    def predict(self, payload: bytes, confidence: float | None) -> dict:
        if not self.lock.acquire(blocking=False):
            raise HTTPException(429, "Model is busy. Try the next frame.")
        try:
            if self.model is None:
                raise HTTPException(503, self.status["message"])
            image = decode_image(payload)
            width, height = image.size
            started = perf_counter()
            try:
                result = self.model.predict(
                    source=image,
                    conf=self.settings.confidence if confidence is None else confidence,
                    imgsz=self.settings.image_size,
                    device=self.settings.device,
                    max_det=30,
                    save=False,
                    verbose=False,
                )[0]
                detections = []
                boxes = result.boxes
                if boxes is not None:
                    # One CPU transfer per tensor, rather than one per detection.
                    coordinates = boxes.xyxy.cpu().tolist()
                    scores = boxes.conf.cpu().tolist()
                    identifiers = boxes.cls.cpu().tolist()
                    valid = [index for index, (xyxy, score) in enumerate(zip(coordinates, scores))
                             if all(math.isfinite(float(value)) for value in [*xyxy, score])
                             and xyxy[2] > xyxy[0] and xyxy[3] > xyxy[1]]
                    coordinates = [coordinates[index] for index in valid]
                    scores = [scores[index] for index in valid]
                    identifiers = [identifiers[index] for index in valid]
                    if self.classifier:
                        crops = [image.crop((max(0, math.floor(x1)), max(0, math.floor(y1)),
                                    min(width, max(math.floor(x1) + 1, math.ceil(x2))),
                                    min(height, max(math.floor(y1) + 1, math.ceil(y2)))))
                                 for x1, y1, x2, y2 in coordinates]
                        classified = self.classifier.predict(crops, device=self.settings.device)
                        if len(classified) != len(coordinates):
                            raise ValueError("The classifier did not return one label per crop.")
                        identifiers = [item[0] for item in classified]
                        scores = [item[1] for item in classified]
                    names = {entry["id"]: entry["name"] for entry in self.status["classes"]}
                    for xyxy, score, identifier in zip(coordinates, scores, identifiers):
                        identifier = int(identifier)
                        if not all(math.isfinite(float(value)) for value in [*xyxy, score]):
                            continue
                        if self.classifier and score < (self.settings.confidence if confidence is None else confidence):
                            continue
                        box = [max(0.0, min(1.0, float(value) / divisor))
                               for value, divisor in zip(xyxy, [width, height, width, height])]
                        detections.append({
                            "class_id": identifier,
                            "class_name": names.get(identifier, str(identifier)),
                            "confidence": round(float(score), 5),
                            "box": box,
                        })
                detections.sort(key=lambda item: item["confidence"], reverse=True)
                return {
                    "detections": detections,
                    "inference_ms": round((perf_counter() - started) * 1000, 1),
                    "width": width,
                    "height": height,
                }
            except Exception as exc:
                LOGGER.exception("Inference failed")
                raise HTTPException(500, "Inference failed. Check the server console and model configuration.") from exc
        finally:
            self.lock.release()


def decode_image(payload: bytes) -> Image.Image:
    if not payload:
        raise HTTPException(400, "Send a JPEG, PNG, or WebP image.")
    if len(payload) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image is too large. Maximum upload is 5 MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(payload)) as image:
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise HTTPException(413, "Image exceeds 12 megapixels. Choose a smaller image.")
                if image.format not in {"JPEG", "PNG", "WEBP"}:
                    raise HTTPException(400, "Use a JPEG, PNG, or WebP image.")
                return ImageOps.exif_transpose(image).convert("RGB")
    except HTTPException:
        raise
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise HTTPException(413, "Image dimensions are too large.") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(400, "Could not read this image. Use JPEG, PNG, or WebP.") from exc


def create_app(runtime: ModelRuntime | None = None) -> FastAPI:
    runtime = runtime or ModelRuntime(APP_DIR / "config.json")

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        await run_in_threadpool(runtime.reload)
        yield

    application = FastAPI(title="Trash Lens", lifespan=lifespan)
    application.state.runtime = runtime

    @application.middleware("http")
    async def response_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Permissions-Policy"] = "camera=(self), microphone=()"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @application.get("/api/model")
    def model_info():
        return runtime.metadata()

    @application.post("/api/model/reload")
    async def reload_model(model_id: str | None = Query(None)):
        return await run_in_threadpool(runtime.reload, model_id)

    @application.post("/api/predict")
    async def predict(request: Request, confidence: float | None = Query(None, ge=0, le=1)):
        if not runtime.metadata()["ready"]:
            raise HTTPException(503, runtime.metadata()["message"])
        length = request.headers.get("content-length")
        if length:
            try:
                if int(length) > MAX_IMAGE_BYTES:
                    raise HTTPException(413, "Image is too large. Maximum upload is 5 MB.")
            except ValueError as exc:
                raise HTTPException(400, "Invalid content length.") from exc
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > MAX_IMAGE_BYTES:
                raise HTTPException(413, "Image is too large. Maximum upload is 5 MB.")
            body.extend(chunk)
        return await run_in_threadpool(runtime.predict, bytes(body), confidence)

    @application.get("/", include_in_schema=False)
    def index():
        return FileResponse(APP_DIR / "web" / "index.html", headers={"Cache-Control": "no-cache"})

    @application.get("/setup/local-ca.crt", include_in_schema=False)
    def local_certificate():
        certificate = APP_DIR / ".certs" / "local-ca.crt"
        if not certificate.is_file():
            raise HTTPException(404, "Start the app with --https to create the local certificate.")
        return FileResponse(certificate, media_type="application/x-x509-ca-cert", filename="sort-local-ca.crt")

    application.mount("/static", StaticFiles(directory=APP_DIR / "web", check_dir=False), name="static")
    return application


app = create_app()
