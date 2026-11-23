"""Build the phone-only GitHub Pages app from the existing SORT interface.

uv pip install --python .venv/Scripts/python.exe onnx onnxruntime onnxslim
npm install --prefix app/.runtime/pages-tools --ignore-scripts onnxruntime-web@1.31.1
python -B app/build_pages.py --export
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


APP = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(export=False):
    catalog = json.loads((APP / "models.json").read_text(encoding="utf-8"))
    models = []
    assets = {}
    exported = set()
    for entry in catalog["models"]:
        detector_path = entry.get("detector_path", entry["path"])
        detector = next(item for item in catalog["models"] if item["path"] == detector_path)
        checkpoint = APP / detector_path
        converted = checkpoint.with_suffix(".onnx")
        if export and detector_path not in exported:
            os.environ.setdefault("YOLO_CONFIG_DIR", str(APP / ".runtime/yolo-settings"))
            from ultralytics import YOLO

            model = YOLO(str(checkpoint))
            if list(model.names.values()) != detector["classes"]:
                raise ValueError(f"{checkpoint.name} has unexpected material classes")
            model.export(format="onnx", imgsz=320, batch=1, dynamic=False,
                         simplify=True, opset=17, device="cpu")
            exported.add(detector_path)
        if not converted.is_file():
            raise FileNotFoundError(f"Run app/build_pages.py --export first: {converted.name}")
        if converted.stat().st_size >= 100 * 1024 * 1024:
            raise ValueError(f"{converted.name} exceeds GitHub's individual-file size limit")
        assets[converted.name] = converted
        metadata = {
            "id": entry["id"], "name": entry["name"],
            "model_url": f"./models/{converted.name}", "classes": entry["classes"],
            "image_size": 320, "confidence": 0.35,
            "model_bytes": converted.stat().st_size, "model_sha256": digest(converted),
            "source_checkpoint": checkpoint.name, "source_sha256": digest(checkpoint),
        }
        if entry.get("type") == "yolo_resnet":
            try:
                from app.classifier import CropClassifier
            except ModuleNotFoundError:
                from classifier import CropClassifier
            classifier_path = APP / entry["path"]
            classifier_onnx = classifier_path.with_suffix(".onnx")
            classifier = CropClassifier(classifier_path)
            if classifier.classes != entry["classes"]:
                raise ValueError("The ResNet checkpoint has unexpected material classes")
            if export:
                classifier.export(classifier_onnx)
            if not classifier_onnx.is_file():
                raise FileNotFoundError("Run app/build_pages.py --export to export the ResNet checkpoint")
            if classifier_onnx.stat().st_size >= 100 * 1024 * 1024:
                raise ValueError("The ResNet export exceeds GitHub's individual-file size limit")
            assets[classifier_onnx.name] = classifier_onnx
            metadata.update(type="yolo_resnet", description=entry["description"],
                detector_classes=detector["classes"], classifier_url=f"./models/{classifier_onnx.name}",
                classifier_preprocessing=classifier.config["preprocessing"],
                classifier_bytes=classifier_onnx.stat().st_size, classifier_sha256=digest(classifier_onnx),
                classifier_source_checkpoint=classifier_path.name, classifier_source_sha256=digest(classifier_path))
        models.append(metadata)
    runtime = APP / ".runtime/pages-tools/node_modules/onnxruntime-web"
    if json.loads((runtime / "package.json").read_text(encoding="utf-8"))["version"] != "1.31.1":
        raise ValueError("Install the pinned onnxruntime-web@1.31.1")
    output = APP / ".pages-dist"
    output.mkdir(exist_ok=True)
    web = APP / "web"
    revision = hashlib.sha256(b"".join((web / name).read_bytes() for name in
        ("index.html", "styles.css", "app.js", "browser-runtime.js", "model-worker.js"))
        + (APP / "models.json").read_bytes()).hexdigest()[:12]
    for name in ("app.js", "styles.css", "icon.svg", "browser-runtime.js", "model-worker.js"):
        shutil.copyfile(web / name, output / name)
    (output / "models").mkdir(exist_ok=True)
    for name, source in assets.items():
        shutil.copyfile(source, output / "models" / name)
    (output / "vendor").mkdir(exist_ok=True)
    runtime_files = ["ort.webgpu.min.mjs"]
    for variant in ("", ".jsep", ".asyncify", ".jspi"):
        runtime_files.extend(f"ort-wasm-simd-threaded{variant}.{extension}" for extension in ("mjs", "wasm"))
    for name in runtime_files:
        shutil.copyfile(runtime / "dist" / name, output / "vendor" / name)
    import onnxruntime

    notices = Path(onnxruntime.__file__).parent
    shutil.copyfile(notices / "LICENSE", output / "vendor/ONNXRUNTIME-LICENSE.txt")
    shutil.copyfile(notices / "ThirdPartyNotices.txt", output / "vendor/ThirdPartyNotices.txt")
    page = (web / "index.html").read_text(encoding="utf-8")
    page = page.replace('/static/', './').replace('href="/"', 'href="./"')
    page = page.replace('<script src="./app.js" defer></script>',
                        '<script src="./browser-runtime.js" defer></script>\n    <script src="./app.js" defer></script>')
    for name in ("styles.css", "app.js", "browser-runtime.js"):
        page = page.replace(f'./{name}"', f'./{name}?v={revision}"')
    page = page.replace('</head>', f'  <meta name="sort-build" content="{revision}">\n  </head>')
    page = page.replace('Connect your YOLO checkpoint', 'Choose a detection model')
    page = page.replace('app/models/best.pt', 'Loading your selected model…')
    page = page.replace('Choose a detection model above. Labels update automatically when you switch.',
                        'Models run on your phone. The YOLO + ResNet option loads both models.')
    page = page.replace('Frames processed on your server. Nothing saved.',
                        'Camera images stay on your device. Nothing uploaded.')
    page = page.replace('A simple live view of your trash detection model.',
                        'Live trash detection on your phone. Camera images stay on your device.')
    (output / "index.html").write_text(page, encoding="utf-8")
    manifest = json.loads((web / "manifest.webmanifest").read_text(encoding="utf-8"))
    manifest.update(start_url="./", scope="./")
    manifest["icons"][0]["src"] = "./icon.svg"
    (output / "manifest.webmanifest").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / ".nojekyll").write_text("", encoding="utf-8")
    metadata = {
        "build_id": revision,
        "default_model": catalog["default_model"], "models": models,
        "runtime_version": "1.31.1", "processing": "browser-only",
    }
    (output / "model.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), **metadata}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", action="store_true", help="Export the registered YOLO and ResNet checkpoints")
    build(parser.parse_args().export)
