"""Build the phone-only GitHub Pages app from the existing SORT interface.

uv pip install --python .venv/Scripts/python.exe onnx onnxruntime onnxslim
npm install --prefix app/.runtime/pages-tools --ignore-scripts onnxruntime-web@1.31.1
python -B app/build_pages.py --export
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


APP = Path(__file__).resolve().parent
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(export=False):
    checkpoint = APP / "models/YOLO_full_TACO.pt"
    converted = checkpoint.with_suffix(".onnx")
    if export:
        from ultralytics import YOLO

        model = YOLO(str(checkpoint))
        if list(model.names.values()) != CLASSES:
            raise ValueError("The TACO checkpoint has unexpected material classes")
        model.export(format="onnx", imgsz=320, batch=1, dynamic=False,
                     simplify=True, opset=17, device="cpu")
    if not converted.is_file():
        raise FileNotFoundError("Run app/build_pages.py --export first")
    if converted.stat().st_size >= 100 * 1024 * 1024:
        raise ValueError("The model exceeds GitHub's individual-file size limit")
    runtime = APP / ".runtime/pages-tools/node_modules/onnxruntime-web"
    if json.loads((runtime / "package.json").read_text(encoding="utf-8"))["version"] != "1.31.1":
        raise ValueError("Install the pinned onnxruntime-web@1.31.1")
    output = APP / ".pages-dist"
    output.mkdir(exist_ok=True)
    web = APP / "web"
    for name in ("app.js", "styles.css", "icon.svg", "browser-runtime.js", "model-worker.js"):
        shutil.copyfile(web / name, output / name)
    (output / "models").mkdir(exist_ok=True)
    shutil.copyfile(converted, output / "models/YOLO_full_TACO.onnx")
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
    page = page.replace('Connect your YOLO checkpoint', 'YOLO_full_TACO · on your device')
    page = page.replace('UPDATE YOUR MODEL HERE', 'YOUR MODEL')
    page = page.replace('app/models/best.pt', 'YOLO_full_TACO')
    page = page.replace('Replace this checkpoint on your server, then reload. Class names update automatically.',
                        'The TACO-trained model runs on your phone. The first download is about 78 MB; speed depends on your device.')
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
        "model_url": "./models/YOLO_full_TACO.onnx",
        "classes": CLASSES, "image_size": 320, "confidence": 0.35,
        "model_bytes": converted.stat().st_size, "model_sha256": digest(converted),
        "source_checkpoint": checkpoint.name, "source_sha256": digest(checkpoint),
        "runtime_version": "1.31.1", "processing": "browser-only",
    }
    (output / "model.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), **metadata}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", action="store_true", help="Export the current TACO checkpoint again")
    build(parser.parse_args().export)
