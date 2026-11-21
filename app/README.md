# SORT — live trash detection

A minimal phone interface for this project's YOLO detector. Open it in Safari or
Chrome, start the rear camera, and see detected objects with their labels and
confidence. You can also analyze a photo.

The phone sends JPEG frames to a Python server on your computer. YOLO runs on the
computer, and the browser draws the returned boxes. Frames and predictions are
not saved. There is one request at a time, so a slower CPU does not build up a
queue. Live update speed depends on your computer and checkpoint.

## Start on this computer

### GitHub Pages on a phone

Open [SORT on GitHub Pages](https://mgvillafane.github.io/TrashClassifierYoloCNN/)
in Chrome on Android, wait for the model to load, tap **Start camera**, and allow
camera access.

The Pages build runs `YOLO_full_TACO` inside the phone's browser, using the
exported ONNX model. It needs no Python server, tunnel, local Wi-Fi connection,
or certificate installation. Camera frames are processed in a worker on the
device and are not uploaded. The model and runtime files are public downloads.
The first model download is about 78 MiB, plus the browser runtime. Performance
depends on the phone; GPU acceleration is used when available, with a CPU fallback.

To rebuild the static artifact:

```powershell
uv pip install --python .venv/Scripts/python.exe onnx onnxruntime onnxslim
npm install --prefix app/.runtime/pages-tools --ignore-scripts --save-exact onnxruntime-web@1.31.1
.\.venv\Scripts\python.exe -B app/build_pages.py --export
```

Publish only `app/.pages-dist/` to the Pages source branch, including the ONNX
model, runtime files, and `.nojekyll`. All asset URLs are relative so the app
works at a repository subpath. The export uses the preserved TACO weights with
320-pixel inputs for phone inference. The Python app below keeps 640-pixel inputs.

### Python server

From the repository root in PowerShell, use the existing Python environment:

```powershell
uv pip install --python .venv\Scripts\python.exe -r app/requirements.txt
.\.venv\Scripts\python.exe -B app/run.py
```

Open **http://localhost:8000**. Browser camera access works on localhost.
The launcher also works from other directories; file paths are resolved relative
to `app`, not your current shell directory.

For a fresh environment instead:

```powershell
uv venv app/.venv --python 3.13
uv pip install --python app/.venv/Scripts/python.exe -r app/requirements.txt
.\app\.venv\Scripts\python.exe -B app/run.py
```

## Use your phone's camera

Phone browsers require a **trusted HTTPS connection** for live camera access.
Plain HTTP at your computer's LAN address supports the photo input, but cannot
open the live camera. See [the browser requirement](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia#privacy_and_security).

1. Connect the phone and computer to the same Wi-Fi network.
2. Stop the HTTP server with Ctrl+C, then run:

   ```powershell
   .\.venv\Scripts\python.exe -B app/run.py --https
   ```

3. The terminal prints two addresses. On your phone, open the **certificate setup
   address** (`http://YOUR_COMPUTER_IP:8001`) first. Download its public certificate.
4. Install and trust that local development certificate on your phone:
   - **iPhone:** Settings → General → VPN & Device Management → downloaded SORT
     profile → Install. Then Settings → General → About → Certificate Trust
     Settings → enable full trust for **SORT local development CA**.
   - **Android:** install the downloaded certificate as a **CA certificate** in
     Settings, usually Security & privacy → More security settings → Encryption
     & credentials → Install a certificate → CA certificate. Menu names vary.
5. Open the **app address** printed in the terminal
   (`https://YOUR_COMPUTER_IP:8000`), allow the camera, and tap **Start camera**.

Allow Python through Windows Firewall for your private network if Windows asks.
Keep the computer awake and the server running. A guest Wi-Fi network may block
connections between devices. If the page still has a certificate warning, camera
access may remain blocked until the certificate is trusted.

The setup server on port 8001 serves only instructions and the public certificate.
Private keys remain in ignored `app/.certs/`; never share those keys. Trust this
CA only on your own development devices and remove it when you stop using it.
The CA stays the same between runs; the server certificate picks up current LAN
addresses each time. This is a local prototype without sign-in: keep it on a
trusted network, rather than forwarding its ports to the internet.

If you already have a certificate trusted by your phone, use it directly:

```powershell
.\.venv\Scripts\python.exe -B app/run.py --certfile C:/path/server.pem --keyfile C:/path/server-key.pem
```

You can add the page to your phone's home screen. The server is still required;
this is not an offline detector or an App Store application.

## Update the model here

**Replace `app/models/best.pt`, then tap “Reload model” in the app's Model panel.**
You do not need to rebuild the interface or restart Python. Stop the camera
before replacing a checkpoint, and finish copying the file before reloading.
Only load checkpoints you trust: PyTorch model files may execute Python code.

To keep a checkpoint somewhere else, edit **`app/config.json`**:

```json
{
  "model_path": "models/best.pt",
  "confidence": 0.35,
  "image_size": 640,
  "device": "cpu"
}
```

Relative model paths start at `app/`; absolute paths also work. Tap Reload model
after changing the configuration. Use `"device": "0"` with a CUDA-enabled PyTorch
environment for the first GPU. The included environment uses CPU inference.

The current training defines **cardboard, glass, metal, paper, plastic, trash**.
The app reads class IDs and labels from the checkpoint itself, so a later model
can change the class list. Detection and segmentation checkpoints are supported;
the interface uses their bounding boxes. It does not change material labels into
local recycling or disposal instructions.

Use the **trained `best.pt`**, not the pretrained `yolo26m.pt` (COCO labels).
The older local `demo-2` model uses 60 TACO categories. Model provenance, when
downloaded for this app, is recorded in `app/models/source.json`; checkpoint
binaries are ignored by Git.

If the weights are absent or invalid, the app explains the problem and shows no
invented predictions. The expected six classes are labeled separately until a
checkpoint is loaded.

## Controls

- **Start / Stop camera:** rear camera by default; stopping releases the camera.
- **Flip camera:** switch between rear and front cameras.
- **Analyze photo:** choose or capture a photo without a continuous camera stream.
- **Confidence:** hide detections below the chosen threshold.
- **Model:** inspect active labels and path, then reload a replacement checkpoint.

Switching away from the page stops the camera. Tap Start camera to resume.
The first prediction can take longer while PyTorch initializes. A photo with no
boxes means the model found no objects above the confidence threshold.

## Verify

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s app/tests -v
node --check app/web/app.js
```

The API accepts a raw JPEG, PNG, or WebP image body at
`POST /api/predict?confidence=0.35` and returns normalized boxes and inference
timing. `GET /api/model` reports readiness and actual checkpoint labels;
`POST /api/model/reload` reloads the local configuration and trusted weights.
Uploaded frames are limited to 5 MiB. The backend rejects invalid images and
returns a busy response while another inference or reload is running.
