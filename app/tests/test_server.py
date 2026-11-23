"""API checks use a fake detector: no GPU, checkpoint, or network is needed."""

from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fastapi.testclient import TestClient
from PIL import Image

from app.server import MAX_IMAGE_BYTES, ModelRuntime, create_app


class Tensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class Boxes:
    xyxy = Tensor([[10, 5, 90, 45], [-5, -5, 110, 55]])
    conf = Tensor([0.9, 0.4])
    cls = Tensor([1, 0])


class Result:
    boxes = Boxes()


class Detector:
    task = "detect"
    names = {0: "cardboard", 1: "plastic"}

    def predict(self, **kwargs):
        self.last_options = kwargs
        return [Result()]


def image_bytes(size=(100, 50)):
    output = BytesIO()
    Image.new("RGB", size, "white").save(output, format="JPEG")
    return output.getvalue()


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.config = self.root / "config.json"
        self.write_config()
        self.detector = Detector()
        self.loaded = []

        def loader(path):
            self.loaded.append(path)
            return self.detector

        self.runtime = ModelRuntime(self.config, loader)

    def tearDown(self):
        self.directory.cleanup()

    def write_config(self, **overrides):
        data = {"model_path": "models/best.pt", "confidence": 0.35, "image_size": 640, "device": "cpu"}
        data.update(overrides)
        self.config.write_text(json.dumps(data), encoding="utf-8")

    def checkpoint(self, name="models/best.pt"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake trusted checkpoint")
        return path

    def test_missing_model_starts_without_downloading_weights(self):
        with TestClient(create_app(self.runtime)) as client:
            metadata = client.get("/api/model").json()
            self.assertFalse(metadata["ready"])
            self.assertEqual(metadata["classes"], [])
            self.assertIn("model_path", metadata["message"])
            self.assertEqual(self.loaded, [])
            self.assertEqual(client.post("/api/predict", content=image_bytes()).status_code, 503)

    def test_metadata_uses_the_checkpoint_classes(self):
        path = self.checkpoint()
        with TestClient(create_app(self.runtime)) as client:
            response = client.get("/api/model")
            self.assertEqual(response.headers["cache-control"], "no-store")
            data = response.json()
            self.assertTrue(data["ready"])
            self.assertEqual(data["classes"], [{"id": 0, "name": "cardboard"}, {"id": 1, "name": "plastic"}])
            self.assertEqual(self.loaded, [path.resolve()])

    def test_predict_normalizes_boxes_and_passes_threshold(self):
        self.checkpoint()
        with TestClient(create_app(self.runtime)) as client:
            response = client.post("/api/predict?confidence=0.6", content=image_bytes(), headers={"Content-Type": "image/jpeg"})
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            self.assertEqual((result["width"], result["height"]), (100, 50))
            self.assertEqual(result["detections"][0]["box"], [0.1, 0.1, 0.9, 0.9])
            self.assertEqual(result["detections"][1]["box"], [0, 0, 1, 1])
            self.assertEqual(result["detections"][0]["class_name"], "plastic")
            self.assertEqual(self.detector.last_options["conf"], 0.6)
            self.assertFalse(self.detector.last_options["save"])

    def test_invalid_image_and_confidence_are_rejected(self):
        self.checkpoint()
        with TestClient(create_app(self.runtime)) as client:
            self.assertEqual(client.post("/api/predict", content=b"not an image").status_code, 400)
            self.assertEqual(client.post("/api/predict", content=b"").status_code, 400)
            self.assertEqual(client.post("/api/predict?confidence=2", content=image_bytes()).status_code, 422)
            self.assertEqual(client.post("/api/predict?confidence=nan", content=image_bytes()).status_code, 422)

    def test_large_body_is_rejected(self):
        self.checkpoint()
        with TestClient(create_app(self.runtime)) as client:
            response = client.post("/api/predict", content=b"a" * (MAX_IMAGE_BYTES + 1))
            self.assertEqual(response.status_code, 413)

    def test_large_image_dimensions_are_rejected(self):
        self.checkpoint()
        with TestClient(create_app(self.runtime)) as client:
            response = client.post("/api/predict", content=image_bytes((4000, 3100)))
            self.assertEqual(response.status_code, 413)

    def test_reload_rereads_configuration(self):
        self.checkpoint()
        with TestClient(create_app(self.runtime)) as client:
            self.checkpoint("replacement.pt")
            self.write_config(model_path="replacement.pt", confidence=0.55)
            response = client.post("/api/model/reload")
            self.assertTrue(response.json()["ready"])
            self.assertEqual(response.json()["filename"], "replacement.pt")
            self.assertEqual(response.json()["confidence"], 0.55)
            self.assertEqual(len(self.loaded), 2)

    def test_busy_inference_and_reload_are_rejected(self):
        self.checkpoint()
        with TestClient(create_app(self.runtime)) as client:
            self.runtime.lock.acquire()
            try:
                self.assertEqual(client.post("/api/predict", content=image_bytes()).status_code, 429)
                self.assertEqual(client.post("/api/model/reload").status_code, 429)
            finally:
                self.runtime.lock.release()

    def model_catalog(self):
        choices = [
            {"id": "taco", "name": "TACO", "path": "models/best.pt", "classes": ["cardboard", "plastic"]},
            {"id": "sortwaste", "name": "SortWaste", "path": "models/sortwaste.pt", "classes": ["cardboard", "carton"]},
        ]
        (self.root / "models.json").write_text(json.dumps({"default_model": "taco", "models": choices}))

    def test_switching_models_updates_inference_labels_and_can_switch_back(self):
        self.checkpoint()
        self.checkpoint("models/sortwaste.pt")
        self.model_catalog()
        alternate = Detector()
        alternate.names = {0: "cardboard", 1: "carton"}
        self.runtime.loader = lambda path: alternate if path.name == "sortwaste.pt" else self.detector
        with TestClient(create_app(self.runtime)) as client:
            initial = client.get("/api/model").json()
            self.assertEqual(initial["selected_model"], "taco")
            self.assertEqual(len(initial["available_models"]), 2)
            switched = client.post("/api/model/reload?model_id=sortwaste").json()
            self.assertTrue(switched["ready"])
            self.assertEqual(switched["selected_model"], "sortwaste")
            self.assertEqual(switched["classes"][1]["name"], "carton")
            self.assertEqual(switched["confidence"], 0.35)
            prediction = client.post("/api/predict", content=image_bytes()).json()
            self.assertEqual(prediction["detections"][0]["class_name"], "carton")
            switched_back = client.post("/api/model/reload?model_id=taco").json()
            self.assertEqual(switched_back["classes"][1]["name"], "plastic")

    def test_unknown_model_cannot_load_arbitrary_paths_or_discard_active_model(self):
        self.checkpoint()
        self.model_catalog()
        with TestClient(create_app(self.runtime)) as client:
            self.assertEqual(client.post("/api/model/reload?model_id=../../other.pt").status_code, 400)
            self.assertTrue(client.get("/api/model").json()["ready"])
            self.assertEqual(len(self.loaded), 1)
            missing = client.post("/api/model/reload?model_id=sortwaste").json()
            self.assertFalse(missing["ready"])
            self.assertFalse(missing["available_models"][1]["available"])
            self.assertEqual(client.post("/api/predict", content=image_bytes()).status_code, 503)
            self.assertTrue(client.post("/api/model/reload?model_id=taco").json()["ready"])

    def hybrid_catalog(self):
        self.model_catalog()
        catalog_path = self.root / "models.json"
        catalog = json.loads(catalog_path.read_text())
        catalog["models"].append({
            "id": "post", "name": "YOLO + post-trained ResNet", "type": "yolo_resnet",
            "path": "models/post.pt", "detector_path": "models/best.pt",
            "description": "YOLO boxes with post-trained material labels.",
            "classes": ["cardboard", "plastic"],
        })
        catalog_path.write_text(json.dumps(catalog))

    def test_hybrid_uses_resnet_labels_and_scores_with_the_original_yolo_boxes(self):
        self.checkpoint()
        self.checkpoint("models/post.pt")
        self.hybrid_catalog()
        calls = []

        class Classifier:
            classes = ["cardboard", "plastic"]

            def predict(self, crops, device):
                calls.append(([crop.size for crop in crops], device))
                return [(0, .87), (1, .76)]

        self.runtime.classifier_loader = lambda path: Classifier()
        with TestClient(create_app(self.runtime)) as client:
            switched = client.post("/api/model/reload?model_id=post").json()
            self.assertTrue(switched["ready"])
            self.assertEqual(switched["selected_model"], "post")
            self.assertEqual(len(switched["available_models"]), 3)
            predictions = client.post("/api/predict", content=image_bytes()).json()["detections"]
            self.assertEqual(predictions[0]["class_name"], "cardboard")
            self.assertEqual(predictions[0]["confidence"], .87)
            self.assertEqual(predictions[0]["box"], [.1, .1, .9, .9])
            self.assertEqual(predictions[1]["box"], [0, 0, 1, 1])
            self.assertEqual(calls, [([(80, 40), (100, 50)], "cpu")])
            self.assertTrue(client.post("/api/model/reload?model_id=taco").json()["ready"])
            self.assertIsNone(self.runtime.classifier)
            original = client.post("/api/predict", content=image_bytes()).json()["detections"]
            self.assertEqual(original[0]["class_name"], "plastic")
            self.assertEqual(original[0]["confidence"], .9)

    def test_hybrid_requires_both_checkpoints(self):
        self.checkpoint()
        self.hybrid_catalog()
        with TestClient(create_app(self.runtime)) as client:
            choices = client.get("/api/model").json()["available_models"]
            self.assertFalse(choices[2]["available"])
            self.assertFalse(client.post("/api/model/reload?model_id=post").json()["ready"])
        self.checkpoint("models/post.pt")
        (self.root / "models/best.pt").unlink()
        self.assertFalse(self.runtime.model_choices()[2]["available"])

    def test_hybrid_confidence_slider_also_filters_uncertain_material_labels(self):
        self.checkpoint()
        self.checkpoint("models/post.pt")
        self.hybrid_catalog()

        class Classifier:
            classes = ["cardboard", "plastic"]

            def predict(self, crops, device):
                return [(0, .2), (1, .8)]

        self.runtime.classifier_loader = lambda path: Classifier()
        with TestClient(create_app(self.runtime)) as client:
            client.post("/api/model/reload?model_id=post")
            predictions = client.post("/api/predict?confidence=0.6", content=image_bytes()).json()["detections"]
            self.assertEqual(len(predictions), 1)
            self.assertEqual(predictions[0]["class_name"], "plastic")
            self.assertEqual(predictions[0]["confidence"], .8)

    def test_incompatible_checkpoint_is_not_marked_ready(self):
        self.checkpoint()
        self.detector.task = "classify"
        with TestClient(create_app(self.runtime)) as client:
            metadata = client.get("/api/model").json()
            self.assertFalse(metadata["ready"])
            self.assertIn("classification", metadata["message"])

    def test_bad_configuration_still_serves_model_status(self):
        self.config.write_text("{broken", encoding="utf-8")
        with TestClient(create_app(self.runtime)) as client:
            self.assertFalse(client.get("/api/model").json()["ready"])
            self.assertEqual(self.loaded, [])


if __name__ == "__main__":
    unittest.main()
