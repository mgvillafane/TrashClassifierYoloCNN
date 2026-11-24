"""Load and export the app's original and post-trained material classifiers."""

from pathlib import Path


class CropClassifier:
    def __init__(self, path):
        import torch
        from torch import nn
        from torchvision import transforms
        from torchvision.models import resnet18

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self.config = checkpoint["config"]
        self.classes = list(self.config["classes"])
        self.epoch = checkpoint["epoch"]
        if self.config["architecture"] != "resnet-layer4":
            raise ValueError("Expected a ResNet-layer4 material classifier")
        settings = self.config["preprocessing"]
        self.transform = transforms.Compose([
            transforms.Resize(tuple(settings["image_size"]),
                              interpolation=transforms.InterpolationMode(settings["interpolation"])),
            transforms.ToTensor(), transforms.Normalize(settings["mean"], settings["std"]),
        ])
        self.model = resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(self.classes))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict(self, crops, device="cpu"):
        import torch

        if not crops:
            return []
        device = "cuda:" + device if str(device).isdigit() else device
        self.model.to(device)
        inputs = torch.stack([self.transform(crop.convert("RGB")) for crop in crops]).to(device)
        with torch.inference_mode():
            scores, labels = self.model(inputs).softmax(dim=1).max(dim=1)
        return [(int(label), float(score)) for label, score in zip(labels.cpu(), scores.cpu())]

    def export(self, path):
        import torch

        shape = [1, 3, *self.config["preprocessing"]["image_size"]]
        torch.onnx.export(self.model.cpu().eval(), torch.zeros(shape), str(Path(path)),
                          input_names=["images"], output_names=["logits"], opset_version=17,
                          dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}}, dynamo=False)
