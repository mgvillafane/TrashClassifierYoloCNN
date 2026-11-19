"""Model definitions and RGB crop inference for the prepared classifier data."""

CLASSES = ('cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash')
MODELS = ('cnn', 'resnet-head', 'resnet-layer4')
PREPROCESSING = {
    'color_mode': 'RGB',
    'image_size': [224, 224],
    'interpolation': 'bilinear',
    'pixel_scale': 255,
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
}
TRAINING_AUGMENTATION = {
    'crop_scale': [0.8, 1.0],
    'crop_aspect_ratio': [0.75, 4 / 3],
    'horizontal_flip_probability': 0.5,
    'vertical_flip_probability': 0.5,
    'rotation_degrees': 25,
    'brightness': 0.25,
    'contrast': 0.25,
}


def image_transform(settings, augmentation=None):
    from torchvision import transforms

    size = tuple(settings['image_size'])
    interpolation = transforms.InterpolationMode(settings['interpolation'])
    if augmentation is None:
        steps = [transforms.Resize(size, interpolation=interpolation)]
    else:
        steps = [
            transforms.RandomResizedCrop(size, scale=tuple(augmentation['crop_scale']),
                                         ratio=tuple(augmentation['crop_aspect_ratio']),
                                         interpolation=interpolation),
            transforms.RandomHorizontalFlip(p=augmentation['horizontal_flip_probability']),
            transforms.RandomVerticalFlip(p=augmentation['vertical_flip_probability']),
            transforms.RandomRotation(augmentation['rotation_degrees'], interpolation=interpolation),
            transforms.ColorJitter(brightness=augmentation['brightness'], contrast=augmentation['contrast']),
        ]
    steps.extend([transforms.ToTensor(), transforms.Normalize(settings['mean'], settings['std'])])
    return transforms.Compose(steps)


def build_model(architecture, pretrained_weights=None):
    if architecture == 'cnn':
        from cnn_from_scratch import TrashCNN

        return TrashCNN(num_classes=len(CLASSES))
    if architecture not in MODELS:
        raise ValueError(f'Unknown classifier architecture: {architecture}')

    import torch
    from torch import nn
    from torchvision.models import resnet18

    model = resnet18(weights=None)
    if pretrained_weights is not None:
        model.load_state_dict(torch.load(pretrained_weights, map_location='cpu', weights_only=True))
    model.requires_grad_(False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model


class CropClassifier:
    """Load a training checkpoint and classify a batch of PIL images or HWC RGB uint8 arrays."""

    def __init__(self, checkpoint_path, device=None):
        import torch

        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        config = checkpoint['config']
        self.classes = tuple(config['classes'])
        if self.classes != CLASSES:
            raise ValueError(f'Unexpected checkpoint label order: {self.classes}')
        self.color_mode = config['preprocessing']['color_mode']
        self.transform = image_transform(config['preprocessing'])
        self.model = build_model(config['architecture'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device).eval()

    def predict(self, crops):
        """Return one label ID, label and softmax confidence per crop, in input order."""
        from PIL import Image
        import torch

        images = [crop if isinstance(crop, Image.Image) else Image.fromarray(crop) for crop in crops]
        if not images:
            return []
        batch = torch.stack([self.transform(image.convert(self.color_mode)) for image in images])
        with torch.inference_mode():
            confidence, label_ids = self.model(batch.to(self.device)).softmax(dim=1).max(dim=1)
        return [dict(label_id=label_id, label=self.classes[label_id], confidence=score)
                for label_id, score in zip(label_ids.tolist(), confidence.tolist())]
