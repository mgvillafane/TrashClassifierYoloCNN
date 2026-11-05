import torch
from torchvision.models import resnet18, ResNet18_Weights


torch.hub.set_dir("/Volumes/workspace/default/models/resnet_pretrained/torch_cache")

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)