import torchvision
from torchvision.models import resnet50, ResNet50_Weights
import torch

model = torchvision.models.resnet50(weights = ResNet50_Weights.DEFAULT)
torch.save(model.state_dict(), 'resnet_50_model.pth')
