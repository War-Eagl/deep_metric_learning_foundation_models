import torch
import timm
import torch.nn as nn
from config import Config

class OmniglotModel():
    def __init__(self, model_name):
        self.model_name = model_name
        self.num_classes = None
        self.embedding_size = Config.EMBEDDING_SIZE
        self.margin = Config.MARGIN
        self.trunk = timm.create_model(self.model_name, pretrained=True, in_chans=1, num_classes=0)
        self.embedder = nn.Sequential(
            nn.Linear(in_features=self.trunk.num_features, out_features=self.embedding_size),
            nn.ReLU(),
            nn.Linear(in_features=self.embedding_size, out_features=self.num_classes)
        )