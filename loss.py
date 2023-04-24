import torch
import pytorch_metric_learning.losses as losses
from config import Config

class OmniglotLoss:
    def __init__(self):
        self.loss_fn = losses.AngularLoss()

    def compute_loss(self, embeddings, labels):
        return self.loss_fn(embeddings, labels)