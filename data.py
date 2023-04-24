import torch
import torchvision
from torch.utils.data import random_split
from torchvision import transforms
from config import Config

class OmniglotData:
    def __init__(self):
        self.root = Config.ROOT
        self.dataset = Config.DATASET
        self.background = Config.BACKGROUND
        self.transform = Config.TRANSFORM

    def load_data(self):
        if self.dataset == "omniglot":
            return torchvision.datasets.Omniglot(root=self.root, background=self.background, download=True, transform=self.transform["train"])
        else:
            raise ValueError(f"Invalid dataset {self.dataset}")

    def split_data(self, dataset):
        split = Config.SPLIT
        num_train = int(len(dataset) * split[0])
        num_val = len(dataset) - num_train
        return random_split(dataset, [num_train, num_val])