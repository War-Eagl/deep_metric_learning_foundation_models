import torch
import pytorch_metric_learning.miners as miners
from config import Config

class OmniglotMiner:
    def __init__(self):
        miner_name = Config.MINER
        miner_params = Config.MINER_PARAMS
        self.miner = getattr(miners, miner_name)(**miner_params)

    def mine(self, embeddings, labels):
        indices_tuple = self.miner(embeddings, labels)
        return indices_tuple