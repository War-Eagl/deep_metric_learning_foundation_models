import torch
import pytorch_metric_learning.samplers as samplers
from config import Config

class OmniglotSampler:
    def __init__(self, labels):
        sampler_name = Config.SAMPLER
        sampler_params = Config.SAMPLER_PARAMS
        self.sampler = getattr(samplers, sampler_name)(labels, **sampler_params)

    def sample(self):
        return self.sampler()