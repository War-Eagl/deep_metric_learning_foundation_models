import torch
from torchvision import transforms
class Config:
    # Data
    ROOT = "/content"
    DATASET = "omniglot"
    BACKGROUND = True
    TRANSFORM = {"train": transforms.Compose([transforms.Resize(224), transforms.ToTensor()]),
                 "val": transforms.Compose([transforms.Resize(224), transforms.ToTensor()])}
    BATCH_SIZE = 32
    SPLIT = [0.8, 0.2]

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model
    MODEL_NAMES = ["resnet18",'resnet34','resnet50','resnet101','efficientnet_b0','efficientnet_b1','efficientnet_b2','efficientnet_b3','efficientnet_b4','convnext_tiny','convnext_small','convnext_base', 'vit_small_patch16_224','vit_relpos_medium_patch16_224','vit_base_patch16_224','deit_tiny_patch16_224','deit_small_patch16_224','deit_base_patch16_224','swin_s3_tiny_224','swin_s3_small_224','swin_s3_base_224' ]
    EMBEDDING_SIZE = 128
    MARGIN = 0.2
    LOSS = "Angular_Loss"

    # Training
    NUM_EPOCHS = 20
    LEARNING_RATE = {"trunk": 0.00001, "embedder": 0.0001}
    WEIGHT_DECAY = 0.0001
    MINER = "triplet_margin"
    MINER_PARAMS = {}
    SAMPLER = "m_per_class"
    SAMPLER_PARAMS = {"m": 5}
    USE_WANDB = True
    WANDB_PROJECT = "omniglot"
    WANDB_ENTITY = "your-entity-name"