import torch
from loss import OmniglotLoss
from miner import OmniglotMiner
from sampler import OmniglotSampler
from config import Config
from data import OmniglotData
from model import OmniglotModel
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import umap
from cycler import cycler
import pytorch_metric_learning.utils.logging_presets as logging_presets
from pytorch_metric_learning import testers, trainers
import numpy as np
import os
import wandb

import logging
from matplotlib import pyplot as plt

wandb.login()

omniglot_dataset = OmniglotData().load_data()

train_dataset, val_dataset = OmniglotData().split_data(omniglot_dataset)

model_names = Config.MODEL_NAMES
for model_name in model_names:

    run = wandb.init(project = 'deep-metric-learning-foundation-models',
                     config = {'model':model_name})
    itr = 0 


    experiment_name = model_name + '_' + Config.LOSS
    experiment_path = './experiments/' + experiment_name + '/' 
    os.makedirs(experiment_path)

    embedding_size = Config.EMBEDDING_SIZE 	
    device = Config.DEVICE

    model = OmniglotModel(model_name=model_name)
    trunk = model.trunk
    embedder = model.embedder

    trunk.to(device)
    embedder.to(device)

    loss_fn = OmniglotLoss().compute_loss
    miner = OmniglotMiner().mine

    trunk_optimizer = torch.optim.Adam(trunk.parameters(), lr=Config.LEARNING_RATE['trunk'], weight_decay=0.0001)
    embedder_optimizer = torch.optim.Adam(embedder.parameters(), lr=Config.LEARNING_RATE['embedder'], weight_decay=0.0001)

    num_epochs = Config.NUM_EPOCHS  # number of epochs to run the model for.
    batch_size = Config.BATCH_SIZE

    models = {"trunk": trunk, "embedder": embedder}
    optimizers = {
        "trunk_optimizer": trunk_optimizer,
        "embedder_optimizer": embedder_optimizer,
    }
    loss_funcs = {"metric_loss": loss_fn}
    mining_funcs = {"tuple_miner": miner}



    logging.getLogger().setLevel(logging.INFO)
    logging.info("VERSION %s" % pytorch_metric_learning.__version__)

    record_keeper, _, _ = logging_presets.get_record_keeper(
        experiment_path+"example_logs", experiment_path+"example_tensorboard"
    )
    hooks = logging_presets.get_hook_container(record_keeper)
    dataset_dict = {"val": val_dataset}
    model_folder = experiment_path + "example_saved_models"


    def visualizer_hook(umapper, umap_embeddings, labels, split_name, keyname, *args):
        logging.info(
            "UMAP plot for the {} split and label set {}".format(split_name, keyname)
        )
        label_set = np.unique(labels)
        num_classes = len(label_set)
        plt.figure(figsize=(20, 15))
        plt.gca().set_prop_cycle(
            cycler(
                "color", [plt.cm.nipy_spectral(i) for i in np.linspace(0, 0.9, num_classes)]
            )
        )
        for i in range(num_classes):
            idx = labels == label_set[i]
            plt.plot(umap_embeddings[idx, 0], umap_embeddings[idx, 1], ".", markersize=1)
        plt.show()
        plt.savefig(experiment_path + f'UMAP_{itr}.png')
        itr = itr + 1
        wandb.log({'umap_embeddings': wandb.Image(plt), 'iteration':itr})
        


    # Create the tester
    tester = testers.GlobalEmbeddingSpaceTester(
        end_of_testing_hook=hooks.end_of_testing_hook,
        visualizer=umap.UMAP(),
        visualizer_hook=visualizer_hook,
        dataloader_num_workers=2,
        accuracy_calculator=AccuracyCalculator(k="max_bin_count"),
    )

    end_of_epoch_hook = hooks.end_of_epoch_hook(
        tester, dataset_dict, model_folder, test_interval=1, patience=1
    )


    trainer = trainers.MetricLossOnly(
        models,
        optimizers,
        batch_size,
        loss_funcs,
        train_dataset,
        mining_funcs=mining_funcs,
        #sampler=sampler,
        dataloader_num_workers=4,
        end_of_iteration_hook=hooks.end_of_iteration_hook,
        end_of_epoch_hook=end_of_epoch_hook,
    )


    trainer.train(num_epochs=num_epochs)