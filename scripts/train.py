import os
from pathlib import Path

import mlflow
import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

try:
    from torchvision.datasets.mnist import MNIST
    from torchvision import transforms
except Exception as e:
    from tests.base.datasets import MNIST

from pytorch_lightning import Trainer
from pytorch_lightning import callbacks
from pytorch_lightning import loggers
from models import SimpleClassifier
from omegaconf import OmegaConf

# collect relevant project files locations
SCRIPT_PATH = os.path.abspath(__file__)
PROJECT_PATH = Path(SCRIPT_PATH).parent.parent
DATASET_PATH = PROJECT_PATH / 'dataset'
DATASET_PATH.mkdir(exist_ok=True)
TRAIN_CONFIG_PATH = PROJECT_PATH / 'run_configs/train.yaml'
PROJECT_CONFIG_FILE = PROJECT_PATH / 'MLProject'

# load training configuration
conf = OmegaConf.load(TRAIN_CONFIG_PATH)
projconf = OmegaConf.load(PROJECT_CONFIG_FILE)
conf.experiment.name = projconf.name

# define hyper parameters
hparams = dict(conf.model)
batch_size = conf.train.batch_size
num_epochs = conf.train.epochs

# define dataset and dataset loaders
dataset = MNIST(DATASET_PATH, train=True, download=True, transform=transforms.ToTensor())
mnist_test = MNIST(DATASET_PATH, train=False, download=True, transform=transforms.ToTensor())
mnist_train, mnist_val = random_split(dataset, [55000, 5000])

train_loader = DataLoader(mnist_train, batch_size=batch_size)
val_loader = DataLoader(mnist_val, batch_size=batch_size)
test_loader = DataLoader(mnist_test, batch_size=batch_size)

# define logger(s)
logger = []

if conf.experiment.tf_log is not None: # use tensorboard logger
    (PROJECT_PATH / conf.experiment.tf_log).mkdir(exist_ok=True, parents=True)
    tf_logger = loggers.TensorBoardLogger(save_dir=str(PROJECT_PATH / conf.experiment.tf_log),
                                      name=conf.experiment.name)
    logger.append(tf_logger)

# use mlflow logger
(PROJECT_PATH / conf.experiment.mlflow_log).mkdir(exist_ok=True, parents=True)
mlf_logger = loggers.MLFlowLogger(experiment_name=conf.experiment.name,
                                  tracking_uri=f"file:{str(PROJECT_PATH / conf.experiment.mlflow_log)}")
logger.append(mlf_logger)

# define checkpoint callback
CHECKPOINTS_PATH = PROJECT_PATH / conf.experiment.checkpoints
CHECKPOINTS_PATH.mkdir(exist_ok=True, parents=True)

checkpoint_callback = callbacks.ModelCheckpoint(dirpath=str(CHECKPOINTS_PATH),
                                                save_top_k=-1) # save all epochs

# define model
model = SimpleClassifier(**hparams)

# define trainer
trainer = pl.Trainer(max_epochs=num_epochs,  
                     gpus=1, 
                     logger=logger,
                     callbacks=[checkpoint_callback])

# run training
trainer.fit(model, train_loader, val_loader)




