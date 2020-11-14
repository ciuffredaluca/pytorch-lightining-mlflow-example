from models import SimpleClassifier

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

try:
    from torchvision.datasets.mnist import MNIST
    from torchvision import transforms
except Exception as e:
    from tests.base.datasets import MNIST


batch_size = 16
hparams = { 'hidden_dim': 256, 'learning_rate': 0.01 }
num_epochs = 10

dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
mnist_test = MNIST('', train=False, download=True, transform=transforms.ToTensor())
mnist_train, mnist_val = random_split(dataset, [55000, 5000])


train_loader = DataLoader(mnist_train, batch_size=batch_size)
val_loader = DataLoader(mnist_val, batch_size=batch_size)
test_loader = DataLoader(mnist_test, batch_size=batch_size)


from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger

mlf_logger = MLFlowLogger(
    experiment_name="mnist-hello-world",
    tracking_uri="file:./ml-runs"
)

# ------------
model = SimpleClassifier(**hparams)
# ------------
# training
# ------------
trainer = pl.Trainer(max_epochs=num_epochs,  
                        gpus=1, logger=mlf_logger)
trainer.fit(model, train_loader, val_loader)

# ------------
trainer.test(test_dataloaders=test_loader)




