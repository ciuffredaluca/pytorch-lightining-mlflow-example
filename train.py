import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

from pytorch_lightning import Trainer
import pytorch_lightning as pl

class CoolSystem(pl.LightningModule):

    def __init__(self, classes=10):
        super().__init__()
        self.save_hyperparameters()

        # not the best model...
        self.l1 = torch.nn.Linear(28 * 28, self.hparams.classes)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def prepare_data(self):
        MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())

    def train_dataloader(self):
        mnist_train = MNIST(os.getcwd(), train=True, download=False, transform=transforms.ToTensor())
        loader = DataLoader(mnist_train, batch_size=32, num_workers=4)
        return loader


model = CoolSystem()

# most basic trainer, uses good defaults
trainer = Trainer(precision=16, progress_bar_refresh_rate=5, max_epochs=10, amp_backend=None)
trainer.fit(model)