from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


class Classifier(pl.LightningModule):
    def __init__(self, c, w, h, num_classes=10, hidden_dim=64, lr=0.0001):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Linear(self.hparams.c * self.hparams.w * self.hparams.h, self.hparams.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hparams.hidden_dim, self.hparams.num_classes)
        )
        self.accuracy_metric = pl.metrics.Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        outputs = self(x)
        loss = F.cross_entropy(outputs, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        outputs = self(x)
        loss = F.cross_entropy(outputs, y)
        preds = outputs.argmax(dim=1)
        acc = self.accuracy_metric(preds, y)
        self.log("val_loss", loss)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer


def parse_args(args=None):
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--data_dir", type=str, default="./")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args(args)


def cli_main(args):
    pl.seed_everything(1234)
    dataset = MNIST(
        args.data_dir, train=True, download=True, transform=transforms.ToTensor()
    )
    cifar_test = MNIST(
        args.data_dir, train=False, download=True, transform=transforms.ToTensor()
    )
    cifar_train, cifar_val = random_split(dataset, [55000, 5000])
    train_loader = DataLoader(cifar_train, batch_size=args.batch_size)
    val_loader = DataLoader(cifar_val, batch_size=args.batch_size)
    test_loader = DataLoader(cifar_test, batch_size=args.batch_size)

    model = Classifier(1, 28, 28, lr=args.lr, hidden_dim=args.hidden_dim)

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, val_loader)

    result = trainer.test(test_dataloaders=test_loader)
    print(result)


if __name__ == "__main__":
    cli_main(parse_args())
