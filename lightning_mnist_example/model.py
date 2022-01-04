import pytorch_lightning as pl
import torch
from torch import nn


class Classifier(pl.LightningModule):
    def __init__(self, c, h, w, num_classes=10, hidden_dim=64, lr=0.0001):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                self.hparams.c * self.hparams.h * self.hparams.w,
                self.hparams.hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(self.hparams.hidden_dim, self.hparams.num_classes),
        )
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy_metric = pl.metrics.Accuracy()

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch, split="train"):
        x, y = batch
        outputs = self(x)
        loss = self.criterion(outputs, y)
        self.log(f"{split}_loss", loss)

        if split in ["val", "test"]:
            preds = outputs.argmax(dim=1)
            acc = self.accuracy_metric(preds, y)
            self.log(f"{split}_acc", acc, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


if __name__ == "__main__":
    b, c, h, w = 4, 1, 28, 28
    num_classes = 10
    x = torch.rand(b, c, h, w)
    model = Classifier(c, h, w, num_classes=num_classes)
    out = model(x)
    assert out.shape == (b, num_classes)
