import fire
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

try:
    from model import Classifier
except:
    from .model import Classifier


def main(batch_size: int = 32, num_workers: int = 0, hidden_dim: int = 64, data_dir: str = "./", lr: float = 1e-3):
    pl.seed_everything(1234)

    # Prepare MNIST Dataset + DataLoaders
    mnist_dataset = MNIST(data_dir, train=True, download=True, transform=ToTensor())
    mnist_test = MNIST(data_dir, train=False, download=True, transform=ToTensor())
    mnist_train, mnist_val = random_split(mnist_dataset, [55000, 5000])
    train_loader = DataLoader(mnist_train, batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(mnist_val, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, num_workers=num_workers)

    # Initialize Model
    model = Classifier(1, 28, 28, lr=lr, hidden_dim=hidden_dim)

    # Initialize Trainer
    trainer = pl.Trainer.from_argparse_args()

    # Train Model
    trainer.fit(model, train_loader, val_loader)

    # Test Model
    trainer.test(test_dataloaders=test_loader)

    return model, trainer


if __name__ == "__main__":
    fire.Fire(main)
