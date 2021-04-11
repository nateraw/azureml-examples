from argparse import ArgumentParser

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

try:
    from model import Classifier
except:
    from .model import Classifier


def parse_args(args=None):
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--data_dir", type=str, default="./")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args(args)


def main(args):
    pl.seed_everything(1234)

    # Prepare MNIST Dataset + DataLoaders
    mnist_dataset = MNIST(args.data_dir, train=True, download=True, transform=ToTensor())
    mnist_test = MNIST(args.data_dir, train=False, download=True, transform=ToTensor())
    mnist_train, mnist_val = random_split(mnist_dataset, [55000, 5000])
    train_loader = DataLoader(mnist_train, batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = DataLoader(mnist_val, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = DataLoader(mnist_test, batch_size=args.batch_size, num_workers=args.num_workers)

    # Initialize Model
    model = Classifier(1, 28, 28, lr=args.lr, hidden_dim=args.hidden_dim)

    # Initialize Trainer
    trainer = pl.Trainer.from_argparse_args(args)

    # Train Model
    trainer.fit(model, train_loader, val_loader)

    # Test Model
    trainer.test(test_dataloaders=test_loader)

    return model, trainer


if __name__ == "__main__":
    args = parse_args()
    model, trainer = main(args)
