from argparse import ArgumentParser
from pathlib import Path

from tensorflow import keras


def parse_args(args=None):
    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--logdir", type=str, default="logs/")
    return parser.parse_args(args)


def main(args):
    data_dir = Path(args.data_dir).absolute()
    data_dir.mkdir(parents=True, exist_ok=True)
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(data_dir / "mnist.npz")
    train_data, test_data = (x_train.astype("float32") / 255, y_train), (
        x_test.astype("float32") / 255,
        y_test,
    )

    model = keras.models.Sequential(
        [
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    _ = model.fit(
        *train_data,
        batch_size=args.batch_size,
        epochs=args.max_epochs,
        validation_split=0.1,
        callbacks=[keras.callbacks.TensorBoard(log_dir=args.logdir)],
    )
    model.evaluate(*test_data)
    model.save(str(Path(args.logdir) / "saved_model/"))


if __name__ == "__main__":
    main(parse_args())
