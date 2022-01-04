from pathlib import Path

import fire
from tensorflow import keras


def main(lr: float = 1e-3, batch_size: int = 32, max_epochs: int = 5, data_dir: str = "data/", logdir: str = "logs/"):
    data_dir = Path(data_dir).absolute()
    data_dir.mkdir(parents=True, exist_ok=True)
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(data_dir / "mnist.npz")
    train_data, test_data = (x_train.astype("float32") / 255, y_train), (x_test.astype("float32") / 255, y_test)

    model = keras.models.Sequential(
        [
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    _ = model.fit(
        *train_data,
        batch_size=batch_size,
        epochs=max_epochs,
        validation_split=0.1,
        callbacks=[keras.callbacks.TensorBoard(log_dir=logdir)],
    )
    model.evaluate(*test_data)
    model.save(str(Path(logdir) / "saved_model/"))


if __name__ == "__main__":
    fire.Fire(main)
