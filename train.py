import os
import tensorflow as tf
from absl import app, flags

from models import XChest
from data_loader import data_gen
import config

flags.DEFINE_string('model', default='densenet', help='Model name')

_flags = flags.FLAGS

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def get_callbacks(model_name):
    callbacks = []
    tensor_board = tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32)
    callbacks.append(tensor_board)
    if model_name != 'ensemble':
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config.model_path, f'model.{model_name}.h5'),
            verbose=1,
            save_best_only=True)
        callbacks.append(checkpoint)

    early = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                             mode="min",
                                             patience=3)
    callbacks.append(early)

    return callbacks


def main(_):
    x_chest = XChest(_flags.model, input_shape=(config.image_height, config.image_width, 3))
    model = x_chest.build()
    train_gen, valid_gen, test_X, test_Y, train_len, test_len = data_gen()

    callbacks = get_callbacks(_flags.model)

    model.fit_generator(train_gen,
                        steps_per_epoch=100,
                        validation_data=(test_X, test_Y),
                        epochs=50,
                        callbacks=callbacks)


if __name__ == '__main__':
    app.run(main)
