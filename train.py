import os
import tensorflow as tf
from absl import app, flags

from models import XChest
from data_loader import data_gen
import config

flags.DEFINE_string('model', default='densenet', help='Model name')
flags.DEFINE_string('input', default='../x-ray-data', help='Data Path')

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

    return callbacks


def main(_):
    train_gen, valid_gen, test_X, test_Y, train_len, test_len = data_gen(_flags.input)
    t_x, t_y = next(train_gen)

    x_chest = XChest(_flags.model, input_shape=t_x.shape[1:])
    model = x_chest.build()

    callbacks = get_callbacks(_flags.model)

    model.fit_generator(train_gen,
                        steps_per_epoch=train_len // 32,
                        validation_data=(test_X, test_Y),
                        epochs=50,
                        callbacks=callbacks)


if __name__ == '__main__':
    app.run(main)
