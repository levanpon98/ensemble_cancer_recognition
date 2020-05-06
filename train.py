import os
import tensorflow as tf
import numpy as np
from absl import app, flags
from sklearn.metrics import roc_curve, auc, roc_auc_score
from models import XChest
from matplotlib import pyplot as plt
from data_loader import data_gen
import config

flags.DEFINE_string('model', default='densenet', help='Model name')
flags.DEFINE_string('input', default='/home/levanpon/data/ChestXray-NIHCC/', help='Data Path')
flags.DEFINE_integer('epochs', default=10, help='Number of epochs')
_flags = flags.FLAGS

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def get_callbacks(model_name):
    callbacks = []
    tensor_board = tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0)
    callbacks.append(tensor_board)
    if model_name != 'ensemble':
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config.model_path, f'model.{model_name}.h5'),
            verbose=1,
            save_best_only=True)
        callbacks.append(checkpoint)

    return callbacks


def main(_):
    train_gen, valid_gen, test_X, test_Y, train_len, test_len, all_labels = data_gen(_flags.input)

    x_chest = XChest(_flags.model, input_shape=(config.image_height, config.image_width, 3))
    model = x_chest.build()

    callbacks = get_callbacks(_flags.model)

    if _flags.model != 'ensemble':
        model.fit_generator(train_gen,
                            steps_per_epoch=train_len // 32,
                            validation_data=(test_X, test_Y),
                            epochs=_flags.epochs,
                            callbacks=callbacks)

    y_pred = model.predict(test_X)
    for c_label, p_count, t_count in zip(all_labels,
                                         100 * np.mean(y_pred, 0),
                                         100 * np.mean(test_Y, 0)):
        print('%s: actual: %2.2f%%, predicted: %2.2f%%' % (c_label, t_count, p_count))

    fig, c_ax = plt.subplots(1, 1, figsize=(9, 9))
    for (idx, c_label) in enumerate(all_labels):
        fpr, tpr, thresholds = roc_curve(test_Y[:, idx].astype(int), y_pred[:, idx])
        c_ax.plot(fpr, tpr, label='%s (AUC:%0.2f)' % (c_label, auc(fpr, tpr)))
    c_ax.legend()
    c_ax.set_xlabel('False Positive Rate')
    c_ax.set_ylabel('True Positive Rate')
    fig.savefig('trained_net.png')

    roc_auc_score(test_Y.astype(int), y_pred)


if __name__ == '__main__':
    app.run(main)
