import os
import tensorflow as tf
import numpy as np
from absl import app, flags
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score, average_precision_score
from models import XChest
from matplotlib import pyplot as plt
from data_loader import data_gen
import config

flags.DEFINE_string('model', default='inception_resnet_v2', help='Model name')
flags.DEFINE_string('input', default='/home/levanpon/data/ChestXray-NIHCC/', help='Data Path')
flags.DEFINE_integer('epochs', default=10, help='Number of epochs')
flags.DEFINE_integer('batch_size', default=32, help='Number of epochs')
flags.DEFINE_integer('image_size', default=256, help='Number of epochs')
_flags = flags.FLAGS

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def get_callbacks(model_name):
    callbacks = []
    tensor_board = tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0)
    callbacks.append(tensor_board)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(config.model_path, f'model.{model_name}.h5'),
        verbose=1,
        save_best_only=True)
    # erly = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    callbacks.append(checkpoint)
    # callbacks.append(erly)
    return callbacks


def main(_):
    if not os.path.exists(config.model_path):
        os.mkdir(config.model_path)

    train_gen, valid_gen, test_X, test_Y, train_len, test_len, all_labels = data_gen(_flags.input, _flags.batch_size,
                                                                                     _flags.image_size)

    x_chest = XChest(classes=len(all_labels), model_name=_flags.model,
                     input_shape=(_flags.image_size, _flags.image_size, 3))
    model = x_chest.build()
    model.summary()
    callbacks = get_callbacks(_flags.model)

    model.fit(train_gen,
              steps_per_epoch=100,
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

    print('================')
    print('ROC auc score: {:.3f}'.format(roc_auc_score(test_Y.astype(int), y_pred)))
    print('Accuracy score: {:.3f}'.format(accuracy_score(test_Y.astype(int), y_pred)))
    print('Average precision score, micro-averaged over all classes: {:.3f}'.format(
        average_precision_score(test_Y.astype(int), y_pred, average="micro")))


if __name__ == '__main__':
    app.run(main)

# Atelectasis: actual: 10.64%, predicted: 14.13%
# Cardiomegaly: actual: 2.25%, predicted: 3.50%
# Consolidation: actual: 3.91%, predicted: 5.64%
# Edema: actual: 1.46%, predicted: 4.11%
# Effusion: actual: 13.28%, predicted: 13.94%
# Emphysema: actual: 3.32%, predicted: 2.76%
# Fibrosis: actual: 1.66%, predicted: 1.32%
# Infiltration: actual: 20.02%, predicted: 22.66%
# Mass: actual: 5.18%, predicted: 6.34%
# No Finding: actual: 50.98%, predicted: 50.57%
# Nodule: actual: 6.45%, predicted: 7.68%
# Pleural_Thickening: actual: 2.83%, predicted: 2.69%
# Pneumonia: actual: 1.07%, predicted: 1.83%
# Pneumothorax: actual: 5.47%, predicted: 4.98%
