import os
import tensorflow as tf
import numpy as np
from absl import app, flags
from sklearn.metrics import roc_curve, auc, roc_auc_score
from matplotlib import pyplot as plt
from data_loader import data_gen
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import Activation
from tensorflow.keras.backend import sigmoid

flags.DEFINE_string('input', default='/home/levanpon/data/covid-chestxray-dataset/', help='Data Path')
flags.DEFINE_integer('epochs', default=10, help='Number of epochs')
flags.DEFINE_integer('batch_size', default=32, help='Number of epochs')
flags.DEFINE_integer('image_size', default=32, help='Image size')

_flags = flags.FLAGS

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class SwishActivation(Activation):

    def __init__(self, activation, **kwargs):
        super(SwishActivation, self).__init__(activation, **kwargs)
        self.__name__ = 'swish_act'


def swish_act(x, beta=1):
    return x * sigmoid(beta * x)


def get_callbacks():
    callbacks = []
    tensor_board = tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0)
    callbacks.append(tensor_board)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join('saved/model.h5'),
        verbose=1,
        save_best_only=True)
    callbacks.append(checkpoint)
    return callbacks


def get_model(classes=1000, input_shape=(32, 32, 3)):
    model = Xception(include_top=False, input_shape=input_shape, pooling='avg', weights='imagenet')

    # Adding 2 fully-connected layers to B0.
    x = model.output

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.7)(x)

    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = Activation(swish_act)(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = Activation(swish_act)(x)

    # Output layer
    predictions = tf.keras.layers.Dense(classes, activation="softmax")(x)

    model_final = tf.keras.Model(inputs=model.input, outputs=predictions)
    model_final.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.001),
                        metrics=['accuracy', 'mse'])
    model_final.summary()
    return model_final


def main(_):
    train_gen, valid_gen, test_X, test_Y, train_len, test_len, all_labels = data_gen(_flags.input, _flags.batch_size,
                                                                                     _flags.image_size)

    model = get_model(len(all_labels), input_shape=(_flags.image_size, _flags.image_size, 3))

    callbacks = get_callbacks()

    model.fit_generator(train_gen,
                        steps_per_epoch=train_len // _flags.epochs,
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

    print(auc(test_Y.astype(int), y_pred))


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
