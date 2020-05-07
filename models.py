import os
import tensorflow as tf
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.xception import Xception

import config


class XChest():
    def __init__(self, model_name=None, input_shape=(256, 256, 3)):
        self.model_name = model_name
        self.input_shape = input_shape
        self.optimizer = tf.keras.optimizers.Adam(lr=0.001)
        self.loss = 'binary_crossentropy'
        self.list_model = {
            'densenet': DenseNet121,
            'xception': Xception,
            'inceptionv3': InceptionV3
        }

    def build(self):
        if self.model_name in self.list_model:
            base_model = self.list_model[self.model_name](include_top=False, weights='imagenet',
                                                          input_shape=self.input_shape,
                                                          pooling ='avg')

            x = base_model.output
            # x = tf.keras.layers.GlobalAveragePooling2D()(x)
            # x = tf.keras.layers.Dropout(0.3)(x)
            # x = tf.keras.layers.Dense(units=512, activation='relu')(x)
            # x = tf.keras.layers.Dropout(0.3)(x)
            # x = tf.keras.layers.Dense(units=512, activation='relu')(x)
            output = tf.keras.layers.Dense(units=13, activation="sigmoid")(x)

            model = tf.keras.Model(base_model.input, output)
            # model.summary()

            model.compile(optimizer=self.optimizer, loss=self.loss,
                          metrics=['binary_accuracy', 'mae'])
            return model
        elif self.model_name == 'ensemble':
            input_tensor = tf.keras.layers.Input(shape=self.input_shape)
            models = []
            for model_ in self.list_model:
                base_model = self.list_model[model_](include_top=False, weights=None, input_tensor=input_tensor)

                x = base_model.output
                x = tf.keras.layers.GlobalAveragePooling2D()(x)
                x = tf.keras.layers.Dropout(0.3)(x)
                x = tf.keras.layers.Dense(units=512, activation='relu')(x)
                x = tf.keras.layers.Dropout(0.3)(x)
                x = tf.keras.layers.Dense(units=512, activation='relu')(x)
                output = tf.keras.layers.Dense(units=13, activation='sigmoid')(x)

                model_ = tf.keras.Model(base_model.input, output)
                # model_.summary()

                model_.compile(optimizer=self.optimizer, loss=self.loss,
                               metrics=['binary_accuracy', 'mae'])

                model_.load_weights(os.path.join(config.model_path, f'model.{self.model_name}.h5'))
                models.append(model_)

            outputs = [model.outputs[0] for model in models]

            x = tf.keras.layers.Average()(outputs)
            model = tf.keras.Model(input_tensor, x, name='ensemble')
            return model
        else:
            raise Exception('Model name does not match, use (densenet, xception, inceptionv3, ensemble)')
