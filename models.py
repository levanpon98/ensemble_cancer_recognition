import os
import tensorflow as tf
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.nasnet import NASNetMobile
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from dpn import DPN137, DPN92, DPN98, DPN107


class XChest():
    def __init__(self, classes=1000, model_name=None, input_shape=(256, 256, 3)):
        self.model_name = model_name
        self.classes = classes
        self.input_shape = input_shape
        self.optimizer = tf.keras.optimizers.RMSprop()
        self.loss = 'binary_crossentropy'
        self.list_model = {
            'densenet': DenseNet121,
            'xception': Xception,
            'inceptionv3': InceptionV3,
            'nasnet': NASNetMobile,
            'inception_resnet_v2': InceptionResNetV2,
            'dpn': DPN137
        }

    def build(self):
        if self.model_name in self.list_model:
            base_model = self.list_model[self.model_name](include_top=False, weights='imagenet',
                                                          input_shape=self.input_shape, )

            x = base_model.output

            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            output = tf.keras.layers.Dense(units=self.classes, activation="sigmoid")(x)

            model = tf.keras.Model(base_model.input, output)

            model.compile(optimizer=self.optimizer, loss=self.loss,
                          metrics=['accuracy'])
            return model
