import cv2
import numpy as np
from models import XChest
from absl import flags, app

from tensorflow.keras.preprocessing import image

flags.DEFINE_string('image', default=None, help='Path of image')
flags.DEFINE_string('efficientnet', default='densenet', help='Model name')

_flags = flags.FLAGS

list_labels = ['Atelectasis',
               'Cardiomegaly',
               'Consolidation',
               'Edema',
               'Effusion',
               'Emphysema',
               'Fibrosis',
               'Infiltration',
               'Mass',
               'Nodule',
               'Pleural_Thickening',
               'Pneumonia',
               'Pneumothorax']


def main(_):
    image_path = _flags.image
    model_name = _flags.model

    x_chest = XChest(model_name=model_name)
    model = x_chest.build()

    if model_name != 'ensemble':
        model.load_weights(f'saved/efficientnet.{model_name}.h5')

    img = image.load_img(image_path, target_size=(256, 256))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0).astype('float') / 255.

    out = model.predict(img)
    # print(out)
    for i, score in enumerate(out[0]):
        print(list_labels[i], ': {:.2f}'.format(score))


if __name__ == '__main__':
    app.run(main)
