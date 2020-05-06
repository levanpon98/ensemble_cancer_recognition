import os
import numpy as np
import pandas as pd

from itertools import chain
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import config


def data_gen(data_path):
    data, all_labels = prepare_data(data_path)
    train_df, valid_df = train_test_split(data,
                                          test_size=0.20,
                                          random_state=2018,
                                          stratify=data['Finding Labels'].map(lambda x: x[:4]))
    print('train', train_df.shape[0], 'validation', valid_df.shape[0])

    train_df['labels'] = train_df.apply(lambda x: x['Finding Labels'].split('|'), axis=1)
    valid_df['labels'] = valid_df.apply(lambda x: x['Finding Labels'].split('|'), axis=1)

    core_idg = ImageDataGenerator(rescale=1 / 255,
                                  samplewise_center=True,
                                  samplewise_std_normalization=True,
                                  horizontal_flip=True,
                                  vertical_flip=False,
                                  height_shift_range=0.05,
                                  width_shift_range=0.1,
                                  rotation_range=5,
                                  shear_range=0.1,
                                  fill_mode='reflect',
                                  zoom_range=0.15)

    train_gen = core_idg.flow_from_dataframe(dataframe=train_df,
                                             directory=None,
                                             x_col='path',
                                             y_col='labels',
                                             class_mode='categorical',
                                             batch_size=32,
                                             classes=all_labels,
                                             target_size=(config.image_height, config.image_width))

    valid_gen = core_idg.flow_from_dataframe(dataframe=valid_df,
                                             directory=None,
                                             x_col='path',
                                             y_col='labels',
                                             class_mode='categorical',
                                             batch_size=256,
                                             classes=all_labels,
                                             target_size=(config.image_height, config.image_width))
    
    test_X, test_Y = next(core_idg.flow_from_dataframe(dataframe=valid_df,
                                                       directory=None,
                                                       x_col='path',
                                                       y_col='labels',
                                                       class_mode='categorical',
                                                       batch_size=1024,
                                                       classes=all_labels,
                                                       target_size=(config.image_height, config.image_width)))

    return train_gen, valid_gen, test_X, test_Y, train_df.shape[0], valid_df.shape[0], all_labels


def prepare_data(data_path):
    data = pd.read_csv(os.path.join(data_path, 'Data_Entry_2017.csv'))

    data_image_paths = {os.path.basename(x): x for x in
                        glob(os.path.join(data_path, 'images', '*.png'))}

    print('Scans found:', len(data_image_paths), ', Total Headers', data.shape[0])
    data['path'] = data['Image Index'].map(data_image_paths.get)
    # data['Finding Labels'] = data['Finding Labels'].map(lambda x: x.replace('No Finding', ''))

    all_labels = np.unique(list(chain(*data['Finding Labels'].map(lambda x: x.split('|')).tolist())))
    all_labels = [x for x in all_labels if len(x) > 0]
    print('All Labels ({}): {}'.format(len(all_labels), all_labels))

    for c_label in all_labels:
        if len(c_label) > 1:  # leave out empty labels
            data[c_label] = data['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)

    all_labels = [c_label for c_label in all_labels if data[c_label].sum() > config.min_cases]
    print('Clean Labels ({})'.format(len(all_labels)), [(c_label, int(data[c_label].sum())) for c_label in all_labels])

    return data, all_labels
