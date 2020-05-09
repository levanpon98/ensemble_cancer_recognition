import os
import numpy as np
import pandas as pd
from itertools import chain
from glob import glob
from absl import app, flags
from absl.flags import FLAGS
from tqdm import tqdm
import shutil

flags.DEFINE_string('input', default='/home/levanpon/data/ChestXray-NIHCC', help='input path')
flags.DEFINE_string('output', default='/home/levanpon/data/xray-data', help='input path')


def main(_):
    data = pd.read_csv(os.path.join(FLAGS.input, 'Data_Entry_2017.csv'))
    new_data = []
    data_image_paths = {os.path.basename(x): x for x in
                        glob(os.path.join(FLAGS.input, '*', '*.png'))}
    data['path'] = data['Image Index'].map(data_image_paths.get)
    data['Finding Labels'] = data['Finding Labels'].map(lambda x: x.replace('No Finding', ''))
    all_labels = np.unique(list(chain(*data['Finding Labels'].map(lambda x: x.split('|')).tolist())))
    all_labels = [x for x in all_labels if len(x) > 0]
    for c_label in all_labels:
        if len(c_label) > 1:  # leave out empty labels
            data[c_label] = data['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)

    all_labels = [c_label for c_label in all_labels if data[c_label].sum() > 1000]

    for index, item in tqdm(data.iterrows()):
        for c_label in all_labels:
            if item[c_label]:
                new_data.append((item['path'], c_label))
    new_data = pd.DataFrame(new_data, columns=['path', 'label'])
    print(new_data.head())

    for index, item in tqdm(new_data.iterrows()):
        class_path = os.path.join(FLAGS.output, str(item['label']).lower())
        if not os.path.exists(class_path):
            os.mkdir(class_path)
        shutil.copy(item['path'], class_path)


if __name__ == '__main__':
    app.run(main)
