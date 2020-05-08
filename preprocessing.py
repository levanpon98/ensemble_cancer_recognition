import os
import shutil
import numpy as np
import pandas as pd
from itertools import chain
from absl import flags, app
from absl.flags import FLAGS

flags.DEFINE_string('input', default='/home/levanpon/data/covid-chestxray-dataset/', help='')


def main(_):
    df = pd.read_csv(os.path.join(FLAGS.input, 'metadata.csv'))

    all_labels = np.unique(list(chain(*df['finding'].map(lambda x: x.split(',')).tolist())))
    all_labels = [x for x in all_labels if len(x) > 0]

    # for row in df:



if __name__ == '__main__':
    app.run(main)
