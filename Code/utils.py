import os
import numpy as np
import h5py
import json
import torch
from skimage.transform import resize
from skimage.io import imread
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample


def create_input_files(dataset, karpathy_json_path, image_folder, captions_per_image, min_word_freq,
                       output_folder, max_len=100):

    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)

    word_freq = Counter()

    for img in data['images']:
        captions = []
        for c in img['sentences']:
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])

        if len(captions) == 0:
            continue

