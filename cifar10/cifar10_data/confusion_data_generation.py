from __future__ import print_function
from PIL import Image
import os
import os.path
import errno
import copy
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
import os
import argparse


train_list = [
    ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
    ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
    ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
    ['data_batch_4', '634d18415352ddfa80567beed471001a'],
    ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
]

base_folder = 'cifar-10-batches-py'


def compare_two_images(a, b):
    cnt = 0
    for i in range(32):
        for j in range(32):
            for k in range(3):
                if a[i,j,k] == b[i,j,k]:
                    cnt += 1
    return cnt


def compare(confusion_data, actual_data, duplicate_num):
    for i in range(50000):
        raw_ = actual_data[i]
        for j in range(duplicate_num):
            generate_ = confusion_data[i*duplicate_num+j]
            print(j, '\t', 1.*(32*32*3-compare_two_images(raw_, generate_)) / (32*32*3))
        if i >= 10:
            break
    return


def load_actual_train_data_and_label(root):
    root = os.path.expanduser(root)

    actual_train_data = []
    actual_train_labels = []
    for fentry in train_list:
        f = fentry[0]
        file = os.path.join(root, base_folder, f)
        fo = open(file, 'rb')
        if sys.version_info[0] == 2:
            entry = pickle.load(fo)
        else:
            entry = pickle.load(fo, encoding='latin1')
        actual_train_data.append(entry['data'])
        if 'labels' in entry:
            actual_train_labels += entry['labels']
        else:
            actual_train_labels += entry['fine_labels']
        fo.close()

    actual_train_data = np.concatenate(actual_train_data)
    actual_train_data = actual_train_data.reshape((50000, 3, 32, 32))
    actual_train_data = actual_train_data.transpose((0, 2, 3, 1))
    return actual_train_data, actual_train_labels


def generate_random_data(confusion_R, zero_out_ratio):
    root = './data'
    actual_train_data, actual_train_labels = load_actual_train_data_and_label(root)

    duplicate_num = confusion_R

    confusion_data = []
    confusion_labels = []
    for idx in range(50000):
        l = 32 * 32 * 3
        n = int(l * zero_out_ratio)
        for _ in range(duplicate_num):
            list_ = np.arange(l)
            np.random.shuffle(list_)
            sample_index = list_[:n]
            data = copy.deepcopy(actual_train_data[idx])
            for index in sample_index:
                a = index % 32
                c = int(index / (32 * 32))
                b = int((index - a - c * 32 * 32) / 32)
                data[a, b, c] = 0
            confusion_data.append(data)
            confusion_labels.append(np.random.randint(10))

    compare(confusion_data, actual_train_data, duplicate_num)

    confusion_data = np.array(confusion_data)
    confusion_labels = np.array(confusion_labels)
    print('Training data size\t', confusion_data.shape)
    print('Training label size\t', confusion_labels.shape)
    print(confusion_labels[:100])
    print()

    np.savez_compressed('./confusion_random_train_label/zero_out_{}_{}'.format(confusion_R, zero_out_ratio),
                        training_data=confusion_data,
                        training_label=confusion_labels)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
    parser.add_argument('--confusion_R', type=int, default=10)
    parser.add_argument('--zero_out_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    confusion_R = args.confusion_R
    zero_out_ratio = args.zero_out_ratio
    seed = args.seed
    np.random.seed(seed)

    print('Regenerating confuion data: confusion R={}, zero out ratio={}.\nRandom seed: {}.'.format(confusion_R, zero_out_ratio, seed))

    generate_random_data(confusion_R=confusion_R, zero_out_ratio=zero_out_ratio)
