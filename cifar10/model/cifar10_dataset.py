from __future__ import print_function

from PIL import Image
import os
import os.path
import numpy as np
import sys
import copy
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch
import torch.utils.data as data
from torchvision import datasets, transforms


class CIFAR10(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 confusion_R=0, zero_out_ratio=0,
                 confusion=False, actual=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        if self.train:
            if confusion:
                # Use confusion training data
                confusion_path = '{}/zero_out_{}_{}.npz'.format(root, confusion_R, zero_out_ratio)
                confusion_data = np.load(confusion_path)
                self.train_data = confusion_data['training_data']
                self.train_labels = confusion_data['training_label']
                print('Load confusion training data from {}'.format(confusion_path))
            elif actual:
                # Use actual training data
                self.train_data = []
                self.train_labels = []
                for fentry in self.train_list:
                    f = fentry[0]
                    file = os.path.join(self.root, self.base_folder, f)
                    fo = open(file, 'rb')
                    if sys.version_info[0] == 2:
                        entry = pickle.load(fo)
                    else:
                        entry = pickle.load(fo, encoding='latin1')
                    self.train_data.append(entry['data'])
                    if 'labels' in entry:
                        self.train_labels += entry['labels']
                    else:
                        self.train_labels += entry['fine_labels']
                    fo.close()
                print('Load actual training data from {}'.format(root))

                self.train_data = np.concatenate(self.train_data)
                self.train_data = self.train_data.reshape((50000, 3, 32, 32))
                self.train_data = self.train_data.transpose((0, 2, 3, 1))
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


def get_dataloader(args):
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

    if args.DA_for_train:
        transforms_ = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transforms_ = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    if args.mode.startswith('random_init'):
        print('Apply random initialization.')
        train_dataset = CIFAR10('./cifar10_data/data', train=True, transform=transforms_, actual=True)
    elif args.mode.startswith('adversarial_init'):
        print('Apply adversarial initialization.')
        if args.Is_Init:
            print('In Adversarial Initialization / Pre Training.')
            train_dataset = CIFAR10('./cifar10_data/confusion_random_train_label', train=True, transform=transforms_,
                                    confusion_R=args.confusion_R, zero_out_ratio=args.zero_out_ratio, confusion=True)
        else:
            print('In Main Training / Fine Tuning.')
            train_dataset = CIFAR10('./cifar10_data/data', train=True, transform=transforms_, actual=True)

    test_dataset = CIFAR10('./cifar10_data/data', train=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                           ]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader