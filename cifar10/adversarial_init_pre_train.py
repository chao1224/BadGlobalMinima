from __future__ import print_function

import argparse
import time as time
import copy
import os

import sys
sys.path.insert(0, 'model')
import resnet
import densenet
import vgg
from cifar10_dataset import get_dataloader
from util import *

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--no-cuda', action='store_true', default=False)

parser.add_argument('--epochs', type=int, default=350)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 250])
parser.add_argument('--gamma', type=float, default=0.1)

parser.add_argument('--use-DA', dest='DA_for_train', action='store_true')
parser.add_argument('--no-DA', dest='DA_for_train', action='store_false')
parser.set_defaults(DA_for_train=False)
parser.add_argument('--momentum', type=float, default=0)
parser.add_argument('--weight-decay', type=float, default=0)

parser.add_argument('--model', type=str, default='resnet18')
parser.add_argument('--running_idx', type=int, default=1)
parser.add_argument('--mode', type=str, default='adversarial_init_00')
parser.set_defaults(Is_Init=True)
parser.add_argument('--confusion_R', type=int, default=1)
parser.add_argument('--zero_out_ratio', type=float, default=0.1)

parser.add_argument('--model_weight_dir', type=str, default='model_weight/{}/model_weight_{}/{}')
parser.add_argument('--pre_model_weight_dir', type=str, default='model_weight/{}/model_weight_{}/{}')
parser.add_argument('--output_dir', type=str, default='output/{}/output_{}/{}')
parser.add_argument('--epoch_weight_path', type=str, default='{}/pre_epoch_{}_confusion_{}_zero_out_{}.pt')
parser.add_argument('--pre_model_weight_path', type=str, default='{}/pre_confusion_{}_zero_out_{}.pt')

parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--freq_report', type=int, default=25)


def get_model(args):
    if args.model == 'resnet18':
        model = resnet.ResNet18(num_classes=10)
    elif args.model == 'resnet50':
        model = resnet.ResNet50(num_classes=10)
    elif args.model == 'densenet40':
        model = densenet.DenseNet3(depth=40, num_classes=10)
    elif args.model == 'vgg16':
        model = vgg.VGG('VGG16')
    return model


def create_dir(args):
    model_weight_dir = args.model_weight_dir.format(args.model, args.mode, args.running_idx)
    if not os.path.exists(model_weight_dir):
        os.makedirs(model_weight_dir)
    print('Saving model weight to {}'.format(model_weight_dir))
    pre_model_weight_dir = args.pre_model_weight_dir.format(args.model, args.mode, args.running_idx)
    if not os.path.exists(pre_model_weight_dir):
        os.makedirs(pre_model_weight_dir)
    print('Saving pre-trained model to {}'.format(pre_model_weight_dir))
    output_dir = args.output_dir.format(args.model, args.mode, args.running_idx)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print('Saving output to {}'.format(output_dir))
    print()
    return model_weight_dir, pre_model_weight_dir, output_dir


if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_loader, test_loader = get_dataloader(args)

    model = get_model(args)
    if args.cuda:
        model.cuda()
        cudnn.benchmark = True
    init_model = copy.deepcopy(model)

    model_weight_dir, pre_model_weight_dir, output_dir = create_dir(args)

    print('Training strategy: DA: {}\tl2: {}\tmomentum: {}'.format(args.DA_for_train, args.weight_decay, args.momentum))
    global_learning_rate = args.lr
    optimizer = optim.SGD(model.parameters(), lr=global_learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    histogram_epoch_list = []
    training_acc_list, training_loss_list = [], []
    test_acc_list, test_loss_list = [], []

    ################## Initial State ##################
    print('Epoch: Initial')
    training_acc, training_loss = test(model, train_loader, args)
    test_acc, test_loss = test(model, test_loader, args)
    histogram_epoch_list.append(0)
    training_acc_list.append(training_acc)
    training_loss_list.append(training_loss)
    test_acc_list.append(test_acc)
    test_loss_list.append(test_loss)
    print('training acc: {},\t\ttraining loss: {}\ntest acc: {},\t\ttest loss: {}\n'.
          format(training_acc, training_loss, test_acc, test_loss))
    with open(args.epoch_weight_path.format(model_weight_dir, 0, args.confusion_R, args.zero_out_ratio), 'wb') as f_:
        torch.save(model.state_dict(), f_)

    for i in range(1, 1 + args.epochs):
        print('Epoch: {}'.format(i))
        start_time = time.time()
        train(model, train_loader, optimizer, args)
        end_time = time.time()
        print('Training Process Time:\t', (end_time - start_time))

        if i in args.schedule:
            print('Changing learning rate, from\t', global_learning_rate),
            global_learning_rate *= args.gamma
            print('to\t', global_learning_rate)
            for param_group in optimizer.param_groups:
                param_group['lr'] = global_learning_rate

        if i % args.freq_report == 0:
            training_acc, training_loss = test(model, train_loader, args)
            test_acc, test_loss = test(model, test_loader, args)
            training_acc_list.append(training_acc)
            training_loss_list.append(training_loss)
            test_acc_list.append(test_acc)
            test_loss_list.append(test_loss)
            histogram_epoch_list.append(i)
            print('training acc: {},\t\ttraining loss: {}\ntest acc: {},\t\ttest loss: {}\n'.
                  format(training_acc, training_loss, test_acc, test_loss))
            with open(args.epoch_weight_path.format(model_weight_dir, i, args.confusion_R, args.zero_out_ratio), 'wb') as f_:
                torch.save(model.state_dict(), f_)

            if training_acc >= 99.99:
                print('Hitting training acc at 100%')
                break

    with open(args.pre_model_weight_path.format(pre_model_weight_dir, args.confusion_R, args.zero_out_ratio), 'wb') as f_:
        torch.save(model.state_dict(), f_)

    np.savez_compressed('{}/pre_confusion_{}_zero_out_{}'.format(output_dir, args.confusion_R, args.zero_out_ratio),
                        training_acc_list=training_acc_list,
                        training_loss_list=training_loss_list,
                        test_acc_list=test_acc_list,
                        test_loss_list=test_loss_list)
