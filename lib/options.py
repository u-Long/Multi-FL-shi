#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--rounds1', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--rounds2', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--rounds3', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=5,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.04,
                        help='the fraction of clients: C')
    parser.add_argument('--train_ep', type=int, default=5,
                        help="the number of local episodes: E")
    parser.add_argument('--local_bs', type=int, default=4,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--alg', type=str, default='fedproto', help="algorithms")
    parser.add_argument('--mode', type=str, default='task_heter', help="mode")
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")

    # other arguments
    parser.add_argument('--data_dir', type=str, default='../data/', help="directory of dataset")
    parser.add_argument('--dataset', type=str, default='UTD', help="name \
                        of dataset")
    # python federated_main.py --dataset MMAct --num_classes 37 --num_users 20
    # 
    parser.add_argument('--num_classes', type=int, default=27, help="number \
                        of classes")
    parser.add_argument('--gpu', default=0, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--test_ep', type=int, default=10, help="num of test episodes for evaluation")

    # Local arguments
    parser.add_argument('--ways', type=int, default=3, help="num of classes")
    parser.add_argument('--shots', type=int, default=100, help="num of shots")
    parser.add_argument('--train_shots_max', type=int, default=110, help="num of shots")
    parser.add_argument('--test_shots', type=int, default=15, help="num of shots")
    parser.add_argument('--stdev', type=int, default=2, help="stdev of ways")
    parser.add_argument('--ld', type=float, default=1, help="weight of proto loss")
    parser.add_argument('--ft_round', type=int, default=10, help="round of fine tuning")

    # new
    parser.add_argument('--num_train_basic', type=int, default=16, help='num_test_basic')
    parser.add_argument('--num_test_basic', type=int, default=8, help='num_test_basic') #UTD-32
    parser.add_argument('--num_unlabel_basic', type=int, default=8, help='num_unlabel_basic')
    parser.add_argument('--label_rate', type=int, default=5, help='label_rate')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16, help='num of workers to use')
    parser.add_argument('--temp', type=float, default=0.07, help='temperature for loss function')
    parser.add_argument('--num_positive', type=int, default=2, help='num_positive')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--cosine', action='store_true', help='using cosine annealing')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,175,200',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.9,
                        help='decay rate for learning rate')
    parser.add_argument('--epochs', type=int, default=200,
                    help='number of training epochs')
    args = parser.parse_args()
    return args
