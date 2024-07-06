#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.metrics import accuracy_score


import copy, sys
import time
import math
import numpy as np
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
import random
import torch.utils.model_zoo as model_zoo
from pathlib import Path
import torch.backends.cudnn as cudnn
lib_dir = (Path(__file__).parent / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
mod_dir = (Path(__file__).parent / "lib" / "models").resolve()
if str(mod_dir) not in sys.path:
    sys.path.insert(0, str(mod_dir))

from resnet import resnet18
from options import args_parser
from update import LocalUpdate, save_protos, LocalTest, test_inference_new_het_lt, test_proto, agg_model_m1, agg_model_m2, agg_linear_classifier_attn, agg_feature_classifier, test_global, test_unimodal, aggregate_global_models
from models import CNNMnist, CNNFemnist, MyUTDModelFeature1, MyUTDModelFeature2, MyUTDModelFeature, SkeletonClassifier, InertialClassifier, FeatureClassifier, DualModalClassifier, LinearClassifierAttn, CustomDataset, TXTFeature, TXTDecoder, ImageFeature, IMGClassifier, UnitFeature,\
cnn_layers_1, cnn_layers_2, HeadModule
from utils import get_dataset, average_weights, exp_details, proto_aggregation, agg_func, average_weights_per, average_weights_sem, save_model_parameters_to_log, visualize_prototypes_with_tsne
from data_modules.mmact_data_module import MMActDataModule

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def FedProto_taskheter(args, train_dataset, test_dataset1, test_noisy_1, test_dataset2, testdataset12, test_noisy_12, user_groups, local_model_list, local_classifier_list):
    summary_writer = SummaryWriter('../tensorboard/'+ args.dataset +'_fedproto_' + str(args.ways) + 'w' + str(args.shots) + 's' + str(args.stdev) + 'e_' + str(args.num_users) + 'u_' + str(args.rounds1) + 'r')

    global_protos = []
    idxs_users = np.arange(args.num_users)

    train_loss, train_accuracy = [], []

    for round in tqdm(range(args.rounds1), disable=True):
        local_weights, local_w1, local_losses, local_protos = [], [], [], {}
        print(f'\n | Global Training Round : {round + 1} |\n')
        proto_loss = 0

        # idx = 16
        # local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
        # w, w1, loss, acc, protos = local_model.update_weights_het_mm(args, idx, global_protos, model=copy.deepcopy(local_model_list[idx]), classifier=copy.deepcopy(local_classifier_list[idx]), global_round=round)
        
        
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            if args.dataset == 'UMPC' or args.dataset == 'MMAct':
                if idx<8:
                    w, w1, loss, acc, protos = local_model.update_weights_het_m1(args, idx, global_protos, model=copy.deepcopy(local_model_list[idx]), classifier=copy.deepcopy(local_classifier_list[idx]), global_round=round)
                elif idx>=8 and idx<16:
                    w, w1, loss, acc, protos = local_model.update_weights_het_m2(args, idx, global_protos, model=copy.deepcopy(local_model_list[idx]), classifier=copy.deepcopy(local_classifier_list[idx]), global_round=round)
                else:
                    w, w1, loss, acc, protos = local_model.update_weights_het_mm(args, idx, global_protos, model=copy.deepcopy(local_model_list[idx]), classifier=copy.deepcopy(local_classifier_list[idx]), global_round=round)
            elif args.dataset == 'UTD':
                if idx<5:
                    w, w1, loss, acc, protos = local_model.update_weights_het_m1(args, idx, global_protos, model=copy.deepcopy(local_model_list[idx]), classifier=copy.deepcopy(local_classifier_list[idx]), global_round=round)
                elif idx>=5 and idx<11:
                    w, w1, loss, acc, protos = local_model.update_weights_het_m2(args, idx, global_protos, model=copy.deepcopy(local_model_list[idx]), classifier=copy.deepcopy(local_classifier_list[idx]), global_round=round)
                else:
                    w, w1, loss, acc, protos = local_model.update_weights_het_mm(args, idx, global_protos, model=copy.deepcopy(local_model_list[idx]), classifier=copy.deepcopy(local_classifier_list[idx]), global_round=round)
            agg_protos = agg_func(protos) # 做一个取平均的操作，对每个label

            local_weights.append(copy.deepcopy(w))
            local_w1.append(copy.deepcopy(w1))
            local_losses.append(copy.deepcopy(loss['total']))
            local_protos[idx] = agg_protos

            summary_writer.add_scalar('Train/Loss/user' + str(idx + 1), loss['total'], round)
            summary_writer.add_scalar('Train/Loss1/user' + str(idx + 1), loss['1'], round)
            summary_writer.add_scalar('Train/Loss2/user' + str(idx + 1), loss['2'], round)
            summary_writer.add_scalar('Train/Acc/user' + str(idx + 1), acc, round)

            proto_loss += loss['2']
        # print('local_protos[0]')
        # print(local_protos[0])
        # print('local_protos[2]')
        # print(local_protos[2])
        # print('local_protos[4]')
        # print(local_protos[4])
        # update global weights
        local_weights_list = local_weights
        local_w1_list = local_w1
        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights_list[idx], strict=True)
            local_model_list[idx] = local_model
            local_classifier = copy.deepcopy(local_classifier_list[idx])
            local_classifier.load_state_dict(local_w1_list[idx], strict=True)
            local_classifier_list[idx] = local_classifier
        # update global weights
        global_protos = proto_aggregation(local_protos) # 汇总取平均


        # 啊哈，我来加一个模型聚合(utd)
        local_model_list, local_classifier_list, local_model, local_classifier = aggregate_global_models(local_model_list, local_classifier_list, user_groups, args)
        

        # print(global_protos)
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
    # print("Global Prototypes:")
    # for label in sorted(global_protos.keys(), key=int):
    #     protos = global_protos[label]
    #     print(f"\nLabel: {label}")
    #     for i, proto in enumerate(protos, start=1):
    #         print(f"  Modality {i}: {proto}")
    # visualize_prototypes_with_tsne(global_protos, save_img=True, img_path='./tsne_visualization.png')
    
    acc_list_g1, acc_list_g2, acc_list_g12, loss_list12 = test_proto(args, local_model_list, test_dataset1, test_dataset2, testdataset12, global_protos, local_model)
    # print("Test with proto on modality1, acc is {:.5f}".format(np.array(acc_list_g1)))
    # print("Test with proto on modality2, acc is {:.5f}".format(np.array(acc_list_g2)))
    # print("Test with proto on modality12, acc is {:.5f}, loss is {:.5f}".format(np.array(acc_list_g12), np.array(loss_list12).mean()))
    # print('For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_g),np.std(acc_list_g)))
    # print('For all users (with protos), mean of proto loss is {:.5f}, std of loss acc is {:.5f}'.format(np.mean(loss_list), np.std(loss_list)))
    # 假设您的模型列表是local_model_list，您可以像这样保存它们的参数
    # log_file = "model_parameters_1_1.log"
    # for i, model in enumerate(local_model_list):
    #     save_model_parameters_to_log(model, f"Model_{i}", log_file)
    # test_unimodal(args, local_model_list, test_noisy_1, local_classifier_list)
    return global_protos, local_model_list, local_classifier_list
    # save protos
    # if args.dataset == 'mnist':
    #     save_protos(args, local_model_list, test_dataset, user_groups_lt)


def FedProto_taskheter2(args, train_dataset, test_dataset1, test_dataset2, test_dataset12, test_noisy_12, user_groups, local_model_list, local_classifier_list, global_protos):
    summary_writer = SummaryWriter('../tensorboard/'+ args.dataset +'_fedproto2_' + str(args.ways) + 'w' + str(args.shots) + 's' + str(args.stdev) + 'e_' + str(args.num_users) + 'u_' + str(args.rounds2) + 'r')

    idxs_users = np.arange(args.num_users)

    train_loss, train_accuracy = [], []

    for round in tqdm(range(args.rounds2), disable=True):
        local_weights, local_w1, local_losses, local_protos, local_features = [], [], [], {}, {}
        print(f'\n | Global Training Round : {round + 1} |\n')

        proto_loss = 0
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            if args.dataset == 'UTD':
                if idx<5:
                    w, loss, protos, features = local_model.update_weights_het_m1_2(args, idx, global_protos, model=copy.deepcopy(local_model_list[idx]), global_round=round)
                elif idx>=5 and idx<11:
                    w, loss, protos, features = local_model.update_weights_het_m2_2(args, idx, global_protos, model=copy.deepcopy(local_model_list[idx]), global_round=round)
                else:
                    w, loss, protos, features = local_model.update_weights_het_mm_2(args, idx, global_protos, model=copy.deepcopy(local_model_list[idx]), global_round=round)
            elif args.dataset == 'UMPC' or args.dataset == 'MMAct':
                if idx<5:
                    w, loss, protos, features = local_model.update_weights_het_m1_2(args, idx, global_protos, model=copy.deepcopy(local_model_list[idx]), global_round=round)
                elif idx>=5 and idx<11:
                    w, loss, protos, features = local_model.update_weights_het_m2_2(args, idx, global_protos, model=copy.deepcopy(local_model_list[idx]), global_round=round)
                else:
                    w, loss, protos, features = local_model.update_weights_het_mm_2(args, idx, global_protos, model=copy.deepcopy(local_model_list[idx]), global_round=round)
            agg_protos = agg_func(protos) # 做一个取平均的操作，对每个label
            agg_features = agg_func(features) 

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            local_protos[idx] = agg_protos
            local_features[idx] = agg_features
            
            summary_writer.add_scalar('Train/Loss/user' + str(idx + 1), loss, round)

            proto_loss += loss

        # update global weights
        local_weights_list = local_weights
        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights_list[idx], strict=True)
            local_model_list[idx] = local_model

        # 啊哈，我来加一个模型聚合(utd)
        local_model_list, local_classifier_list, local_model, local_classifier = aggregate_global_models(local_model_list, local_classifier_list, user_groups, args)
        # update global weights
        global_protos = proto_aggregation(local_protos) # 汇总取平均
        global_features = proto_aggregation(local_features) 
        
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
    # print("Global Prototypes:")
    # for label in sorted(global_protos.keys(), key=int):
    #     protos = global_protos[label]
    #     print(f"\nLabel: {label}")
    #     for i, proto in enumerate(protos, start=1):
    #         print(f"  Modality {i}: {proto}")
    # visualize_prototypes_with_tsne(global_protos, save_img=True, img_path='./tsne_visualization_2.png')
    acc_list_g1, acc_list_g2, acc_list_g12, loss_list12 = test_proto(args, local_model_list, test_dataset1, test_dataset2, test_dataset12, global_protos, local_model) # local_model就是聚合的多模态model
    # print("Test with proto on modality1, acc is {:.5f}".format(np.array(acc_list_g1)))
    # print("Test with proto on modality2, acc is {:.5f}".format(np.array(acc_list_g2)))
    # print("Test with proto on modality12, acc is {:.5f}, loss is {:.5f}".format(np.array(acc_list_g12), np.array(loss_list12).mean()))

    # 假设您的模型列表是local_model_list，您可以像这样保存它们的参数
    # log_file = "model_parameters_2_1.log"
    # for i, model in enumerate(local_model_list):
    #     save_model_parameters_to_log(model, f"Model_{i}", log_file)

    # save protos
    # if args.dataset == 'mnist':
    #     save_protos(args, local_model_list, test_dataset, user_groups_lt)
    
    return global_protos, local_model_list, global_features

def FedProto_taskheter3(args, train_dataset, test_dataset1, test_dataset2, test_dataset12, test_noisy_12, user_groups, local_model_list, local_classifier_list, global_protos):
    summary_writer = SummaryWriter('../tensorboard/'+ args.dataset +'_fedproto3_' + str(args.ways) + 'w' + str(args.shots) + 's' + str(args.stdev) + 'e_' + str(args.num_users) + 'u_' + str(args.rounds3) + 'r')

    idxs_users = np.arange(args.num_users)

    train_loss, train_accuracy = [], []
    flag = 0
    for round in tqdm(range(args.rounds3), disable=True):
        local_weights, local_w1, local_losses, local_protos = [], [], [], {}
        print(f'\n | Global Training Round : {round + 1} |\n')

        proto_loss = 0
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            if args.dataset == 'UTD':
                if idx<5:
                    w, w1, loss, acc, protos = local_model.update_weights_het_m1_3(args, idx, global_protos, model=copy.deepcopy(local_model_list[idx]), classifier=copy.deepcopy(local_classifier_list[idx]), global_round=round)
                elif idx>=5 and idx<11:
                    w, w1, loss, acc, protos = local_model.update_weights_het_m2_3(args, idx, global_protos, model=copy.deepcopy(local_model_list[idx]), classifier=copy.deepcopy(local_classifier_list[idx]), global_round=round)
                else:
                    w, w1, loss, acc, protos = local_model.update_weights_het_mm_3(args, idx, global_protos, model=copy.deepcopy(local_model_list[idx]), classifier=copy.deepcopy(local_classifier_list[idx]), global_round=round)
            elif args.dataset == 'UMPC' or args.dataset == 'MMAct':
                if idx<8:
                    w, w1, loss, acc, protos = local_model.update_weights_het_m1_3(args, idx, global_protos, model=copy.deepcopy(local_model_list[idx]), classifier=copy.deepcopy(local_classifier_list[idx]), global_round=round)
                elif idx>=8 and idx<16:
                    w, w1, loss, acc, protos = local_model.update_weights_het_m2_3(args, idx, global_protos, model=copy.deepcopy(local_model_list[idx]), classifier=copy.deepcopy(local_classifier_list[idx]), global_round=round)
                else:
                    w, w1, loss, acc, protos = local_model.update_weights_het_mm_3(args, idx, global_protos, model=copy.deepcopy(local_model_list[idx]), classifier=copy.deepcopy(local_classifier_list[idx]), global_round=round)
            agg_protos = agg_func(protos) # 做一个取平均的操作，对每个label

            local_weights.append(copy.deepcopy(w))
            local_w1.append(copy.deepcopy(w1))
            local_losses.append(copy.deepcopy(loss['total']))
            local_protos[idx] = agg_protos

            summary_writer.add_scalar('Train/Loss/user' + str(idx + 1), loss['total'], round)
            # summary_writer.add_scalar('Train/Loss1/user' + str(idx + 1), loss['1'], round)
            summary_writer.add_scalar('Train/Loss2/user' + str(idx + 1), loss['2'], round)
            summary_writer.add_scalar('Train/Acc/user' + str(idx + 1), acc, round)

            proto_loss += loss['2']

        # update global weights
        local_weights_list = local_weights
        local_w1_list = local_w1
        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights_list[idx], strict=True)
            local_model_list[idx] = local_model
            local_classifier = copy.deepcopy(local_classifier_list[idx])
            local_classifier.load_state_dict(local_w1_list[idx], strict=True)
            local_classifier_list[idx] = local_classifier
        # update global weights
        global_protos = proto_aggregation(local_protos) # 汇总取平均
        # print(global_protos)
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
    # print("Global Prototypes:")
    # for label in sorted(global_protos.keys(), key=int):
    #     protos = global_protos[label]
    #     print(f"\nLabel: {label}")
    #     for i, proto in enumerate(protos, start=1):
    #         print(f"  Modality {i}: {proto}")
        # acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt(flag, args, local_model_list, local_classifier_list, test_dataset1, test_dataset2, test_dataset12, global_protos)
        local_model_list, local_classifier_list, local_model, local_classifier = aggregate_global_models_c3(local_model_list,   , user_groups, args)


        flag = 1
        acc_list_g1, acc_list_g2, acc_list_g12, acc_list_l = test_inference_new_het_lt(flag, args, local_model_list, local_classifier_list, test_dataset1, test_dataset2, test_noisy_12, global_protos)
    #     print('For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_g),np.std(acc_list_g)))
    #     print('For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_l), np.std(acc_list_l)))
    #     print('For all users (with protos), mean of proto loss is {:.5f}, std of loss acc is {:.5f}'.format(np.mean(loss_list), np.std(loss_list)))
    # flag = 1
    # acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt(flag, args, local_model_list, local_classifier_list, test_dataset1, test_dataset2, test_dataset12, global_protos)
    # print('For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_g),np.std(acc_list_g)))
    # print('For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_l), np.std(acc_list_l)))
    # print('For all users (with protos), mean of proto loss is {:.5f}, std of loss acc is {:.5f}'.format(np.mean(loss_list), np.std(loss_list)))

    # 假设您的模型列表是local_model_list，您可以像这样保存它们的参数
    # log_file = "model_parameters_3_1.log"
    # for i, model in enumerate(local_model_list):
    #     save_model_parameters_to_log(model, f"Model_{i}", log_file)

    # log_file = "classifier_parameters_1.log"
    # for i, model in enumerate(local_classifier_list):
    #     save_model_parameters_to_log(model, f"Model_{i}", log_file)


def FedProto_modelheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list):
    summary_writer = SummaryWriter('../tensorboard/'+ args.dataset +'_fedproto_mh_' + str(args.ways) + 'w' + str(args.shots) + 's' + str(args.stdev) + 'e_' + str(args.num_users) + 'u_' + str(args.rounds) + 'r')

    global_protos = []
    idxs_users = np.arange(args.num_users)

    train_loss, train_accuracy = [], []

    for round in tqdm(range(args.rounds)):
        local_weights, local_losses, local_protos = [], [], {}
        print(f'\n | Global Training Round : {round + 1} |\n')

        proto_loss = 0
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w, loss, acc, protos = local_model.update_weights_het(args, idx, global_protos, model=copy.deepcopy(local_model_list[idx]), global_round=round)
            agg_protos = agg_func(protos)

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss['total']))

            local_protos[idx] = agg_protos
            summary_writer.add_scalar('Train/Loss/user' + str(idx + 1), loss['total'], round)
            summary_writer.add_scalar('Train/Loss1/user' + str(idx + 1), loss['1'], round)
            summary_writer.add_scalar('Train/Loss2/user' + str(idx + 1), loss['2'], round)
            summary_writer.add_scalar('Train/Acc/user' + str(idx + 1), acc, round)
            proto_loss += loss['2']

        # update global weights
        local_weights_list = local_weights

        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights_list[idx], strict=True)
            local_model_list[idx] = local_model

        # update global protos
        global_protos = proto_aggregation(local_protos)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

    acc_list_l, acc_list_g = test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_lt, global_protos)
    print('For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_g),np.std(acc_list_g)))
    print('For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_l), np.std(acc_list_l)))

def set_model(args):
    model = MyUTDModelFeature(input_size=1)
    classifier = LinearClassifierAttn(num_classes=args.num_classes)
    criterion = torch.nn.CrossEntropyLoss()

    model.to(args.device)
    classifier.to(args.device)

    # model.load_state_dict(state_dict)

    #freeze the MLP in pretrained feature encoders
    for name, param in model.named_parameters():
        if "head" in name:
            param.requires_grad = False
        
    return model, classifier, criterion


def train_global(train_loader, model, classifier, criterion, optimizer, epoch, local_model_list, local_classifier_list, args):
    """one epoch training"""

    model.train()
    classifier.train() 

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    pseudo_label_acc_meter = AverageMeter()  # 新增：用于跟踪伪标签准确率的计量器


    end = time.time()
    for idx, (input_data1, input_data2, labels) in enumerate(train_loader):
        if input_data1.size(0) < 2:
            continue  # 跳过只有一个样本的批次
        input_data1, input_data2, labels = input_data1.to(args.device), input_data2.to(args.device), labels.to(args.device)
        data_time.update(time.time() - end)
        log_probs = []
        for idx in range(args.num_users):
            model_ = copy.deepcopy(local_model_list[idx])
            classifier_ = copy.deepcopy(local_classifier_list[idx])
            if idx < 2:
                output = classifier_(model_(input_data1))
            elif idx < 4:
                output = classifier_(model_(input_data2))
            else:
                w1, w2 = model_(input_data1, input_data2)
                output1, output2 = classifier_(w1, w2)
                output = (output1 + output2)/2
            log_probs.append(output[:, 0:args.num_classes])
        
        _, pre_labels = torch.stack(log_probs, dim=0).mean(dim=0).max(1)
        if torch.cuda.is_available():
            pre_labels.cuda()
        accuracy = (pre_labels == labels).float().mean()
        # if accuracy < 0.75:
        #     continue
        # print(f"Accuracy of pseudo-labels: {accuracy.item() * 100:.2f}%")
        pseudo_label_acc_meter.update(accuracy.item(), input_data1.size(0))

        bsz = pre_labels.shape[0]
        
        # compute loss
        feature1, feature2 = model.encoder(input_data1, input_data2)
        # feature1: torch.Size([27, 16, 118, 4])
        # feature2: torch.Size([27, 16, 24, 7, 1])
        output, weight1, weight2 = classifier(feature1, feature2)
        loss = criterion(output, pre_labels)


        # update metric
        losses.update(loss.item(), bsz)
        acc, _ = accuracy_(output, pre_labels, topk=(1, 5))
        top1.update(acc[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()
    # 打印整个epoch的伪标签平均准确率
    print(f'Epoch: [{epoch}] Pseudo-label Accuracy: {pseudo_label_acc_meter.avg * 100:.2f}%')
    
    return losses.avg, top1.avg

def accuracy_(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr
    lr_decay_epochs = np.asarray(args.lr_decay_epochs.split(','), dtype=int)
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > lr_decay_epochs)
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    # TODO
    '''个人感觉特征和原型这里不要分开搞，有可能圆形拉近了，但特征不近（没试）'''
    '''数据集单模态划分的时候舍弃了一部分数据，这里可以优化'''
    def set_seed(seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    set_seed(42)
    start_time = time.time()

    args = args_parser()
    exp_details(args)

    # set random seeds
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda':
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # load dataset and user groups 每个用户将要处理的类别数量和每个类别的样本数量
    n_list = np.random.randint(max(2, args.ways - args.stdev), min(args.num_classes, args.ways + args.stdev + 1), args.num_users)
    if args.dataset == 'mnist':
        k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev - 1, args.num_users)
    elif args.dataset == 'cifar10':
        k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev + 1, args.num_users)
    elif args.dataset =='cifar100':
        k_list = np.random.randint(args.shots, args.shots + 1, args.num_users)
    elif args.dataset == 'femnist':
        k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev + 1, args.num_users)
    elif args.dataset == 'UTD':
        k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev + 1, args.num_users) #还没用
    elif args.dataset == 'UMPC':
        k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev + 1, args.num_users) #还没用
    elif args.dataset == 'MMAct':
        k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev + 1, args.num_users) #还没用
    train_dataloader_single_modality_1, train_dataset, test_dataset1, test_dataset2, test_dataset12, \
       test_noisy_1, test_noisy_12, global_dataset, user_groups = get_dataset(args, n_list, k_list) # 其实都是dataloader
    # train_dataloader, test_dataloader, user_groups = get_dataset(args, n_list, k_list)

    
    # for i in range(5):
    #     print(len(user_groups[i])) #52 !!!!!!!!!!!!!!!!!
    # ## unimodel
    # # 定义超参数
    # num_epochs = 20
    # learning_rate = 0.001

    # # 实例化模型
    # input_size = 1  # 根据您的数据调整
    # num_classes = 6  # 根据您的数据调整
    # encoder_model = MyUTDModelFeature1(input_size=1).to(args.device)
    # decoder_model = FeatureClassifier(args).to(args.device)

    # # 定义损失函数和优化器
    # criterion = nn.CrossEntropyLoss()
    # encoder_optimizer = optim.Adam(encoder_model.parameters(), lr=learning_rate)
    # decoder_optimizer = optim.Adam(decoder_model.parameters(), lr=learning_rate)

    # # 训练模型
    # def train_model(encoder, decoder, train_loader, criterion, encoder_optimizer, decoder_optimizer, num_epochs):
    #     encoder.train()
    #     decoder.train()
    #     for epoch in range(num_epochs):
    #         total_loss = 0
    #         for i, (inputs, labels) in enumerate(train_loader):
    #             inputs, labels = inputs.to(args.device), labels.to(args.device)

    #             # 前向传播
    #             features = encoder(inputs)
    #             outputs = decoder(features)
    #             loss = criterion(outputs, labels)

    #             # 反向传播和优化
    #             encoder_optimizer.zero_grad()
    #             decoder_optimizer.zero_grad()
    #             loss.backward()
    #             encoder_optimizer.step()
    #             decoder_optimizer.step()

    #             total_loss += loss.item()

    #         avg_loss = total_loss / len(train_loader)
    #         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    # # 预测
    # def test_model(encoder, decoder, test_loader):
    #     encoder.eval()
    #     decoder.eval()
    #     all_preds = []
    #     all_labels = []
    #     with torch.no_grad():
    #         for inputs, labels in test_loader:
    #             inputs, labels = inputs.to(args.device), labels.to(args.device)
    #             features = encoder(inputs)
    #             outputs = decoder(features)
    #             _, preds = torch.max(outputs, 1)
    #             all_preds.extend(preds.cpu().numpy())
    #             all_labels.extend(labels.cpu().numpy())

    #     accuracy = accuracy_score(all_labels, all_preds)
    #     print(f'Test Accuracy: {accuracy:.4f}')

    # # 运行训练和测试
    # train_model(encoder_model, decoder_model, train_dataloader_single_modality_1, criterion, encoder_optimizer, decoder_optimizer, num_epochs)
    # test_model(encoder_model, decoder_model, test_noisy_1)







    # Build models
    local_model_list = []
    local_classifier_list = []
    local_classifier_list_3 = []
    # 声明组件模型参数
    imu_cnn = cnn_layers_1(input_size=1)
    ske_cnn = cnn_layers_2(1, args.dataset)
    head1 = HeadModule(7552, 128)
    head2 = HeadModule(2688, 128)
    head1_mmact = HeadModule(23680, 128)
    head2_mmact = HeadModule(8576, 128)

    classifier_uni = FeatureClassifier(args)

    for i in range(args.num_users):
        if args.dataset == 'mnist':
            if args.mode == 'model_heter':
                if i<7:
                    args.out_channels = 18
                elif i>=7 and i<14:
                    args.out_channels = 20
                else:
                    args.out_channels = 22
            else:
                args.out_channels = 20

            local_model = CNNMnist(args=args)

        elif args.dataset == 'femnist':
            if args.mode == 'model_heter':
                if i<7:
                    args.out_channels = 18
                elif i>=7 and i<14:
                    args.out_channels = 20
                else:
                    args.out_channels = 22
            else:
                args.out_channels = 20
            local_model = CNNFemnist(args=args)

        elif args.dataset == 'cifar100' or args.dataset == 'cifar10':
            if args.mode == 'model_heter':
                if i<10:
                    args.stride = [1,4]
                else:
                    args.stride = [2,2]
            else:
                args.stride = [2, 2]
            resnet = resnet18(args, pretrained=False, num_classes=args.num_classes)
            initial_weight = model_zoo.load_url(model_urls['resnet18'])
            local_model = resnet
            initial_weight_1 = local_model.state_dict()
            for key in initial_weight.keys():
                if key[0:3] == 'fc.' or key[0:5]=='conv1' or key[0:3]=='bn1':
                    initial_weight[key] = initial_weight_1[key]

            local_model.load_state_dict(initial_weight)

        elif args.dataset == 'UTD':
            if i<5:
                local_model = MyUTDModelFeature1(input_size=1, p1_size=7552)
                local_model.encoder.imu_cnn_layers.load_state_dict(imu_cnn.state_dict())
                local_model.head_1.load_state_dict(head1.state_dict())
                local_classifier = FeatureClassifier(args)
                local_classifier.load_state_dict(classifier_uni.state_dict())
            elif i>=5 and i<10:
                local_model = MyUTDModelFeature2(1, 2688, args.dataset)
                local_model.encoder.skeleton_cnn_layers.load_state_dict(ske_cnn.state_dict())
                local_model.head_2.load_state_dict(head2.state_dict())
                local_classifier = FeatureClassifier(args)
                local_classifier.load_state_dict(classifier_uni.state_dict())
            else:
                local_model = MyUTDModelFeature(1, 7552, 2688, args.dataset)
                local_model.encoder.imu_cnn_layers.load_state_dict(imu_cnn.state_dict())
                local_model.encoder.skeleton_cnn_layers.load_state_dict(ske_cnn.state_dict())
                local_model.head_1.load_state_dict(head1.state_dict())
                local_model.head_2.load_state_dict(head2.state_dict())
                local_classifier = DualModalClassifier(args)
                local_classifier.classifier_modality_1.load_state_dict(classifier_uni.state_dict())
                local_classifier.classifier_modality_2.load_state_dict(classifier_uni.state_dict())
        elif args.dataset == 'UMPC':

            if i<8:
                local_model = ImageFeature()
                local_classifier = FeatureClassifier(args)
            elif i>=8 and i<16:
                bidirectional = True
                local_model = TXTFeature(hidden_dim=256, lstm_layers=2, bidirectional=bidirectional)
                local_classifier = FeatureClassifier(args)
            else:
                local_model = UnitFeature(hidden_dim=256, lstm_layers=2, bidirectional=bidirectional)
                local_classifier = DualModalClassifier(args)         
        elif args.dataset == "MMAct":
            if i<8:
                local_model = MyUTDModelFeature1(input_size=1, p1_size=23680)
                local_model.encoder.imu_cnn_layers.load_state_dict(imu_cnn.state_dict())
                local_model.head_1.load_state_dict(head1_mmact.state_dict())
                local_classifier = FeatureClassifier(args)
                local_classifier.load_state_dict(classifier_uni.state_dict())
            elif i>=8 and i<16:
                local_model = MyUTDModelFeature2(1, 8576, args.dataset)
                local_model.encoder.skeleton_cnn_layers.load_state_dict(ske_cnn.state_dict())
                local_model.head_2.load_state_dict(head2_mmact.state_dict())
                local_classifier = FeatureClassifier(args)
                local_classifier.load_state_dict(classifier_uni.state_dict())
            else:
                local_model = MyUTDModelFeature(1, 23680, 8576, args.dataset)
                local_model.encoder.imu_cnn_layers.load_state_dict(imu_cnn.state_dict())
                local_model.encoder.skeleton_cnn_layers.load_state_dict(ske_cnn.state_dict())
                local_model.head_1.load_state_dict(head1_mmact.state_dict())
                local_model.head_2.load_state_dict(head2_mmact.state_dict())
                local_classifier = DualModalClassifier(args)
                local_classifier.classifier_modality_1.load_state_dict(classifier_uni.state_dict())
                local_classifier.classifier_modality_2.load_state_dict(classifier_uni.state_dict())      


        local_model.to(args.device)
        local_model.train()
        local_model_list.append(local_model)
        local_classifier.to(args.device)
        local_classifier.train()
        local_classifier_list.append(local_classifier)
        if args.dataset == 'UTD':
            classifier = LinearClassifierAttn(num_classes=args.num_classes, input_size1=7552, input_size2=2688)
        elif args.dataset == 'UMPC':
            classifier = LinearClassifierAttn(num_classes=args.num_classes, input_size1=25088, input_size2=512)
        elif args.dataset == 'MMAct':
            classifier = LinearClassifierAttn(num_classes=args.num_classes, input_size1=23680, input_size2=8576)
        classifier.to(args.device)
        classifier.train()
        local_classifier_list_3.append(classifier)

    # log_file = "model_parameters_init.log"
    # for i, model in enumerate(local_model_list):
    #     save_model_parameters_to_log(model, f"Model_{i}", log_file)

    if args.mode == 'task_heter':
        global_protos, local_model_list, local_classifier_list = FedProto_taskheter(args, train_dataset, test_dataset1, test_noisy_1, test_dataset2, test_dataset12, test_noisy_12, user_groups, local_model_list, local_classifier_list)
        
        # model_m1 = agg_model_m1(local_model_list)
        # model_m2 = agg_model_m2(local_model_list)
        # classifier_m1 = agg_feature_classifier(local_classifier_list[:2], args)
        # classifier_m2 = agg_feature_classifier(local_classifier_list[2:4], args)
        # model_m1.to(args.device)
        # model_m2.to(args.device)
        # classifier_m1.to(args.device)
        # classifier_m2.to(args.device)
        # model, classifier, criterion = set_model(args)
        # optimizer = torch.optim.SGD([ 
        #             {'params': model.parameters(), 'lr': 1e-4},   # 0
        #             {'params': classifier.parameters(), 'lr': args.lr}],
        #             momentum=args.momentum,
        #             weight_decay=args.weight_decay)
        # # weight_model, weight_classifier = [], []
        # best_acc = 0.0
        # best_model_path = "/home/shayulong/proto/FedProto/lib/models/best_global_model.pth"
        # best_classifier_path = "/home/shayulong/proto/FedProto/lib/models/best_global_classifier.pth"

        # for epoch in range(1, args.epochs + 1):
        #     adjust_learning_rate(args, optimizer, epoch)

        #     # train for one epoch
        #     time1 = time.time()
        #     loss, acc = train_global(global_dataset, model, classifier, criterion,
        #                       optimizer, epoch, local_model_list, local_classifier_list, args)
        #     current_acc = test_global(model, classifier, test_dataset, args)
        #     if current_acc > best_acc:  # 如果当前准确率高于之前最高的，则保存模型和分类器
        #         best_acc = current_acc
        #         torch.save(model.state_dict(), best_model_path)
        #         torch.save(classifier.state_dict(), best_classifier_path)
        #         print(f"New best model saved with accuracy: {best_acc:.3f}")
        
           
    # else:
    #     FedProto_modelheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list)

    global_protos, local_model_list, global_features = FedProto_taskheter2(args, train_dataset, test_dataset1, test_dataset2, test_dataset12, test_noisy_12, user_groups, local_model_list, local_classifier_list, global_protos)
    FedProto_taskheter3(args, train_dataset, test_dataset1, test_dataset2, test_dataset12, test_noisy_12, user_groups, local_model_list, local_classifier_list_3, global_features)