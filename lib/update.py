#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import copy
import numpy as np
from models import CNNFemnist, MyUTDModelFeature1, MyUTDModelFeature2, MyUTDModelFeature, LinearClassifierAttn, FeatureClassifier, cnn_layers_1, cnn_layers_2, HeadModule
from utils import global_test, extract_data_from_dataloader, average_state_dicts, distillation_loss, visualize_prototypes_with_tsne
from cosmo.cmc_design import FeatureConstructor, CMCLoss

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.trainloader = idxs # self.train_val_test(dataset, list(idxs))
        self.device = args.device
        self.criterion = nn.NLLLoss().to(self.device)
        
    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        idxs_train = idxs[:int(1 * len(idxs))]
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True, drop_last=True)

        return trainloader

    def update_weights(self, idx, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.train_ep):
            batch_loss = []
            for batch_idx, (images, labels_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels_g.to(self.device)

                model.zero_grad()
                log_probs, protos = model(images)
                loss = self.criterion(log_probs, labels)

                loss.backward()
                optimizer.step()

                _, y_hat = log_probs.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f} | Acc: {:.3f}'.format(
                        global_round, idx, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader),
                        loss.item(),
                        acc_val.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))


        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), acc_val.item()

    def update_weights_prox(self, idx, local_weights, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []
        if idx in local_weights.keys():
            w_old = local_weights[idx]
        w_avg = model.state_dict()
        loss_mse = nn.MSELoss().to(self.device)

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.train_ep):
            batch_loss = []
            for batch_idx, (images, labels_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels_g.to(self.device)

                model.zero_grad()
                log_probs, protos = model(images)
                loss = self.criterion(log_probs, labels)
                if idx in local_weights.keys():
                    loss2 = 0
                    for para in w_avg.keys():
                        loss2 += loss_mse(w_avg[para].float(), w_old[para].float())
                    loss2 /= len(local_weights)
                    loss += loss2 * 150
                loss.backward()
                optimizer.step()

                _, y_hat = log_probs.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f} | Acc: {:.3f}'.format(
                        global_round, idx, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader),
                        loss.item(),
                        acc_val.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))


        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), acc_val.item()


    def update_weights_het_m1(self, args, idx, global_protos, model, classifier, global_round=round):
        # Set mode to train model
        model.train()
        classifier.train()
        epoch_loss = {'total':[],'1':[], '2':[], '3':[]}

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            # optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
            #                             momentum=0.5)
            optimizer = torch.optim.SGD([ 
                    {'params': model.parameters(), 'lr': args.lr},   # 0
                    {'params': classifier.parameters(), 'lr': args.lr}],
                    momentum=0.9, weight_decay=1e-4)
        elif self.args.optimizer == 'adam':
            # optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
            #                              weight_decay=1e-4)
            optimizer = torch.optim.Adam([ 
                    {'params': model.parameters(), 'lr': args.lr},   # 0
                    {'params': classifier.parameters(), 'lr': args.lr}],
                    weight_decay=1e-4)
        for iter in range(self.args.train_ep):
            batch_loss = {'total':[],'1':[], '2':[], '3':[]}
            agg_protos_label = {}
            if args.dataset == 'MMAct':
                for batch_idx, batch in enumerate(self.trainloader):
                    m1 = batch['inertial']
                    label_g = batch['label']
                    if m1.size(0) < 2:
                        continue  # 跳过只有一个样本的批次
                    m1, labels = m1.to(self.device), label_g.to(self.device)
                    
                    model.zero_grad()
                    classifier.zero_grad()
                    optimizer.zero_grad()
                    protos = model(m1.unsqueeze(1)) # (bsz,特征维度)
                    log_probs = classifier(protos)
                    loss1 = self.criterion(log_probs, labels)
                    # food torch.Size([64, 128, 14, 14])
                    # print("m1", protos.shape) # [8, 16, 118, 4],[8,128]
                    # mmact [bsz,16,148,10]
                    loss_mse = nn.MSELoss()
                    
                    if len(global_protos) == 0:
                        loss2 = torch.tensor(0.0, device=self.device, requires_grad=True)
                    else:
                        proto_new = copy.deepcopy(protos.data)
                        i = 0
                        for label in labels: # （bsz，）
                            if label.item() in global_protos.keys():
                                proto_new[i, :] = global_protos[label.item()][0].data
                            i += 1
                        loss2 = loss_mse(proto_new, protos)

                    loss = loss1 + loss2 * args.ld
                    loss.backward()
                    optimizer.step()
                    
                    for i in range(len(labels)):
                        label = label_g[i].item()
                        if label not in agg_protos_label:
                            agg_protos_label[label] = [[], []]  # 为每个标签初始化两个空列表

                        agg_protos_label[label][0].append(protos[i,:])

                    log_probs = log_probs[:, 0:args.num_classes]
                    _, y_hat = log_probs.max(1)
                    acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()

                    if batch_idx == len(self.trainloader) - 2:
                        print('| Global Round : {} | User: {} | Local Epoch : {} | Loss: {:.3f} | Acc: {:.3f}'.format(
                            global_round, idx, iter, 
                            loss.item(),
                            acc_val.item()))
                    batch_loss['total'].append(loss.item())
                    batch_loss['1'].append(loss1.item())
                    batch_loss['2'].append(loss2.item())      
            else:          
                for batch_idx, (m1, _, label_g) in enumerate(self.trainloader):
                    if m1.size(0) < 2:
                        continue  # 跳过只有一个样本的批次
                    m1, labels = m1.to(self.device), label_g.to(self.device)
                    
                    model.zero_grad()
                    classifier.zero_grad()
                    optimizer.zero_grad()
                    protos = model(m1) # (bsz,特征维度)
                    log_probs = classifier(protos)
                    loss1 = self.criterion(log_probs, labels)
                    # food torch.Size([64, 128, 14, 14])
                    # print("m1", protos.shape) # [8, 16, 118, 4],[8,128]
                    # mmact [bsz,16,148,10]
                    loss_mse = nn.MSELoss()
                    
                    if len(global_protos) == 0:
                        loss2 = torch.tensor(0.0, device=self.device, requires_grad=True)
                    else:
                        proto_new = copy.deepcopy(protos.data)
                        i = 0
                        for label in labels: # （bsz，）
                            if label.item() in global_protos.keys():
                                proto_new[i, :] = global_protos[label.item()][0].data
                            i += 1
                        loss2 = loss_mse(proto_new, protos)

                    loss = loss1 + loss2 * args.ld
                    loss.backward()
                    optimizer.step()
                    
                    for i in range(len(labels)):
                        label = label_g[i].item()
                        if label not in agg_protos_label:
                            agg_protos_label[label] = [[], []]  # 为每个标签初始化两个空列表

                        agg_protos_label[label][0].append(protos[i,:])

                    log_probs = log_probs[:, 0:args.num_classes]
                    _, y_hat = log_probs.max(1)
                    acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()

                    if batch_idx == len(self.trainloader) - 2:
                        print('| Global Round : {} | User: {} | Local Epoch : {} | Loss: {:.3f} | Acc: {:.3f}'.format(
                            global_round, idx, iter, 
                            loss.item(),
                            acc_val.item()))
                    batch_loss['total'].append(loss.item())
                    batch_loss['1'].append(loss1.item())
                    batch_loss['2'].append(loss2.item())
            epoch_loss['total'].append(sum(batch_loss['total'])/len(batch_loss['total']))
            epoch_loss['1'].append(sum(batch_loss['1']) / len(batch_loss['1']))
            epoch_loss['2'].append(sum(batch_loss['2']) / len(batch_loss['2']))

        epoch_loss['total'] = sum(epoch_loss['total']) / len(epoch_loss['total'])
        epoch_loss['1'] = sum(epoch_loss['1']) / len(epoch_loss['1'])
        epoch_loss['2'] = sum(epoch_loss['2']) / len(epoch_loss['2'])
        # print(agg_protos_label)没问题
        return model.state_dict(), classifier.state_dict(), epoch_loss, acc_val.item(), agg_protos_label

    def update_weights_het_m2(self, args, idx, global_protos, model, classifier, global_round=round):
        # Set mode to train model
        model.train()
        classifier.train()
        epoch_loss = {'total':[],'1':[], '2':[], '3':[]}

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            # optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
            #                             momentum=0.5)
            optimizer = torch.optim.SGD([ 
                    {'params': model.parameters(), 'lr': args.lr},   # 0
                    {'params': classifier.parameters(), 'lr': args.lr}],
                    momentum=0.9, weight_decay=1e-4)
        elif self.args.optimizer == 'adam':
            # optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
            #                              weight_decay=1e-4)
            optimizer = torch.optim.Adam([ 
                    {'params': model.parameters(), 'lr': args.lr},   # 0
                    {'params': classifier.parameters(), 'lr': args.lr}],
                    weight_decay=1e-4)

        for iter in range(self.args.train_ep):
            batch_loss = {'total':[],'1':[], '2':[], '3':[]}
            agg_protos_label = {}

            if args.dataset == 'MMAct':
                for batch_idx, batch in enumerate(self.trainloader):
                    m2 = batch['skeleton']
                    label_g = batch['label']
                    if m2.size(0) < 2:
                        continue  # 跳过只有一个样本的批次
                    m2, labels = m2.to(self.device), label_g.to(self.device)
                    model.zero_grad()
                    classifier.zero_grad()
                    optimizer.zero_grad()
                    # print(m2.shape) # torch.Size([8, 17, 2, 150])
                    protos = model(m2.permute(0, 3, 1, 2).unsqueeze(1)) # (bsz,特征维度)
                    log_probs = classifier(protos)
                    loss1 = self.criterion(log_probs, labels)
                    # print("m2", protos.shape)[8, 16, 24, 7, 1]
                    #mmact (bsz, 16, 134, 4, 1)
                    loss_mse = nn.MSELoss()
                    if len(global_protos) == 0:
                        loss2 = torch.tensor(0.0, device=self.device, requires_grad=True)
                    else:
                        proto_new = copy.deepcopy(protos.data)
                        i = 0
                        for label in labels: # （bsz，）
                            if label.item() in global_protos.keys():
                                proto_new[i, :] = global_protos[label.item()][1].data
                            i += 1
                        loss2 = loss_mse(proto_new, protos)

                    loss = loss1 + loss2 * args.ld
                    loss.backward()
                    optimizer.step()

                    for i in range(len(labels)):
                        label = label_g[i].item()
                        if label not in agg_protos_label:
                            agg_protos_label[label] = [[], []]  # 为每个标签初始化两个空列表

                        agg_protos_label[label][1].append(protos[i,:])
                    log_probs = log_probs[:, 0:args.num_classes]
                    _, y_hat = log_probs.max(1)
                    acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()
                    if batch_idx == len(self.trainloader) - 2:
                        print('| Global Round : {} | User: {} | Local Epoch : {} | Loss: {:.3f} | Acc: {:.3f}'.format(
                            global_round, idx, iter, 
                            loss.item(),
                            acc_val.item()))
                    batch_loss['total'].append(loss.item())
                    batch_loss['1'].append(loss1.item())
                    batch_loss['2'].append(loss2.item())
            else:
                for batch_idx, (_, m2, label_g) in enumerate(self.trainloader):
                    # print(m2['input_ids'].shape)[64,40]
                    if args.dataset == 'UMPC':
                        if m2['input_ids'].size(0) < 2:
                            continue  # 跳过只有一个样本的批次
                        input_ids, attention_mask, labels = m2['input_ids'].to(self.device), m2['attention_mask'].to(self.device), label_g.to(self.device)
                        model.zero_grad()
                        classifier.zero_grad()
                        optimizer.zero_grad()
                        protos = model(input_ids, attention_mask) # (bsz,特征维度)
                        # print("m2", protos.shape) # [64, 512]
                    elif args.dataset == 'UTD':
                        if m2.size(0) < 2:
                            continue  # 跳过只有一个样本的批次
                        m2, labels = m2.to(self.device), label_g.to(self.device)
                        model.zero_grad()
                        classifier.zero_grad()
                        optimizer.zero_grad()
                        protos = model(m2) # (bsz,特征维度)
                    log_probs = classifier(protos)
                    loss1 = self.criterion(log_probs, labels)
                    # print("m2", protos.shape)[8, 16, 24, 7, 1]
                    #mmact (bsz, 16, 134, 4, 1)
                    loss_mse = nn.MSELoss()
                    if len(global_protos) == 0:
                        loss2 = torch.tensor(0.0, device=self.device, requires_grad=True)
                    else:
                        proto_new = copy.deepcopy(protos.data)
                        i = 0
                        for label in labels: # （bsz，）
                            if label.item() in global_protos.keys():
                                proto_new[i, :] = global_protos[label.item()][1].data
                            i += 1
                        loss2 = loss_mse(proto_new, protos)

                    loss = loss1 + loss2 * args.ld
                    loss.backward()
                    optimizer.step()

                    for i in range(len(labels)):
                        label = label_g[i].item()
                        if label not in agg_protos_label:
                            agg_protos_label[label] = [[], []]  # 为每个标签初始化两个空列表

                        agg_protos_label[label][1].append(protos[i,:])
                    log_probs = log_probs[:, 0:args.num_classes]
                    _, y_hat = log_probs.max(1)
                    acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()
                    if batch_idx == len(self.trainloader) - 2:
                        print('| Global Round : {} | User: {} | Local Epoch : {} | Loss: {:.3f} | Acc: {:.3f}'.format(
                            global_round, idx, iter, 
                            loss.item(),
                            acc_val.item()))
                    batch_loss['total'].append(loss.item())
                    batch_loss['1'].append(loss1.item())
                    batch_loss['2'].append(loss2.item())
            epoch_loss['total'].append(sum(batch_loss['total'])/len(batch_loss['total']))
            epoch_loss['1'].append(sum(batch_loss['1']) / len(batch_loss['1']))
            epoch_loss['2'].append(sum(batch_loss['2']) / len(batch_loss['2']))

        epoch_loss['total'] = sum(epoch_loss['total']) / len(epoch_loss['total'])
        epoch_loss['1'] = sum(epoch_loss['1']) / len(epoch_loss['1'])
        epoch_loss['2'] = sum(epoch_loss['2']) / len(epoch_loss['2'])
        
        return model.state_dict(), classifier.state_dict(), epoch_loss, acc_val.item(), agg_protos_label
    
    def update_weights_het_mm(self, args, idx, global_protos, model, classifier, global_round=round):
        # Set mode to train model
        model.train()
        classifier.train()
        epoch_loss = {'total':[],'1':[], '2':[], '3':[]}

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            # optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
            #                             momentum=0.5)
            optimizer = torch.optim.SGD([ 
                    {'params': model.parameters(), 'lr': args.lr},
                    {'params': classifier.parameters(), 'lr': args.lr}],
                    momentum=0.9, weight_decay=1e-4)
            # optimizer_modality_1 = torch.optim.SGD([ 
            #         {'params': model.encoder.imu_cnn_layers.parameters(), 'lr': args.lr},   # 0
            #         {'params': model.head_1.parameters(), 'lr': args.lr},
            #         {'params': classifier.classifier_modality_1.parameters(), 'lr': args.lr}],
            #         momentum=0.9, weight_decay=1e-4)
        # elif self.args.optimizer == 'adam':
            # optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
            #                              weight_decay=1e-4)
            # optimizer_modality_2 = torch.optim.SGD([ 
            #         {'params': model.encoder.skeleton_cnn_layers.parameters(), 'lr': args.lr},   # 0
            #         {'params': model.head_2.parameters(), 'lr': args.lr},
            #         {'params': classifier.classifier_modality_2.parameters(), 'lr': args.lr}],
            #         weight_decay=1e-4)

        for iter in range(self.args.train_ep):
            batch_loss = {'total':[],'1':[], '2':[], '3':[]}
            agg_protos_label = {}
            if args.dataset == 'MMAct':
                for batch_idx, batch in enumerate(self.trainloader):
                    m1 = batch['inertial']
                    m2 = batch['skeleton']
                    label_g = batch['label']
                    if m1.size(0) < 2:
                        continue  # 跳过只有一个样本的批次
                    m1, m2, labels = m1.to(self.device), m2.to(self.device), label_g.to(self.device)
                    model.zero_grad()
                    classifier.zero_grad()
                    optimizer.zero_grad()

                    protos1, protos2 = model(m1.unsqueeze(1), m2.permute(0, 3, 1, 2).unsqueeze(1)) # (bsz,特征维度)
                    log_probs1, log_probs2 = classifier(protos1, protos2)
                    # loss1_1 = self.criterion(log_probs1, labels)
                    # loss1_2 = self.criterion(log_probs2, labels)
                    loss1 = 1/2*self.criterion(log_probs1, labels)+1/2*self.criterion(log_probs2, labels)
                    loss_mse = nn.MSELoss()
                    if len(global_protos) == 0:
                        loss2 = torch.tensor(0.0, device=self.device, requires_grad=True)
                        # loss2_1 = torch.tensor(0.0, device=self.device, requires_grad=True)
                        # loss2_2 = torch.tensor(0.0, device=self.device, requires_grad=True)
                    else:
                        proto_new1 = copy.deepcopy(protos1.data)
                        proto_new2 = copy.deepcopy(protos2.data)
                        i = 0
                        for label in labels: # （bsz，）
                            if label.item() in global_protos.keys():
                                proto_new1[i, :] = global_protos[label.item()][0].data
                                proto_new2[i, :] = global_protos[label.item()][1].data
                            i += 1
                        loss2 = 1/2*loss_mse(proto_new1, protos1)+1/2*loss_mse(proto_new2, protos2)
                        # loss2_1 = loss_mse(proto_new1, protos1)
                        # loss2_2 = loss_mse(proto_new2, protos2)
                    loss = loss1 + loss2 * args.ld
                    loss.backward()
                    optimizer.step()
                    # loss_1 = loss1_1#+loss2_1*args.ld
                    # loss_2 = loss1_2#+loss2_2*args.ld
                    # loss_1.backward()
                    # optimizer_modality_1.step()
                    # loss_2.backward()
                    # optimizer_modality_2.step()

                    for i in range(len(labels)):
                        label = label_g[i].item()
                        if label not in agg_protos_label:
                            agg_protos_label[label] = [[], []]  # 为每个标签初始化两个空列表

                        agg_protos_label[label][0].append(protos1[i,:])
                        agg_protos_label[label][1].append(protos2[i,:])

                    log_probs1 = log_probs1[:, 0:args.num_classes]
                    _, y_hat1 = log_probs1.max(1)
                    log_probs2 = log_probs2[:, 0:args.num_classes]
                    _, y_hat2 = log_probs2.max(1)
                    acc_val = (torch.eq(y_hat1, labels.squeeze()).float().mean()+torch.eq(y_hat2, labels.squeeze()).float().mean())/2

                    if batch_idx == len(self.trainloader) - 2:
                        print('| Global Round : {} | User: {} | Local Epoch : {} | Loss: {:.3f} | Acc: {:.3f}'.format(
                            global_round, idx, iter, 
                            loss.item(),
                            acc_val.item()))
                    batch_loss['total'].append(loss.item())
                    batch_loss['1'].append(loss1.item())
                    batch_loss['2'].append(loss2.item())
            else:
                for batch_idx, (m1, m2, label_g) in enumerate(self.trainloader):
                    if m1.size(0) < 2:
                        continue  # 跳过只有一个样本的批次
                    if args.dataset == 'UMPC':
                        m1, input_ids, attention_mask, labels = m1.to(self.device), m2['input_ids'].to(self.device), m2['attention_mask'].to(self.device), label_g.to(self.device)
                        model.zero_grad()
                        classifier.zero_grad()
                        optimizer.zero_grad()

                        protos1, protos2 = model(m1, input_ids, attention_mask) # (bsz,特征维度)
                    elif args.dataset == 'UTD':
                        m1, m2, labels = m1.to(self.device), m2.to(self.device), label_g.to(self.device)
                        model.zero_grad()
                        classifier.zero_grad()
                        optimizer.zero_grad()

                        protos1, protos2 = model(m1, m2) # (bsz,特征维度)
                    log_probs1, log_probs2 = classifier(protos1, protos2)
                    # loss1_1 = self.criterion(log_probs1, labels)
                    # loss1_2 = self.criterion(log_probs2, labels)
                    loss1 = 1/2*self.criterion(log_probs1, labels)+1/2*self.criterion(log_probs2, labels)
                    loss_mse = nn.MSELoss()
                    if len(global_protos) == 0:
                        loss2 = torch.tensor(0.0, device=self.device, requires_grad=True)
                        # loss2_1 = torch.tensor(0.0, device=self.device, requires_grad=True)
                        # loss2_2 = torch.tensor(0.0, device=self.device, requires_grad=True)
                    else:
                        proto_new1 = copy.deepcopy(protos1.data)
                        proto_new2 = copy.deepcopy(protos2.data)
                        i = 0
                        for label in labels: # （bsz，）
                            if label.item() in global_protos.keys():
                                proto_new1[i, :] = global_protos[label.item()][0].data
                                proto_new2[i, :] = global_protos[label.item()][1].data
                            i += 1
                        loss2 = 1/2*loss_mse(proto_new1, protos1)+1/2*loss_mse(proto_new2, protos2)
                        # loss2_1 = loss_mse(proto_new1, protos1)
                        # loss2_2 = loss_mse(proto_new2, protos2)
                    loss = loss1 + loss2 * args.ld
                    loss.backward()
                    optimizer.step()
                    # loss_1 = loss1_1#+loss2_1*args.ld
                    # loss_2 = loss1_2#+loss2_2*args.ld
                    # loss_1.backward()
                    # optimizer_modality_1.step()
                    # loss_2.backward()
                    # optimizer_modality_2.step()

                    for i in range(len(labels)):
                        label = label_g[i].item()
                        if label not in agg_protos_label:
                            agg_protos_label[label] = [[], []]  # 为每个标签初始化两个空列表

                        agg_protos_label[label][0].append(protos1[i,:])
                        agg_protos_label[label][1].append(protos2[i,:])

                    log_probs1 = log_probs1[:, 0:args.num_classes]
                    _, y_hat1 = log_probs1.max(1)
                    log_probs2 = log_probs2[:, 0:args.num_classes]
                    _, y_hat2 = log_probs2.max(1)
                    acc_val = (torch.eq(y_hat1, labels.squeeze()).float().mean()+torch.eq(y_hat2, labels.squeeze()).float().mean())/2

                    if batch_idx == len(self.trainloader) - 2:
                        print('| Global Round : {} | User: {} | Local Epoch : {} | Loss: {:.3f} | Acc: {:.3f}'.format(
                            global_round, idx, iter, 
                            loss.item(),
                            acc_val.item()))
                    batch_loss['total'].append(loss.item())
                    batch_loss['1'].append(loss1.item())
                    batch_loss['2'].append(loss2.item())
            epoch_loss['total'].append(sum(batch_loss['total'])/len(batch_loss['total']))
            epoch_loss['1'].append(sum(batch_loss['1']) / len(batch_loss['1']))
            epoch_loss['2'].append(sum(batch_loss['2']) / len(batch_loss['2']))

        epoch_loss['total'] = sum(epoch_loss['total']) / len(epoch_loss['total'])
        epoch_loss['1'] = sum(epoch_loss['1']) / len(epoch_loss['1'])
        epoch_loss['2'] = sum(epoch_loss['2']) / len(epoch_loss['2'])
        
        return model.state_dict(), classifier.state_dict(), epoch_loss, acc_val.item(), agg_protos_label
    
    def update_weights_het_m1_2(self, args, idx, global_protos, model, global_round=round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.9, weight_decay=1e-4)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        for iter in range(self.args.train_ep):
            batch_loss = []
            agg_protos_label = {}
            agg_features_label = {}
            for batch_idx, (m1, _, label_g) in enumerate(self.trainloader):
                if m1.size(0) < 2:
                    continue  # 跳过只有一个样本的批次
                m1, labels = m1.to(self.device), label_g.to(self.device)
                
                model.zero_grad()
                optimizer.zero_grad()
                protos = model(m1) # (bsz,特征维度)
                missing_protos = torch.stack([global_protos[label.item()][1] for label in labels], dim=0).to(self.device)
                # print(missing_protos.shape)
                features = FeatureConstructor(protos, missing_protos, args.num_positive)
                criterion = CMCLoss(temperature=args.temp)
                loss1 = criterion(features)
                # print("m1", protos.shape)[8, 16, 118, 4],[8,128]
                loss_mse = nn.MSELoss()
                if len(global_protos) == 0:
                    loss2 = torch.tensor(0.0, device=self.device, requires_grad=True)
                else:
                    proto_new = copy.deepcopy(protos.data)
                    i = 0
                    for label in labels: # （bsz，）
                        if label.item() in global_protos.keys():
                            proto_new[i, :] = global_protos[label.item()][0].data
                        i += 1
                    loss2 = loss_mse(proto_new, protos)

                loss = loss1 #+ loss2 * args.ld

                loss.backward()
                optimizer.step()
                protos_new = model.encoder(m1)
                for i in range(len(labels)):
                    label = label_g[i].item()
                    if label not in agg_protos_label:
                        agg_protos_label[label] = [[], []]  # 为每个标签初始化两个空列表

                    agg_protos_label[label][0].append(protos[i,:])
                    

                for i in range(len(labels)):
                    label = label_g[i].item()
                    if label not in agg_features_label:
                        agg_features_label[label] = [[], []]  # 为每个标签初始化两个空列表

                    agg_features_label[label][0].append(protos_new[i,:])


                if batch_idx == len(self.trainloader) - 2:
                    print('| Global Round : {} | User: {} | Local Epoch : {} | Loss: {:.3f}'.format(
                        global_round, idx, iter, 
                        loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        epoch_loss = sum(epoch_loss) / len(epoch_loss)

        return model.state_dict(), epoch_loss, agg_protos_label, agg_features_label
    
    def update_weights_het_m2_2(self, args, idx, global_protos, model, global_round=round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.9, weight_decay=1e-4)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        for iter in range(self.args.train_ep):
            batch_loss = []
            agg_protos_label = {}
            agg_features_label = {}
            for batch_idx, (_, m2, label_g) in enumerate(self.trainloader):
                if args.dataset == 'UMPC':
                    if m2['input_ids'].size(0) < 2:
                        continue  # 跳过只有一个样本的批次
                    input_ids, attention_mask, labels = m2['input_ids'].to(self.device), m2['attention_mask'].to(self.device), label_g.to(self.device)
                    model.zero_grad()
                    optimizer.zero_grad()
                    protos = model(input_ids, attention_mask) # (bsz,特征维度)
                    # print("m2", protos.shape) # [64, 512]
                elif args.dataset == 'UTD':
                    if m2.size(0) < 2:
                        continue  # 跳过只有一个样本的批次
                    m2, labels = m2.to(self.device), label_g.to(self.device)
                    model.zero_grad()
                    optimizer.zero_grad()
                    protos = model(m2) # (bsz,特征维度)
                missing_protos = torch.stack([global_protos[label.item()][0] for label in labels], dim=0).to(self.device)
                features = FeatureConstructor(missing_protos, protos, args.num_positive)
                criterion = CMCLoss(temperature=args.temp)
                loss1 = criterion(features)
                # print("m1", protos.shape)[8, 16, 118, 4],[8,128]
                loss_mse = nn.MSELoss()
                if len(global_protos) == 0:
                    loss2 = torch.tensor(0.0, device=self.device, requires_grad=True)
                else:
                    proto_new = copy.deepcopy(protos.data)
                    i = 0
                    for label in labels: # （bsz，）
                        if label.item() in global_protos.keys():
                            proto_new[i, :] = global_protos[label.item()][1].data
                        i += 1
                    loss2 = loss_mse(proto_new, protos)

                loss = loss1 #+ loss2 * args.ld
                loss.backward()
                optimizer.step()
                if args.dataset == 'UMPC':
                    protos_new = model.encoder(input_ids, attention_mask)
                elif args.dataset == 'UTD':
                    protos_new = model.encoder(m2)
                for i in range(len(labels)):
                    label = label_g[i].item()
                    if label not in agg_protos_label:
                        agg_protos_label[label] = [[], []]  # 为每个标签初始化两个空列表

                    agg_protos_label[label][1].append(protos[i,:])

                for i in range(len(labels)):
                    label = label_g[i].item()
                    if label not in agg_features_label:
                        agg_features_label[label] = [[], []]  # 为每个标签初始化两个空列表

                    agg_features_label[label][1].append(protos_new[i,:])

                if batch_idx == len(self.trainloader) - 2:
                    print('| Global Round : {} | User: {} | Local Epoch : {} | Loss: {:.3f}'.format(
                        global_round, idx, iter, 
                        loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        epoch_loss = sum(epoch_loss) / len(epoch_loss)

        return model.state_dict(), epoch_loss, agg_protos_label, agg_features_label
    
    def update_weights_het_mm_2(self, args, idx, global_protos, model, global_round=round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.9, weight_decay=1e-4)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        for iter in range(self.args.train_ep):
            batch_loss = []
            agg_protos_label = {}
            agg_features_label = {}
            for batch_idx, (m1, m2, label_g) in enumerate(self.trainloader):
                if m1.size(0) < 2:
                    continue  # 跳过只有一个样本的批次
                if args.dataset == 'UMPC':
                    m1, input_ids, attention_mask, labels = m1.to(self.device), m2['input_ids'].to(self.device), m2['attention_mask'].to(self.device), label_g.to(self.device)
                    model.zero_grad()
                    optimizer.zero_grad()

                    protos1, protos2 = model(m1, input_ids, attention_mask) # (bsz,特征维度)
                elif args.dataset == 'UTD':
                    m1, m2, labels = m1.to(self.device), m2.to(self.device), label_g.to(self.device)
                    model.zero_grad()
                    optimizer.zero_grad()

                    protos1, protos2 = model(m1, m2) # (bsz,特征维度)
                
                features = FeatureConstructor(protos1, protos2, args.num_positive)
                criterion = CMCLoss(temperature=args.temp)
                loss1 = criterion(features)
                # print("m1", protos.shape)[8, 16, 118, 4],[8,128]
                
                loss_mse = nn.MSELoss()
                if len(global_protos) == 0:
                    loss2 = torch.tensor(0.0, device=self.device, requires_grad=True)
                else:
                    proto_new1 = copy.deepcopy(protos1.data)
                    proto_new2 = copy.deepcopy(protos2.data)
                    i = 0
                    for label in labels: # （bsz，）
                        if label.item() in global_protos.keys():
                            proto_new1[i, :] = global_protos[label.item()][0].data
                            proto_new2[i, :] = global_protos[label.item()][1].data
                        i += 1
                    loss2 = 1/2*loss_mse(proto_new1, protos1)+1/2*loss_mse(proto_new2, protos2)

                loss = loss1 #+ loss2 * args.ld
                
                loss.backward()
                optimizer.step()
                if args.dataset == 'UMPC':
                    protos1_new, protos2_new = model.encoder(m1, input_ids, attention_mask)
                elif args.dataset == 'UTD':
                    protos1_new, protos2_new = model.encoder(m1, m2)

                for i in range(len(labels)):
                    label = label_g[i].item()
                    if label not in agg_protos_label:
                        agg_protos_label[label] = [[], []]  # 为每个标签初始化两个空列表

                    agg_protos_label[label][0].append(protos1[i,:])
                    agg_protos_label[label][1].append(protos2[i,:])

                for i in range(len(labels)):
                    label = label_g[i].item()
                    if label not in agg_features_label:
                        agg_features_label[label] = [[], []]  # 为每个标签初始化两个空列表

                    agg_features_label[label][0].append(protos1_new[i,:])
                    agg_features_label[label][1].append(protos2_new[i,:])

                if batch_idx == len(self.trainloader) - 2:
                    print('| Global Round : {} | User: {} | Local Epoch : {} | Loss: {:.3f}'.format(
                        global_round, idx, iter, 
                        loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        epoch_loss = sum(epoch_loss) / len(epoch_loss)

        return model.state_dict(), epoch_loss, agg_protos_label, agg_features_label
    
    def update_weights_het_m1_3(self, args, idx, global_protos, model, classifier, global_round=round):
        # Set mode to train model
        model.train()
        classifier.train()
        epoch_loss = {'total':[],'1':[], '2':[], '3':[]}

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            # optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
            #                             momentum=0.5)
            optimizer = torch.optim.SGD([ 
                    {'params': model.parameters(), 'lr': 1e-4},   # 0
                    {'params': classifier.parameters(), 'lr': args.lr}],
                    momentum=args.momentum,
                    weight_decay=args.weight_decay)
        elif self.args.optimizer == 'adam':
            # optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
            #                              weight_decay=1e-4)
            optimizer = torch.optim.Adam([ 
                    {'params': model.parameters(), 'lr': args.lr},   # 0
                    {'params': classifier.parameters(), 'lr': args.lr}],
                    weight_decay=1e-4)
        for iter in range(self.args.train_ep):
            batch_loss = {'total':[],'1':[], '2':[], '3':[]}
            agg_protos_label = {}
            
            for batch_idx, (m1, _, label_g) in enumerate(self.trainloader):
                if m1.size(0) < 2:
                    continue  # 跳过只有一个样本的批次
                m1, labels = m1.to(self.device), label_g.to(self.device)
                
                model.zero_grad()
                classifier.zero_grad()
                optimizer.zero_grad()
                protos = model.encoder(m1) # (bsz,特征维度)
                missing_protos = torch.stack([global_protos[label.item()][1] for label in labels], dim=0).to(self.device)
                log_probs, weight1, weight2 = classifier(protos, missing_protos)
                loss1 = self.criterion(log_probs, labels)
                # print("m1", protos.shape)[8, 16, 118, 4],[8,128]
                loss_mse = nn.MSELoss()
                
                if len(global_protos) == 0:
                    loss2 = torch.tensor(0.0, device=self.device, requires_grad=True)
                else:
                    proto_new = copy.deepcopy(protos.data)
                    i = 0
                    for label in labels: # （bsz，）
                        if label.item() in global_protos.keys():
                            proto_new[i, :] = global_protos[label.item()][0].data
                        i += 1
                    loss2 = loss_mse(proto_new, protos)

                loss = loss1 + loss2*args.ld # loss2不知道有没有必要
                loss.backward()
                optimizer.step()
                
                for i in range(len(labels)):
                    label = label_g[i].item()
                    if label not in agg_protos_label:
                        agg_protos_label[label] = [[], []]  # 为每个标签初始化两个空列表

                    agg_protos_label[label][0].append(protos[i,:])

                log_probs = log_probs[:, 0:args.num_classes]
                _, y_hat = log_probs.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()

                if batch_idx == len(self.trainloader) - 2:
                    print('| Global Round : {} | User: {} | Local Epoch : {} | Loss: {:.3f} | Acc: {:.3f}'.format(
                        global_round, idx, iter, 
                        loss.item(),
                        acc_val.item()))
                batch_loss['total'].append(loss.item())
                batch_loss['1'].append(loss1.item())
                batch_loss['2'].append(loss2.item())
            epoch_loss['total'].append(sum(batch_loss['total'])/len(batch_loss['total']))
            epoch_loss['1'].append(sum(batch_loss['1']) / len(batch_loss['1']))
            epoch_loss['2'].append(sum(batch_loss['2']) / len(batch_loss['2']))

        epoch_loss['total'] = sum(epoch_loss['total']) / len(epoch_loss['total'])
        epoch_loss['1'] = sum(epoch_loss['1']) / len(epoch_loss['1'])
        epoch_loss['2'] = sum(epoch_loss['2']) / len(epoch_loss['2'])
        # print(agg_protos_label)没问题
        return model.state_dict(), classifier.state_dict(), epoch_loss, acc_val.item(), agg_protos_label

    def update_weights_het_m2_3(self, args, idx, global_protos, model, classifier, global_round=round):
        # Set mode to train model
        model.train()
        classifier.train()
        epoch_loss = {'total':[],'1':[], '2':[], '3':[]}

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            # optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
            #                             momentum=0.5)
            optimizer = torch.optim.SGD([ 
                    {'params': model.parameters(), 'lr': 1e-4},   # 0
                    {'params': classifier.parameters(), 'lr': args.lr}],
                    momentum=args.momentum,
                    weight_decay=args.weight_decay)
        elif self.args.optimizer == 'adam':
            # optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
            #                              weight_decay=1e-4)
            optimizer = torch.optim.Adam([ 
                    {'params': model.parameters(), 'lr': args.lr},   # 0
                    {'params': classifier.parameters(), 'lr': args.lr}],
                    weight_decay=1e-4)
        for iter in range(self.args.train_ep):
            batch_loss = {'total':[],'1':[], '2':[], '3':[]}
            agg_protos_label = {}
            
            for batch_idx, (_, m2, label_g) in enumerate(self.trainloader):
                if args.dataset == 'UTD':
                    if m2.size(0) < 2:
                        continue  # 跳过只有一个样本的批次
                    m2, labels = m2.to(self.device), label_g.to(self.device)
                    
                    model.zero_grad()
                    classifier.zero_grad()
                    optimizer.zero_grad()
                    protos = model.encoder(m2) # (bsz,特征维度)
                elif args.dataset == 'UMPC':
                    if m2['input_ids'].size(0) < 2:
                        continue  # 跳过只有一个样本的批次
                    input_ids, attention_mask, labels = m2['input_ids'].to(self.device), m2['attention_mask'].to(self.device), label_g.to(self.device)
                    model.zero_grad()
                    classifier.zero_grad()
                    optimizer.zero_grad()
                    protos = model(input_ids, attention_mask) # (bsz,特征维度)
                missing_protos = torch.stack([global_protos[label.item()][0] for label in labels], dim=0).to(self.device)
                log_probs, weight1, weight2 = classifier(missing_protos, protos)
                loss1 = self.criterion(log_probs, labels)
                # print("m1", protos.shape)[8, 16, 118, 4],[8,128]
                loss_mse = nn.MSELoss()
                
                if len(global_protos) == 0:
                    loss2 = torch.tensor(0.0, device=self.device, requires_grad=True)
                else:
                    proto_new = copy.deepcopy(protos.data)
                    i = 0
                    for label in labels: # （bsz，）
                        if label.item() in global_protos.keys():
                            proto_new[i, :] = global_protos[label.item()][1].data
                        i += 1
                    loss2 = loss_mse(proto_new, protos)

                loss = loss1 + loss2*args.ld # loss2不知道有没有必要
                loss.backward()
                optimizer.step()
                
                for i in range(len(labels)):
                    label = label_g[i].item()
                    if label not in agg_protos_label:
                        agg_protos_label[label] = [[], []]  # 为每个标签初始化两个空列表

                    agg_protos_label[label][1].append(protos[i,:])

                log_probs = log_probs[:, 0:args.num_classes]
                _, y_hat = log_probs.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()

                if batch_idx == len(self.trainloader) - 2:
                    print('| Global Round : {} | User: {} | Local Epoch : {} | Loss: {:.3f} | Acc: {:.3f}'.format(
                        global_round, idx, iter, 
                        loss.item(),
                        acc_val.item()))
                batch_loss['total'].append(loss.item())
                batch_loss['1'].append(loss1.item())
                batch_loss['2'].append(loss2.item())
            epoch_loss['total'].append(sum(batch_loss['total'])/len(batch_loss['total']))
            epoch_loss['1'].append(sum(batch_loss['1']) / len(batch_loss['1']))
            epoch_loss['2'].append(sum(batch_loss['2']) / len(batch_loss['2']))

        epoch_loss['total'] = sum(epoch_loss['total']) / len(epoch_loss['total'])
        epoch_loss['1'] = sum(epoch_loss['1']) / len(epoch_loss['1'])
        epoch_loss['2'] = sum(epoch_loss['2']) / len(epoch_loss['2'])
        # print(agg_protos_label)没问题
        return model.state_dict(), classifier.state_dict(), epoch_loss, acc_val.item(), agg_protos_label

    def update_weights_het_mm_3(self, args, idx, global_protos, model, classifier, global_round=round):
        # Set mode to train model
        model.train()
        classifier.train()
        # teacher_model_path = '/home/shayulong/proto/FedProto/lib/models/best_global_model.pth'
        # teacher_classifier_path = '/home/shayulong/proto/FedProto/lib/models/best_global_classifier.pth'

        # teacher_model = MyUTDModelFeature(input_size=1)
        # teacher_classifier = LinearClassifierAttn(num_classes=args.num_classes)

        # # 加载教师模型和分类器的状态
        # teacher_model_state = torch.load(teacher_model_path)
        # teacher_classifier_state = torch.load(teacher_classifier_path)

        # teacher_model.load_state_dict(teacher_model_state)
        # teacher_classifier.load_state_dict(teacher_classifier_state)    

        # teacher_model.to(args.device)
        # teacher_classifier.to(args.device)
        # teacher_model.eval()
        # teacher_classifier.eval()

        epoch_loss = {'total':[],'1':[], '2':[], '3':[]}

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            # optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
            #                             momentum=0.5)
            optimizer = torch.optim.SGD([ 
                    {'params': model.parameters(), 'lr': 1e-4},   # 0
                    {'params': classifier.parameters(), 'lr': args.lr}],
                    momentum=args.momentum,
                    weight_decay=args.weight_decay)
        elif self.args.optimizer == 'adam':
            # optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
            #                              weight_decay=1e-4)
            optimizer = torch.optim.Adam([ 
                    {'params': model.parameters(), 'lr': args.lr},   # 0
                    {'params': classifier.parameters(), 'lr': args.lr}],
                    weight_decay=1e-4)
        for iter in range(self.args.train_ep):
            batch_loss = {'total':[],'1':[], '2':[], '3':[]}
            agg_protos_label = {}
            
            for batch_idx, (m1, m2, label_g) in enumerate(self.trainloader):
                if m1.size(0) < 2:
                    continue  # 跳过只有一个样本的批次
                loss_mse = nn.MSELoss()
                if args.dataset == 'UTD':
                    m1, m2, labels = m1.to(self.device), m2.to(self.device), label_g.to(self.device)
                    
                    model.zero_grad()
                    classifier.zero_grad()
                    optimizer.zero_grad()
                    protos1, protos2 = model.encoder(m1, m2) # (bsz,特征维度)
                elif args.dataset == 'UMPC':
                    m1, input_ids, attention_mask, labels = m1.to(self.device), m2['input_ids'].to(self.device), m2['attention_mask'].to(self.device), label_g.to(self.device)
                    model.zero_grad()
                    classifier.zero_grad()
                    optimizer.zero_grad()
                    protos1, protos2 = model(m1, input_ids, attention_mask) # (bsz,特征维度)
                # protos1_global, protos2_global = teacher_model.encoder(m1, m2)
                # loss0 = 1/2*loss_mse(protos1_global, protos1)+1/2*loss_mse(protos2_global, protos2)

                log_probs, weight1, weight2 = classifier(protos1, protos2)
                loss1_ = self.criterion(log_probs, labels)
                # with torch.no_grad():
                    # # teacher_output1, teacher_output2 = teacher_model.encoder(m1, m2)  # 假设教师模型和客户端模型结构相同
                    # teacher_output, w1, w2 = teacher_classifier(protos1_global, protos2_global)

                # loss1 = distillation_loss(log_probs, labels, teacher_output)
                # # print("m1", protos.shape)[8, 16, 118, 4],[8,128]
                
                
                if len(global_protos) == 0:
                    loss2 = torch.tensor(0.0, device=self.device, requires_grad=True)
                else:
                    proto_new1 = copy.deepcopy(protos1.data)
                    proto_new2 = copy.deepcopy(protos2.data)
                    i = 0
                    for label in labels: # （bsz，）
                        if label.item() in global_protos.keys():
                            proto_new1[i, :] = global_protos[label.item()][0].data
                            proto_new2[i, :] = global_protos[label.item()][1].data
                        i += 1
                    loss2 = 1/2*loss_mse(proto_new1, protos1)+1/2*loss_mse(proto_new2, protos2)

                loss = loss1_ + loss2 # loss2不知道有没有必要
                loss.backward()
                optimizer.step()
                
                for i in range(len(labels)):
                    label = label_g[i].item()
                    if label not in agg_protos_label:
                        agg_protos_label[label] = [[], []]  # 为每个标签初始化两个空列表

                    agg_protos_label[label][0].append(protos1[i,:])
                    agg_protos_label[label][1].append(protos2[i,:])

                log_probs = log_probs[:, 0:args.num_classes]
                _, y_hat = log_probs.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()

                if batch_idx == len(self.trainloader) - 2:
                    print('| Global Round : {} | User: {} | Local Epoch : {} | Loss: {:.3f} | Acc: {:.3f}'.format(
                        global_round, idx, iter, 
                        loss.item(),
                        acc_val.item()))
                batch_loss['total'].append(loss.item())
                # batch_loss['1'].append(loss1.item())
                batch_loss['2'].append(loss2.item())
            epoch_loss['total'].append(sum(batch_loss['total'])/len(batch_loss['total']))
            # epoch_loss['1'].append(sum(batch_loss['1']) / len(batch_loss['1']))
            epoch_loss['2'].append(sum(batch_loss['2']) / len(batch_loss['2']))

        epoch_loss['total'] = sum(epoch_loss['total']) / len(epoch_loss['total'])
        # epoch_loss['1'] = sum(epoch_loss['1']) / len(epoch_loss['1'])
        epoch_loss['2'] = sum(epoch_loss['2']) / len(epoch_loss['2'])
        # print(agg_protos_label)没问题
        return model.state_dict(), classifier.state_dict(), epoch_loss, acc_val.item(), agg_protos_label
    

    def update_weights_het(self, args, idx, global_protos, model, global_round=round):
        # Set mode to train model
        model.train()
        epoch_loss = {'total':[],'1':[], '2':[], '3':[]}

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.train_ep):
            batch_loss = {'total':[],'1':[], '2':[], '3':[]}
            agg_protos_label = {}
            for batch_idx, (images, label_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), label_g.to(self.device)

                # loss1: cross-entrophy loss, loss2: proto distance loss
                model.zero_grad()
                log_probs, protos = model(images) # (bsz,特征维度)
                loss1 = self.criterion(log_probs, labels)

                loss_mse = nn.MSELoss()
                if len(global_protos) == 0:
                    loss2 = 0*loss1
                else:
                    proto_new = copy.deepcopy(protos.data)
                    i = 0
                    for label in labels: # （bsz，）
                        if label.item() in global_protos.keys():
                            proto_new[i, :] = global_protos[label.item()][0].data
                        i += 1
                    loss2 = loss_mse(proto_new, protos)

                loss = loss1 + loss2 * args.ld
                loss.backward()
                optimizer.step()

                for i in range(len(labels)):
                    if label_g[i].item() in agg_protos_label:
                        agg_protos_label[label_g[i].item()].append(protos[i,:])
                    else:
                        agg_protos_label[label_g[i].item()] = [protos[i,:]]
                # agg_protos_label会含有这一个batch里所有protos，[标签号,protos]
                log_probs = log_probs[:, 0:args.num_classes]
                _, y_hat = log_probs.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f} | Acc: {:.3f}'.format(
                        global_round, idx, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader),
                        loss.item(),
                        acc_val.item()))
                batch_loss['total'].append(loss.item())
                batch_loss['1'].append(loss1.item())
                batch_loss['2'].append(loss2.item())
            epoch_loss['total'].append(sum(batch_loss['total'])/len(batch_loss['total']))
            epoch_loss['1'].append(sum(batch_loss['1']) / len(batch_loss['1']))
            epoch_loss['2'].append(sum(batch_loss['2']) / len(batch_loss['2']))

        epoch_loss['total'] = sum(epoch_loss['total']) / len(epoch_loss['total'])
        epoch_loss['1'] = sum(epoch_loss['1']) / len(epoch_loss['1'])
        epoch_loss['2'] = sum(epoch_loss['2']) / len(epoch_loss['2'])

        return model.state_dict(), epoch_loss, acc_val.item(), agg_protos_label

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss

class LocalTest(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.testloader = self.test_split(dataset, list(idxs))
        self.device = args.device
        self.criterion = nn.NLLLoss().to(args.device)

    def test_split(self, dataset, idxs):
        idxs_test = idxs[:int(1 * len(idxs))]

        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                 batch_size=64, shuffle=False)
        return testloader

    def get_result(self, args, idx, classes_list, model):
        # Set mode to train model
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)
            model.zero_grad()
            outputs, protos = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # prediction
            outputs = outputs[: , 0 : args.num_classes]
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        acc = correct / total

        return loss, acc

    def fine_tune(self, args, dataset, idxs, model):
        trainloader = self.test_split(dataset, list(idxs))
        device = args.device
        criterion = nn.NLLLoss().to(device)
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.5)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

        model.train()
        for i in range(args.ft_round):
            for batch_idx, (images, label_g) in enumerate(trainloader):
                images, labels = images.to(device), label_g.to(device)

                # compute loss
                model.zero_grad()
                log_probs, protos = model(images)
                loss = criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

        return model.state_dict()


def test_inference(args, model, test_dataset, global_protos):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = args.device
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs, protos = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss

def test_inference_new(args, local_model_list, test_dataset, classes_list, global_protos=[]):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0

    device = args.device
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        prob_list = []
        for idx in range(args.num_users):
            images = images.to(args.device)
            model = local_model_list[idx]
            probs, protos = model(images)  # outputs 64*6
            prob_list.append(probs)

        outputs = torch.zeros(size=(images.shape[0], 10)).to(device)  # outputs 64*10
        cnt = np.zeros(10)
        for i in range(10):
            for idx in range(args.num_users):
                if i in classes_list[idx]:
                    tmp = np.where(classes_list[idx] == i)[0][0]
                    outputs[:,i] += prob_list[idx][:,tmp]
                    cnt[i]+=1
        for i in range(10):
            if cnt[i]!=0:
                outputs[:, i] = outputs[:,i]/cnt[i]

        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)


    acc = correct/total

    return loss, acc

def test_inference_new_cifar(args, local_model_list, test_dataset, classes_list, global_protos=[]):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0

    device = args.device
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        prob_list = []
        for idx in range(args.num_users):
            images = images.to(args.device)
            model = local_model_list[idx]
            probs, protos = model(images)  # outputs 64*6
            prob_list.append(probs)

        outputs = torch.zeros(size=(images.shape[0], 100)).to(device)  # outputs 64*10
        cnt = np.zeros(100)
        for i in range(100):
            for idx in range(args.num_users):
                if i in classes_list[idx]:
                    tmp = np.where(classes_list[idx] == i)[0][0]
                    outputs[:,i] += prob_list[idx][:,tmp]
                    cnt[i]+=1
        for i in range(100):
            if cnt[i]!=0:
                outputs[:, i] = outputs[:,i]/cnt[i]

        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)


    acc = correct/total

    return loss, acc


def test_inference_new_het(args, local_model_list, test_dataset, global_protos=[]):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0
    loss_mse = nn.MSELoss()

    device = args.device
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    cnt = 0
    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        prob_list = []
        protos_list = []
        for idx in range(args.num_users):
            images = images.to(args.device)
            model = local_model_list[idx]
            _, protos = model(images)
            protos_list.append(protos)

        ensem_proto = torch.zeros(size=(images.shape[0], protos.shape[1])).to(device)
        # protos ensemble
        for protos in protos_list:
            ensem_proto += protos
        ensem_proto /= len(protos_list)

        a_large_num = 100
        outputs = a_large_num * torch.ones(size=(images.shape[0], 10)).to(device)  # outputs 64*10
        for i in range(images.shape[0]):
            for j in range(10):
                if j in global_protos.keys():
                    dist = loss_mse(ensem_proto[i,:],global_protos[j][0])
                    outputs[i,j] = dist

        # Prediction
        _, pred_labels = torch.min(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    acc = correct/total

    return acc

def test_global(model, classifier, test_dataset, args):
    test_dataset = extract_data_from_dataloader(test_dataset)
    testloader = global_test(args, test_dataset)
    total, correct = 0.0, 0.0
    # test (local model)
    model.eval()
    classifier.eval()
    for batch_idx, (m1, m2, labels) in enumerate(testloader):
        if m1.size(0) < 2:
            continue  # 跳过只有一个样本的批次
        m1, m2, labels = m1.to(args.device), m2.to(args.device), labels.to(args.device)
        model.zero_grad()
        classifier.zero_grad()
        feature1, feature2 = model.encoder(m1, m2) # (bsz,特征维度)
        log_probs, weight1, weight2 = classifier(feature1, feature2)

        # prediction
        _, pred_labels = torch.max(log_probs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    acc = correct / total
    print('| Global Test Acc w/o protos: {:.3f}'.format(acc))
    return acc

def test_inference_new_het_lt(flag, args, local_model_list, local_classifier_list, test_dataset1, test_dataset2, test_dataset12, global_protos=[]):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0 ##
    loss_mse = nn.MSELoss()

    device = args.device
    criterion = nn.NLLLoss().to(device)

    acc_list_g = []
    acc_list_l = []
    loss_list = []
    model1 = agg_model_m1(local_model_list, args)
    model2 = agg_model_m2(local_model_list, args)
    model12 = agg_model_mm(local_model_list, args) #local_model_list[-1] # agg_model(local_model_list)
    classifier1, classifier2, classifier12 = aggregate_classifiers(local_classifier_list, args) # agg_linear_classifier_attn(local_classifier_list, args) # local_classifier_list[-1]
    model1.to(args.device)
    model2.to(args.device)
    model12.to(args.device)
    classifier1.to(args.device)
    classifier2.to(args.device)
    classifier12.to(args.device)
    # test_dataset = extract_data_from_dataloader(test_dataset)
    # testloader = global_test(args, test_dataset)
    # with proto
    def modality1(model, classifier, dataloader, global_protos, mdl):
        model.eval()
        correct, total, acc_list, loss_list = 0.0, 0.0, [], []
        for (m, labels) in dataloader:
            if m.size(0) < 2:
                continue
            m, labels = m.to(device), labels.to(device)
            model.zero_grad()
            feature = model.encoder(m)
            # compute the dist between protos and global_protos
            a_large_num = 100
            dist = a_large_num * torch.ones(size=(m.shape[0], args.num_classes)).to(device)  # initialize a distance matrix
            for i in range(m.shape[0]):
                for j in range(args.num_classes):
                    if j in global_protos.keys():
                        if mdl == 1:
                            d = loss_mse(feature[i, :], global_protos[j][0])
                        else:
                            d = loss_mse(feature[i, :], global_protos[j][1])
                        dist[i, j] = d
            # prediction
            _, pred_labels = torch.min(dist, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        acc = correct / total
        # if flag == 1:
            # print('| Global Test Unimodality Acc with protos: {:.5f}'.format(acc))
        acc_list_g.append(acc)

        # classifier.eval()
        # correct, total, acc_list, loss_list = 0.0, 0.0, [], []
        # for (m, labels) in dataloader:
        #     if m.size(0) < 2:
        #         continue
        #     m, labels = m.to(device), labels.to(device)
        #     model.zero_grad()
        #     feature = model.encoder(m)
        #     if mdl == 1:
        #         missing_feature = torch.zeros_like(global_protos[0][1])
        #     else:
        #         missing_feature = torch.zeros_like(global_protos[0][0])
        #     for i in range(m.shape[0]):
        #         if mdl == 1:
        #             closest_proto = min(
        #                 [global_protos[j][1] for j in global_protos.keys()],
        #                 key=lambda x: torch.norm(feature[i, :] - x, p=2)
        #             )
        #             missing_feature[i, :] = closest_proto
        #         else:
        #             closest_proto = min(
        #                 [global_protos[j][0] for j in global_protos.keys()],
        #                 key=lambda x: torch.norm(feature[i, :] - x, p=2)
        #             )
        #             missing_feature[i, :] = closest_proto
        #     if mdl == 1:      
        #         log_probs, weight1, weight2 = classifier(feature, missing_feature)
        #     else:
        #         log_probs, weight1, weight2 = classifier(missing_feature, feature)
        #     # prediction
        #     _, pred_labels = torch.max(log_probs, 1)
        #     pred_labels = pred_labels.view(-1)
        #     correct += torch.sum(torch.eq(pred_labels, labels)).item()
        #     total += len(labels)
        # acc = correct / total
        # if flag == 1:
        #     print('| Global Test Unimodality Acc w/o protos: {:.5f}'.format(acc))
       
        
        return acc_list_g
    
    def multi(model, classifier, dataloader, global_protos):
        model.eval()
        classifier.eval()
        correct, total, acc_list, loss_list = 0.0, 0.0, [], []
        for (m1, m2, labels) in dataloader:
            if m1.size(0) < 2:
                continue
            m1, m2, labels = m1.to(device), m2.to(device), labels.to(device)
            model.zero_grad()
            feature1, feature2 = model.encoder(m1, m2)
            # compute the dist between protos and global_protos
            a_large_num = 100
            dist = a_large_num * torch.ones(size=(m1.shape[0], args.num_classes)).to(device)  # initialize a distance matrix
            for i in range(m1.shape[0]):
                for j in range(args.num_classes):
                    if j in global_protos.keys():
                        d = 1/2*loss_mse(feature1[i, :], global_protos[j][0])+1/2*loss_mse(feature2[i, :], global_protos[j][1])
                        dist[i, j] = d
            # prediction
            _, pred_labels = torch.min(dist, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        acc = correct / total
        if flag == 1:
            print('| Global Test Multimodality Acc with protos: {:.5f}'.format(acc))
        acc_list_g.append(acc)


        correct, total, acc_list, loss_list = 0.0, 0.0, [], []
        for (m1, m2, labels) in dataloader:
            if m1.size(0) < 2:
                continue
            m1, m2, labels = m1.to(device), m2.to(device), labels.to(device)
            model.zero_grad()
            feature1, feature2 = model.encoder(m1, m2)
            log_probs, weight1, weight2 = classifier(feature1, feature2)

            # prediction
            _, pred_labels = torch.max(log_probs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
        acc = correct / total
        if flag == 1:
            print('| Global Test Multimodality Acc w/o protos: {:.5f}'.format(acc))
        acc_list_l.append(acc)
        return acc_list_g, acc_list_l
    acc_list_g1 = modality1(model1, classifier1, test_dataset1, global_protos, 1)
    acc_list_g2 = modality1(model2, classifier2, test_dataset2, global_protos, 2)
    acc_list_g12, acc_list_l = multi(model12, classifier12, test_dataset12, global_protos)
    return acc_list_g1, acc_list_g2, acc_list_g12, acc_list_l
    # test (local model)
    model.eval()
    classifier.eval()
    for batch_idx, (m1, m2, labels) in enumerate(testloader):
        if m1.size(0) < 2:
            continue  # 跳过只有一个样本的批次
        m1, m2, labels = m1.to(args.device), m2.to(args.device), labels.to(args.device)
        model.zero_grad()
        classifier.zero_grad()
        feature1, feature2 = model.encoder(m1, m2) # (bsz,特征维度)
        log_probs, weight1, weight2 = classifier(feature1, feature2)
        batch_loss = criterion(log_probs, labels)
        loss += batch_loss.item()

        # prediction
        _, pred_labels = torch.max(log_probs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    acc = correct / total
    if flag == 1:
        print('| Global Test Acc w/o protos: {:.3f}'.format(acc))
    acc_list_l.append(acc)

    # Before starting the test with global prototypes
    correct = 0.0
    total = 0.0

    # test (use global proto)
    if global_protos!=[]:
        for batch_idx, (m1, m2, labels) in enumerate(testloader):
            if m1.size(0) < 2:
                continue  # 跳过只有一个样本的批次
            m1, m2, labels = m1.to(args.device), m2.to(args.device), labels.to(args.device)
            model.zero_grad()
            feature1, feature2 = model.encoder(m1, m2) # (bsz,特征维度)

            # compute the dist between protos and global_protos
            a_large_num = 100
            dist = a_large_num * torch.ones(size=(m1.shape[0], args.num_classes)).to(device)  # initialize a distance matrix
            for i in range(m1.shape[0]):
                for j in range(args.num_classes):
                    if j in global_protos.keys():
                        d = 1/2*loss_mse(feature1[i, :], global_protos[j][0])+1/2*loss_mse(feature2[i, :], global_protos[j][1])
                        dist[i, j] = d

            # prediction
            _, pred_labels = torch.min(dist, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

            # compute loss
            feature_new1 = copy.deepcopy(feature1.data)
            feature_new2 = copy.deepcopy(feature2.data)
            i = 0
            for label in labels:
                if label.item() in global_protos.keys():
                    feature_new1[i, :] = global_protos[label.item()][0].data
                    feature_new2[i, :] = global_protos[label.item()][1].data
                i += 1
            loss2 = 1/2*loss_mse(feature_new1, feature1)+1/2*loss_mse(feature_new2, feature2)
            if args.device == 'cuda':
                loss2 = loss2.cpu().detach().numpy()
            else:
                loss2 = loss2.detach().numpy()

        acc = correct / total
        if flag == 1:
            print('| Global Test Acc with protos: {:.5f}'.format(acc))
        acc_list_g.append(acc)
        loss_list.append(loss2)

    return acc_list_l, acc_list_g, loss_list

def agg_model_m1(local_model_list, args):
    # 筛选出所有的 MyUTDModelFeature 实例
    my_utd_model_features = [model for model in local_model_list if isinstance(model, MyUTDModelFeature1)]
    
    # 确保至少有一个 MyUTDModelFeature 实例
    if not my_utd_model_features:
        raise ValueError("No MyUTDModelFeature1 instances found in the provided local_model_list.")
    
    # 初始化新的 MyUTDModelFeature
    if args.dataset == 'UTD':
        aggregated_model = MyUTDModelFeature1(input_size=1, p1_size=7552)
    elif args.dataset == 'MMAct':
        aggregated_model = MyUTDModelFeature1(input_size=1, p1_size=23680)
    # 聚合 head_1 的权重
    head_1_weights = [model.head_1.state_dict() for model in my_utd_model_features]
    avg_head_1_weights = average_state_dicts(head_1_weights)
    aggregated_model.head_1.load_state_dict(avg_head_1_weights)

    # 聚合 imu_cnn_layers 的权重
    imu_cnn_weights = [model.encoder.imu_cnn_layers.state_dict() for model in my_utd_model_features]
    avg_imu_cnn_weights = average_state_dicts(imu_cnn_weights)
    aggregated_model.encoder.imu_cnn_layers.load_state_dict(avg_imu_cnn_weights)

    return aggregated_model

def agg_model_m2(local_model_list, args):
    # 筛选出所有的 MyUTDModelFeature 实例
    my_utd_model_features = [model for model in local_model_list if isinstance(model, MyUTDModelFeature2)]
    
    # 确保至少有一个 MyUTDModelFeature 实例
    if not my_utd_model_features:
        raise ValueError("No MyUTDModelFeature2 instances found in the provided local_model_list.")
    
    # 初始化新的 MyUTDModelFeature
    if args.dataset == 'UTD':
        aggregated_model = MyUTDModelFeature2(1, 2688, args.dataset)
    elif args.dataset == 'MMAct':
        aggregated_model = MyUTDModelFeature2(1, 8576, args.dataset)

    # 聚合 head_2 的权重
    head_2_weights = [model.head_2.state_dict() for model in my_utd_model_features]
    avg_head_2_weights = average_state_dicts(head_2_weights)
    aggregated_model.head_2.load_state_dict(avg_head_2_weights)

    # 聚合 skeleton_cnn_layers 的权重
    skeleton_cnn_weights = [model.encoder.skeleton_cnn_layers.state_dict() for model in my_utd_model_features]
    avg_skeleton_cnn_weights = average_state_dicts(skeleton_cnn_weights)
    aggregated_model.encoder.skeleton_cnn_layers.load_state_dict(avg_skeleton_cnn_weights)

    return aggregated_model

def agg_model_mm(local_model_list, args):
    # 筛选出所有的 MyUTDModelFeature 实例
    my_utd_model_features = [model for model in local_model_list if isinstance(model, MyUTDModelFeature)]
    
    # 确保至少有一个 MyUTDModelFeature 实例
    if not my_utd_model_features:
        raise ValueError("No MyUTDModelFeature instances found in the provided local_model_list.")
    
    # 初始化新的 MyUTDModelFeature
    if args.dataset == 'UTD':
        aggregated_model = MyUTDModelFeature(1, 7552, 2688, args.dataset)
    elif args.dataset == 'MMAct':
        aggregated_model = MyUTDModelFeature(1, 23680, 8576, args.dataset)

    # 聚合 head_1 的权重
    head_1_weights = [model.head_1.state_dict() for model in my_utd_model_features]
    avg_head_1_weights = average_state_dicts(head_1_weights)
    aggregated_model.head_1.load_state_dict(avg_head_1_weights)

    # 聚合 head_2 的权重
    head_2_weights = [model.head_2.state_dict() for model in my_utd_model_features]
    avg_head_2_weights = average_state_dicts(head_2_weights)
    aggregated_model.head_2.load_state_dict(avg_head_2_weights)

    # 聚合 imu_cnn_layers 的权重
    imu_cnn_weights = [model.encoder.imu_cnn_layers.state_dict() for model in my_utd_model_features]
    avg_imu_cnn_weights = average_state_dicts(imu_cnn_weights)
    aggregated_model.encoder.imu_cnn_layers.load_state_dict(avg_imu_cnn_weights)

    # 聚合 skeleton_cnn_layers 的权重
    skeleton_cnn_weights = [model.encoder.skeleton_cnn_layers.state_dict() for model in my_utd_model_features]
    avg_skeleton_cnn_weights = average_state_dicts(skeleton_cnn_weights)
    aggregated_model.encoder.skeleton_cnn_layers.load_state_dict(avg_skeleton_cnn_weights)

    return aggregated_model


def agg_model(local_model_list):
    # 初始化新的 MyUTDModelFeature
    aggregated_model = MyUTDModelFeature(input_size=1)
   
    # # 聚合 head_1 的权重
    # head_1_weights = [model.head_1.state_dict() for model in local_model_list if hasattr(model, 'head_1')]
    # avg_head_1_weights = average_state_dicts(head_1_weights)
    # aggregated_model.head_1.load_state_dict(avg_head_1_weights)

    # # 聚合 head_2 的权重
    # head_2_weights = [model.head_2.state_dict() for model in local_model_list if hasattr(model, 'head_2')]
    # avg_head_2_weights = average_state_dicts(head_2_weights)
    # aggregated_model.head_2.load_state_dict(avg_head_2_weights)

    # 聚合 imu_cnn_layers 的权重
    imu_cnn_weights = [model.encoder.imu_cnn_layers.state_dict() for model in local_model_list if hasattr(model.encoder, 'imu_cnn_layers')]
    avg_imu_cnn_weights = average_state_dicts(imu_cnn_weights)
    aggregated_model.encoder.imu_cnn_layers.load_state_dict(avg_imu_cnn_weights)

    # 聚合 skeleton_cnn_layers 的权重
    skeleton_cnn_weights = [model.encoder.skeleton_cnn_layers.state_dict() for model in local_model_list if hasattr(model.encoder, 'skeleton_cnn_layers')]
    avg_skeleton_cnn_weights = average_state_dicts(skeleton_cnn_weights)
    aggregated_model.encoder.skeleton_cnn_layers.load_state_dict(avg_skeleton_cnn_weights)
    return aggregated_model

def agg_linear_classifier_attn(local_model_list, args):
    # 初始化新的 LinearClassifierAttn
    aggregated_model = LinearClassifierAttn(num_classes=args.num_classes, input_size1=7552, input_size2=2688)

    # 聚合 attn 层的权重
    attn_weights = [model.attn.state_dict() for model in local_model_list]
    avg_attn_weights = average_state_dicts(attn_weights)
    aggregated_model.attn.load_state_dict(avg_attn_weights)

    # 聚合 gru 层的权重
    gru_weights = [model.gru.state_dict() for model in local_model_list]
    avg_gru_weights = average_state_dicts(gru_weights)
    aggregated_model.gru.load_state_dict(avg_gru_weights)

    # 聚合 classifier 层的权重
    classifier_weights = [model.classifier.state_dict() for model in local_model_list]
    avg_classifier_weights = average_state_dicts(classifier_weights)
    aggregated_model.classifier.load_state_dict(avg_classifier_weights)

    return aggregated_model

def aggregate_classifiers(local_model_list, args):
    """ 返回三个聚合后的模型。 """
    if args.dataset == 'UTD':
        if len(local_model_list) < 5:
            raise ValueError("local_model_list should have at least 5 models.")

        model1 = agg_linear_classifier_attn(local_model_list[:2], args)
        model2 = agg_linear_classifier_attn(local_model_list[2:4], args)
        model3 = agg_linear_classifier_attn([local_model_list[4]], args)
    elif args.dataset == 'MMAct':
        if len(local_model_list) < 20:
            raise ValueError("local_model_list should have at least 5 models.")

        model1 = agg_linear_classifier_attn(local_model_list[:8], args)
        model2 = agg_linear_classifier_attn(local_model_list[8:16], args)
        model3 = agg_linear_classifier_attn([local_model_list[16:]], args)

    return model1, model2, model3

def agg_feature_classifier(local_model_list, args):
    """聚合多个 FeatureClassifier 模型的 classifier 层的权重。"""
    aggregated_model = FeatureClassifier(args)
    
    # 聚合 classifier 层的权重
    classifier_weights = [model.classifier.state_dict() for model in local_model_list]
    avg_classifier_weights = average_state_dicts(classifier_weights)
    aggregated_model.classifier.load_state_dict(avg_classifier_weights)

    return aggregated_model

def test_proto(args, local_model_list, test_dataloader1, test_dataloader2, test_dataloader12, global_protos=[]):
    """ Returns the test accuracy and loss. """
    loss_mse = nn.MSELoss()
    device = args.device

    # Initialize models
    model1 = agg_model_m1(local_model_list, args)
    model2 = agg_model_m2(local_model_list, args)
    model12 = agg_model_mm(local_model_list, args)
    model1.to(device)
    model2.to(device)
    model12.to(device)

    def evaluate_single_modality(model, dataloader, global_protos, mdl):
        model.eval()
        correct, total, acc_list, loss_list = 0.0, 0.0, [], []
        visualize_done = False  # 设置一个标志
        for (m, labels) in dataloader:
            if m.size(0) < 2:
                continue
            m, labels = m.to(device), labels.to(device)
            model.zero_grad()
            if args.dataset == 'MMAct' and mdl == 1:
                protos = model(m.unsqueeze(1))
            elif args.dataset == 'MMAct' and mdl == 2:
                protos = model(m.permute(0,3,1,2).unsqueeze(1))
            else:
                protos = model(m.unsqueeze(1))
            if not visualize_done:  # 如果函数还没有执行过
                # visualize_prototypes_with_tsne(global_protos, protos, labels, save_img=True, img_path="./img_sne/")
                visualize_done = True  # 改变标志的值
            a_large_num = 100
            dist = a_large_num * torch.ones(size=(m.shape[0], args.num_classes)).to(device)
            for i in range(m.shape[0]):
                for j in range(args.num_classes):
                    if j in global_protos.keys():
                        if mdl == 1:
                            d = loss_mse(protos[i, :], global_protos[j][0])
                        else:
                            d = loss_mse(protos[i, :], global_protos[j][1])
                        # if mdl == 1:
                        #     all_other_protos = [global_protos[k][1] for k in global_protos.keys()]
                        #     closest_proto_other = min(all_other_protos, key=lambda x: torch.norm(protos[i, :] - x, p=2))
                        #     d = 0.5 * loss_mse(protos[i, :], global_protos[j][0]) + 0.5 * loss_mse(closest_proto_other, global_protos[j][1])
                        # else:
                        #     all_other_protos = [global_protos[k][0] for k in global_protos.keys()]
                        #     closest_proto_other = min(all_other_protos, key=lambda x: torch.norm(protos[i, :] - x, p=2))
                        #     d = 0.5 * loss_mse(protos[i, :], global_protos[j][1]) + 0.5 * loss_mse(closest_proto_other, global_protos[j][0])
                        dist[i, j] = d

            _, pred_labels = torch.min(dist, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        acc = correct / total
        acc_list.append(acc)
        print(f'| Test Accuracy with single modality: {acc:.5f}')
        return acc_list

    def evaluate_multimodality(model, dataloader, global_protos):
        model.eval()
        correct, total, acc_list, loss_list = 0.0, 0.0, [], []
        visualize_done = False  # 设置一个标志
        if args.dataset == 'MMAct':
            for batch_idx, batch in enumerate(dataloader):
                m1 = batch['inertial']
                m2 = batch['skeleton']
                labels = batch['label']
                if m1.size(0) < 2:
                    continue  # 跳过只有一个样本的批次
                m1, m2, labels = m1.to(device), m2.to(device), labels.to(device)
                model.zero_grad()
                protos1, protos2 = model(m1.unsqueeze(1), m2.permute(0,3,1,2).unsqueeze(1))
                if not visualize_done:  # 如果函数还没有执行过
                    # visualize_prototypes_with_tsne(global_protos, protos1, labels, save_img=True, img_path="./img_sne/")
                    # visualize_prototypes_with_tsne(global_protos, protos2, labels, save_img=True, img_path="./img_sne/")
                    visualize_done = True  # 改变标志的值
                a_large_num = 100
                dist = a_large_num * torch.ones(size=(m1.shape[0], args.num_classes)).to(device)
                for i in range(m1.shape[0]):
                    for j in range(args.num_classes):
                        if j in global_protos.keys():
                            d = 0.5 * loss_mse(protos1[i, :], global_protos[j][0]) + 0.5 * loss_mse(protos2[i, :], global_protos[j][1])
                            dist[i, j] = d

                _, pred_labels = torch.min(dist, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels) 

                proto_new1 = copy.deepcopy(protos1.data)
                proto_new2 = copy.deepcopy(protos2.data)
                for i, label in enumerate(labels):
                    if label.item() in global_protos.keys():
                        proto_new1[i, :] = global_protos[label.item()][0].data
                        proto_new2[i, :] = global_protos[label.item()][1].data

                loss2 = 0.5 * loss_mse(proto_new1, protos1) + 0.5 * loss_mse(proto_new2, protos2)
                if args.device == 'cuda':
                    loss2 = loss2.cpu().detach().numpy()
                else:
                    loss2 = loss2.detach().numpy()
                loss_list.append(loss2)
        else:
            for (m1, m2, labels) in dataloader:
                if m1.size(0) < 2:
                    continue
                m1, m2, labels = m1.to(device), m2.to(device), labels.to(device)
                model.zero_grad()
                protos1, protos2 = model(m1, m2)
                if not visualize_done:  # 如果函数还没有执行过
                    # visualize_prototypes_with_tsne(global_protos, protos1, labels, save_img=True, img_path="./img_sne/")
                    # visualize_prototypes_with_tsne(global_protos, protos2, labels, save_img=True, img_path="./img_sne/")
                    visualize_done = True  # 改变标志的值
                a_large_num = 100
                dist = a_large_num * torch.ones(size=(m1.shape[0], args.num_classes)).to(device)
                for i in range(m1.shape[0]):
                    for j in range(args.num_classes):
                        if j in global_protos.keys():
                            d = 0.5 * loss_mse(protos1[i, :], global_protos[j][0]) + 0.5 * loss_mse(protos2[i, :], global_protos[j][1])
                            dist[i, j] = d

                _, pred_labels = torch.min(dist, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels) 

                proto_new1 = copy.deepcopy(protos1.data)
                proto_new2 = copy.deepcopy(protos2.data)
                for i, label in enumerate(labels):
                    if label.item() in global_protos.keys():
                        proto_new1[i, :] = global_protos[label.item()][0].data
                        proto_new2[i, :] = global_protos[label.item()][1].data

                loss2 = 0.5 * loss_mse(proto_new1, protos1) + 0.5 * loss_mse(proto_new2, protos2)
                if args.device == 'cuda':
                    loss2 = loss2.cpu().detach().numpy()
                else:
                    loss2 = loss2.detach().numpy()
                loss_list.append(loss2)

        acc = correct / total
        acc_list.append(acc)
        print(f'| Test Accuracy with multimodality: {acc:.5f}')
        return acc_list, loss_list
    # 不聚合模型的话，就是这样单客户端测试
    # print("client:0")
    # acc_list_g1 = evaluate_single_modality(local_model_list[0], test_dataloader1, global_protos, 1)
    # print("client:1")
    # acc_list_g1 = evaluate_single_modality(local_model_list[1], test_dataloader1, global_protos, 1)
    # print("client:2")
    # acc_list_g2 = evaluate_single_modality(local_model_list[2], test_dataloader2, global_protos, 2)
    # print("client:3")
    # acc_list_g2 = evaluate_single_modality(local_model_list[3], test_dataloader2, global_protos, 2)
    # print("client:4")
    # acc_list_g12, loss_list12 = evaluate_multimodality(local_model_list[4], test_dataloader12, global_protos)
    
    # 聚合模型后可以一类测试
    acc_list_g1 = evaluate_single_modality(model1, test_dataloader1, global_protos, 1)
    acc_list_g2 = evaluate_single_modality(model2, test_dataloader2, global_protos, 2)
    acc_list_g12, loss_list12 = evaluate_multimodality(model12, test_dataloader12, global_protos)
    return acc_list_g1, acc_list_g2, acc_list_g12, loss_list12

def test_unimodal(args, local_model_list, test_dataset1, local_classifier_list):
    loss_fn = nn.NLLLoss()
    device = args.device

    # Initialize models
    model1 = agg_model_m1(local_model_list)
    model1.to(device)
    model1.eval()
    def agg_classifier(local_model_list, args):
        # 初始化新的 LinearClassifierAttn
        aggregated_model = FeatureClassifier(args)

        # 聚合 classifier 层的权重
        classifier_weights = [model.classifier.state_dict() for model in local_model_list]
        avg_classifier_weights = average_state_dicts(classifier_weights)
        aggregated_model.classifier.load_state_dict(avg_classifier_weights)

        return aggregated_model
    classifier1 = agg_classifier(local_classifier_list[:2], args)

    classifier1.to(device)
    classifier1.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0

    # Iterate over test dataset
    for data, labels in test_dataset1:
        if data.size(0) < 2:
            continue
        data, labels = data.to(device), labels.to(device)

        # Forward pass
        with torch.no_grad():
            outputs = classifier1(model1(data))
            loss = loss_fn(outputs, labels)

        # Accumulate loss
        total_loss += loss.item()

        # Compute accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(test_dataset1)
    accuracy = correct / total

    print(f'Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}')

    return avg_loss, accuracy

def save_protos(args, local_model_list, test_dataset, user_groups_gt):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0

    device = args.device
    criterion = nn.NLLLoss().to(device)

    agg_protos_label = {}
    for idx in range(args.num_users):
        agg_protos_label[idx] = {}
        model = local_model_list[idx]
        model.to(args.device)
        testloader = DataLoader(DatasetSplit(test_dataset, user_groups_gt[idx]), batch_size=64, shuffle=True)

        model.eval()
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            model.zero_grad()
            outputs, protos = model(images)

            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

            for i in range(len(labels)):
                if labels[i].item() in agg_protos_label[idx]:
                    agg_protos_label[idx][labels[i].item()].append(protos[i, :])
                else:
                    agg_protos_label[idx][labels[i].item()] = [protos[i, :]]

    x = []
    y = []
    d = []
    for i in range(args.num_users):
        for label in agg_protos_label[i].keys():
            for proto in agg_protos_label[i][label]:
                if args.device == 'cuda':
                    tmp = proto.cpu().detach().numpy()
                else:
                    tmp = proto.detach().numpy()
                x.append(tmp)
                y.append(label)
                d.append(i)

    x = np.array(x)
    y = np.array(y)
    d = np.array(d)
    np.save('./' + args.alg + '_protos.npy', x)
    np.save('./' + args.alg + '_labels.npy', y)
    np.save('./' + args.alg + '_idx.npy', d)

    print("Save protos and labels successfully.")

def test_inference_new_het_cifar(args, local_model_list, test_dataset, global_protos=[]):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0
    loss_mse = nn.MSELoss()

    device = args.device
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    cnt = 0
    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        prob_list = []
        for idx in range(args.num_users):
            images = images.to(args.device)
            model = local_model_list[idx]
            probs, protos = model(images)  # outputs 64*6
            prob_list.append(probs)

        a_large_num = 1000
        outputs = a_large_num * torch.ones(size=(images.shape[0], 100)).to(device)  # outputs 64*10
        for i in range(images.shape[0]):
            for j in range(100):
                if j in global_protos.keys():
                    dist = loss_mse(protos[i,:],global_protos[j][0])
                    outputs[i,j] = dist

        _, pred_labels = torch.topk(outputs, 5)
        for i in range(pred_labels.shape[1]):
            correct += torch.sum(torch.eq(pred_labels[:,i], labels)).item()
        total += len(labels)

        cnt+=1
        if cnt==20:
            break

    acc = correct/total

    return acc

def aggregate_global_models(local_model_list, local_classifier_list, user_groups, args):
    # 初始化全局模型并移动到指定设备上
    imu_cnn_global = cnn_layers_1(input_size=1).to(args.device)
    ske_cnn_global = cnn_layers_2(1, args.dataset).to(args.device)
    if args.dataset == "UTD":
        head1_global = HeadModule(7552, 128).to(args.device)
        head2_global = HeadModule(2688, 128).to(args.device)
    elif args.dataset == "MMAct":
        head1_global = HeadModule(23680, 128).to(args.device)
        head2_global = HeadModule(8576, 128).to(args.device)        
    classifier_1_global = FeatureClassifier(args).to(args.device)
    classifier_2_global = FeatureClassifier(args).to(args.device)

    # 计算选的客户端的数据样本总量
    client_data_sizes = [len(user_groups[idx]) for idx in user_groups]

    # 定义加权平均函数
    def weighted_average_state_dicts(state_dicts, weights):
        """
        Compute weighted average of state dicts.
        Args:
        - state_dicts (list of OrderedDict): List of state dicts from different models.
        - weights (list of float): List of weights for each state dict.
        Returns:
        - averaged_state_dict (OrderedDict): Averaged state dict.
        """
        # Initialize an empty state dict to store the averaged weights
        averaged_state_dict = state_dicts[0].copy()
        
        for key in averaged_state_dict.keys():
            averaged_state_dict[key] = sum(state_dict[key] * weight for state_dict, weight in zip(state_dicts, weights))

        return averaged_state_dict

    # 记录哪些客户端参与了 imu_cnn_layers 的聚合，并计算相应的权重
    imu_cnn_indices = [i for i, model in enumerate(local_model_list) if hasattr(model.encoder, 'imu_cnn_layers')]
    imu_cnn_list = [model.encoder.imu_cnn_layers.state_dict() for i, model in enumerate(local_model_list) \
                    if hasattr(model.encoder, 'imu_cnn_layers')]

    imu_cnn_data_sizes = [client_data_sizes[i] for i in imu_cnn_indices]
    total_imu_cnn_data_size = sum(imu_cnn_data_sizes)
    imu_cnn_weights = [size / total_imu_cnn_data_size for size in imu_cnn_data_sizes]

    # 计算并加载加权平均的 imu_cnn_global 权重
    if imu_cnn_list:
        avg_imu_cnn_weights = weighted_average_state_dicts(imu_cnn_list, imu_cnn_weights)
        imu_cnn_global.load_state_dict(avg_imu_cnn_weights)

    # 对 ske_cnn_layers 重复相同的步骤
    ske_cnn_indices = [i for i, model in enumerate(local_model_list) if hasattr(model.encoder, 'skeleton_cnn_layers')]
    ske_cnn_list = [model.encoder.skeleton_cnn_layers.state_dict() for i, model in enumerate(local_model_list) \
                    if hasattr(model.encoder, 'skeleton_cnn_layers')]

    ske_cnn_data_sizes = [client_data_sizes[i] for i in ske_cnn_indices]
    total_ske_cnn_data_size = sum(ske_cnn_data_sizes)
    ske_cnn_weights = [size / total_ske_cnn_data_size for size in ske_cnn_data_sizes]

    if ske_cnn_list:
        avg_ske_cnn_weights = weighted_average_state_dicts(ske_cnn_list, ske_cnn_weights)
        ske_cnn_global.load_state_dict(avg_ske_cnn_weights)

    # 对 head1_global 重复相同的步骤
    head1_indices = [i for i, model in enumerate(local_model_list) if hasattr(model, 'head_1')]
    head1_list = [model.head_1.state_dict() for i, model in enumerate(local_model_list) \
                  if hasattr(model, 'head_1')]

    head1_data_sizes = [client_data_sizes[i] for i in head1_indices]
    total_head1_data_size = sum(head1_data_sizes)
    head1_weights = [size / total_head1_data_size for size in head1_data_sizes]

    if head1_list:
        avg_head1_weights = weighted_average_state_dicts(head1_list, head1_weights)
        head1_global.load_state_dict(avg_head1_weights)

    # 对 head2_global 重复相同的步骤
    head2_indices = [i for i, model in enumerate(local_model_list) if hasattr(model, 'head_2')]
    head2_list = [model.head_2.state_dict() for i, model in enumerate(local_model_list) \
                  if hasattr(model, 'head_2')]

    head2_data_sizes = [client_data_sizes[i] for i in head2_indices]
    total_head2_data_size = sum(head2_data_sizes)
    head2_weights = [size / total_head2_data_size for size in head2_data_sizes]

    if head2_list:
        avg_head2_weights = weighted_average_state_dicts(head2_list, head2_weights)
        head2_global.load_state_dict(avg_head2_weights)

    # 聚合 classifier_1_global 的权重
    if args.dataset == 'UTD':
        classifier_1_indices = [0, 1, 4]
        classifier_list_1 = [local_classifier_list[i].state_dict() if i < 2 else local_classifier_list[i].classifier_modality_1.state_dict() for i in classifier_1_indices]
    elif args.dataset == 'MMAct':
        classifier_1_indices = [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19]
        classifier_list_1 = [local_classifier_list[i].state_dict() if i < 8 else local_classifier_list[i].classifier_modality_1.state_dict() for i in classifier_1_indices]
    
    c1_data_sizes = [client_data_sizes[i] for i in classifier_1_indices]
    total_c1_data_size = sum(c1_data_sizes)
    c1_weights = [size / total_c1_data_size for size in c1_data_sizes]
    avg_classifier_1_weights = weighted_average_state_dicts(classifier_list_1, c1_weights)
    classifier_1_global.load_state_dict(avg_classifier_1_weights)

    # 聚合 classifier_2_global 的权重
    if args.dataset == 'UTD':
        classifier_2_indices = [2, 3, 4]
        classifier_list_2 = [local_classifier_list[i].state_dict() if i < 4 else local_classifier_list[i].classifier_modality_2.state_dict() for i in classifier_2_indices]
    elif args.dataset == 'MMAct':
        classifier_2_indices = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        classifier_list_2 = [local_classifier_list[i].state_dict() if i < 16 else local_classifier_list[i].classifier_modality_2.state_dict() for i in classifier_2_indices]
    
    c2_data_sizes = [client_data_sizes[i] for i in classifier_2_indices]
    total_c2_data_size = sum(c2_data_sizes)
    c2_weights = [size / total_c2_data_size for size in c2_data_sizes]
    avg_classifier_2_weights = weighted_average_state_dicts(classifier_list_2, c2_weights)
    classifier_2_global.load_state_dict(avg_classifier_2_weights)

    # 将聚合后的权重重新分配给本地模型
    for i, local_model in enumerate(local_model_list):
        if hasattr(local_model.encoder, 'imu_cnn_layers'):
            local_model.encoder.imu_cnn_layers.load_state_dict(imu_cnn_global.state_dict())
        if hasattr(local_model.encoder, 'skeleton_cnn_layers'):
            local_model.encoder.skeleton_cnn_layers.load_state_dict(ske_cnn_global.state_dict())
        if hasattr(local_model, 'head_1'):
            local_model.head_1.load_state_dict(head1_global.state_dict())
        if hasattr(local_model, 'head_2'):
            local_model.head_2.load_state_dict(head2_global.state_dict())

    # 将聚合后的分类器权重重新分配给本地分类器
    for i, local_classifier in enumerate(local_classifier_list):
        if i in classifier_1_indices:
            if (args.dataset == 'UTD' and i == 4) or (args.dataset == 'MMAct' and i in [16, 17, 18, 19]):
                local_classifier.classifier_modality_1.load_state_dict(classifier_1_global.state_dict())
            else:
                local_classifier.load_state_dict(classifier_1_global.state_dict())
        if i in classifier_2_indices:
            if (args.dataset == 'UTD' and i == 4) or (args.dataset == 'MMAct' and i in [16, 17, 18, 19]):
                local_classifier.classifier_modality_2.load_state_dict(classifier_2_global.state_dict())
            else:
                local_classifier.load_state_dict(classifier_2_global.state_dict())

    return local_model_list, local_classifier_list