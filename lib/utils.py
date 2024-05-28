#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torch.utils.data.dataset import random_split
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal, mnist_noniid_lt
from sampling import femnist_iid, femnist_noniid, femnist_noniid_unequal, femnist_noniid_lt
from sampling import cifar_iid, cifar100_noniid, cifar10_noniid, cifar100_noniid_lt, cifar10_noniid_lt
from cosmo.data_pre import load_data, Multimodal_dataset, SingleModalityDataset, uni_CustomDataset # CustomDataset
import femnist
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from UMPC_Dataset import UMPC_FoodDataset, CustomDataset
import os
from datetime import datetime

trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
trans_cifar100_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                               std=[0.267, 0.256, 0.276])])
trans_cifar100_val = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                              std=[0.267, 0.256, 0.276])])



def visualize_prototypes_with_tsne(global_protos, encoded_protos=None, labels=None, save_img=False, img_path='/mnt/data/tsne_visualization.png'):
    """
    使用t-SNE对全局原型和可选的编码后原型进行降维并可视化，支持将图像保存到文件。
    
    参数:
    - global_protos: 一个字典，包含每个标签下每种模态的原型向量。
    - encoded_protos: 一个列表，包含经过编码后的原型向量，可选。
    - labels: 一个列表，包含每个原型对应的标签，仅当encoded_protos不为空时需要。
    - save_img: 是否保存图像，默认为False。
    - img_path: 保存图像的路径，默认为'/mnt/data/tsne_visualization.png'。
    
    返回:
    - 无，直接在函数内部进行绘图，可选择保存图像。
    """
    # 获取标签数量
    num_labels = len(global_protos)
    
    # 将全局原型数据转换为适用于t-SNE的格式
    global_data = np.array([proto.cpu().numpy() for label, protos in global_protos.items() for proto in protos])
    global_labels = [label for label, protos in global_protos.items() for _ in protos]
    
    # 使用t-SNE进行降维
    tsne = TSNE(n_components=2, perplexity=min(30, num_labels - 1), random_state=42)
    reduced_global_data = tsne.fit_transform(global_data)
    
    # 准备颜色
    base_colors = plt.cm.tab20.colors  # 从预定义的颜色映射中获取颜色
    colors = [base_colors[i % len(base_colors)] for i in range(num_labels)]  # 分配颜色
    
    # 绘制降维后的全局原型数据
    plt.figure(figsize=(12, 10))
    for i, label in enumerate(sorted(set(global_labels))):
        idxs = [j for j, x in enumerate(global_labels) if x == label]
        modality_colors = [colors[i], mcolors.to_rgba(colors[i], alpha=0.5)]  # 为模态创建颜色和颜色变种
        for idx in idxs:
            modality = idx % 2
            plt.scatter(reduced_global_data[idx, 0], reduced_global_data[idx, 1], label=f'Label {label} Modality {modality + 1}',
                        alpha=0.8, color=modality_colors[modality], edgecolors='black', linewidths=0.5)
    
    # 如果提供了编码后原型数据，则进行处理
    if encoded_protos is not None and labels is not None:
        encoded_data = np.array([proto.cpu().detach().numpy() for proto in encoded_protos])
        # 在原型数量较少的情况下设置合适的perplexity值
        perplexity = min(30, len(encoded_data) - 1)
        tsne_encoded = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        reduced_encoded_data = tsne_encoded.fit_transform(encoded_data)   # 再次运行t-SNE以包括编码后原型
        
        for i, proto in enumerate(reduced_encoded_data):
            plt.scatter(proto[0], proto[1], label=f'Encoded Proto {labels[i]}', alpha=0.8, color='red', edgecolors='black', linewidths=0.5)
    
    plt.title('Global Prototypes and Encoded Prototypes in t-SNE Reduced Feature Space')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='x-small')
    
    # 根据参数决定是否保存图像
    if save_img:
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        img_path = os.path.join(img_path, f'tsne_visualization_{timestamp}.png')
        plt.savefig(img_path, bbox_inches='tight')
        print(f"Image saved to {img_path}")
    plt.show()
    plt.close()


def utd_iid(args, dataset, num_users, first_modality_users, second_modality_users):
    """
    Sample IID user data from UTD dataset, allowing for single modality for some users.
    :param dataset: Complete dataset with two modalities.
    :param num_users: Total number of users.
    :param single_modality_users: List of users that should receive only one modality.
    :return: Dictionary of user data indices.
    """
    num_items = int(len(dataset)*0.8) # int(len(dataset) / num_users)
    user_groups = {}
    user_dataloaders = {}
    for i in range(num_users):
        indices = np.random.choice(range(len(dataset)), num_items, replace=False)
        if i in first_modality_users:
            # Select only one modality for these users
            # Assuming dataset[idx] returns (modality1, modality2, label)
            user_groups[i] = [(dataset[idx][0], None, dataset[idx][2]) for idx in indices]
        elif i in second_modality_users:
            user_groups[i] = [(None, dataset[idx][1], dataset[idx][2]) for idx in indices]
        else:
            user_groups[i] = [dataset[idx] for idx in indices]
    
    for user_idx, user_data in user_groups.items():
        user_dataset = CustomDataset(user_data)
        user_dataloader = torch.utils.data.DataLoader(
            user_dataset, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=True, shuffle=True)
        user_dataloaders[user_idx] = user_dataloader
    return user_dataloaders


def global_test(args, dataset):
    test_dataset = CustomDataset(dataset)
    test_dataloader = torch.utils.data.DataLoader(        
        test_dataset, batch_size=args.batch_size,        
        num_workers=args.num_workers, pin_memory=True, shuffle=True)     
    return test_dataloader

def extract_data_from_dataloader(dataloader):
    data_list = []
    for m1, m2, label in dataloader:
        for i in range(len(label)):
            data_list.append((m1[i], m2[i], label[i]))
    return data_list

# 辅助函数：计算平均权重
# def average_state_dicts(state_dicts):
#     """计算状态字典的平均值。"""
#     avg_state_dict = {}
#     for key in state_dicts[0].keys():
#         avg_state_dict[key] = sum(d[key] for d in state_dicts) / len(state_dicts)
#     return avg_state_dict

def average_state_dicts(state_dicts):
    """计算状态字典的平均值。"""
    avg_state_dict = {}
    for key in state_dicts[0].keys():
        # 检查张量数据类型
        if state_dicts[0][key].dtype in [torch.float16, torch.float32, torch.float64]:
            # 使用torch.stack来堆叠所有状态字典中相同键的张量
            stacked_tensors = torch.stack([d[key] for d in state_dicts])
            # 计算这些张量的平均值
            avg_state_dict[key] = torch.mean(stacked_tensors, dim=0)
        else:
            # 对于非浮点型张量，我们可以选择直接取第一个值或者其他逻辑
            avg_state_dict[key] = state_dicts[0][key]
    return avg_state_dict

def save_model_parameters_to_log(model, model_name, log_file):
    with open(log_file, 'a') as f:
        f.write(f"Parameters of {model_name}:\n")
        for name, param in model.named_parameters():
            f.write(f"{name}: {param.size()} - mean: {param.data.mean()}, std: {param.data.std()}\n")
        f.write("\n")

        
def get_dataset(args, n_list, k_list):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    data_dir = args.data_dir + args.dataset
    if args.dataset == 'mnist':
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(args, train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups, classes_list = mnist_noniid(args, train_dataset, args.num_users, n_list, k_list)
                user_groups_lt = mnist_noniid_lt(args, test_dataset, args.num_users, n_list, k_list, classes_list)
                classes_list_gt = classes_list

    elif args.dataset == 'femnist':
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = femnist.FEMNIST(args, data_dir, train=True, download=True,
                                        transform=apply_transform)
        test_dataset = femnist.FEMNIST(args, data_dir, train=False, download=True,
                                       transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = femnist_iid(train_dataset, args.num_users)
            # print("TBD")
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                # user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
                user_groups = femnist_noniid_unequal(args, train_dataset, args.num_users)
                # print("TBD")
            else:
                # Chose euqal splits for every user
                user_groups, classes_list, classes_list_gt = femnist_noniid(args, args.num_users, n_list, k_list)
                user_groups_lt = femnist_noniid_lt(args, args.num_users, classes_list)

    elif args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=trans_cifar10_train)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=trans_cifar10_val)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups, classes_list, classes_list_gt = cifar10_noniid(args, train_dataset, args.num_users, n_list, k_list)
                user_groups_lt = cifar10_noniid_lt(args, test_dataset, args.num_users, n_list, k_list, classes_list)

    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True, transform=trans_cifar100_train)
        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=trans_cifar100_val)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups, classes_list = cifar100_noniid(args, train_dataset, args.num_users, n_list, k_list)
                user_groups_lt = cifar100_noniid_lt(test_dataset, args.num_users, classes_list)
    
    # elif args.dataset == 'flickr30k':
        

    elif args.dataset == 'UTD':
        # load data (already normalized)
        num_of_train = (1 * (args.num_train_basic) * np.ones(args.num_classes)).astype(int)
        num_of_test = (1*(args.num_test_basic) * np.ones(args.num_classes)).astype(int)
        # num_of_global = (1 * (16) * np.ones(args.num_classes)).astype(int)
        #load labeled train and test data
        print("train labeled data:")
        x_train_label_1, x_train_label_2, y_train = load_data(args.num_classes, num_of_train, 3, args.label_rate, args)
        print("test data:")
        x_test_1, x_test_2, y_test = load_data(args.num_classes, num_of_test, 2, args.label_rate, args)
        # print("global data:")
        # x_global_1, x_global_2, y_global = load_data(args.num_classes, num_of_global, 4, args.label_rate)

        train_dataset = Multimodal_dataset(x_train_label_1, x_train_label_2, y_train)
        test_dataset = Multimodal_dataset(x_test_1, x_test_2, y_test)
        # global_dataset = Multimodal_dataset(x_global_1, x_global_2, y_global)
        train_dataset_single_modality_1 = SingleModalityDataset(x_train_label_1, y_train)
        train_dataloader_single_modality_1 = torch.utils.data.DataLoader(
            train_dataset_single_modality_1, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=True, shuffle=True)
        
        # Split test dataset into single-modality and multi-modality datasets
        num_test_samples = len(y_test)
        indices = np.arange(num_test_samples)
        np.random.shuffle(indices)
        split_point = num_test_samples // 2

        single_modality_indices = indices[:split_point]
        multi_modality_indices = indices[split_point:]

        x_test_1_single = x_test_1[single_modality_indices]
        x_test_2_single = x_test_2[single_modality_indices]
        y_test_single = y_test[single_modality_indices]

        x_test_1_multi = x_test_1[multi_modality_indices]
        x_test_2_multi = x_test_2[multi_modality_indices]
        y_test_multi = y_test[multi_modality_indices]

        # Create single-modality datasets for modality 1 and modality 2
        test_dataset_single_modality_1 = SingleModalityDataset(x_test_1_single, y_test_single)
        test_dataset_single_modality_2 = SingleModalityDataset(x_test_2_single, y_test_single)
        test_dataset_multi_modality = Multimodal_dataset(x_test_1_multi, x_test_2_multi, y_test_multi)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=True, shuffle=True)
        test_dataloader_single_modality_1 = torch.utils.data.DataLoader(
            test_dataset_single_modality_1, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=True, shuffle=True)
        test_dataloader_single_modality_2 = torch.utils.data.DataLoader(
            test_dataset_single_modality_2, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=True, shuffle=True)
        test_dataloader_multi_modality = torch.utils.data.DataLoader(
            test_dataset_multi_modality, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=True, shuffle=True)
        
        # 添加噪声函数
        def add_noise(data, noise_level):
            data_std = np.std(data)
            noise = np.random.normal(0, noise_level * data_std, data.shape)
            noisy_data = data + noise
            return noisy_data


        # 创建带噪声数据集
        noise_level = 0.5  # 可调整的噪声水平
        x_test_1_noisy = add_noise(x_test_1, noise_level)
        x_test_2_noisy = add_noise(x_test_2, noise_level)

        test_dataset_noisy_single_modality_1 = SingleModalityDataset(x_test_1_noisy[single_modality_indices], y_test_single)
        test_dataset_noisy_multi_modality = Multimodal_dataset(x_test_1_noisy[multi_modality_indices], x_test_2_noisy[multi_modality_indices], y_test_multi)

        test_dataloader_noisy_single_modality_1 = DataLoader(test_dataset_noisy_single_modality_1, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True)
        test_dataloader_noisy_multi_modality = DataLoader(test_dataset_noisy_multi_modality, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True)


        # global_dataloader = torch.utils.data.DataLoader(
        #     global_dataset, batch_size=args.batch_size,
        #     num_workers=args.num_workers, pin_memory=True, shuffle=True)
        class EmptyDataset(Dataset):
            def __len__(self):
                return 0

            def __getitem__(self, index):
                raise IndexError

        empty_dataset = EmptyDataset()
        global_dataloader = DataLoader(empty_dataset, batch_size=1)    


        # Define which users should get only one modality
        first_modality_users = list(range(2))  
        second_modality_users = list(range(2, 4))

        # Sample training data amongst users
        if args.iid:
            extracted_train_data = extract_data_from_dataloader(train_dataloader)
            user_dataloaders = utd_iid(args, extracted_train_data, args.num_users, first_modality_users, second_modality_users)
        else:
            # class CusDataset(Dataset):
            #     def __init__(self, data):
            #         self.data = data

            #     def __getitem__(self, index):
            #         return self.data[index]

            #     def __len__(self):
            #         return len(self.data)
            class CusDataset(Dataset):
                def __init__(self, data, modality):
                    self.data = data
                    self.modality = modality

                def __len__(self):
                    return len(self.data)

                def __getitem__(self, index):
                    # 检查当前数据集是针对哪种模态
                    if self.modality == 1:  # 模态一
                        modality1, label = self.data[index]
                        modality2 = None  # 模态二的数据不存在
                    elif self.modality == 2:  # 模态二
                        modality2, label = self.data[index]
                        modality1 = None  # 模态一的数据不存在
                    else:  # 如果是多模态客户端
                        modality1, modality2, label = self.data[index]
        
                    # 如果 modality1 或 modality2 是 None，创建一个占位符张量
                    if modality1 is None:
                        modality1 = torch.zeros(1, 120, 6)  # 占位符的形状应该与模态一的数据形状相匹配
                    if modality2 is None:
                        modality2 = torch.zeros(1, 40, 20, 3)  # 占位符的形状应该与模态二的数据形状相匹配
                    
                    return modality1, modality2, label
            def dirichlet_distribute_modal_data(dataset, num_users, alpha, modality):
                # 获取数据集的大小
                dataset_size = len(dataset)
                # 使用狄利克雷分布生成数据索引的概率
                idx_prob = np.random.dirichlet(alpha=np.ones(num_users) * alpha, size=1).reshape(-1)
                # 计算每个用户应得的数据量
                data_per_user = (idx_prob * dataset_size).astype(int)
                # 确保分配的数据总量不超过实际数据量
                data_per_user[-1] = dataset_size - np.sum(data_per_user[:-1])
                
                # 随机分配数据集索引给每个用户
                indices = torch.randperm(dataset_size).tolist()
                user_data_indices = [indices[sum(data_per_user[:i]):sum(data_per_user[:i+1])] for i in range(num_users)]
                
                modality_data_indices = []
                for user_indices in user_data_indices:
                    if modality == 1:
                        # 模态一
                        modality_data_indices.append([(dataset[i][0], dataset[i][2]) for i in user_indices])
                    elif modality == 2:
                        # 模态二
                        modality_data_indices.append([(dataset[i][1], dataset[i][2]) for i in user_indices])
                        
                return modality_data_indices

            def create_non_iid_modal_dataloaders(dataset, num_users_group1, num_users_group2, alpha):
                user_dataloaders = {}
                
                # 客户端1和2，模态一
                user_data_indices_group1 = dirichlet_distribute_modal_data(dataset, num_users_group1, alpha, modality=1)
                # 客户端3和4，模态二
                user_data_indices_group2 = dirichlet_distribute_modal_data(dataset, num_users_group2, alpha, modality=2)
                
                # 为每组客户端创建DataLoader
                for user_idx, data_indices in enumerate(user_data_indices_group1):
                    user_dataset = CusDataset(data_indices, 1)
                    user_dataloader = DataLoader(user_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
                    user_dataloaders[user_idx] = user_dataloader
                for user_idx, data_indices in enumerate(user_data_indices_group2):
                    user_dataset = CusDataset(data_indices, 2)
                    user_dataloader = DataLoader(user_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
                    user_dataloaders[user_idx+2] = user_dataloader
                return user_dataloaders

            def allocate_data_for_multimodal_client(dataset, multimodal_client_share=0.2):
                dataset_size = len(dataset)
                multimodal_data_size = int(dataset_size * multimodal_client_share)
                
                indices = torch.randperm(dataset_size).tolist()
                multimodal_data_indices = indices[:multimodal_data_size]
                remaining_indices = indices[multimodal_data_size:]
                
                multimodal_dataset = [dataset[i] for i in multimodal_data_indices]
                remaining_dataset = [dataset[i] for i in remaining_indices]

                return multimodal_dataset, remaining_dataset

            def print_client_dataset_classes(user_dataloaders):
                for user_idx, dataloader in user_dataloaders.items():
                    all_labels = []
                    for _, _, labels in dataloader:  # 假设标签在第三个位置
                        all_labels.extend(labels.numpy())  # 假设标签已经是Tensor类型，转换为numpy数组进行处理
                    unique_labels = set(all_labels)
                    print(f"客户端 {user_idx+1} 的数据集包含的类别: {unique_labels}")

            multimodal_dataset, remaining_dataset = allocate_data_for_multimodal_client(train_dataset)
            user_dataloaders = {}
            multimodal_dataloader = DataLoader(CusDataset(multimodal_dataset, 3), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
            
            user_dataloaders = create_non_iid_modal_dataloaders(remaining_dataset, 2, 2, 2)
            user_dataloaders[4] = multimodal_dataloader

            # 调用函数打印每个客户端的数据集类别
            print_client_dataset_classes(user_dataloaders)
    
    elif args.dataset == 'UMPC':
        # 加载IID分布的客户端数据集
        client_datasets_iid = torch.load('../client_datasets_iid.pt')
        # 加载Non-IID分布的客户端数据集
        client_datasets_noniid = torch.load('../client_datasets_noniid_0.5.pt')
        # 打印数据集中的一些信息以确认加载正确
        print("Number of client datasets (IID):", len(client_datasets_iid))
        print("Number of samples in first client dataset (IID):", len(client_datasets_iid[0]))

        print("Number of client datasets (Non-IID):", len(client_datasets_noniid))
        print("Number of samples in first client dataset (Non-IID):", len(client_datasets_noniid[0]))
        if args.iid:
            # 为IID数据集创建DataLoader
            user_dataloaders = [DataLoader(dataset, batch_size=32, num_workers=16, shuffle=True) for dataset in client_datasets_iid]
        else:
            # 为Non-IID数据集创建DataLoader
            user_dataloaders = [DataLoader(dataset, batch_size=32, num_workers=16, shuffle=True) for dataset in client_datasets_noniid]
        tf = transforms.Compose([transforms.Resize((224,224)),
                                # transforms.CenterCrop(resize),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                    [0.229, 0.224, 0.225])])
        Train_set=UMPC_FoodDataset(targ_dir="../data/food" ,phase="train", mode="all",transform=tf)
        Test_set=UMPC_FoodDataset(targ_dir="../data/food" ,phase="test", mode="all",transform=tf)
        train_dataloader = DataLoader(Train_set, batch_size=32, shuffle=False,num_workers=16)
        test_dataloader = DataLoader(Test_set, batch_size=32, shuffle=False,num_workers=16)  
        # 创建一个空的 TensorDataset
        empty_dataset = TensorDataset(torch.Tensor([]), torch.Tensor([]))

        # 使用空的 TensorDataset 创建 DataLoader
        global_dataloader = DataLoader(empty_dataset, batch_size=1)
        pass
        # tf = transforms.Compose([transforms.Resize((224,224)),
        #                         # transforms.CenterCrop(resize),
        #                         transforms.ToTensor(),
        #                         transforms.Normalize([0.485, 0.456, 0.406],
        #                                             [0.229, 0.224, 0.225])])
        # Train_set=UMPC_FoodDataset(targ_dir="./data/food" ,phase="train", mode="all",transform=tf)
        # Test_set=UMPC_FoodDataset(targ_dir="./data/food" ,phase="test", mode="all",transform=tf)
        # # 假设 CustomDataset 已经根据之前的定义实现
        # class CustomDataset(Dataset):
        #     def __init__(self, data):
        #         self.data = data

        #     def __len__(self):
        #         return len(self.data)

        #     def __getitem__(self, index):
        #         modality1, modality2, label = self.data[index]
        #         if modality1 is None:
        #             modality1 = torch.zeros(3, 224, 224)
        #         if modality2 is None:
        #             modality2 = {
        #                 'input_ids': torch.zeros((40,), dtype=torch.long),
        #                 'token_type_ids': torch.zeros((40,), dtype=torch.long),
        #                 'attention_mask': torch.zeros((40,), dtype=torch.long)
        #             }
        #         return modality1, modality2, label

        # # 定义函数用于创建IID或Non-IID数据分布
        # def federated_split(dataset, num_clients=20, alpha=None):
        #     total_len = len(dataset)
        #     indices = list(range(total_len))
        #     np.random.shuffle(indices)

        #     if alpha:  # 使用Dirichlet分布进行非IID分布
        #         labels = [label for _, _, _, label in dataset]
        #         n_classes = len(set(labels))
        #         label_to_indices = {label: np.where(np.array(labels) == label)[0] for label in range(n_classes)}
                
        #         per_client_probs = np.random.dirichlet([alpha]*n_classes, num_clients)
        #         client_indices = [[] for _ in range(num_clients)]
        #         for c, probs in enumerate(per_client_probs):
        #             for label, prob in enumerate(probs):
        #                 indices = label_to_indices[label]
        #                 client_label_indices = np.random.choice(indices, int(prob * len(indices)), replace=False)
        #                 client_indices[c].extend(client_label_indices)
        #                 label_to_indices[label] = np.setdiff1d(label_to_indices[label], client_label_indices)
        #     else:  # IID分布
        #         per_client = total_len // num_clients
        #         client_indices = [indices[i * per_client:(i + 1) * per_client] for i in range(num_clients)]

        #     client_datasets = []
        #     for i, indices in enumerate(client_indices):
        #         subset = Subset(dataset, indices)
        #         client_data = [(x[0], x[1], x[3]) for x in subset]
        #         if i < 8:  # 只有图片
        #             client_data = [(x[0], None, x[2]) for x in client_data]
        #         elif 8 <= i < 16:  # 只有文本
        #             client_data = [(None, x[1], x[2]) for x in client_data]
        #         # 客户端17-20保留两种模态
        #         client_datasets.append(CustomDataset(client_data))
        #     return client_datasets

        # # 使用这些函数
        # # 假设 Train_set 是原始训练数据集
        # if args.iid:
        #     client_datasets_iid = federated_split(Train_set, num_clients=20, alpha=None)
        #     user_dataloaders = [DataLoader(dataset, batch_size=100, num_workers=16, shuffle=True) for dataset in client_datasets_iid]
        # else:
        #     client_datasets_noniid = federated_split(Train_set, num_clients=20, alpha=0.5)
        #     user_dataloaders = [DataLoader(dataset, batch_size=100, num_workers=16, shuffle=True) for dataset in client_datasets_noniid]

    return train_dataloader_single_modality_1, train_dataloader, test_dataloader_single_modality_1, test_dataloader_single_modality_2, test_dataloader_multi_modality, \
        test_dataloader_noisy_single_modality_1, test_dataloader_noisy_multi_modality, global_dataloader, user_dataloaders

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w)
    for key in w[0].keys():
        if key[0:4] != '....':
            for i in range(1, len(w)):
                w_avg[0][key] += w[i][key]
            # w_avg[0][key] = torch.true_divide(w_avg[0][key], len(w))
            w_avg[0][key] = torch.div(w_avg[0][key], len(w))
            for i in range(1, len(w)):
                w_avg[i][key] = w_avg[0][key]
    return w_avg

def average_weights_sem(w, n_list):
    """
    Returns the average of the weights.
    """
    k = 2
    model_dict = {}
    for i in range(k):
        model_dict[i] = []

    idx = 0
    for i in n_list:
        if i< np.mean(n_list):
            model_dict[0].append(idx)
        else:
            model_dict[1].append(idx)
        idx += 1

    ww = copy.deepcopy(w)
    for cluster_id in model_dict.keys():
        model_id_list = model_dict[cluster_id]
        w_avg = copy.deepcopy(w[model_id_list[0]])
        for key in w_avg.keys():
            for j in range(1, len(model_id_list)):
                w_avg[key] += w[model_id_list[j]][key]
            w_avg[key] = torch.true_divide(w_avg[key], len(model_id_list))
            # w_avg[key] = torch.div(w_avg[key], len(model_id_list))
        for model_id in model_id_list:
            for key in ww[model_id].keys():
                ww[model_id][key] = w_avg[key]

    return ww

def average_weights_per(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w)
    for key in w[0].keys():
        if key[0:2] != 'fc':
            for i in range(1, len(w)):
                w_avg[0][key] += w[i][key]
            w_avg[0][key] = torch.true_divide(w_avg[0][key], len(w))
            # w_avg[0][key] = torch.div(w_avg[0][key], len(w))
            for i in range(1, len(w)):
                w_avg[i][key] = w_avg[0][key]
    return w_avg

def average_weights_het(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w)
    for key in w[0].keys():
        if key[0:4] != 'fc2.':
            for i in range(1, len(w)):
                w_avg[0][key] += w[i][key]
            # w_avg[0][key] = torch.true_divide(w_avg[0][key], len(w))
            w_avg[0][key] = torch.div(w_avg[0][key], len(w))
            for i in range(1, len(w)):
                w_avg[i][key] = w_avg[0][key]
    return w_avg

def distillation_loss(y_pred_log_softmax, y_true, teacher_pred_log_softmax, T=2.0, alpha=0.5):
    """
    修改后的知识蒸馏损失函数，适用于输入已是对数概率形式的情况。
    
    y_pred_log_softmax: 学生模型的输出，已通过F.log_softmax处理
    y_true: 真实标签
    teacher_pred_log_softmax: 教师模型的输出，已通过F.log_softmax处理
    T: 温度参数
    alpha: 软目标损失和硬目标损失之间的权重平衡参数
    """
    # 软目标损失。注意：教师模型的输出需要先经过exp()回到概率形式，再调整温度
    soft_loss = nn.KLDivLoss(reduction='batchmean')(y_pred_log_softmax / T,
                                                    torch.exp(teacher_pred_log_softmax / T)) * (T * T * alpha)
    # 硬目标损失。由于y_pred_log_softmax已经是对数形式，直接使用NLLLoss
    hard_loss = F.nll_loss(y_pred_log_softmax, y_true) * (1. - alpha)
    return soft_loss + hard_loss


def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for label, proto_lists in protos.items():
        proto1_list, proto2_list = proto_lists[0], proto_lists[1]

        if proto1_list:  # 检查 proto1_list 是否为空
            if len(proto1_list) > 1:
                proto1 = 0 * proto1_list[0].data
                for i in proto1_list:
                    # print(i.shape)
                    proto1 += i.data
                protos[label][0] = proto1 / len(proto1_list)
            else:
                protos[label][0] = proto1_list[0]

        if proto2_list:  # 检查 proto2_list 是否为空
            if len(proto2_list) > 1:
                proto2 = 0 * proto2_list[0].data
                for i in proto2_list:
                    proto2 += i.data
                protos[label][1] = proto2 / len(proto2_list)
            else:
                protos[label][1] = proto2_list[0]
    
    return protos

# def proto_aggregation(local_protos_list):
#     agg_protos_label = dict()


#     local_protos = local_protos_list[4]
#     for label, protos in local_protos.items():
#         if label not in agg_protos_label:
#             agg_protos_label[label] = [[], []]

#         # 检查 protos[0] 是单个张量还是张量列表
#         if isinstance(protos[0], list):
#             agg_protos_label[label][0].extend(protos[0])
#         else:
#             agg_protos_label[label][0].append(protos[0])

#         # 检查 protos[1] 是单个张量还是张量列表
#         if isinstance(protos[1], list):
#             agg_protos_label[label][1].extend(protos[1])
#         else:
#             agg_protos_label[label][1].append(protos[1])

#     for label, proto_lists in agg_protos_label.items():
#         proto1_list, proto2_list = proto_lists[0], proto_lists[1]

#         if proto1_list:
#             if len(proto1_list) > 1:
#                 proto1 = 0 * proto1_list[0].data
#                 for i in proto1_list:
#                     proto1 += i.data
#                 agg_protos_label[label][0] = proto1 / len(proto1_list)
#             else:
#                 agg_protos_label[label][0] = proto1_list[0]

#         if proto2_list:
#             if len(proto2_list) > 1:
#                 proto2 = 0 * proto2_list[0].data
#                 for i in proto2_list:
#                     proto2 += i.data
#                 agg_protos_label[label][1] = proto2 / len(proto2_list)
#             else:
#                 agg_protos_label[label][1] = proto2_list[0]

#     return agg_protos_label



def proto_aggregation(local_protos_list):
    agg_protos_label = dict()
    
    for idx in local_protos_list:
        local_protos = local_protos_list[idx]
        for label, protos in local_protos.items():
            
            if label not in agg_protos_label:
                agg_protos_label[label] = [[], []]

            # 检查 protos[0] 是单个张量还是张量列表
            if isinstance(protos[0], list):
                agg_protos_label[label][0].extend(protos[0])
            else:
                agg_protos_label[label][0].append(protos[0])

            # 检查 protos[1] 是单个张量还是张量列表
            if isinstance(protos[1], list):
                agg_protos_label[label][1].extend(protos[1])
            else:
                agg_protos_label[label][1].append(protos[1])


    for label, proto_lists in agg_protos_label.items():
        
        proto1_list, proto2_list = proto_lists[0], proto_lists[1]
        # 检查所有张量的形状是否一致
        # for tensor in proto1_list:
        #     if tensor.shape != proto1_list[0].shape:
        #         print(f"形状不匹配: {tensor.shape} vs {proto1_list[0].shape}")

        if proto1_list:
            if len(proto1_list) > 1:
                proto1 = 0 * proto1_list[0].data
                for i in proto1_list:
                    # print(i.shape)
                    proto1 += i.data
                agg_protos_label[label][0] = proto1 / len(proto1_list)
            else:
                agg_protos_label[label][0] = proto1_list[0]
        if proto2_list:
            if len(proto2_list) > 1:
                proto2 = 0 * proto2_list[0].data
                for i in proto2_list:
                    proto2 += i.data
                agg_protos_label[label][1] = proto2 / len(proto2_list)
            else:
                agg_protos_label[label][1] = proto2_list[0]

    return agg_protos_label

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.rounds3}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.train_ep}\n')
    return