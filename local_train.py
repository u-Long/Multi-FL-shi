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
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from lib.cosmo.data_pre import CustomDataset

args = args_parser()
exp_details(args)

args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.device == 'cuda':
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
else:
    torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

train_dataloader_single_modality_1, train_dataset, test_dataset1, test_dataset2, test_dataset12, \
    test_noisy_1, test_noisy_12, global_dataset, user_groups, user_unlabels = get_dataset(args, 1, 1)

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
    if args.dataset == 'UTD':
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

# 只有第一阶段训练
idxs_users = np.arange(args.num_users)
for round in tqdm(range(args.rounds1), disable=True):
    round_loss = []
    local_weights, local_w1, local_losses, local_protos = [], [], [], {}
    print(f'\n | Global Training Round : {round + 1} |\n')
    for idx in idxs_users:
        model=copy.deepcopy(local_model_list[idx])
        classifier=copy.deepcopy(local_classifier_list[idx])
        trainloader = user_groups[idx]
        criterion = nn.NLLLoss().to(args.device)
        model.train()
        classifier.train()
        optimizer = torch.optim.SGD([ 
            {'params': model.parameters(), 'lr': args.lr},   # 0
            {'params': classifier.parameters(), 'lr': args.lr}],
            momentum=0.9, weight_decay=1e-4) 
        for iter in range(args.train_ep):
            epoch_loss = []
            if idx < 5:
                for batch_idx, (m1, _, label_g) in enumerate(trainloader):
                    if m1.size(0) < 2:
                        continue  # 跳过只有一个样本的批次
                    m1, labels = m1.to(args.device), label_g.to(args.device)
                    
                    model.zero_grad()
                    classifier.zero_grad()
                    optimizer.zero_grad()
                    protos = model(m1) # (bsz,特征维度)
                    log_probs = classifier(protos)
                    loss = criterion(log_probs, labels)
                    loss.backward()
                    optimizer.step()
                    log_probs = log_probs[:, 0:args.num_classes]
                    _, y_hat = log_probs.max(1)
                    acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()
                    if batch_idx == len(trainloader) - 2:
                        print('| Global Round : {} | User: {} | Local Epoch : {} | Loss: {:.3f} | Acc: {:.3f}'.format(
                            round, idx, iter, 
                            loss.item(),
                            acc_val.item()))
            else:
                for batch_idx, (_, m2, label_g) in enumerate(trainloader):
                    if m2.size(0) < 2:
                        continue  # 跳过只有一个样本的批次
                    m2, labels = m2.to(args.device), label_g.to(args.device)
                    
                    model.zero_grad()
                    classifier.zero_grad()
                    optimizer.zero_grad()
                    protos = model(m2) # (bsz,特征维度)
                    log_probs = classifier(protos)
                    loss = criterion(log_probs, labels)
                    loss.backward()
                    optimizer.step()
                    log_probs = log_probs[:, 0:args.num_classes]
                    _, y_hat = log_probs.max(1)
                    acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()
                    if batch_idx == len(trainloader) - 2:
                        print('| Global Round : {} | User: {} | Local Epoch : {} | Loss: {:.3f} | Acc: {:.3f}'.format(
                            round, idx, iter, 
                            loss.item(),
                            acc_val.item()))
        local_model_list[idx].load_state_dict(model.state_dict()) 
        local_classifier_list[idx].load_state_dict(classifier.state_dict())                    

print("testing")
for idx in idxs_users:
    model=copy.deepcopy(local_model_list[idx])
    classifier=copy.deepcopy(local_classifier_list[idx])
    model.eval()
    classifier.eval()
    all_preds = []
    all_labels = []
    if idx < 5:
        testloader = test_dataset1
    else:
        testloader = test_dataset2
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            features = model(inputs)
            outputs = classifier(features)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    print(f'Client:{idx}  Test Accuracy: {accuracy:.4f}')

print("Pseudo-labeling with modalities")
for idx in idxs_users:
    model = copy.deepcopy(local_model_list[idx])
    classifier = copy.deepcopy(local_classifier_list[idx])
    pseuloader = user_unlabels[idx]  # 未标记数据的加载器
    trainloader = user_groups[idx]   # 已标记数据的加载器
    threshold = 0.9  # 置信度阈值
    pseudo_labels = []
    valid_indices = []
    real_labels = []

    model.eval()
    classifier.eval()

    # 生成伪标签
    new_data = []
    for batch_idx, data in enumerate(pseuloader):
        if idx < 5:  # 模态一
            inputs, _, labels = data
        else:        # 模态二
            _, inputs, labels = data

        if inputs.size(0) < 2:
            continue  # 跳过只有一个样本的批次
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        with torch.no_grad():
            protos = model(inputs)
            log_probs = classifier(protos)
            probs = torch.softmax(log_probs, dim=1)
            max_probs, pseudo = torch.max(probs, dim=1)
            valid = max_probs > threshold
            valid_indices.extend(valid.nonzero(as_tuple=True)[0].tolist())
            pseudo_labels.extend(pseudo[valid].tolist())
            real_labels.extend(labels[valid].tolist())
            # 仅添加置信度高于阈值的数据
            for i in valid.nonzero(as_tuple=True)[0]:
                if idx < 5:
                    new_data.append((inputs[i].cpu(), None, pseudo[i].cpu()))
                else:
                    new_data.append((None, inputs[i].cpu(), pseudo[i].cpu()))

    # 计算伪标签准确率
    pseudo_labels = torch.tensor(pseudo_labels, device=args.device)
    real_labels = torch.tensor(real_labels, device=args.device)
    accuracy = (pseudo_labels == real_labels).float().mean().item()
    print(f"User {idx}: Pseudo-labeling accuracy: {accuracy:.2f}")

    # # 混合有标签数据和伪标签数据进行训练
    # if new_data:
    #     pseuloader.dataset.data = new_data  # 假设数据集支持直接赋值更新
    #     pseuloader = DataLoader(pseuloader.dataset, batch_size=args.batch_size, shuffle=True)

    # 合并有标签数据和伪标签数据的加载器
    if new_data:
        combined_data = list(trainloader.dataset) + new_data
        combined_dataset = CustomDataset(combined_data)
        mixed_loader = DataLoader(combined_dataset, batch_size=args.batch_size, shuffle=True)
    else:
        mixed_loader = trainloader

    model.train()
    classifier.train()
    optimizer = torch.optim.SGD([
        {'params': model.parameters(), 'lr': args.lr},
        {'params': classifier.parameters(), 'lr': args.lr}],
        momentum=0.9, weight_decay=1e-4)

    for epoch in range(args.train_ep):
        for batch_data in mixed_loader:
            if idx < 5:  # 模态一
                inputs, _, labels = batch_data
            else:        # 模态二
                _, inputs, labels = batch_data
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            if inputs.size(0) < 2:
                continue  # 跳过只有一个样本的批次
            optimizer.zero_grad()
            protos = model(inputs)
            log_probs = classifier(protos)
            loss = criterion(log_probs, labels)
            loss.backward()
            optimizer.step()
        print(f'User {idx}: Additional training epoch {epoch + 1}, Loss: {loss.item()}')

    # 更新模型状态
    local_model_list[idx].load_state_dict(model.state_dict())
    local_classifier_list[idx].load_state_dict(classifier.state_dict())



print("testing")
for idx in idxs_users:
    model=copy.deepcopy(local_model_list[idx])
    classifier=copy.deepcopy(local_classifier_list[idx])
    model.eval()
    classifier.eval()
    all_preds = []
    all_labels = []
    if idx < 5:
        testloader = test_dataset1
    else:
        testloader = test_dataset2
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            features = model(inputs)
            outputs = classifier(features)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    print(f'Client:{idx}  Test Accuracy: {accuracy:.4f}')