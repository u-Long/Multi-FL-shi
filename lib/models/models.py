#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import BertModel
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)

class CNNFemnist(nn.Module):
    def __init__(self, args):
        super(CNNFemnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, args.out_channels, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(int(16820/20*args.out_channels), 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x1 = F.relu(self.fc1(x))
        x = F.dropout(x1, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1), x1

class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, args.out_channels, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(int(320/20*args.out_channels), 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x1 = F.relu(self.fc1(x))
        x = F.dropout(x1, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1), x1

class CNNFashion_Mnist(nn.Module):
    def __init__(self, args):
        super(CNNFashion_Mnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc0 = nn.Linear(16 * 5 * 5, 120)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, args.num_classes)

    # def forward(self, x):
    #     x = self.pool(F.relu(self.conv1(x)))
    #     x = self.pool(F.relu(self.conv2(x)))
    #     x = x.view(-1, 16 * 5 * 5)
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     return F.log_softmax(x, dim=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x1 = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x1))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1), x1


class Lenet(nn.Module):
    def __init__(self, args):
        super(Lenet, self).__init__()
        self.n_cls = 10
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 5 * 5, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, self.n_cls)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x1 = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x1))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1), x1
    

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    else:
        # Default initialization for other layers
        for param in m.parameters():
            if param.dim() > 1:
                nn.init.kaiming_uniform_(param, nonlinearity='relu')
            else:
                nn.init.constant_(param, 0)

class cnn_layers_1(nn.Module):
    """
    CNN layers applied on acc sensor data to generate pre-softmax
    ---
    params for __init__():
        input_size: e.g. 1
        num_classes: e.g. 6
    forward():
        Input: data
        Output: pre-softmax
    """
    def __init__(self, input_size):
        super().__init__()

        # Extract features, 2D conv layers
        self.features = nn.Sequential(
            nn.Conv2d(input_size, 64, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Conv2d(64, 64, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Conv2d(64, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Conv2d(32, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            )
        # self.apply(init_weights)

    def forward(self, x):
        x = self.features(x)

        return x


class cnn_layers_2(nn.Module):
    """
    CNN layers applied on skeletal data with different configurations
    based on the dataset type.
    ---
    params for __init__():
        input_size: e.g. 1
        dataset: e.g. 'UTD' or 'MMAct'
    forward():
        Input: data
        Output: pre-softmax
    """
    def __init__(self, input_size, dataset):
        super().__init__()

        if dataset == 'UTD':
            conv2 = nn.Conv3d(64, 64, [5, 5, 2])
        elif dataset == 'MMAct':
            conv2 = nn.Conv3d(64, 64, [5, 5, 1])
        else:
            raise ValueError("Unsupported dataset type")

        # Extract features, 3D conv layers
        self.features = nn.Sequential(
            nn.Conv3d(input_size, 64, [5, 5, 2]),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            conv2,
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Conv3d(64, 32, [5, 5, 1]),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Conv3d(32, 16, [5, 2, 1]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        return x
# 因为mmact是17个骨骼点xy二维数据

    
class Encoder1(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.imu_cnn_layers = cnn_layers_1(input_size)

    def forward(self, x1):

        imu_output = self.imu_cnn_layers(x1)

        return imu_output


class Encoder2(nn.Module):
    def __init__(self, input_size, dataset):
        super().__init__()

        self.skeleton_cnn_layers = cnn_layers_2(input_size, dataset)

    def forward(self, x2):

        skeleton_output = self.skeleton_cnn_layers(x2)

        return skeleton_output


class Encoder(nn.Module):
    def __init__(self, input_size, dataset):
        super().__init__()

        self.imu_cnn_layers = cnn_layers_1(input_size)
        self.skeleton_cnn_layers = cnn_layers_2(input_size, dataset)

    def forward(self, x1, x2):

        imu_output = self.imu_cnn_layers(x1)
        skeleton_output = self.skeleton_cnn_layers(x2)

        return imu_output, skeleton_output

class HeadModule(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_size, 1280),
            nn.BatchNorm1d(1280),
            nn.ReLU(inplace=True),
            nn.Linear(1280, output_size),
        )
        # self.apply(init_weights)

    def forward(self, x):
        return F.normalize(self.seq(x.view(x.size(0), -1)), dim=1)
    
    
class MyUTDModelFeature1(nn.Module):
    """Model for human-activity-recognition."""
    def __init__(self, input_size, p1_size):
        super().__init__()

        self.encoder = Encoder1(input_size)

        self.head_1 = HeadModule(p1_size, 128) #7552 23680
        # self.apply(init_weights)

    def forward(self, x1):

        imu_output = self.encoder(x1)
        # (16,118,4) 
        imu_output = F.normalize(self.head_1(imu_output.view(imu_output.size(0), -1)), dim=1)

        return imu_output


class MyUTDModelFeature2(nn.Module):
    """Model for human-activity-recognition."""
    def __init__(self, input_size, p2_size, dataset):
        super().__init__()

        self.encoder = Encoder2(input_size, dataset)

        self.head_2 = HeadModule(p2_size, 128)#2688 8576
        # self.apply(init_weights)

    def forward(self, x2):

        skeleton_output = self.encoder(x2)
        # (16, 24, 7, 1)
        skeleton_output = F.normalize(self.head_2(skeleton_output.view(skeleton_output.size(0), -1)), dim=1)

        return skeleton_output
    

class MyUTDModelFeature(nn.Module):
    """Model for human-activity-recognition."""
    def __init__(self, input_size, p1_size, p2_size, dataset):
        super().__init__()

        self.encoder = Encoder(input_size, dataset)

        self.head_1 = HeadModule(p1_size, 128)

        self.head_2 = HeadModule(p2_size, 128)
        # self.apply(init_weights)

    def forward(self, x1, x2):

        imu_output, skeleton_output = self.encoder(x1, x2)
        # (16,118,4) (16, 24, 7, 1)
        imu_output = F.normalize(self.head_1(imu_output.view(imu_output.size(0), -1)), dim=1)
        skeleton_output = F.normalize(self.head_2(skeleton_output.view(skeleton_output.size(0), -1)), dim=1)

        return imu_output, skeleton_output

class DualModalClassifier(nn.Module):
    """Classifier for handling two modalities."""
    def __init__(self, args):
        super(DualModalClassifier, self).__init__()
        
        # 为每种模态创建一个FeatureClassifier
        self.classifier_modality_1 = FeatureClassifier(args)
        self.classifier_modality_2 = FeatureClassifier(args)

    def forward(self, x1, x2):
        # 分别处理两种模态的数据
        output_modality_1 = self.classifier_modality_1(x1)
        output_modality_2 = self.classifier_modality_2(x2)
        
        return output_modality_1, output_modality_2

class FeatureClassifier(nn.Module):
    def __init__(self, args):
        super(FeatureClassifier, self).__init__()

        # 定义分类器
        # 由于特征已经被预处理和降维，网络可以更简单
        self.classifier = nn.Sequential(
            nn.Linear(128, 1280),
            nn.BatchNorm1d(1280),
            nn.ReLU(inplace=True),
            nn.Linear(1280, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, args.num_classes),
        )
        # self.apply(init_weights)

    def forward(self, x):
        # x的形状是 [batch_size, 128]
        # 通过分类器得到最终的输出
        output = self.classifier(x)
        return F.log_softmax(output, dim=1)

class SkeletonClassifier(nn.Module):
    def __init__(self, args):
        super(SkeletonClassifier, self).__init__()

        # 定义GRU层
        # 假设每个骨骼点的特征维度为3（XYZ坐标），共20个骨骼点，因此输入特征是60
        self.gru = nn.GRU(input_size=60, hidden_size=120, num_layers=2, batch_first=True)

        # 定义分类器
        self.classifier = nn.Sequential(
            nn.Linear(120, 1280),
            nn.BatchNorm1d(1280),
            nn.ReLU(inplace=True),
            nn.Linear(1280, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, args.num_classes),
        )

    def forward(self, x):
        # 假设x的形状是 [batch_size, frames, joints, 3]
        # 将其调整为适合GRU输入的形状 [batch_size, frames, features]
        batch_size, frames, joints, _ = x.shape
        x = x.view(batch_size, frames, -1)

        # 通过GRU处理
        x, _ = self.gru(x)

        # 取最后一个时间步的输出
        x = x[:, -1, :]

        # 通过分类器得到最终的输出
        output = self.classifier(x)
        return output


class InertialClassifier(nn.Module):
    def __init__(self, args):
        super(InertialClassifier, self).__init__()

        # 定义GRU层
        # 输入特征为6（每帧的特征数）
        self.gru = nn.GRU(input_size=6, hidden_size=120, num_layers=2, batch_first=True)

        # 定义分类器
        self.classifier = nn.Sequential(
            nn.Linear(120, 1280),
            nn.BatchNorm1d(1280),
            nn.ReLU(inplace=True),
            nn.Linear(1280, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, args.num_classes),
        )

    def forward(self, x):
        # 假设x的形状是 [batch_size, frames, features]，即 [batch_size, 120, 6]
        # 通过GRU处理
        x, _ = self.gru(x)

        # 取最后一个时间步的输出
        x = x[:, -1, :]

        # 通过分类器得到最终的输出
        output = self.classifier(x)
        return output

"""
Attention block
Reference: https://github.com/philipperemy/keras-attention-mechanism
"""
class Attn(nn.Module):
    def __init__(self, input_size1, input_size2):
        super().__init__()

        self.reduce_d1 = nn.Linear(input_size1, 1280)#7552 25088(UPMC) 

        self.reduce_d2 = nn.Linear(input_size2, 1280)#2688 512

        self.weight = nn.Sequential(

            nn.Linear(2560, 1280),
            nn.BatchNorm1d(1280),
            nn.Tanh(),

            nn.Linear(1280, 128),
            nn.BatchNorm1d(128),
            nn.Tanh(),

            nn.Linear(128, 2),
            nn.BatchNorm1d(2),
            nn.Tanh(),

            )
        # self.apply(init_weights)

    def forward(self, hidden_state_1, hidden_state_2):

        new_1 = self.reduce_d1(hidden_state_1.view(hidden_state_1.size(0), -1))#hidden_state_1, [bsz, 16, 472]
        new_2 = self.reduce_d2(hidden_state_2.view(hidden_state_2.size(0), -1))#hidden_state_2, [bsz, 16, 168]

        concat_feature = torch.cat((new_1, new_2), dim=1) #[bsz, 1280*2]
        activation = self.weight(concat_feature)#[bsz, 2]

        score = F.softmax(activation, dim=1)

        attn_feature_1 = hidden_state_1 * (score[:, 0].view(-1, 1, 1)) 
        attn_feature_2 = hidden_state_2 * (score[:, 1].view(-1, 1, 1))
        # print("attn_feature_1:", attn_feature_1.shape)
        # print("attn_feature_2:", attn_feature_2.shape)
        # attn_feature_1: torch.Size([27, 16, 472])
        # attn_feature_2: torch.Size([27, 16, 168])
        fused_feature = torch.cat( (attn_feature_1, attn_feature_2), dim=2)

        return fused_feature, score[:, 0], score[:, 1]


class LinearClassifierAttn(nn.Module):
    """Linear classifier"""
    def __init__(self, num_classes, input_size1, input_size2):
        super(LinearClassifierAttn, self).__init__()

        self.attn = Attn(input_size1, input_size2)

        self.gru = nn.GRU(640, 120, 2, batch_first=True)

        # Classify output, fully connected layers
        self.classifier = nn.Sequential(

            nn.Linear(1920, 1280),
            nn.BatchNorm1d(1280),
            nn.ReLU(inplace=True),

            nn.Linear(1280, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Linear(128, num_classes),
            )
        # self.apply(init_weights)

    def forward(self, feature1, feature2):

        feature1 = feature1.view(feature1.size(0), 16, -1)
        feature2 = feature2.view(feature2.size(0), 16, -1)
        
        fused_feature, weight1, weight2 = self.attn(feature1, feature2)

        fused_feature, _ = self.gru(fused_feature)
        fused_feature = fused_feature.contiguous().view(fused_feature.size(0), -1)

        output = self.classifier(fused_feature)

        return F.log_softmax(output, dim=1), weight1, weight2
 
    
class model_(nn.Module):
    def __init__(self, input_size, num_classes):
        super(model_, self).__init__()
        self.model = MyUTDModelFeature(input_size)
        self.classifier = LinearClassifierAttn(num_classes)

    def forward(self, images, captions, cap_lens):
        image_code = self.encoder(images)
        return self.decoder(image_code, captions, cap_lens)

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention_weights = self.softmax(torch.bmm(Q, K.transpose(1, 2)) / (K.size(-1) ** 0.5))
        return torch.bmm(attention_weights, V)

class TXTEncoder(nn.Module):
    def __init__(self, hidden_dim, lstm_layers, bidirectional, bert_model='bert-base-uncased', dropout_rate=0.1):
        super(TXTEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(input_size=768, hidden_size=hidden_dim, num_layers=lstm_layers, 
                            bidirectional=bidirectional, batch_first=True)
        self.attention = SelfAttention(hidden_dim * 2 if bidirectional else hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, hidden_dim * 2 if bidirectional else hidden_dim)
        # self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def forward(self, input_ids, attention_mask):
        # 使用BERT模型获取嵌入
        embedded = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # 应用dropout
        embedded = self.dropout(embedded)
        # 使用LSTM处理BERT的输出
        self.lstm.flatten_parameters()
        lstm_output, _ = self.lstm(embedded)
        # 应用自注意力机制
        attention_output = self.attention(lstm_output)
        return attention_output.mean(dim=1)
        # 应用全连接层
        features = self.fc(attention_output.mean(dim=1))
        return features

class TXTFeature(nn.Module):
    def __init__(self, hidden_dim, lstm_layers, bidirectional):
        super(TXTFeature, self).__init__()
        
        self.encoder = TXTEncoder(hidden_dim, lstm_layers, bidirectional)

        self.head_2 = nn.Sequential(

            nn.Linear(512, 1280),
            nn.BatchNorm1d(1280),
            nn.ReLU(inplace=True),

            nn.Linear(1280, 128),            
            )
    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids, attention_mask)
        output = F.normalize(self.head_2(hidden.view(hidden.size(0), -1)), dim=1)

        return output

class UnitEncoder(nn.Module):
    def __init__(self, hidden_dim, lstm_layers, bidirectional):
        super().__init__()

        self.encoder1 = IMGEncoder()
        self.encoder2 = TXTEncoder(hidden_dim, lstm_layers, bidirectional)

    def forward(self, x1, x2, x3):

        output_img = self.encoder1(x1)
        output_txt = self.encoder2(x2, x3)

        return output_img, output_txt

class UnitFeature(nn.Module):
    def __init__(self, hidden_dim, lstm_layers, bidirectional):
        super(UnitFeature, self).__init__()

        self.encoder = UnitEncoder(hidden_dim, lstm_layers, bidirectional)
        self.head_1 = nn.Sequential(
            nn.Linear(14*14*128, 1280),
            nn.BatchNorm1d(1280),
            nn.ReLU(inplace=True),
            nn.Linear(1280, 128),            
            )
        self.head_2 = nn.Sequential(

            nn.Linear(512, 1280),
            nn.BatchNorm1d(1280),
            nn.ReLU(inplace=True),

            nn.Linear(1280, 128),            
            )
    def forward(self, x1, x2, x3):
        output_img, output_txt = self.encoder(x1,x2,x3)
        output_img = F.normalize(self.head_1(output_img.view(output_img.size(0), -1)), dim=1)
        output_txt = F.normalize(self.head_2(output_txt.view(output_txt.size(0), -1)), dim=1)
        return output_img, output_txt
    
class TXTDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TXTDecoder, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # 使用全连接层进行分类
        output = self.fc(x)
        return output

class CustomTextModel(nn.Module):
    def __init__(self):
        super(CustomTextModel, self).__init__()
        bidirectional = True
        hidden_dim = 256
        lstm_layers = 2
        self.encoder = TXTEncoder(hidden_dim=hidden_dim, lstm_layers=lstm_layers, bidirectional=bidirectional)
        self.decoder = TXTDecoder(input_dim=hidden_dim * 2 if bidirectional else hidden_dim, output_dim=101)

    def forward(self, input_ids, attention_mask):
        encoded_features = self.encoder(input_ids, attention_mask)
        output = self.decoder(encoded_features)
        return output
    
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        modality1, modality2, label = self.data[index]
        if modality1 is None:
            modality1 = torch.zeros(3, 224, 224)
        if modality2 is None:
            modality2 = {
                'input_ids': torch.zeros((40,), dtype=torch.long),
                'token_type_ids': torch.zeros((40,), dtype=torch.long),
                'attention_mask': torch.zeros((40,), dtype=torch.long)
            }
        return modality1, modality2, label
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class IMGEncoder(nn.Module):
    def __init__(self):
        super(IMGEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        # Assuming the input image size is 224x224, the output feature map size here is 14x14x128

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return x

class IMGClassifier(nn.Module):
    def __init__(self, num_classes):
        super(IMGClassifier, self).__init__()
        # Convert feature maps to a single vector per image
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(14*14*128, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ImageFeature(nn.Module):
    def __init__(self):
        super(ImageFeature, self).__init__()
        self.encoder = IMGEncoder()
        self.head_1 = nn.Sequential(
            nn.Linear(14*14*128, 1280),
            nn.BatchNorm1d(1280),
            nn.ReLU(inplace=True),
            nn.Linear(1280, 128),            
            )

    def forward(self, x):
        x = self.encoder(x)
        x = F.normalize(self.head_1(x.view(x.size(0), -1)), dim=1)
        return x

class IMGTXTEncoder(nn.Module):
    def __init__(self, hidden_dim, lstm_layers, bidirectional):
        super(IMGTXTEncoder, self).__init__()
        self.encoder1 = IMGEncoder()
        self.encoder2 = TXTEncoder(hidden_dim=hidden_dim, lstm_layers=lstm_layers, bidirectional=bidirectional)
        