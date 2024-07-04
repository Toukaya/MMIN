import os
import json
from typing import List
import torch
import numpy as np
import h5py
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence

from data.base_dataset import BaseDataset
from data.msp_multimodal_dataset import MSPMultimodalDataset


class CMUMOSIMultimodalDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, isTrain=None):
        """
        函数功能：修改命令行选项参数

        参数：
        parser：ArgumentParser对象，用于处理命令行参数
        isTrain：可选参数，默认值为None，表示是否处于训练阶段

        返回值：
        parser：更新后的ArgumentParser对象，包含新的命令行选项
        """

        # 添加'--cvNo'选项，类型为整数，用于指定交叉验证集编号
        parser.add_argument('--cvNo', type=int, help='选择哪个交叉验证集合')

        # 添加'--output_dim'选项，类型为整数，表示数据集中标签的种类数量
        parser.add_argument('--output_dim', type=int, help='该数据集中有多少种标签类型')

        # 添加'--norm_method'选项，类型为字符串，可选值为'utt'或'trn'
        # 用于指定如何规范化输入的comparE特征
        parser.add_argument('--norm_method', type=str, choices=['utt', 'trn'], help='如何规范化输入的比较特征')

        # 返回更新后的命令行解析器
        return parser

    # 初始化函数，用于创建IEMOCAP数据集读取器
    def __init__(self, opt, set_name):
        ''' 初始化IEMOCAP数据集读取器
            参数：
                opt：包含配置信息的对象
                set_name：数据集子集名称，可选['trn', 'val', 'tst']
        '''
        # 调用父类的初始化方法
        super().__init__(opt)

        # 记录和加载基本设置
        cvNo = opt.cvNo  # 获取交叉验证编号
        self.set_name = set_name  # 保存数据集子集名称
        # 获取当前文件的绝对路径
        pwd = os.path.abspath(__file__)
        # 获取当前文件所在目录
        pwd = os.path.dirname(pwd)
        # 加载配置文件
        config = json.load(open(os.path.join(pwd, 'config', 'CMUMOSI_config.json')))
        # 获取特征归一化方法
        self.norm_method = opt.norm_method

        # 加载特征数据
        self.all_A = np.load(  # 加载音频特征
            os.path.join(config['feature_root'], 'A', str(opt.cvNo), f'{set_name}.npy'),
            'r'
        )
        self.all_V = np.load(  # 加载视觉特征
            os.path.join(config['feature_root'], 'V', str(opt.cvNo), f'{set_name}.npy'),
            'r'
        )
        self.all_L = np.load(  # 加载语言特征
            os.path.join(config['feature_root'], 'L', str(opt.cvNo), f'{set_name}.npy'),
            'r'
        )

        # 加载目标标签
        label_path = os.path.join(config['target_root'], f'{cvNo}', f"{set_name}_label.npy")  # 标签文件路径
        int2name_path = os.path.join(config['target_root'], f'{cvNo}', f"{set_name}_int2name.npy")  # 类别名映射文件路径
        self.label = np.load(label_path)  # 加载标签数据
        # self.label = np.argmax(self.label, axis=1)  # 可选操作：将多分类标签转换为单标签（取最大概率类别）
        self.int2name = np.load(int2name_path)  # 加载类别名映射

        # 是否使用自定义的collate_fn（数据打包函数）
        self.manual_collate_fn = False

    def __getitem__(self, index):
        # 将索引映射到对应的名称
        int2name = self.int2name[index]

        # 将标签转换为torch tensor，这里将其转换为float类型
        label = torch.tensor(self.label[index]).float()

        # 处理A特征，将其转换为浮点数类型的torch tensor
        A_feat = torch.FloatTensor(self.all_A[index])

        # 处理V特征，将其转换为浮点数类型的torch tensor
        V_feat = torch.FloatTensor(self.all_V[index])

        # 提供L特征，将其转换为浮点数类型的torch tensor
        L_feat = torch.FloatTensor(self.all_L[index])

        # 返回字典包含所有特征和标签
        return {
            'A_feat': A_feat,  # A特征
            'V_feat': V_feat,  # V特征
            'L_feat': L_feat,  # L特征
            'label': label,  # 标签
            'int2name': int2name  # 索引到名称的映射
        }

    def __len__(self):
        """
        返回标签(label)的长度，即标签列表的元素数量。
        """
        return len(self.label)

    def normalize_on_utt(self, features):
        # 计算特征矩阵在维度0上的平均值，并将其展成一个单列向量
        mean_f = torch.mean(features, dim=0).unsqueeze(0).float()

        # 计算特征矩阵在维度0上的标准差，并同样将其展成一个单列向量
        std_f = torch.std(features, dim=0).unsqueeze(0).float()

        # 将标准差为0的元素设置为1，以避免除以0的错误
        std_f[std_f == 0.0] = 1.0

        # 对特征进行标准化操作：减去平均值，然后除以标准差
        features = (features - mean_f) / std_f

        # 返回标准化后的特征
        return features

    def normalize_on_trn(self, features):
        """
        此函数用于对训练集（trn）中的特征进行归一化处理。

        参数:
        features: 输入的特征数据

        返回:
        features: 归一化后的特征数据
        """
        # 计算特征数据与均值的差值
        features = features - self.mean
        # 将差值除以标准差，完成归一化操作
        features = features / self.std
        # 返回归一化后的特征数据
        return features

    def collate_fn(self, batch):  # 定义一个用于数据加载时合并批次样本的函数
        A = [sample['A_feat'] for sample in batch]  # 提取批次中每个样本的'A_feat'特征
        V = [sample['V_feat'] for sample in batch]  # 提取批次中每个样本的'V_feat'特征
        L = [sample['L_feat'] for sample in batch]  # 提取批次中每个样本的'L_feat'特征

        lengths = torch.tensor([len(sample) for sample in A]).long()  # 计算并转换'A_feat'的长度为torch长整型张量
        A = pad_sequence(A, batch_first=True, padding_value=0)  # 对'A_feat'进行填充，使批次内的序列长度一致，以batch_first方式排列，用0填充
        V = pad_sequence(V, batch_first=True, padding_value=0)  # 对'V_feat'进行填充，同上
        L = pad_sequence(L, batch_first=True, padding_value=0)  # 对'L_feat'进行填充，同上

        label = torch.tensor([sample['label'] for sample in batch])  # 将批次中每个样本的'label'标签转换为torch张量
        int2name = [sample['int2name'] for sample in batch]  # 获取批次中每个样本的'int2name'映射

        return {  # 返回一个字典，包含处理后的特征和标签
            'A_feat': A,
            'V_feat': V,
            'L_feat': L,
            'label': label,
            'lengths': lengths,
            'int2name': int2name
        }

if __name__ == '__main__':
    class test:
        cvNo = 1
        A_type = "comparE"
        V_type = "denseface"
        L_type = "bert_large"
        norm_method = 'trn'

    
    opt = test()
    print('Reading from dataset:')
    a = MSPMultimodalDataset(opt, set_name='trn')
    data = next(iter(a))
    for k, v in data.items():
        if k not in ['int2name', 'label']:
            print(k, v.shape)
        else:
            print(k, v)
    print('Reading from dataloader:')
    x = [a[100], a[34], a[890]]
    print('each one:')
    for i, _x in enumerate(x):
        print(i, ':')
        for k, v in _x.items():
            if k not in ['int2name', 'label']:
                print(k, v.shape)
            else:
                print(k, v)
    print('packed output')
    x = a.collate_fn(x)
    for k, v in x.items():
        if k not in ['int2name', 'label']:
            print(k, v.shape)
        else:
            print(k, v)
    