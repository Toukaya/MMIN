import os
import json
from typing import List
import torch
import numpy as np
import h5py
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence

from data.base_dataset import BaseDataset


class ComparEDataset(BaseDataset):
    @staticmethod
    # 定义一个函数 modify_commandline_options，用于处理命令行参数
    def modify_commandline_options(parser, isTrain=None):
        # 添加一个参数 '--cvNo'，类型为整数，用于指定交叉验证集编号
        parser.add_argument('--cvNo', type=int, help='which cross validation set')

        # 添加一个参数 '--output_dim'，默认值为4，表示数据集中标签的种类数量
        parser.add_argument('--output_dim', type=int, default=4, help='how many label types in this dataset')

        # 添加一个参数 '--norm_method'，可选值为 'utt' 或 'trn'，用于指定输入比较特征的归一化方法
        parser.add_argument('--norm_method', type=str, choices=['utt', 'trn'],
                            help='how to normalize input comparE feature')

        # 返回处理后的参数解析器
        return parser

    def __init__(self, opt, set_name):
        """
        初始化 IEMOCAP 数据集读取器
        参数:
            opt: 包含配置信息的对象
            set_name: 数据集名称，可选 ['trn', 'val', 'tst']
        """
        # 调用父类构造函数
        super().__init__(opt)

        # 记录并加载基本设置
        cvNo = opt.cvNo
        self.set_name = set_name
        # 获取当前文件的绝对路径
        pwd = os.path.abspath(__file__)
        # 获取当前目录
        pwd = os.path.dirname(pwd)
        # 加载配置文件
        config = json.load(open(os.path.join(pwd, 'config', 'IEMOCAP_config.json')))
        # 设置归一化方法
        self.norm_method = opt.norm_method

        # 加载特征
        self.all_A = h5py.File(os.path.join(config['feature_root'], 'A', 'comparE.h5'), 'r')  # 读取特征文件
        self.mean_std = h5py.File(os.path.join(config['feature_root'], 'A', 'comparE_mean_std.h5'), 'r')  # 读取均值和标准差文件
        # 将h5py数据转换为torch张量
        self.mean = torch.from_numpy(self.mean_std[str(cvNo)]['mean'][()]).unsqueeze(0).float()
        self.std = torch.from_numpy(self.mean_std[str(cvNo)]['std'][()]).unsqueeze(0).float()

        # 加载目标标签
        label_path = os.path.join(config['target_root'], f'{cvNo}', f"{set_name}_label.npy")  # 标签文件路径
        int2name_path = os.path.join(config['target_root'], f'{cvNo}', f"{set_name}_int2name.npy")  # 类别名文件路径
        self.label = np.load(label_path)  # 加载标签
        self.label = np.argmax(self.label, axis=1)  # 获取每个样本的最大概率类别
        self.int2name = np.load(int2name_path)  # 加载类别名映射
        # 使用自定义的collate_fn
        self.manual_collate_fn = True

    def __getitem__(self, index):
        # 将索引转换为对应的名称
        int2name = self.int2name[index][0].decode()
        # 获取对应的标签
        label = torch.tensor(self.label[index])

        # 处理A_feat
        # 从numpy数组转换为torch张量
        A_feat = torch.from_numpy(self.all_A[int2name][()]).float()
        # 根据norm_method进行归一化处理
        A_feat = self.normalize_on_utt(A_feat) if self.norm_method == 'utt' else self.normalize_on_trn(A_feat)

        # 返回字典，包含处理后的A_feat、标签和名称
        return {
            'A_feat': A_feat,
            'label': label,
            'int2name': int2name
        }

    def __len__(self):
        # 返回标签列表的长度，用于确定序列的长度
        return len(self.label)

    def normalize_on_utt(self, features):
        # 计算每一维特征的平均值，并将其转换为一个单元素张量
        mean_f = torch.mean(features, dim=0).unsqueeze(0).float()

        # 计算每一维特征的标准差，并同样将其转换为一个单元素张量
        std_f = torch.std(features, dim=0).unsqueeze(0).float()

        # 标准差为0时，设置为1，以避免除以0的错误
        std_f[std_f == 0.0] = 1.0

        # 对特征进行标准化处理，减去平均值并除以标准差
        features = (features - mean_f) / std_f

        # 返回标准化后的特征
        return features

    def normalize_on_trn(self, features):
        """
        对训练集特征进行归一化处理

        参数:
        - features: 输入的特征数据

        返回:
        - features: 归一化后的特征数据
        """
        # 计算特征数据与均值的差值
        features = features - self.mean
        # 将差值除以标准差，完成归一化
        features = features / self.std
        # 返回归一化后的特征数据
        return features

    def collate_fn(self, batch):  # 定义一个用于数据加载时合并批次样本的函数
        A = [sample['A_feat'] for sample in batch]  # 将批次中每个样本的'A_feat'键对应的值提取到列表A中
        lengths = torch.tensor([len(sample) for sample in A]).long()  # 计算每个样本的长度，并转换为torch.long类型的张量
        A = pad_sequence(A, batch_first=True, padding_value=0)  # 对列表A进行填充，使所有样本长度相同，以batch_first=True的方式排列，用0填充
        # A = pack_padded_sequence(A, lengths=lengths, batch_first=True, enforce_sorted=False)
        # 注释掉的这行代码原本用于处理变长序列，但在这里未使用

        label = torch.tensor([sample['label'] for sample in batch])  # 将批次中每个样本的'label'键对应的值转换为torch.tensor
        int2name = [sample['int2name'] for sample in batch]  # 将批次中每个样本的'int2name'键对应的值提取到列表中

        return {  # 返回一个字典，包含处理后的样本数据
            'A_feat': A,  # 填充后的特征序列
            'label': label,  # 样本标签
            'lengths': lengths,  # 每个样本原始的长度信息
            'int2name': int2name  # 样本的'int2name'信息
        }

if __name__ == '__main__':
    class test:
        cvNo = 1
        norm_method = 'trn'
    
    opt = test()
    print('Reading from dataset:')
    a = ComparEDataset(opt, set_name='trn')
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
    