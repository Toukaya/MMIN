import os
import json
from typing import List
import torch
import numpy as np
import h5py
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence

from data.base_dataset import BaseDataset


# 定义WordAlignedDataset类，用于处理单词对齐的数据集
class WordAlignedDataset(BaseDataset):
    # 修改命令行选项
    @staticmethod
    def modify_commandline_options(parser, isTrain=None):
        """
        此函数用于根据传入参数修改命令行选项
        :param parser: argparse.ArgumentParser对象，用于处理命令行参数
        :param isTrain: 可选参数，布尔值，表示是否处于训练模式。默认值可能由外部决定
        :return: 返回修改后的命令行解析器，以便后续使用

        在函数内部，向命令行解析器添加了以下参数：
        - --cvNo: int类型，指定交叉验证集的编号
        - --A_type: str类型，选择使用的音频特征
        - --V_type: str类型，选择使用的视觉特征
        - --L_type: str类型，选择使用的词汇特征
        - --output_dim: int类型，数据集中标签类型的数量
        - --norm_method: str类型，可选值为'utt'或'trn'，指定输入compare特征的归一化方法
        """
        parser.add_argument('--cvNo', type=int, help='选择哪个交叉验证集合')
        parser.add_argument('--A_type', type=str, help='使用的音频特征类型')
        parser.add_argument('--V_type', type=str, help='使用的视觉特征类型')
        parser.add_argument('--L_type', type=str, help='使用的词汇特征类型')
        parser.add_argument('--output_dim', type=int, help='数据集中标签类别的数量')
        parser.add_argument('--norm_method', type=str, choices=['utt', 'trn'], help='输入compare特征的归一化方式')

        return parser

    # 初始化函数，用于加载IEMOCAP数据集
    def __init__(self, opt, set_name):
        ''' IEMOCAP数据集读取器
            set_name: ['trn', 'val', 'tst']之一
        '''
        # 调用父类的初始化方法
        super().__init__(opt)

        # 记录和加载基本设置
        cvNo = opt.cvNo  # 获取交叉验证编号
        self.set_name = set_name  # 设置名称
        pwd = os.path.abspath(__file__)  # 获取当前文件的绝对路径
        pwd = os.path.dirname(pwd)  # 获取当前文件所在目录
        # 加载配置文件
        config = json.load(open(os.path.join(pwd, 'config', 'IEMOCAP_config.json')))

        # 正则化方法
        self.norm_method = opt.norm_method

        # 加载特征
        self.A_type = opt.A_type  # 特征类型A
        self.all_A = h5py.File(os.path.join(config['feature_root'], 'aligned', 'A', f'aligned_{self.A_type}.h5'),
                               'r')  # 加载特征A文件

        # 如果特征类型为comparE，加载均值和标准差
        if self.A_type == 'comparE':
            self.mean_std = h5py.File(
                os.path.join(config['feature_root'], 'aligned', 'A', 'aligned_comparE_mean_std.h5'), 'r')
            self.mean = torch.from_numpy(self.mean_std[str(cvNo)]['mean'][()]).unsqueeze(0).float()  # 转换并存储均值
            self.std = torch.from_numpy(self.mean_std[str(cvNo)]['std'][()]).unsqueeze(0).float()  # 转换并存储标准差

        self.V_type = opt.V_type  # 特征类型V
        self.all_V = h5py.File(os.path.join(config['feature_root'], 'aligned', 'V', f'aligned_{self.V_type}.h5'),
                               'r')  # 加载特征V文件

        self.L_type = opt.L_type  # 特征类型L
        self.all_L = h5py.File(os.path.join(config['feature_root'], 'aligned', 'L', f'aligned_{self.L_type}.h5'),
                               'r')  # 加载特征L文件

        # 加载目标标签
        label_path = os.path.join(config['target_root'], f'{cvNo}', f"{set_name}_label.npy")  # 标签路径
        int2name_path = os.path.join(config['target_root'], f'{cvNo}', f"{set_name}_int2name.npy")  # 名称映射路径
        self.label = np.load(label_path)  # 加载标签
        self.label = np.argmax(self.label, axis=1)  # 获取每个样本的最大概率标签
        self.int2name = np.load(int2name_path)  # 加载名称映射
        self.int2name = [x[0].decode() for x in self.int2name]  # 解码并转换为字符串列表

        # 检查并移除异常名称
        if 'Ses03M_impro03_M001' in self.int2name:
            idx = self.int2name.index('Ses03M_impro03_M001')
            self.int2name.pop(idx)  # 移除异常名称
            self.label = np.delete(self.label, idx, axis=0)  # 移除对应的标签

        # 使用自定义的collate_fn
        self.manual_collate_fn = True

    def __getitem__(self, index):
        """
        根据给定的索引获取数据集中的一个样本。
        返回样本包含以下内容：
        - A_feat：音频特征，转换为PyTorch张量
        - V_feat：视觉特征，转换为PyTorch张量
        - L_feat：词汇特征，转换为PyTorch张量
        - label：样本的标签，表示为PyTorch张量
        - int2name：索引对应的名称

        音频特征A_feat根据其类型进行处理：
        - 如果A_type为'comparE'，则根据norm_method（'utt'或'trn'）进行归一化处理。

        :param index: 数据集中样本的索引位置
        :return: 一个字典，包含音频、视觉、词汇特征、标签及其对应的名称
        """
        int2name = self.int2name[index]  # 获取索引对应的名称
        label = torch.tensor(self.label[index])  # 将标签转换为PyTorch张量

        # 处理音频特征A_feat
        A_feat = torch.from_numpy(self.all_A[int2name][()]).float()  # 转换为PyTorch张量
        if self.A_type == 'comparE':
            # 如果A_type为'comparE'，根据norm_method进行归一化
            A_feat = self.normalize_on_utt(A_feat) if self.norm_method == 'utt' else self.normalize_on_trn(A_feat)

        # 处理视觉特征V_feat
        V_feat = torch.from_numpy(self.all_V[int2name][()]).float()

        # 提供词汇特征L_feat
        L_feat = torch.from_numpy(self.all_L[int2name][()]).float()

        # 返回包含所有特征和标签的字典
        return {
            'A_feat': A_feat,
            'V_feat': V_feat,
            'L_feat': L_feat,
            'label': label,
            'int2name': int2name
        }

    def __len__(self):
        """
        返回数据集的长度，即标签的数量
        """
        # 计算并返回标签列表的长度，代表数据集中的样本数量
        return len(self.label)

    def normalize_on_utt(self, features):
        """
        对单个语音片段进行特征归一化操作

        参数:
        features (Tensor): 输入的特征张量

        返回:
        features (Tensor): 归一化后的特征张量
        """
        # 计算特征在维度0上的平均值，并将其展成一个单行向量
        mean_f = torch.mean(features, dim=0).unsqueeze(0).float()

        # 计算特征在维度0上的标准差，并同样展成一个单行向量
        std_f = torch.std(features, dim=0).unsqueeze(0).float()

        # 将标准差为0的元素设置为1，以避免除以0的错误
        std_f[std_f == 0.0] = 1.0

        # 使用平均值和标准差对特征进行标准化处理
        features = (features - mean_f) / std_f

        # 返回归一化后的特征
        return features

    def normalize_on_trn(self, features):
        """
        此函数用于在整个训练数据集上执行特征归一化操作。

        参数:
        features (numpy array或类似结构): 待归一化的特征数据

        返回:
        features (numpy array或类似结构): 归一化后的特征数据
        """
        # 计算特征数据与训练集均值的差值
        features = features - self.mean
        # 将差值除以训练集标准差，完成归一化
        features = features / self.std
        return features

    def collate_fn(self, batch):
        """
        此方法用于组合一批样本，包括特征填充和标签整合。

        方法步骤如下：
        1. 从批次中的每个样本中提取音频特征（A_feat）、视觉特征（V_feat）和词汇特征（L_feat）。
        2. 使用pad_sequence函数对这些特征进行填充，确保所有特征的长度相同，以适应批量处理。
        3. 创建一个张量记录每个样本的原始长度（lengths），以便后续处理。
        4. 将所有样本的标签（label）转换为张量。
        5. 获取每个样本的名称映射（int2name）列表。
        6. 最后，将填充后的特征、标签、长度信息和名称映射组合成一个字典并返回。

        返回值：一个包含所有处理后特征和标签的字典。
        """
        # 提取音频特征
        A = [sample['A_feat'] for sample in batch]
        # 提取视觉特征
        V = [sample['V_feat'] for sample in batch]
        # 提取词汇特征
        L = [sample['L_feat'] for sample in batch]
        # 计算每个样本的长度，并转换为张量
        lengths = torch.tensor([len(sample) for sample in A]).long()
        # 填充音频特征，设置batch_first=True，padding_value=0
        A = pad_sequence(A, batch_first=True, padding_value=0)
        # 填充视觉特征，设置batch_first=True，padding_value=0
        V = pad_sequence(V, batch_first=True, padding_value=0)
        # 填充词汇特征，设置batch_first=True，padding_value=0
        L = pad_sequence(L, batch_first=True, padding_value=0)
        # 将所有样本的标签转换为张量
        label = torch.tensor([sample['label'] for sample in batch])
        # 获取每个样本的名称映射
        int2name = [sample['int2name'] for sample in batch]

        # 返回处理后的数据字典
        return {
            'A_feat': A,
            'V_feat': V,
            'L_feat': L,
            'label': label,
            'lengths': lengths,
            'int2name': int2name
        }

# 创建了一个测试类，并实例化了WordAlignedDataset类。
# 然后，从这个实例中获取了一个样本，并打印出了其特征的形状和标签。
# 接下来，从这个实例中获取了三个样本，并使用collate_fn方法将它们组合成一个批次，
# 然后打印出了批次中的特征的形状和标签。
if __name__ == '__main__':
    class test:
        cvNo = 1
        A_type = "comparE"
        V_type = "denseface"
        L_type = "bert"
        norm_method = 'trn'

    
    opt = test()
    print('Reading from dataset:')
    a = WordAlignedDataset(opt, set_name='trn')
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
    