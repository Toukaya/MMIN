import os
import json
from typing import List
import torch
import numpy as np
import random
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from data.base_dataset import BaseDataset
from numpy.random import randint
from sklearn.preprocessing import OneHotEncoder


## copy from cpm-net
def random_mask(view_num, alldata_len, missing_rate):
    """
    随机生成不完整数据信息，用完整视图数据模拟部分视图数据
    :param view_num: 视图数量
    :param alldata_len: 样本数量
    :param missing_rate: 文献中第3.2节定义的缺失率
    :return: Sn [alldata_len, view_num]: 返回的矩阵，表示每个样本在各个视图中的可用性
    """
    # 计算非缺失率
    one_rate = 1 - missing_rate  # 如果missing_rate为0.8，则one_rate为0.2

    # 如果非缺失率小于等于1/视图数，选择一个视图
    if one_rate <= (1 / view_num):
        # 使用OneHot编码器，确保至少有一个视图被选中（避免所有输入为0）
        enc = OneHotEncoder(categories=[np.arange(view_num)])
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
        return view_preserve

    # 如果非缺失率为1，所有视图都存在
    if one_rate == 1:
        # 所有元素都为1的矩阵
        matrix = randint(1, 2, size=(alldata_len, view_num))
        return matrix

    # 对于非缺失率在[1 / view_num, 1]之间的场景，可能存在多个视图输入
    # 确保至少有一个视图可用，增加难度（考虑样本重叠）
    error = 1
    while error >= 0.005:
        # 初始化视图保留矩阵
        enc = OneHotEncoder(categories=[np.arange(view_num)])
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()

        # 进一步生成one_num个样本
        one_num = view_num * alldata_len * one_rate - alldata_len
        ratio = one_num / (view_num * alldata_len)

        # 根据比率生成矩阵_iter
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(int)

        # 计算重叠数量
        a = np.sum(((matrix_iter + view_preserve) > 1).astype(int))

        # 更新one_num
        one_num_iter = one_num / (1 - a / one_num)
        ratio = one_num_iter / (view_num * alldata_len)

        # 重新生成矩阵_iter
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(int)

        # 创建最终矩阵，保留大于0的元素
        matrix = ((matrix_iter + view_preserve) > 0).astype(int)
        ratio = np.sum(matrix) / (view_num * alldata_len)
        error = abs(one_rate - ratio)

    return matrix



class CMUMOSIMissDataset(BaseDataset):
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

        # 添加一个名为'cvNo'的参数，类型为整数，用于指定交叉验证集的编号
        parser.add_argument('--cvNo', type=int, help='选择哪个交叉验证集合')

        # 添加一个名为'output_dim'的参数，类型为整数，表示数据集中标签的种类数量
        parser.add_argument('--output_dim', type=int, help='该数据集中有多少种标签类型')

        # 添加一个名为'norm_method'的参数，类型为字符串，可选值为'utt'或'trn'
        # 用于指定如何对输入的comparE特征进行归一化
        parser.add_argument('--norm_method', type=str, choices=['utt', 'trn'], help='如何规范化输入的comparE特征')

        # 返回更新后的参数解析器对象
        return parser

    # 初始化IEMOCAP数据集读取器
    def __init__(self, opt, set_name):
        ''' 初始化IEMOCAP数据集读取器
            参数:
                opt: 包含配置信息的对象
                set_name: 数据集名称，可以是['trn', 'val', 'tst']
        '''
        # 调用父类构造函数
        super().__init__(opt)

        # 记录和加载基本设置
        cvNo = opt.cvNo  # 获取交叉验证编号
        self.mask_rate = opt.mask_rate  # 获取特征掩码率
        self.set_name = set_name  # 保存数据集名称
        # 获取当前文件的绝对路径
        pwd = os.path.abspath(__file__)
        # 获取当前目录
        pwd = os.path.dirname(pwd)
        # 加载配置文件
        config = json.load(open(os.path.join(pwd, 'config', 'CMUMOSI_config.json')))
        # 加载特征
        self.all_A = np.load(  # 加载A特征
            os.path.join(config['feature_root'], 'A', str(opt.cvNo), f'{set_name}.npy'),
            'r'
        )
        self.all_V = np.load(  # 加载V特征
            os.path.join(config['feature_root'], 'V', str(opt.cvNo), f'{set_name}.npy'),
            'r'
        )
        self.all_L = np.load(  # 加载L特征
            os.path.join(config['feature_root'], 'L', str(opt.cvNo), f'{set_name}.npy'),
            'r'
        )
        # 加载目标标签
        label_path = os.path.join(config['target_root'], f'{cvNo}', f"{set_name}_label.npy")  # 标签路径
        int2name_path = os.path.join(config['target_root'], f'{cvNo}', f"{set_name}_int2name.npy")  # 名称映射路径
        self.label = np.load(label_path)  # 加载标签
        self.int2name = np.load(int2name_path)  # 加载名称映射
        # 创建缺失索引
        samplenum = len(self.label)  # 样本数量
        self.maskmatrix = random_mask(3, samplenum, self.mask_rate)  # 创建3维掩码矩阵，根据掩码率随机填充

        # 设置手动合并函数为False
        self.manual_collate_fn = False

    def __getitem__(self, index):
        # 获取mask矩阵中对应索引的mask序列，形状为(3, )
        maskseq = self.maskmatrix[index]

        # 将mask序列转换为LongTensor类型，形状同样为(3, )
        # 当maskrate为1时，maskseq全为1，转换后为[0, 0, 0]
        missing_index = torch.LongTensor(maskseq)

        # 获取对应索引的int2name值
        int2name = self.int2name[index]

        # 原来使用torch.tensor，现在将其转换为float类型
        label = torch.tensor(self.label[index]).float()

        # 获取对应索引的A特征，转换为float类型的张量
        A_feat = torch.tensor(self.all_A[index]).float()

        # 获取对应索引的V特征，转换为float类型的张量
        V_feat = torch.tensor(self.all_V[index]).float()

        # 获取对应索引的L特征，转换为float类型的张量
        L_feat = torch.tensor(self.all_L[index]).float()

        # 返回一个字典，包含所有相关特征和标签信息
        return {
            'A_feat': A_feat,
            'V_feat': V_feat,
            'L_feat': L_feat,
            'label': label,
            'int2name': int2name,
            'missing_index': missing_index
        }

    def __len__(self):
        """
        返回标签(label)的长度。

        返回值:
            int: 标签列表的长度。
        """
        return len(self.label)