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
    随机生成不完整的数据信息，用完整视图数据模拟部分视图数据
    :param view_num: 视图数量
    :param alldata_len: 样本数量
    :param missing_rate: 在论文的第3.2节中定义的缺失率
    :return: Sn [alldata_len, view_num]：返回一个表示视图存在情况的二维数组
    """
    # 计算存在的视图的比例
    one_rate = 1 - missing_rate  # 计算非缺失视图的比例

    # 检查存在的视图比例是否小于或等于1除以视图数量
    if one_rate <= (1 / view_num):
        # 使用OneHotEncoder生成一个只选择一个视图的数组，避免所有输入都为零
        enc = OneHotEncoder(categories=[np.arange(view_num)])  # 初始化编码器
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()  # 转换并获取结果
        return view_preserve

    # 如果存在的视图比例等于1，生成一个所有元素都为1的矩阵，表示所有视图都存在
    if one_rate == 1:
        matrix = randint(1, 2, size=(alldata_len, view_num))  # 生成全1矩阵
        return matrix

    # 如果存在的视图比例在1 / view_num和1之间，生成一个可以有多个视图输入的矩阵
    # 确保至少有一个视图是可用的，因为一些样本可能会重叠，这会增加处理的难度
    error = 1
    while error >= 0.005:
        # 获取初始的view_preserve
        enc = OneHotEncoder(categories=[np.arange(view_num)])
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()

        # 进一步生成one_num样本
        one_num = view_num * alldata_len * one_rate - alldata_len
        ratio = one_num / (view_num * alldata_len)
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(int)
        a = np.sum(((matrix_iter + view_preserve) > 1).astype(int))
        one_num_iter = one_num / (1 - a / one_num)
        ratio = one_num_iter / (view_num * alldata_len)
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(int)
        matrix = ((matrix_iter + view_preserve) > 0).astype(int)
        ratio = np.sum(matrix) / (view_num * alldata_len)
        error = abs(one_rate - ratio)

    return matrix

class CMUMOSEIMissDataset(BaseDataset):
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

        # 添加'cvNo'参数，类型为整数，用于指定交叉验证集编号
        parser.add_argument('--cvNo', type=int, help='选择哪个交叉验证集合')

        # 添加'output_dim'参数，类型为整数，表示数据集中标签的种类数量
        parser.add_argument('--output_dim', type=int, help='该数据集中有多少种标签类型')

        # 添加'norm_method'参数，类型为字符串，可选值为'utt'或'trn'
        # 用于指定如何规范化输入的comparE特征
        parser.add_argument('--norm_method', type=str, choices=['utt', 'trn'], help='如何规范化输入的比较特征')

        # 返回更新后的参数解析器
        return parser


    # 初始化IEMOCAP数据集读取器
    def __init__(self, opt, set_name):
        ''' 初始化IEMOCAP数据集读取器
            参数:
                opt: 包含配置信息的选项对象
                set_name: 数据集名称，可选['trn', 'val', 'tst']
        '''
        # 调用父类构造函数
        super().__init__(opt)

        # 记录和加载基本设置
        cvNo = opt.cvNo  # 获取交叉验证编号
        self.mask_rate = opt.mask_rate  # 获取特征掩码率
        self.set_name = set_name  # 保存数据集名称

        # 获取当前文件的绝对路径
        pwd = os.path.abspath(__file__)
        # 获取当前目录的父目录
        pwd = os.path.dirname(pwd)

        # 加载配置文件
        config = json.load(open(os.path.join(pwd, 'config', 'CMUMOSEI_config.json')))

        # 加载特征数据
        self.all_A = np.load(  # 加载音频特征
            os.path.join(config['feature_root'], 'A', str(opt.cvNo), f'{set_name}.npy'), 'r'
        )
        self.all_V = np.load(  # 加载视频特征
            os.path.join(config['feature_root'], 'V', str(opt.cvNo), f'{set_name}.npy'), 'r'
        )
        self.all_L = np.load(  # 加载语言特征
            os.path.join(config['feature_root'], 'L', str(opt.cvNo), f'{set_name}.npy'), 'r'
        )

        # 加载目标数据
        label_path = os.path.join(config['target_root'], f'{cvNo}', f"{set_name}_label.npy")  # 标签路径
        int2name_path = os.path.join(config['target_root'], f'{cvNo}', f"{set_name}_int2name.npy")  # 整数到名称映射路径
        self.label = np.load(label_path)  # 加载标签
        self.int2name = np.load(int2name_path)  # 加载整数到名称映射

        # 创建缺失索引
        samplenum = len(self.label)  # 样本数量
        self.maskmatrix = random_mask(3, samplenum, self.mask_rate)  # 创建一个3维的随机掩码矩阵，根据mask_rate确定掩码比例

        # 不使用手动collate函数
        self.manual_collate_fn = False


    def __getitem__(self, index):
        # 获取mask矩阵中对应索引的mask序列，形状为(3, )
        maskseq = self.maskmatrix[index]

        # 将mask序列转换为LongTensor类型，形状同样为(3, )
        # 当maskrate为1时，maskseq全为1，转换后为[0,0,0]
        missing_index = torch.LongTensor(maskseq)

        # 获取对应索引的int到名称的映射
        int2name = self.int2name[index]

        # 原始标签为torch.tensor，这里将其转换为float类型
        # label = torch.tensor(self.label[index])  # 原始代码
        label = torch.tensor(self.label[index]).float()

        # 获取对应索引的所有A特征，转换为float类型
        A_feat = torch.tensor(self.all_A[index]).float()

        # 获取对应索引的所有V特征，转换为float类型
        V_feat = torch.tensor(self.all_V[index]).float()

        # 获取对应索引的所有L特征，转换为float类型
        L_feat = torch.tensor(self.all_L[index]).float()

        # 返回一个字典，包含所有特征和相关信息
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
        返回标签(label)的长度，即标签列表的元素数量。
        """
        return len(self.label)
    