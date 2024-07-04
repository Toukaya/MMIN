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
    :return: Sn [alldata_len, view_num]
    """
    # 计算存在的视图的比例
    one_rate = 1 - missing_rate  # missing_rate: 0.8; one_rate: 0.2

    # 如果存在的视图的比例小于或等于1除以视图数量
    if one_rate <= (1 / view_num):
        # 使用OneHotEncoder生成一个只选择一个视图的数组，避免所有输入都为零
        enc = OneHotEncoder(categories=[np.arange(view_num)])
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
        return view_preserve

        # 如果存在的视图的比例等于1，生成一个所有元素都为1的矩阵，表示所有视图都存在
    if one_rate == 1:
        matrix = randint(1, 2, size=(alldata_len, view_num))
        return matrix

    # 如果存在的视图的比例在1 / view_num和1之间，生成一个可以有多个视图输入的矩阵
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
        parser.add_argument('--cvNo', type=int, help='which cross validation set')
        parser.add_argument('--output_dim', type=int, help='how many label types in this dataset')
        parser.add_argument('--norm_method', type=str, choices=['utt', 'trn'], help='how to normalize input comparE feature')
        return parser
    
    def __init__(self, opt, set_name):
        ''' IEMOCAP dataset reader
            set_name in ['trn', 'val', 'tst']
        '''
        super().__init__(opt)

        # record & load basic settings 
        cvNo = opt.cvNo
        self.mask_rate = opt.mask_rate
        self.set_name = set_name
        pwd = os.path.abspath(__file__)
        pwd = os.path.dirname(pwd)
        config = json.load(open(os.path.join(pwd, 'config', 'CMUMOSEI_config.json')))
        # load feature
        self.all_A = np.load(os.path.join(config['feature_root'], 'A', str(opt.cvNo), f'{set_name}.npy'), 'r')
        self.all_V = np.load(os.path.join(config['feature_root'], 'V', str(opt.cvNo), f'{set_name}.npy'), 'r')
        self.all_L = np.load(os.path.join(config['feature_root'], 'L', str(opt.cvNo), f'{set_name}.npy'), 'r')
        # load target
        label_path = os.path.join(config['target_root'], f'{cvNo}', f"{set_name}_label.npy")
        int2name_path = os.path.join(config['target_root'], f'{cvNo}', f"{set_name}_int2name.npy")
        self.label = np.load(label_path)
        self.int2name = np.load(int2name_path)
        # make missing index
        samplenum = len(self.label)
        self.maskmatrix = random_mask(3, samplenum, self.mask_rate) # [samplenum, view_num]

        self.manual_collate_fn = False

    def __getitem__(self, index):
        
        maskseq = self.maskmatrix[index] # (3, )
        missing_index = torch.LongTensor(maskseq) # (3, ) [1,1,1]; maskrate=1=>[0,0,0]

        int2name = self.int2name[index]
        ###############
        # label = torch.tensor(self.label[index])
        label = torch.tensor(self.label[index]).float()
        ###############
        A_feat = torch.tensor(self.all_A[index]).float()
        V_feat = torch.tensor(self.all_V[index]).float()
        L_feat = torch.tensor(self.all_L[index]).float()
        return {
            'A_feat': A_feat, 
            'V_feat': V_feat,
            'L_feat': L_feat,
            'label': label,
            'int2name': int2name,
            'missing_index': missing_index
        }
    
    def __len__(self):
        return len(self.label)
    