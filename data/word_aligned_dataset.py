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
        修改命令行选项
        :param parser: 命令行解析器
        :param isTrain: 是否为训练模式
        :return: 修改后的命令行解析器
        """
        parser.add_argument('--cvNo', type=int, help='which cross validation set')
        parser.add_argument('--A_type', type=str, help='which audio feat to use')
        parser.add_argument('--V_type', type=str, help='which visual feat to use')
        parser.add_argument('--L_type', type=str, help='which lexical feat to use')
        parser.add_argument('--output_dim', type=int, help='how many label types in this dataset')
        parser.add_argument('--norm_method', type=str, choices=['utt', 'trn'], help='how to normalize input comparE feature')
        return parser
    
    def __init__(self, opt, set_name):
        ''' IEMOCAP dataset reader
            set_name in ['trn', 'val', 'tst']
                    初始化函数
        :param opt: 命令行选项
        :param set_name: 数据集名称
        '''
        super().__init__(opt)

        # record & load basic settings 
        cvNo = opt.cvNo
        self.set_name = set_name
        pwd = os.path.abspath(__file__)
        pwd = os.path.dirname(pwd)
        config = json.load(open(os.path.join(pwd, 'config', 'IEMOCAP_config.json')))
        self.norm_method = opt.norm_method
        # load feature
        self.A_type = opt.A_type
        self.all_A = h5py.File(os.path.join(config['feature_root'], 'aligned', 'A', f'aligned_{self.A_type}.h5'), 'r')
        if self.A_type == 'comparE':
            self.mean_std = h5py.File(os.path.join(config['feature_root'], 'aligned', 'A', 'aligned_comparE_mean_std.h5'), 'r')
            self.mean = torch.from_numpy(self.mean_std[str(cvNo)]['mean'][()]).unsqueeze(0).float()
            self.std = torch.from_numpy(self.mean_std[str(cvNo)]['std'][()]).unsqueeze(0).float()
        self.V_type = opt.V_type
        self.all_V = h5py.File(os.path.join(config['feature_root'], 'aligned', 'V', f'aligned_{self.V_type}.h5'), 'r')
        self.L_type = opt.L_type
        self.all_L = h5py.File(os.path.join(config['feature_root'], 'aligned', 'L', f'aligned_{self.L_type}.h5'), 'r')
        # load target
        label_path = os.path.join(config['target_root'], f'{cvNo}', f"{set_name}_label.npy")
        int2name_path = os.path.join(config['target_root'], f'{cvNo}', f"{set_name}_int2name.npy")
        self.label = np.load(label_path)
        self.label = np.argmax(self.label, axis=1)
        self.int2name = np.load(int2name_path)
        self.int2name = [x[0].decode() for x in self.int2name]
        if 'Ses03M_impro03_M001' in self.int2name:
            idx = self.int2name.index('Ses03M_impro03_M001')
            self.int2name.pop(idx)
            self.label = np.delete(self.label, idx, axis=0)
        self.manual_collate_fn = True

    def __getitem__(self, index):
        """
        根据索引获取数据集中的一个样本，
        包括其音频特征（A_feat）、视觉特征（V_feat）和词汇特征（L_feat）以及标签。
        这些特征都是从对应的数据集中获取的，然后转换为PyTorch的张量。
        如果音频特征的类型是comparE，则会对其进行归一化处理。
        :param index: 索引
        :return: 包含音频、视觉和词汇特征以及标签的样本
        """
        int2name = self.int2name[index]
        label = torch.tensor(self.label[index])
        # process A_feat
        A_feat = torch.from_numpy(self.all_A[int2name][()]).float()
        if self.A_type == 'comparE':
            A_feat = self.normalize_on_utt(A_feat) if self.norm_method == 'utt' else self.normalize_on_trn(A_feat)
        # process V_feat 
        V_feat = torch.from_numpy(self.all_V[int2name][()]).float()
        # proveee L_feat
        L_feat = torch.from_numpy(self.all_L[int2name][()]).float()
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
        return len(self.label)
    
    def normalize_on_utt(self, features):
        """
        单个语句级别上进行归一化
        """
        mean_f = torch.mean(features, dim=0).unsqueeze(0).float()
        std_f = torch.std(features, dim=0).unsqueeze(0).float()
        std_f[std_f == 0.0] = 1.0
        features = (features - mean_f) / std_f
        return features
    
    def normalize_on_trn(self, features):
        """
        在整个训练数据集级别上进行归一化
        """
        features = (features - self.mean) / self.std
        return features

    def collate_fn(self, batch):
        """
        ：这个方法用于将一批样本组合成一个批次，包括对特征的填充和标签的组合。
        这个方法首先从批次中的每个样本中提取出音频特征、视觉特征和词汇特征，
        然后使用pad_sequence函数对这些特征进行填充，使它们的长度一致。
        最后，将这些特征以及标签和名称组合成一个字典，作为该方法的返回值。
        """
        A = [sample['A_feat'] for sample in batch]
        V = [sample['V_feat'] for sample in batch]
        L = [sample['L_feat'] for sample in batch]
        lengths = torch.tensor([len(sample) for sample in A]).long()
        A = pad_sequence(A, batch_first=True, padding_value=0)
        V = pad_sequence(V, batch_first=True, padding_value=0)
        L = pad_sequence(L, batch_first=True, padding_value=0)
        label = torch.tensor([sample['label'] for sample in batch])
        int2name = [sample['int2name'] for sample in batch]
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
    