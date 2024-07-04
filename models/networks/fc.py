import torch
import torch.nn as nn

class FcEncoder(nn.Module):
    def __init__(self, input_dim, layers, dropout=0.5, use_bn=False):
        ''' 初始化全连接分类器
        包含fc层、ReLU激活、批量归一化（可选）和Dropout（可选）
        最后一层不使用ReLU，直接进行分类

        参数:
        --------------------------
        input_dim: 输入特征维度
        layers: [x1, x2, x3] 表示创建3层，每层隐藏节点分别为x1, x2, x3
        dropout: Dropout概率，默认0.5
        use_bn: 是否使用批量归一化，默认False
        '''
        # 调用父类初始化方法
        super().__init__()

        # 初始化所有层的列表
        self.all_layers = []

        # 遍历每一层
        for i in range(0, len(layers)):
            # 添加全连接层
            self.all_layers.append(nn.Linear(input_dim, layers[i]))
            # 添加ReLU激活层
            self.all_layers.append(nn.ReLU())
            # 如果使用批量归一化，添加该层
            if use_bn:
                self.all_layers.append(nn.BatchNorm1d(layers[i]))
            # 如果dropout大于0，添加Dropout层
            if dropout > 0:
                self.all_layers.append(nn.Dropout(dropout))
            # 更新输入维度为当前层的输出维度
            input_dim = layers[i]

        # 将所有层组合成一个Sequential模块
        self.module = nn.Sequential(*self.all_layers)

    def forward(self, x):
        # 将各个层组合成一个完整的模块
        feat = self.module(x)  # 通过模块处理输入x，得到特征
        return feat  # 返回处理后的特征