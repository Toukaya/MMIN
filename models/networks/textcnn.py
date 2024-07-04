import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, input_dim, emb_size=128, in_channels=1, out_channels=128, kernel_heights=[3, 4, 5], dropout=0.5):
        super().__init__()  # 继承父类的初始化方法
        '''
        构建网络结构：将（卷积-激活函数-卷积-激活函数-卷积-激活函数）的结果拼接，加上最大池化层，然后应用dropout，最后进行转置操作
        '''
        # 定义三个不同高度的卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, (kernel_heights[0], input_dim), stride=1, padding=0)  # 第一个卷积层
        self.conv2 = nn.Conv2d(in_channels, out_channels, (kernel_heights[1], input_dim), stride=1, padding=0)  # 第二个卷积层
        self.conv3 = nn.Conv2d(in_channels, out_channels, (kernel_heights[2], input_dim), stride=1, padding=0)  # 第三个卷积层
        # 应用dropout层
        self.dropout = nn.Dropout(dropout)
        # 定义嵌入层，用于将前面卷积层的输出转换为指定维度的向量
        self.embd = nn.Sequential(
            nn.Linear(len(kernel_heights) * out_channels, emb_size),  # 全连接层，输入通道数乘以卷积层数量，输出emb_size维向量
            nn.ReLU(inplace=True),  # 使用ReLU激活函数
        )

    def conv_block(self, input, conv_layer):  # 定义一个卷积块函数，接收输入和卷积层作为参数
        conv_out = conv_layer(input)  # 通过卷积层对输入进行处理，得到conv_out，其尺寸为 (batch_size, out_channels, dim, 1)
        activation = F.relu(
            conv_out.squeeze(3))  # 应用ReLU激活函数，将conv_out的第3维（通道维度）压缩为1，尺寸变为 (batch_size, out_channels, dim1)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(
            2)  # 对激活结果进行最大池化，池化窗口大小为dim1，然后挤压第2维，得到尺寸为 (batch_size, out_channels) 的输出
        return max_out  # 返回最大池化后的输出

    def forward(self, frame_x):
        # 获取输入帧特征的批次大小、序列长度和特征维度
        batch_size, seq_len, feat_dim = frame_x.size()

        # 将输入数据reshape，以便于卷积层处理
        frame_x = frame_x.view(batch_size, 1, seq_len, feat_dim)

        # 通过第一个卷积块进行处理
        max_out1 = self.conv_block(frame_x, self.conv1)

        # 通过第二个卷积块进行处理
        max_out2 = self.conv_block(frame_x, self.conv2)

        # 通过第三个卷积块进行处理
        max_out3 = self.conv_block(frame_x, self.conv3)

        # 拼接三个卷积块的输出，形成一个更丰富的特征向量
        all_out = torch.cat((max_out1, max_out2, max_out3), 1)

        # 应用Dropout层以防止过拟合
        fc_in = self.dropout(all_out)

        # 通过全连接层（嵌入层）进一步处理特征
        embd = self.embd(fc_in)

        # 返回最终的嵌入表示
        return embd