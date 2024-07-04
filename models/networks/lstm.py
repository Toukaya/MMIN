import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMEncoder(nn.Module):
    ''' one directional LSTM encoder
    '''

    # 初始化LSTM编码器类，继承自nn.Module
    def __init__(self, input_size, hidden_size, embd_method='last'):
        # 调用父类初始化方法
        super(LSTMEncoder, self).__init__()

        # 设置输入尺寸和隐藏层尺寸
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 创建一个LSTM模块，参数batch_first=True表示输入数据的第一维是批次维度
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)

        # 检查嵌入方法是否合法，只接受'maxpool', 'attention', 'last'
        assert embd_method in ['maxpool', 'attention', 'last']
        self.embd_method = embd_method

        # 如果选择注意力机制作为嵌入方法
        if self.embd_method == 'attention':
            # 初始化注意力向量权重参数
            self.attention_vector_weight = nn.Parameter(torch.Tensor(hidden_size, 1))

            # 构建注意力层，包括线性变换和Tanh激活函数
            self.attention_layer = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),  # 线性层
                nn.Tanh()  # Tanh激活函数
            )

            # 初始化softmax层用于计算注意力权重
            self.softmax = nn.Softmax(dim=-1)  # 沿着最后一个维度进行softmax运算

    def embd_attention(self, r_out, h_n):
        '''
        参考这篇博客的实现:
        https://blog.csdn.net/dendi_hust/article/details/94435919
        https://blog.csdn.net/fkyyly/article/details/82501126
        论文：Hierarchical Attention Networks for Document Classification
        formulation:  lstm_output*softmax(u * tanh(W*lstm_output + Bias)
        W and Bias 是映射函数，其中 Bias 可加可不加
        u 是 attention vector 大小等于 hidden size
        '''
        # 应用注意力层得到序列的隐藏表示
        hidden_reps = self.attention_layer(r_out)  # [batch_size, seq_len, hidden_size]

        # 计算注意力权重，使用注意力向量与隐藏表示的点积
        atten_weight = (hidden_reps @ self.attention_vector_weight)  # [batch_size, seq_len, 1]

        # 应用softmax函数归一化注意力权重
        atten_weight = self.softmax(atten_weight)  # [batch_size, seq_len, 1]

        # 根据注意力权重对序列的隐藏表示进行加权求和，得到句子向量
        sentence_vector = torch.sum(r_out * atten_weight, dim=1)  # [batch_size, hidden_size]

        # 返回句子向量
        return sentence_vector

    def embd_maxpool(self, r_out, h_n):
        """
        使用最大池化操作对嵌入层的输出进行处理。

        参数:
        - r_out (Tensor): 形状为 [batch_size, seq_len, hidden_size] 的张量，表示循环层的输出。
        - h_n (Tensor): 当前未使用，仅用于保持接口一致性。

        返回:
        - embd (Tensor): 经过最大池化的张量，形状为 [batch_size, hidden_size]。
        """
        # 将r_out转置为形状 [batch_size, hidden_size, seq_len]，以便进行最大池化操作
        in_feat = r_out.transpose(1, 2)

        # 对转置后的特征进行一维最大池化，池化窗口大小等于序列长度
        embd = F.max_pool1d(in_feat, in_feat.size(2), in_feat.size(2))

        # 压缩维度，得到形状为 [batch_size, hidden_size] 的张量
        return embd.squeeze()

    def embd_last(self, r_out, h_n):
        # 只适用于单层和单向的情况
        # 返回最后一个时间步的隐藏状态，去掉不必要的维度
        return h_n.squeeze()

    def forward(self, x):
        """
        此函数执行前向传播操作，输入x是一个序列数据。

        r_out 的形状：seq_len（序列长度），batch（批次大小），num_directions * hidden_size（双向RNN的隐藏层大小）
        hn 和 hc 的形状：num_layers（层数）* num_directions，batch，hidden_size（每个方向的隐藏层大小）
        """
        # 使用self.rnn进行RNN处理，返回输出r_out和最后两层的隐藏状态(h_n, h_c)
        r_out, (h_n, h_c) = self.rnn(x)

        # 根据self.embd_method属性选择相应的嵌入方法，并用它来处理r_out和h_n
        embd = getattr(self, 'embd_' + self.embd_method)(r_out, h_n)

        # 返回处理后的嵌入结果embd
        return embd