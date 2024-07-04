import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, fc1_size, output_size, dropout_rate):
        """
        初始化LSTM分类器。

        参数:
            input_size (int): 输入向量的维度。
            hidden_size (int): LSTM隐藏层的单元数量。
            fc1_size (int): 第一个全连接层的节点数。
            output_size (int): 输出层的节点数，即类别数量。
            dropout_rate (float): Dropout比例，用于正则化。
        """
        # 调用父类（nn.Module）的初始化方法
        super(LSTMClassifier, self).__init__()

        # 存储模型参数
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1_size = fc1_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate

        # 定义模块
        # 第一层双向LSTM，batch_first=True表示输入数据的第一维是批次大小
        self.rnn1 = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)

        # 第二层双向LSTM
        self.rnn2 = nn.LSTM(2 * hidden_size, hidden_size, bidirectional=True, batch_first=True)

        # 全连接层1，将LSTM的输出映射到fc1_size
        self.fc1 = nn.Linear(hidden_size * 4, fc1_size)

        # 全连接层2，用于分类
        self.fc2 = nn.Linear(fc1_size, output_size)

        # ReLU激活函数
        self.relu = nn.ReLU()

        # Dropout层，用于正则化
        self.dropout = nn.Dropout(dropout_rate)

        # 层归一化，用于改善LSTM的性能
        self.layer_norm = nn.LayerNorm((hidden_size * 2,))

        # 批量归一化层，用于加速训练和提高模型稳定性
        self.bn = nn.BatchNorm1d(hidden_size * 4)


    def extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm):  # 定义提取特征的函数
        """
        输入参数:
        - sequence: 序列数据
        - lengths: 序列的长度信息
        - rnn1: 第一个循环神经网络实例
        - rnn2: 第二个循环神经网络实例
        - layer_norm: 层归一化操作实例

        函数功能: 使用两个循环神经网络和层归一化处理输入序列，提取最终的特征向量
        """

        # 使用pack_padded_sequence对序列进行打包，以便在RNN中处理不同长度的序列
        packed_sequence = pack_padded_sequence(sequence, lengths, batch_first=True, enforce_sorted=False)

        # 通过第一个循环神经网络(rnn1)处理打包后的序列
        packed_h1, (final_h1, _) = rnn1(packed_sequence)

        # 对RNN1的输出进行解包，并填充回原始的序列长度
        padded_h1, _ = pad_packed_sequence(packed_h1, batch_first=True)

        # 应用层归一化操作到RNN1的输出上
        normed_h1 = layer_norm(padded_h1)

        # 再次打包经过层归一化的序列，以供第二个循环神经网络(rnn2)使用
        packed_normed_h1 = pack_padded_sequence(normed_h1, lengths, batch_first=True, enforce_sorted=False)

        # 通过第二个循环神经网络(rnn2)处理归一化后的序列
        _, (final_h2, _) = rnn2(packed_normed_h1)

        # 返回两个循环神经网络的最终隐藏状态作为特征向量
        return final_h1, final_h2


    # 定义一个名为rnn_flow的方法，输入参数为x和lengths
    def rnn_flow(self, x, lengths):
        # 获取lengths张量的批次大小
        batch_size = lengths.size(0)

        # 使用self.rnn1和self.rnn2两个RNN层，以及self.layer_norm（可能为层归一化）提取特征
        h1, h2 = self.extract_features(x, lengths, self.rnn1, self.rnn2, self.layer_norm)

        # 拼接h1和h2张量，按维度2连接，然后进行转置和reshape操作，以便于后续处理
        h = torch.cat((h1, h2), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        # 应用批归一化操作
        return self.bn(h)


    def mask2length(self, mask):
        """
        根据输入的二维或三维mask计算序列长度

        参数:
        mask: shape为 [batch_size, seq_length, feat_size] 的张量，其中1表示有效元素，0表示填充元素

        返回:
        length: shape为 [batch_size,] 的张量，表示每个样本的有效序列长度
        """
        _mask = torch.mean(mask, dim=-1).long()  # 计算每行（每个样本）的平均值，得到一个二维张量，shape为 [batch_size, seq_len]
        length = torch.sum(_mask, dim=-1)  # 对每行求和，得到每个样本的有效元素数量，即序列长度，shape为 [batch_size,]
        return length


    def forward(self, x, mask):  # 定义前向传播函数，输入为x和掩码mask
        lengths = self.mask2length(mask)  # 根据掩码计算序列长度
        h = self.rnn_flow(x, lengths)  # 使用RNN流模型处理输入x，考虑了不同序列长度
        h = self.fc1(h)  # 通过第一个全连接层（线性变换）
        h = self.dropout(h)  # 应用dropout层以防止过拟合
        h = self.relu(h)  # 使用ReLU激活函数增加非线性
        o = self.fc2(h)  # 通过第二个全连接层，得到最终输出o
        return o, h  # 返回输出o和隐藏状态h


class SimpleClassifier(nn.Module):
    """
    简单分类器的初始化方法。

    参数:
    - embd_size (int): 输入嵌入维度大小。
    - output_dim (int): 输出层的维度大小。
    - dropout (float): Dropout比例，用于正则化防止过拟合。
    """
    def __init__(self, embd_size, output_dim, dropout):
        # 调用父类（nn.Module）的初始化方法
        super(SimpleClassifier, self).__init__()
        self.dropout = dropout
        # 定义一个线性层，将输入嵌入维度转换为输出维度
        self.C = nn.Linear(embd_size, output_dim)
        # 初始化Dropout层，参数为dropout比例
        self.dropout_op = nn.Dropout(dropout)

    def forward(self, x):
        # 如果dropout比例大于0，则对输入x应用dropout操作
        if self.dropout > 0:
            x = self.dropout_op(x)
        # 使用卷积层C对处理后的x进行运算并返回结果
        return self.C(x)

class Identity(nn.Module):
    # 初始化函数，调用父类的初始化方法
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        向前传播方法，接收一个输入x

        参数:
            x: 输入数据

        返回:
            x: 保持不变的输入数据
        """
        return x
    
class FcClassifier(nn.Module):
    def __init__(self, input_dim, layers, output_dim, dropout=0.3, use_bn=False):
        """
        完全连接的分类器
        参数:
        --------------------------
        input_dim: 输入特征维度
        layers: [x1, x2, x3] 将创建3层，每层分别有x1, x2, x3个隐藏节点。
        output_dim: 输出特征维度
        activation: 激活函数（未在代码中使用，可能是遗漏）
        dropout: 阶段性丢弃率
        """
        # 调用父类初始化方法
        super().__init__()

        # 初始化所有层的列表
        self.all_layers = []

        # 遍历隐藏层设置
        for i in range(0, len(layers)):
            # 添加线性层
            self.all_layers.append(nn.Linear(input_dim, layers[i]))
            # 添加ReLU激活函数
            self.all_layers.append(nn.ReLU())
            # 如果使用批量归一化
            if use_bn:
                self.all_layers.append(nn.BatchNorm1d(layers[i]))
            # 如果dropout大于0，添加Dropout层
            if dropout > 0:
                self.all_layers.append(nn.Dropout(dropout))
            # 更新输入维度
            input_dim = layers[i]

        # 如果没有隐藏层，添加一个保持输入不变的Identity层
        if len(layers) == 0:
            layers.append(input_dim)
            self.all_layers.append(Identity())

        # 添加输出线性层
        self.fc_out = nn.Linear(layers[-1], output_dim)
        # 创建顺序模块，包含所有层
        self.module = nn.Sequential(*self.all_layers)


    def forward(self, x):
        # 使用self.module对输入x进行处理，获取特征
        feat = self.module(x)
        # 将得到的特征通过self.fc_out全连接层，得到输出
        out = self.fc_out(feat)
        # 返回最终输出和中间特征结果
        return out, feat


class EF_model_AL(nn.Module):
    def __init__(self, fc_classifier, lstm_classifier, out_dim_a, out_dim_v, fusion_size, num_class, dropout):
        ''' 早期融合模型分类器
            参数:
            --------------------------
            fc_classifier: 声学分类器
            lstm_classifier: 词汇分类器
            out_dim_a: fc_classifier 输出维度
            out_dim_v: lstm_classifier 输出维度
            fusion_size: 融合模型的输出大小
            num_class: 类别数量
            dropout: dropout 率
        '''
        super(EF_model_AL, self).__init__()

        # 初始化声学和词汇分类器
        self.fc_classifier = fc_classifier
        self.lstm_classifier = lstm_classifier

        # 计算总输出维度
        self.out_dim = out_dim_a + out_dim_v

        # 应用 dropout 层
        self.dropout = nn.Dropout(dropout)

        # 设置类别数量
        self.num_class = num_class

        # 设置融合层大小
        self.fusion_size = fusion_size

        # 替换原有的多层线性结构，使用两个独立的线性层
        self.out1 = nn.Linear(self.out_dim, self.fusion_size)
        self.relu = nn.ReLU()  # 使用 ReLU 激活函数
        self.out2 = nn.Linear(self.fusion_size, self.num_class)  # 输出层，用于分类


    def forward(self, A_feat, L_feat, L_mask):
        # 使用全连接层进行A部分特征的分类，返回索引和输出
        _, A_out = self.fc_classifier(A_feat)

        # 使用LSTM层对L部分特征进行分类，考虑L_mask，返回索引和输出
        _, L_out = self.lstm_classifier(L_feat, L_mask)

        # 沿着特征维度将A和L的输出拼接
        feat = torch.cat([A_out, L_out], dim=-1)

        # 应用dropout层以减少过拟合
        feat = self.dropout(feat)

        # 使用ReLU激活函数增强非线性
        feat = self.relu(self.out1(feat))

        # 再次应用dropout，然后通过最后一层全连接层
        out = self.out2(self.dropout(feat))

        # 返回最终输出和中间特征向量
        return out, feat
    

class MaxPoolFc(nn.Module):
    def __init__(self, hidden_size, num_class=4):
        # 调用父类的初始化方法
        super(MaxPoolFc, self).__init__()

        # 定义隐藏层大小
        self.hidden_size = hidden_size

        # 创建全连接层序列，包含一个线性层和一个ReLU激活函数
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, num_class),  # 线性层，输入维度为hidden_size，输出维度为num_class
            nn.ReLU()  # ReLU激活函数
        )


    def forward(self, x):
        '''
        输入x的形状为 [batch_size, seq_len, hidden_size]
        '''
        # 获取批次大小、序列长度和隐藏层大小
        batch_size, seq_len, hidden_size = x.size()

        # 将输入x的形状从[batch_size, seq_len, hidden_size]调整为[batch_size, hidden_size, seq_len]
        x = x.view(batch_size, hidden_size, seq_len)

        # 打印调整后x的尺寸，用于调试
        # print(x.size())

        # 使用最大池化操作，池化窗口大小为序列长度，获取每个批次的最大值
        out = torch.max_pool1d(x, kernel_size=seq_len)

        # 压缩最大池化后的维度，从[batch_size, hidden_size, 1]变为[batch_size, hidden_size]
        out = out.squeeze()

        # 通过全连接层（线性层）进一步处理输出
        out = self.fc(out)

        # 返回处理后的输出
        return out

if __name__ == '__main__':
    a = FcClassifier(256, [128], 4)
    print(a)