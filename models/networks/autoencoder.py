import torch
import torch.nn as nn
import random
import copy
import torch.nn.functional as F

class BaseAutoencoder(nn.Module):
    def __init__(self):  # 初始化方法，用于构建对象时执行
        super().__init__()  # 调用父类的初始化方法，确保继承链上的初始化正确进行

        # 定义编码器（Encoder）网络，将128维输入转化为32维特征
        self.encoder = nn.Sequential(  # 使用Sequential容器组织多个层
            nn.Linear(128, 64),  # 全连接层，将128维输入映射到64维
            nn.ReLU(),  # ReLU激活函数，引入非线性
            nn.Linear(64, 32),  # 再次全连接层，将64维特征压缩到32维
            nn.ReLU()  # 应用ReLU激活函数
        )

        # 定义解码器（Decoder）网络，将32维特征还原回128维输出
        self.decoder = nn.Sequential(  # 创建另一个Sequential容器
            nn.Linear(32, 64),  # 全连接层，将32维特征扩展到64维
            nn.ReLU(),  # 应用ReLU激活函数
            nn.Linear(64, 128),  # 最后一层全连接层，将64维特征恢复到原始的128维
            nn.ReLU()  # 使用ReLU激活函数
        )


    def forward(self, x):
        # 使用编码器将输入数据x转化为潜在向量
        latent_vector = self.encoder(x)

        # 使用解码器将潜在向量转化为重构的输出
        reconstructed = self.decoder(latent_vector)

        # 返回重构后的数据和潜在向量
        return reconstructed, latent_vector


class LSTMAutoencoder(nn.Module):
    # 初始化函数，用于设置模型参数
    def __init__(self, opt):
        # 调用父类的初始化方法
        super().__init__()

        # 获取输入尺寸
        self.input_size = opt.input_size
        # 获取隐藏层尺寸
        self.hidden_size = opt.hidden_size
        # 获取嵌入层尺寸
        self.embedding_size = opt.embedding_size
        # 设置错误教师率，即使用标签代替前一时间步的输出
        self.false_teacher_rate = opt.false_teacher_rate

        # 再次调用父类的初始化方法，可能是为了确保所有需要的属性都被设置
        super().__init__()

        # 初始化编码器，LSTM单元，输入和隐藏层尺寸分别为input_size和hidden_size
        self.encoder = nn.LSTMCell(self.input_size, self.hidden_size)
        # 初始化编码器全连接层，将隐藏层输出映射到embedding_size维度
        self.enc_fc = nn.Linear(self.hidden_size, self.embedding_size)

        # 初始化解码器，LSTM单元，输入尺寸为hidden_size + input_size，输出尺寸为input_size
        self.decoder = nn.LSTMCell(self.hidden_size + self.input_size, self.input_size)
        # 初始化解码器全连接层，将embedding_size维度的输入映射回hidden_size维度
        self.dec_fc = nn.Linear(self.embedding_size, self.hidden_size)
        # 使用ReLU激活函数
        self.relu = nn.ReLU()


    def forward(self, x):
        ''' 输入x的尺寸为 [batch, timestamp, dim]
        '''
        # 获取时间戳维度的大小
        # inverse_range = range(timestamp_size-1, -1, -1)
        # 反转时间序列
        # inverse_x = x[:, inverse_range, :]

        # 初始化输出列表
        outputs = []

        # 初始化编码器的隐藏状态
        o_t_enc = torch.zeros(x.size(0), self.hidden_size).cuda()
        h_t_enc = torch.zeros(x.size(0), self.hidden_size).cuda()

        # 对输入序列进行分块处理
        for i, input_t in enumerate(x.chunk(x.size(1), dim=1)):
            # 压缩时间步维度
            input_t = input_t.squeeze(1)
            # 运行编码器
            o_t_enc, h_t_enc = self.encoder(input_t, (o_t_enc, h_t_enc))

        # 应用全连接层并激活
        embd = self.relu(self.enc_fc(h_t_enc))
        # 解码器的初始隐藏状态
        dec_first_hidden = self.relu(self.dec_fc(embd))
        # 初始化解码器输入为零向量
        dec_first_zeros = torch.zeros(x.size(0), self.input_size).cuda()
        # 组合解码器的初始隐藏状态和零向量
        dec_input = torch.cat((dec_first_hidden, dec_first_zeros), dim=1)

        # 遍历时间步
        for i in range(x.size(1)):
            # 运行解码器
            o_t_dec, h_t_dec = self.decoder(dec_input, (o_t_dec, h_t_dec))
            # 在训练模式下，以false_teacher_rate的概率使用真实输入
            if self.training and random.random() < self.false_teacher_rate:
                dec_input = torch.cat((dec_first_hidden, x[:, -i - 1, :]), dim=1)
            else:
                # 否则使用解码器的隐藏状态
                dec_input = torch.cat((dec_first_hidden, h_t_dec), dim=1)
            # 将解码器的隐藏状态添加到输出列表
            outputs.append(h_t_dec)

        # 反转输出列表
        outputs.reverse()
        # 将输出列表堆叠成张量
        outputs = torch.stack(outputs, 1)
        # 输出的形状打印（仅用于调试）
        # print(outputs.shape)
        # 返回解码器的输出和嵌入向量
        return outputs, embd


class ResidualAE(nn.Module):
    """
    初始化ResidualAE类，该类继承自nn.Module。
    参数：
        layers: 每个编码器和解码器中的层的数量。
        n_blocks: 重复的残差块的数量。
        input_dim: 输入数据的维度。
        dropout: Dropout比例，默认为0.5。
        use_bn: 是否使用批量归一化，默认为False。
    """

    def __init__(self, layers, n_blocks, input_dim, dropout=0.5, use_bn=False):
        # 调用父类（nn.Module）的初始化方法
        super(ResidualAE, self).__init__()

        # 存储是否使用批量归一化和dropout比例
        self.use_bn = use_bn
        self.dropout = dropout

        # 存储残差块的数量和输入维度
        self.n_blocks = n_blocks
        self.input_dim = input_dim

        # 定义过渡层，包含两个全连接层，中间有一个ReLU激活函数
        self.transition = nn.Sequential(
            nn.Linear(input_dim, input_dim),  # 第一个全连接层
            nn.ReLU(),  # ReLU激活函数
            nn.Linear(input_dim, input_dim)  # 第二个全连接层
        )

        # 遍历n_blocks，为每个残差块创建编码器和解码器
        for i in range(n_blocks):
            # 使用get_encoder方法创建并设置属性'encoder_' + str(i)
            setattr(self, 'encoder_' + str(i), self.get_encoder(layers))

            # 使用get_decoder方法创建并设置属性'decoder_' + str(i)
            setattr(self, 'decoder_' + str(i), self.get_decoder(layers))


    def get_encoder(self, layers):
        """
        根据传入的层结构列表创建编码器网络。

        参数:
        - layers: list, 包含每层神经元数量的列表，定义了编码器的结构。

        返回:
        - nn.Sequential, 构建好的编码器网络模型。
        """

        all_layers = []  # 初始化一个空列表，用于存储构建的网络层
        input_dim = self.input_dim  # 获取输入维度

        # 遍历层结构列表
        for i in range(0, len(layers)):
            # 添加全连接层（线性层）
            all_layers.append(nn.Linear(input_dim, layers[i]))

            # 添加激活函数 LeakyReLU
            all_layers.append(nn.LeakyReLU())

            # 如果使用批量归一化，则添加该层
            if self.use_bn:
                all_layers.append(nn.BatchNorm1d(layers[i]))

            # 如果 dropout 概率大于0，则添加 Dropout 层
            if self.dropout > 0:
                all_layers.append(nn.Dropout(self.dropout))

            # 更新输入维度
            input_dim = layers[i]

        # 删除最后一层的激活函数
        decline_num = 1 + int(self.use_bn) + int(self.dropout > 0)
        all_layers = all_layers[:-decline_num]  # 移除指定数量的层

        # 返回构建好的顺序容器模型
        return nn.Sequential(*all_layers)


    def get_decoder(self, layers):
        # 初始化一个空列表，用于存储解码器的所有层
        all_layers = []

        # 深度复制输入的layers列表，并将其反转，以便从输出维度开始构建解码器
        decoder_layer = copy.deepcopy(layers)
        decoder_layer.reverse()

        # 在解码器层列表的末尾添加输入维度
        decoder_layer.append(self.input_dim)

        # 遍历解码器层，从输出层到输入层
        for i in range(0, len(decoder_layer) - 2):
            # 添加线性变换层
            all_layers.append(nn.Linear(decoder_layer[i], decoder_layer[i + 1]))
            # 添加ReLU激活函数
            all_layers.append(nn.ReLU())

            # 如果使用批量归一化，则添加该层
            if self.use_bn:
                all_layers.append(nn.BatchNorm1d(decoder_layer[i]))

            # 如果dropout比例大于0，则添加Dropout层
            if self.dropout > 0:
                all_layers.append(nn.Dropout(self.dropout))

        # 添加最后一层线性变换层，从倒数第二层到最后一层
        all_layers.append(nn.Linear(decoder_layer[-2], decoder_layer[-1]))

        # 将所有层组合成一个Sequential模块并返回
        return nn.Sequential(*all_layers)


    def forward(self, x):
        # 输入变量x
        x_in = x
        # 初始化输出变量x_out，用零填充的克隆版本
        x_out = x.clone().fill_(0)
        # 存储中间层潜变量的列表
        latents = []

        # 遍历自编码器块的数量
        for i in range(self.n_blocks):
            # 获取当前层的编码器
            encoder = getattr(self, 'encoder_' + str(i))
            # 获取当前层的解码器
            decoder = getattr(self, 'decoder_' + str(i))

            # 将输入x_in与上一步的输出x_out相加
            x_in = x_in + x_out
            # 通过编码器获取当前层的潜变量
            latent = encoder(x_in)
            # 通过解码器恢复潜变量得到输出x_out
            x_out = decoder(latent)
            # 将潜变量添加到列表中
            latents.append(latent)

        # 沿着指定维度堆叠所有潜变量
        latents = torch.cat(latents, dim=-1)

        # 通过过渡层处理输入和输出的组合
        transition_output = self.transition(x_in + x_out)

        # 返回经过过渡层的输出和所有潜变量
        return transition_output, latents

class ResidualUnetAE(nn.Module):
    # 初始化ResidualUnetAE类
    def __init__(self, layers, n_blocks, input_dim, dropout=0.5, use_bn=False, fusion='concat'):
        """
        Unet结构是对称的，因此只输入一半的层数即可
        - 如果layers为[128, 64, 32]，对于'add'融合方式，连接为：[(input_dim, 128), (128, 64), (64, 32), (32, 64), (64, 128), (128, input_dim)]
        - 对于'concat'融合方式，连接为：[(input_dim, 128), (128, 64), (64, 32), (32, 64), (128, 128), (256, input_dim)]
        """
        # 调用父类初始化方法
        super(ResidualUnetAE, self).__init__()

        # 是否使用批量归一化
        self.use_bn = use_bn
        # 随机失活概率
        self.dropout = dropout
        # 块的数量
        self.n_blocks = n_blocks
        # 输入维度
        self.input_dim = input_dim
        # 层的列表
        self.layers = layers
        # 融合方式
        self.fusion = fusion

        # 根据融合方式设置扩展数量
        if self.fusion == 'concat':
            self.expand_num = 2
        elif self.fusion == 'add':
            self.expand_num = 1
        else:
            # 只支持'concat'和'add'两种融合方式
            raise NotImplementedError('Only concat and add are available')

        # 初始化编码器和解码器
        # for循环中使用setattr动态添加属性，如：self.encoder_0, self.decoder_0等
        for i in range(self.n_blocks):
            setattr(self, 'encoder_' + str(i), self.get_encoder(layers))
            setattr(self, 'decoder_' + str(i), self.get_decoder(layers))


    def get_encoder(self, layers):
        """
        创建编码器网络结构，由多个线性层、ReLU激活函数、批量归一化和Dropout层组成。

        参数：
        - layers：列表，表示每个编码层的神经元数量。

        返回：
        - encoder：nn.Sequential对象，构建好的编码器模型。
        """

        encoder = []  # 初始化一个空列表，用于存储编码器的各层
        input_dim = self.input_dim  # 获取输入维度

        # 遍历layers列表，为每个层创建并添加组件
        for i in range(0, len(layers)):
            layer = []  # 初始化当前层的组件列表

            # 添加线性层，输入维度为input_dim，输出维度为layers[i]
            layer.append(nn.Linear(input_dim, layers[i]))

            # 添加ReLU激活函数
            layer.append(nn.ReLU())

            # 如果使用批量归一化，添加该层
            if self.use_bn:
                layer.append(nn.BatchNorm1d(layers[i]))

            # 如果dropout比例大于0，添加Dropout层
            if self.dropout > 0:
                layer.append(nn.Dropout(self.dropout))

            # 将当前层的组件封装为Sequential模块
            layer = nn.Sequential(*layer)

            # 将当前层添加到编码器列表中，并更新输入维度
            encoder.append(layer)
            input_dim = layers[i]

        # 将所有编码器层封装为一个Sequential模块
        encoder = nn.Sequential(*encoder)

        # 返回构建好的编码器模型
        return encoder


    def get_decoder(self, layers):
        """
        创建解码器网络结构，由多个线性层和可选的批量归一化、ReLU激活和Dropout层组成。

        参数:
        - layers: list, 包含不同层的神经元数量列表，从大到小排列

        返回:
        - decoder: nn.Sequential, 构建好的解码器模型
        """

        # 初始化解码器列表
        decoder = []

        # 第一层不需要融合输出
        first_layer = []
        # 添加一个线性层，将上一层的神经元数映射到下一层
        first_layer.append(nn.Linear(layers[-1], layers[-2]))
        # 如果使用批量归一化，则添加该层
        if self.use_bn:
            first_layer.append(nn.BatchNorm1d(layers[-1] * self.expand_num))
        # 如果dropout比例大于0，添加Dropout层
        if self.dropout > 0:
            first_layer.append(nn.Dropout(self.dropout))
        # 将第一层的层放入Sequential容器
        decoder.append(nn.Sequential(*first_layer))

        # 遍历从倒数第二层到第一层（不包括第一层）
        for i in range(len(layers) - 2, 0, -1):
            # 初始化当前层的层列表
            layer = []
            # 添加线性层，将当前层的神经元数映射到下一层
            layer.append(nn.Linear(layers[i] * self.expand_num, layers[i - 1]))
            # 添加ReLU激活函数
            layer.append(nn.ReLU())
            # 如果使用批量归一化，添加该层
            if self.use_bn:
                layer.append(nn.BatchNorm1d(layers[i] * self.expand_num))
            # 如果dropout比例大于0，添加Dropout层
            if self.dropout > 0:
                layer.append(nn.Dropout(self.dropout))
            # 将当前层的层放入Sequential容器
            layer = nn.Sequential(*layer)
            # 将当前层添加到解码器列表
            decoder.append(layer)

        # 最后一层，将第一层的神经元数映射到输入维度，并添加ReLU激活
        decoder.append(
            nn.Sequential(
                nn.Linear(layers[0] * self.expand_num, self.input_dim),
                nn.ReLU()
            )
        )

        # 将所有层放入Sequential容器并返回解码器
        decoder = nn.Sequential(*decoder)
        return decoder


    # 定义一个前向传播的自编码器块函数
    def forward_AE_block(self, x, block_num):
        # 获取对应编号的编码器和解码器
        encoder = getattr(self, 'encoder_' + str(block_num))
        decoder = getattr(self, 'decoder_' + str(block_num))

        # 创建一个字典存储每个层的编码器输出
        encoder_out_lookup = {}

        # 输入到编码器
        x_in = x

        # 遍历编码器的每一层
        for i in range(len(self.layers)):
            # 编码器层的输出
            x_out = encoder[i](x_in)
            # 存储当前层的编码器输出
            encoder_out_lookup[i] = x_out.clone()
            # 更新输入到下一层
            x_in = x_out

        # 遍历解码器的每一层
        for i in range(len(self.layers)):
            # 计算对应编码器输出的索引
            encoder_out_num = len(self.layers) - 1 - i
            # 获取该索引的编码器输出
            encoder_out = encoder_out_lookup[encoder_out_num]

            # 如果是第一层，直接跳过
            if i == 0:
                pass
            # 若融合方式为拼接，则将上一层解码器和当前编码器输出拼接
            elif self.fusion == 'concat':
                x_in = torch.cat([x_in, encoder_out], dim=-1)
            # 若融合方式为相加，则将上一层解码器和当前编码器输出相加
            elif self.fusion == 'add':
                x_in = x_in + encoder_out

            # 解码器层的输出
            x_out = decoder[i](x_in)
            # 更新输入到下一层
            x_in = x_out

        # # 打印解码器最后一层和输入输出的形状（调试用）
        # print(decoder[-1])
        # print(x_in.shape)
        # print(x_out.shape)

        # 返回解码器最后一层的输出
        return x_out


    def forward(self, x):
        # 输入信号
        x_in = x
        # 初始化输出信号，用0填充的副本
        x_out = x.clone().fill_(0)
        # 存储每个编码器块输出的字典
        output = {}
        # 遍历所有编码器块
        for i in range(self.n_blocks):
            # 将输入信号与当前输出信号相加
            x_in = x_in + x_out
            # 通过当前编码器块进行前向传播，并获取输出
            x_out = self.forward_AE_block(x_in, i)
            # 将块的输出保存到字典中
            output[i] = x_out.clone()

        # 返回最终输出和所有块的输出字典
        return x_out, output

class SimpleFcAE(nn.Module):
    def __init__(self, layers, input_dim, dropout=0.5, use_bn=False):
        """
        初始化函数，定义网络结构和参数。

        参数:
        --------------------------
        input_dim: 输入特征维度
        layers: 一个列表，如[x1, x2, x3]，将创建具有x1、x2、x3个隐藏节点的三层网络。
        dropout:Dropout比例
        use_bn: 是否使用批量归一化
        """
        # 调用父类（nn.Module）的初始化方法
        super().__init__()

        # 存储输入维度
        self.input_dim = input_dim
        # 存储dropout比率
        self.dropout = dropout
        # 存储是否使用批量归一化的标志
        self.use_bn = use_bn

        # 构建编码器网络
        self.encoder = self.get_encoder(layers)
        # 构建解码器网络
        self.decoder = self.get_decoder(layers)


    def get_encoder(self, layers):
        """
        创建编码器网络结构的函数，包含多层全连接层、激活函数、批量归一化和Dropout。

        参数：
        - layers：列表，表示各层神经元的数量。

        返回：
        - nn.Sequential：构建好的编码器网络。
        """

        all_layers = []  # 存储所有层的列表
        input_dim = self.input_dim  # 输入维度

        # 遍历layers列表，构建网络层
        for i in range(0, len(layers)):
            # 添加全连接层
            all_layers.append(nn.Linear(input_dim, layers[i]))
            # 添加LeakyReLU激活函数
            all_layers.append(nn.LeakyReLU())

            # 如果使用批量归一化
            if self.use_bn:
                all_layers.append(nn.BatchNorm1d(layers[i]))

            # 如果dropout比例大于0，添加Dropout层
            if self.dropout > 0:
                all_layers.append(nn.Dropout(self.dropout))

            # 更新输入维度
            input_dim = layers[i]

        # 删除最后一层的激活函数
        # decline_num = 1 + int(self.use_bn) + int(self.dropout > 0)
        # all_layers = all_layers[:-decline_num]

        # 返回构建好的编码器网络
        return nn.Sequential(*all_layers)


    def get_decoder(self, layers):
        """
        创建解码器网络结构，由多个线性层、激活函数和可选的批归一化与Dropout层组成。

        参数：
        - layers：列表，表示解码器各层的神经元数量，从输入层到输出层递增

        返回：
        - decoder：nn.Sequential，构建好的解码器模型
        """

        all_layers = []  # 存储所有解码器层的列表

        # 深度复制输入的layers并反转，以便从输出层开始构建解码器
        decoder_layer = copy.deepcopy(layers)
        decoder_layer.reverse()
        # 在解码器层末尾添加输入维度
        decoder_layer.append(self.input_dim)

        # 遍历解码器层，构建线性层、激活函数和批归一化/ Dropout层
        for i in range(0, len(decoder_layer) - 1):
            # 添加线性层
            all_layers.append(nn.Linear(decoder_layer[i], decoder_layer[i + 1]))
            # 如果是倒数第二个层，使用ReLU激活函数
            if i == len(decoder_layer) - 2:
                all_layers.append(nn.ReLU())
            # 否则，使用LeakyReLU激活函数
            else:
                all_layers.append(nn.LeakyReLU())

            # 如果启用了批归一化，添加该层
            if self.use_bn:
                all_layers.append(nn.BatchNorm1d(decoder_layer[i]))

            # 如果dropout比例大于0，添加Dropout层
            if self.dropout > 0:
                all_layers.append(nn.Dropout(self.dropout))

        # 不再需要最后一层的线性层，因为已经在循环中处理了
        # all_layers.append(nn.Linear(decoder_layer[-2], decoder_layer[-1]))

        # 将所有层组合成一个Sequential模型
        return nn.Sequential(*all_layers)


    def forward(self, x):
        # 将层组合成一个完整的模块
        latent = self.encoder(x)  # 通过编码器获取潜在表示
        recon = self.decoder(latent)  # 使用解码器重构输入数据
        return recon, latent  # 返回重构的输出和潜在向量