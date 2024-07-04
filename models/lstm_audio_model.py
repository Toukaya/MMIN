
import torch
import os
import json
import torch.nn.functional as F
from models.base_model import BaseModel
from models.networks.lstm import LSTMEncoder
from models.networks.fc_encoder import FcEncoder


class LSTMAudioModel(BaseModel):
    '''
    A: DNN
    V: denseface + LSTM + maxpool
    L: bert + textcnn
    '''
    @staticmethod
    def modify_commandline_options(parser, is_train=True):  # 定义一个函数，用于根据训练或测试状态修改命令行参数
        parser.add_argument('--input_dim', type=int, default=130)  # 添加参数，输入维度，默认值为130
        parser.add_argument('--cls_layers', type=str, default='256,128')  # 添加参数，分类层的结构，默认为'256,128'
        parser.add_argument('--hidden_size', type=int, default=256)  # 添加参数，隐藏层大小，默认值为256
        parser.add_argument('--embd_method', type=str, default='maxpool')  # 添加参数，嵌入方法，默认为'maxpool'
        return parser  # 返回修改后的参数解析器对象

    def __init__(self, opt):
        """初始化LSTM自动编码器类
        参数:
            opt (Option类) -- 存储所有实验标志，需要是BaseOptions的子类
        """
        super().__init__(opt)
        # 实验设置为10折，教师模型为5折，训练集应匹配
        self.loss_names = ['CE']  # 损失函数名称
        self.model_names = ['A', 'C']  # 模型名称列表

        # 初始化LSTM编码器
        self.netA = LSTMEncoder(opt.input_dim, opt.hidden_size, embd_method=opt.embd_method)

        # 初始化分类器层的维度
        cls_layers = [int(x) for x in opt.cls_layers.split(',')] + [opt.output_dim]
        self.netC = FcEncoder(opt.hidden_size, cls_layers, dropout=0.3)  # 初始化全连接层编码器

        # 如果处于训练模式
        if self.isTrain:
            # 定义交叉熵损失函数
            self.criterion_ce = torch.nn.CrossEntropyLoss()

            # 初始化优化器
            parameters = [{'params': getattr(self, 'net' + net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(parameters, lr=opt.lr, betas=(opt.beta1, 0.998))  # 0.999

            # 将优化器添加到优化器列表
            self.optimizers.append(self.optimizer)
            self.output_dim = opt.output_dim  # 输出维度

        # 修改保存目录
        self.save_dir = os.path.join(self.save_dir, str(opt.cvNo))

        # 如果保存目录不存在，则创建
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    def set_input(self, input):
        """
        解压并预处理输入数据，从数据加载器中获取数据及其元信息。
        参数:
            input (dict): 包含数据本身及其元数据信息。
        """
        # 将特征A的数据转移到设备（GPU或CPU）
        self.A_feat = input['A_feat'].to(self.device)
        # 将标签数据转移到设备（GPU或CPU）
        self.label = input['label'].to(self.device)
        # 保存整个输入数据
        self.input = input

    def forward(self):
        """
        执行前向传播过程；由<optimize_parameters>和<test>两个函数调用。

        参数:
        - self: 当前类的实例

        返回:
        - feat: 通过netA处理后的特征
        - logits: 通过netC处理得到的logits（未归一化的预测得分）
        - pred: 应用softmax激活函数后的预测概率分布，维度为(-1, 类别数)
        """
        # 通过netA网络处理输入的A_feat，得到特征表示
        self.feat = self.netA(self.A_feat)

        # 通过netC网络将特征转换为logits
        self.logits = self.netC(self.feat)

        # 使用softmax函数对logits进行归一化，得到各个类别的预测概率
        self.pred = F.softmax(self.logits, dim=-1)

    def backward(self):
        """
        计算反向传播过程中的损失值
        """
        # 计算交叉熵损失
        self.loss_CE = self.criterion_ce(self.logits, self.label)

        # 将损失值设为要优化的目标
        loss = self.loss_CE

        # 反向传播，计算梯度
        loss.backward()

        # 对每个模型的参数进行梯度裁剪，限制最大范数为5.0，以防止梯度爆炸
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net' + model).parameters(), 5.0)

    def optimize_parameters(self, epoch):
        """
        计算损失、梯度，并更新网络权重；在每个训练迭代中被调用
        """
        # 前向传播，计算预测值
        self.forward()

        # 初始化优化器的梯度为零，准备进行反向传播
        self.optimizer.zero_grad()

        # 反向传播，计算所有参数的梯度
        self.backward()

        # 使用优化器根据梯度更新网络权重
        self.optimizer.step()