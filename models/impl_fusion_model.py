
import torch
import os
import json
import torch.nn.functional as F
from models.base_model import BaseModel
from models.networks.fc import FcEncoder
from models.networks.lstm import LSTMEncoder
from models.networks.textcnn import TextCNN
from models.networks.classifier import FcClassifier

''' Implementation of 
    IMPLICIT FUSION BY JOINT AUDIOVISUAL TRAINING FOR EMOTION RECOGNITION IN MONO MODALITY
    https://ieeexplore.ieee.org/document/8682773
'''

class ImplFusionModel(BaseModel):
    @staticmethod
    # 定义一个函数，用于在命令行参数解析器中添加训练或测试所需的参数
    def modify_commandline_options(parser, is_train=True):
        # 添加声学输入维度参数，默认值为130
        parser.add_argument('--input_dim_a', type=int, default=130, help='声学输入维度')

        # 添加词汇输入维度参数，默认值为1024
        parser.add_argument('--input_dim_l', type=int, default=1024, help='词汇输入维度')

        # 添加视觉输入维度参数，默认值为384
        parser.add_argument('--input_dim_v', type=int, default=384, help='视觉输入维度')

        # 添加音频损失权重参数，默认值为1.0
        parser.add_argument('--weight_a', type=float, default=1.0, help='音频损失权重')

        # 添加视觉损失权重参数，默认值为0.3
        parser.add_argument('--weight_v', type=float, default=0.3, help='视觉损失权重')

        # 添加词汇损失权重参数，默认值为0.3
        parser.add_argument('--weight_l', type=float, default=0.3, help='词汇损失权重')

        # 添加每个模态的嵌入大小参数，默认值为128
        parser.add_argument('--embd_size', default=128, type=int, help='每个模态的嵌入大小')

        # 添加音频嵌入方法参数，默认值为'maxpool'，可选['last', 'maxpool', 'attention']
        parser.add_argument('--embd_method_a', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'],
                            help='音频嵌入方法，可选：last, mean 或 atten')

        # 添加视觉嵌入方法参数，默认值为'maxpool'，可选['last', 'maxpool', 'attention']
        parser.add_argument('--embd_method_v', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'],
                            help='视觉嵌入方法，可选：last, mean 或 atten')

        # 添加分类层节点数参数，默认值为'128,128'
        parser.add_argument('--cls_layers', type=str, default='128,128', help='2层分类层的节点数，如：256,128')

        # 添加Dropout比例参数，默认值为0.3
        parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout比例')

        # 添加批量归一化选项，如果指定，则在全连接层中使用批量归一化
        parser.add_argument('--bn', action='store_true', help='是否使用批量归一化层')

        # 添加训练时使用的模态参数
        parser.add_argument('--trn_modality', type=str, help='模型训练时使用的模态')

        # 添加测试时使用的模态参数
        parser.add_argument('--test_modality', type=str, help='模型测试时使用的模态')

        # 返回更新后的参数解析器
        return parser

    # 初始化LSTM自动编码器类
    def __init__(self, opt):
        """初始化LSTM自动编码器类，接收一个存储实验标志的Option类，需要是BaseOptions的子类
        参数:
            opt (Option class)-- 存储所有实验标志
        """
        # 调用父类（super类）的初始化方法
        super().__init__(opt)

        # 我们的实验设置为10折，教师模型设置为5折，训练集应匹配
        self.loss_names = []  # 存储损失函数名称的列表
        self.model_names = ['C']  # 存储模型名称的列表，初始为'C'

        # 训练和测试的模态
        self.trn_modality = opt.trn_modality
        self.test_modality = opt.test_modality
        # 确保测试模态只有一个
        assert len(self.test_modality) == 1

        # 分类层的层数
        cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))
        # 创建全连接分类器
        self.netC = FcClassifier(opt.embd_size, cls_layers, output_dim=opt.output_dim, dropout=opt.dropout_rate,
                                 use_bn=opt.bn)

        # 音频模型
        if 'A' in self.trn_modality:
            self.model_names.append('A')  # 添加模型名称到列表
            self.loss_names.append('CE_A')  # 添加损失函数名称到列表
            # 创建LSTM编码器
            self.netA = LSTMEncoder(opt.input_dim_a, opt.embd_size, embd_method=opt.embd_method_a)
            # 设置权重
            self.weight_a = opt.weight_a

        # 词汇模型
        if 'L' in self.trn_modality:
            self.model_names.append('L')  # 添加模型名称到列表
            self.loss_names.append('CE_L')  # 添加损失函数名称到列表
            # 创建文本卷积神经网络
            self.netL = TextCNN(opt.input_dim_l, opt.embd_size)
            # 设置权重
            self.weight_l = opt.weight_l

        # 视觉模型
        if 'V' in self.trn_modality:
            self.model_names.append('V')  # 添加模型名称到列表
            self.loss_names.append('CE_V')  # 添加损失函数名称到列表
            # 创建LSTM编码器
            self.netV = LSTMEncoder(opt.input_dim_v, opt.embd_size, opt.embd_method_v)
            # 设置权重
            self.weight_v = opt.weight_v

        # 如果在训练阶段
        if self.isTrain:
            # 初始化交叉熵损失函数
            self.criterion_ce = torch.nn.CrossEntropyLoss()
            # 初始化优化器；调度器将在BaseModel.setup函数中自动创建
            paremeters = [{'params': getattr(self, 'net' + net).parameters()} for net in self.model_names]
            # 使用Adam优化器
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999))
            # 将优化器添加到列表
            self.optimizers.append(self.optimizer)
            # 输出维度
            self.output_dim = opt.output_dim

        # 修改保存目录
        self.save_dir = os.path.join(self.save_dir, str(opt.cvNo))
        # 如果保存目录不存在，则创建
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    def set_input(self, input):
        """
        解包输入数据，从数据加载器中获取，并执行必要的预处理步骤。
        参数:
            input (dict): 包含数据本身及其元数据信息。
        """
        # 如果'A'在训练模式中，将音频特征转换为浮点数并移动到设备上
        if 'A' in self.trn_modality:
            self.acoustic = input['A_feat'].float().to(self.device)

        # 如果'L'在训练模式中，将词汇特征转换为浮点数并移动到设备上
        if 'L' in self.trn_modality:
            self.lexical = input['L_feat'].float().to(self.device)

        # 如果'V'在训练模式中，将视觉特征转换为浮点数并移动到设备上
        if 'V' in self.trn_modality:
            self.visual = input['V_feat'].float().to(self.device)

        # 将标签移动到设备上
        self.label = input['label'].to(self.device)

    def forward(self):
        """
        执行前向传播操作；在<optimize_parameters>和<test>函数中被调用。
        """
        # 根据训练或测试模式选择合适的模态
        modality = self.trn_modality if self.isTrain else self.test_modality

        # 如果'A'模态存在
        if 'A' in modality:
            # 通过网络A处理声学特征
            self.feat_A = self.netA(self.acoustic)
            # 通过共享的分类器网络C处理声学特征
            self.logits_A, _ = self.netC(self.feat_A)
            # 对声学特征的logits进行softmax激活，得到预测概率
            self.pred = F.softmax(self.logits_A, dim=-1)

        # 如果'L'模态存在
        if 'L' in modality:
            # 通过网络L处理词汇特征
            self.feat_L = self.netL(self.lexical)
            # 通过共享的分类器网络C处理词汇特征
            self.logits_L, _ = self.netC(self.feat_L)
            # 对词汇特征的logits进行softmax激活，得到预测概率
            self.pred = F.softmax(self.logits_L, dim=-1)

        # 如果'V'模态存在
        if 'V' in modality:
            # 通过网络V处理视觉特征
            self.feat_V = self.netV(self.visual)
            # 通过共享的分类器网络C处理视觉特征
            self.logits_V, _ = self.netC(self.feat_V)
            # 对视觉特征的logits进行softmax激活，得到预测概率
            self.pred = F.softmax(self.logits_V, dim=-1)

    def backward(self):
        """
        计算反向传播过程中的损失
        """
        losses = []

        # 如果'A'在训练模态中，计算并添加分类交叉熵损失
        if 'A' in self.trn_modality:
            self.loss_CE_A = self.criterion_ce(self.logits_A, self.label) * self.weight_a
            losses.append(self.loss_CE_A)

        # 如果'L'在训练模态中，计算并添加分类交叉熵损失
        if 'L' in self.trn_modality:
            self.loss_CE_L = self.criterion_ce(self.logits_L, self.label) * self.weight_l
            losses.append(self.loss_CE_L)

        # 如果'V'在训练模态中，计算并添加分类交叉熵损失（此处可能有误，应为self.loss_CE_V）
        if 'V' in self.trn_modality:
            self.loss_CE_V = self.criterion_ce(self.logits_V, self.label) * self.weight_v
            losses.append(self.loss_CE_L)  # 应该是self.loss_CE_V，可能是笔误

        # 求和所有损失
        loss = sum(losses)
        # 反向传播损失
        loss.backward()
        # 对每个模型的参数进行梯度裁剪，以防止梯度爆炸
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net' + model).parameters(), 0.5)

    def optimize_parameters(self, epoch):
        """
        计算损失、梯度，并更新网络权重；在每个训练迭代中被调用
        """
        # 前向传播
        self.forward()
        # 反向传播
        self.optimizer.zero_grad()  # 清零优化器的梯度
        self.backward()  # 计算损失和梯度
        self.optimizer.step()  # 根据梯度更新网络参数
