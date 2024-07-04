
import torch
import os
import json
import torch.nn.functional as F
from models.base_model import BaseModel
from models.networks.fc import FcEncoder
from models.networks.lstm import LSTMEncoder
from models.networks.textcnn import TextCNN
from models.networks.classifier import FcClassifier

''' Implement Data augmentation of model fusion
'''
class UttDataAugModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # 添加命令行参数：声学输入维度
        parser.add_argument('--input_dim_a', type=int, default=130, help='acoustic input dimension')

        # 添加命令行参数：词法输入维度
        parser.add_argument('--input_dim_l', type=int, default=1024, help='lexical input dimension')

        # 添加命令行参数：视觉输入维度（与词法输入维度相同）
        parser.add_argument('--input_dim_v', type=int, default=384, help='lexical input dimension')

        # 添加命令行参数：音频模型嵌入大小
        parser.add_argument('--embd_size_a', default=128, type=int, help='audio model embedding size')

        # 添加命令行参数：文本模型嵌入大小
        parser.add_argument('--embd_size_l', default=128, type=int, help='text model embedding size')

        # 添加命令行参数：视觉模型嵌入大小
        parser.add_argument('--embd_size_v', default=128, type=int, help='visual model embedding size')

        # 添加命令行参数：音频嵌入方法，可选'last', 'maxpool', 'attention'
        parser.add_argument('--embd_method_a', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'],
                            help='audio embedding method, last, mean or attention')

        # 添加命令行参数：视觉嵌入方法，可选'last', 'maxpool', 'attention'
        parser.add_argument('--embd_method_v', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'],
                            help='visual embedding method, last, mean or attention')

        # 添加命令行参数：分类层节点数，如'128,128'表示两层，每层分别有128个节点
        parser.add_argument('--cls_layers', type=str, default='128,128',
                            help='256,128 for 2 layers with 256, 128 nodes respectively')

        # 添加命令行参数：Dropout比例
        parser.add_argument('--dropout_rate', type=float, default=0.3, help='dropout rate')

        # 添加命令行参数：是否使用批量归一化层
        parser.add_argument('--bn', action='store_true', help='if specified, use batch normalization layers in FC')

        # 添加命令行参数：模型使用的模态
        parser.add_argument('--modality', type=str, help='which modality to use for the model')

        # 返回解析器对象
        return parser

    # 初始化LSTM自动编码器类
    def __init__(self, opt):
        """初始化LSTM自动编码器类，参数opt是一个包含实验标志的Option类，需要是BaseOptions的子类"""
        super().__init__(opt)
        # 我们的实验设置为10折，教师模型设置为5折，训练集应匹配
        self.loss_names = ['CE']  # 损失函数名称
        self.modality = opt.modality  # 输入模态（如音频、文本、视觉）
        self.model_names = ['C']  # 模型名称列表

        # 分类层的大小
        cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))

        # 输入到分类器的特征维度
        cls_input_size = opt.embd_size_a * int("A" in self.modality) + \
                         opt.embd_size_v * int("V" in self.modality) + \
                         opt.embd_size_l * int("L" in self.modality)

        # 创建分类器
        self.netC = FcClassifier(cls_input_size, cls_layers, output_dim=opt.output_dim, dropout=opt.dropout_rate,
                                 use_bn=opt.bn)

        # 如果模态包含音频
        if 'A' in self.modality:
            self.model_names.append('A')  # 添加模型名称
            self.netA = LSTMEncoder(opt.input_dim_a, opt.embd_size_a, embd_method=opt.embd_method_a)  # 创建音频模型

        # 如果模态包含文本
        if 'L' in self.modality:
            self.model_names.append('L')  # 添加模型名称
            self.netL = TextCNN(opt.input_dim_l, opt.embd_size_l)  # 创建文本模型

        # 如果模态包含视觉
        if 'V' in self.modality:
            self.model_names.append('V')  # 添加模型名称
            self.netV = LSTMEncoder(opt.input_dim_v, opt.embd_size_v, opt.embd_method_v)  # 创建视觉模型

        # 训练模式下
        if self.isTrain:
            self.criterion_ce = torch.nn.CrossEntropyLoss()  # 初始化交叉熵损失函数
            # 初始化优化器，将所有模型的参数添加到优化器
            paremeters = [{'params': getattr(self, 'net' + net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999),
                                              weight_decay=opt.weight_decay)  # 使用Adam优化器
            self.optimizers.append(self.optimizer)  # 添加到优化器列表
            self.output_dim = opt.output_dim  # 输出维度

        # 修改保存目录
        self.save_dir = os.path.join(self.save_dir, str(opt.cvNo))  # 根据交叉验证编号创建保存目录
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)  # 如果目录不存在，创建它

    def set_input(self, input):
        """
        解包从数据加载器获取的输入数据，并执行必要的预处理步骤。
        参数:
            input (dict): 包含数据本身及其元数据信息。
        """
        # 将声学特征转换为浮点数并移动到设备上
        acoustic = input['A_feat'].float().to(self.device)
        # 将词汇特征转换为浮点数并移动到设备上
        lexical = input['L_feat'].float().to(self.device)
        # 将视觉特征转换为浮点数并移动到设备上
        visual = input['V_feat'].float().to(self.device)

        # 如果处于训练模式
        if self.isTrain:
            # 将标签移动到设备上
            self.label = input['label'].to(self.device)
            # 将缺失索引转换为长整型并移动到设备上
            self.missing_index = input['missing_index'].long().to(self.device)

            # 如果包含声学模态
            if 'A' in self.modality:
                # 创建声学模态的缺失索引
                self.A_miss_index = self.missing_index[:, 0].unsqueeze(1).unsqueeze(2)
                # 计算声学模态的缺失部分
                self.A_miss = acoustic * self.A_miss_index

            # 如果包含词汇模态
            if 'L' in self.modality:
                # 创建词汇模态的缺失索引
                self.L_miss_index = self.missing_index[:, 2].unsqueeze(1).unsqueeze(2)
                # 计算词汇模态的缺失部分
                self.L_miss = lexical * self.L_miss_index

            # 如果包含视觉模态
            if 'V' in self.modality:
                # 创建视觉模态的缺失索引
                self.V_miss_index = self.missing_index[:, 1].unsqueeze(1).unsqueeze(2)
                # 计算视觉模态的缺失部分
                self.V_miss = visual * self.V_miss_index

        # 如果处于验证或测试模式
        else:
            # 直接将声学模态设为缺失部分
            self.A_miss = acoustic
            # 直接将视觉模态设为缺失部分
            self.V_miss = visual
            # 直接将词汇模态设为缺失部分
            self.L_miss = lexical

    def forward(self):
        """
        执行前向传播过程；由<optimize_parameters>和<test>两个函数调用。
        """

        final_embd = []  # 初始化最终嵌入向量列表

        # 如果模态包含'A'
        if 'A' in self.modality:
            # 通过netA网络处理缺失的A模态数据
            self.feat_A_miss = self.netA(self.A_miss)
            final_embd.append(self.feat_A_miss)

        # 如果模态包含'L'
        if 'L' in self.modality:
            # 通过netL网络处理缺失的L模态数据
            self.feat_L_miss = self.netL(self.L_miss)
            final_embd.append(self.feat_L_miss)

        # 如果模态包含'V'
        if 'V' in self.modality:
            # 通过netV网络处理缺失的V模态数据
            self.feat_V_miss = self.netV(self.V_miss)
            final_embd.append(self.feat_V_miss)

        # 沿着最后一个维度拼接所有模态的特征
        self.feat = torch.cat(final_embd, dim=-1)

        # 通过netC网络获取模型输出，得到logits和融合特征
        self.logits, self.ef_fusion_feat = self.netC(self.feat)

        # 对logits进行softmax操作，得到预测概率分布
        self.pred = F.softmax(self.logits, dim=-1)

    def backward(self):
        """
        计算用于反向传播的损失值
        """
        # 计算交叉熵损失，其中logits是预测概率，label是真实标签
        self.loss_CE = self.criterion_ce(self.logits, self.label)

        # 将损失值设为要优化的目标
        loss = self.loss_CE

        # 反向传播计算梯度
        loss.backward()

        # 对每个模型，使用梯度裁剪限制参数更新的幅度，防止梯度爆炸
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