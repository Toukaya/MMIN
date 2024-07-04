import torch
import os
import json
import torch.nn.functional as F
from models.base_model import BaseModel
from models.networks.fc import FcEncoder
from models.networks.lstm import LSTMEncoder
from models.networks.textcnn import TextCNN
from models.networks.classifier import FcClassifier


class UttFusionModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # 添加命令行参数：声学输入维度
        parser.add_argument('--input_dim_a', type=int, default=130, help='acoustic input dim')

        # 添加命令行参数：词汇输入维度
        parser.add_argument('--input_dim_l', type=int, default=1024, help='lexical input dim')

        # 添加命令行参数：视觉输入维度
        parser.add_argument('--input_dim_v', type=int, default=384, help='visual input dim')

        # 添加命令行参数：音频模型嵌入大小
        parser.add_argument('--embd_size_a', default=128, type=int, help='audio model embedding size')

        # 添加命令行参数：文本模型嵌入大小
        parser.add_argument('--embd_size_l', default=128, type=int, help='text model embedding size')

        # 添加命令行参数：视觉模型嵌入大小
        parser.add_argument('--embd_size_v', default=128, type=int, help='visual model embedding size')

        # 添加命令行参数：音频嵌入方法，可选'last', 'maxpool', 'attention'
        parser.add_argument('--embd_method_a', default='maxpool', type=str,
                            choices=['last', 'maxpool', 'attention'], help='audio embedding method,last,mean or atten')

        # 添加命令行参数：视觉嵌入方法，可选'last', 'maxpool', 'attention'
        parser.add_argument('--embd_method_v', default='maxpool', type=str,
                            choices=['last', 'maxpool', 'attention'], help='visual embedding method,last,mean or atten')

        # 添加命令行参数：分类层节点数，如'128,128'表示两层，每层分别有128和128个节点
        parser.add_argument('--cls_layers', type=str, default='128,128',
                            help='256,128 for 2 layers with 256, 128 nodes respectively')

        # 添加命令行参数：Dropout比例
        parser.add_argument('--dropout_rate', type=float, default=0.3, help='rate of dropout')

        # 添加命令行参数：是否使用批量归一化层
        parser.add_argument('--bn', action='store_true', help='if specified, use bn layers in FC')

        # 添加命令行参数：模型使用的模态
        parser.add_argument('--modality', type=str, help='which modality to use for model')

        # 返回解析器对象
        return parser

    # 初始化LSTM自动编码器类
    def __init__(self, opt):
        """初始化LSTM自动编码器类
        参数:
            opt (Option class)-- 存储所有实验标志，需要是BaseOptions的子类
        """
        super().__init__(opt)
        # 实验设置为10折，教师设置为5折，训练集应匹配
        self.loss_names = ['CE']  # 损失名称
        self.modality = opt.modality  # 'AVL' - 模态（音频、视频、文本）
        self.model_names = ['C']  # 模型名称列表

        # 将逗号分隔的字符串转换为整数列表
        cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))  # [128, 128]
        # 计算输入尺寸，根据模态组合嵌入尺寸
        cls_input_size = opt.embd_size_a * int("A" in self.modality) + \
                         opt.embd_size_v * int("V" in self.modality) + \
                         opt.embd_size_l * int("L" in self.modality)  # [384]

        # 创建分类器网络，输入维度：模态组合，层：cls_layers，输出维度：output_dim
        self.netC = FcClassifier(cls_input_size, cls_layers, output_dim=opt.output_dim, dropout=opt.dropout_rate,
                                 use_bn=opt.bn)

        # 如果模态包含'A'，则添加声学模型（帧级别）
        if 'A' in self.modality:
            self.model_names.append('A')
            # 使用FcClassifier替换LSTMEncoder
            self.netA = FcClassifier(opt.input_dim_a, cls_layers, output_dim=opt.embd_size_a, dropout=opt.dropout_rate,
                                     use_bn=opt.bn)

        # 如果模态包含'L'，则添加词汇模型（句子级别）
        if 'L' in self.modality:
            self.model_names.append('L')
            # 使用FcClassifier替换TextCNN
            self.netL = FcClassifier(opt.input_dim_l, cls_layers, output_dim=opt.embd_size_l, dropout=opt.dropout_rate,
                                     use_bn=opt.bn)

        # 如果模态包含'V'，则添加视觉模型（帧级别）
        if 'V' in self.modality:
            self.model_names.append('V')
            # 使用FcClassifier替换LSTMEncoder
            self.netV = FcClassifier(opt.input_dim_v, cls_layers, output_dim=opt.embd_size_v, dropout=opt.dropout_rate,
                                     use_bn=opt.bn)

        # 如果处于训练模式
        if self.isTrain:
            # 根据数据集类型选择损失函数
            dataset = opt.dataset_mode.split('_')[0]
            if dataset in ['cmumosi', 'cmumosei']:   self.criterion_ce = torch.nn.MSELoss()
            if dataset in ['boxoflies', 'iemocapfour', 'iemocapsix']: self.criterion_ce = torch.nn.CrossEntropyLoss()

            # 初始化优化器，将所有模型参数放入优化器
            paremeters = [{'params': getattr(self, 'net' + net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
            self.output_dim = opt.output_dim  # 类别数量：4

        # 修改保存目录
        self.save_dir = os.path.join(self.save_dir, str(opt.cvNo))
        # 如果保存目录不存在，则创建
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    def set_input(self, input):
        """
        解包输入数据，从数据加载器中获取，并执行必要的预处理步骤。
        参数:
            input (字典): 包含数据本身及其元数据信息。
        """
        # 如果模态中包含'A'，则将声学特征转换为浮点数并移动到设备上
        if 'A' in self.modality:
            self.acoustic = input['A_feat'].float().to(self.device)

        # 如果模态中包含'L'，则将词汇特征转换为浮点数并移动到设备上
        if 'L' in self.modality:
            self.lexical = input['L_feat'].float().to(self.device)

        # 如果模态中包含'V'，则将视觉特征转换为浮点数并移动到设备上
        if 'V' in self.modality:
            self.visual = input['V_feat'].float().to(self.device)

        # 将标签移动到设备上
        self.label = input['label'].to(self.device)

    def forward(self):
        """
        执行前向传播过程；由<optimize_parameters>和<test>两个函数调用。
        """

        final_embd = []  # 初始化最终嵌入向量列表

        # 如果'A'模态存在
        if 'A' in self.modality:
            self.feat_A, _ = self.netA(self.acoustic)  # self.acoustic: [256, 155, 130]
            final_embd.append(self.feat_A)

        # 如果'L'模态存在
        if 'L' in self.modality:
            self.feat_L, _ = self.netL(self.lexical)
            final_embd.append(self.feat_L)

        # 如果'V'模态存在
        if 'V' in self.modality:
            self.feat_V, _ = self.netV(self.visual)
            final_embd.append(self.feat_V)

        # 合并所有模态的特征
        self.feat = torch.cat(final_embd, dim=-1)  # [batch, dim*3]

        # 获取模型输出
        self.logits, self.ef_fusion_feat = self.netC(self.feat)

        #################################
        # 原本的预测层，使用softmax激活
        # self.pred = F.softmax(self.logits, dim=-1)

        # 将logits挤压成一维
        self.logits = self.logits.squeeze()

        # 直接将挤压后的logits作为预测值
        self.pred = self.logits

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

        # 对每个模型，使用梯度裁剪来限制参数更新的幅度，防止梯度爆炸
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(
                getattr(self, 'net' + model).parameters(), 5.0
            )

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