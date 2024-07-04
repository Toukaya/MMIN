
import torch
import os
import json
import torch.nn.functional as F
from models.base_model import BaseModel
from models.networks.fc import FcEncoder
from models.networks.lstm import LSTMEncoder
from models.networks.textcnn import TextCNN
from models.networks.classifier import FcClassifier


class UttFDataAugModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # 添加命令行参数：声学输入维度
        parser.add_argument('--input_dim_a', type=int, default=130, help='acoustic input dim')
        # 添加命令行参数：词法输入维度
        parser.add_argument('--input_dim_l', type=int, default=1024, help='lexical input dim')
        # 添加命令行参数：视觉输入维度（与词法相同）
        parser.add_argument('--input_dim_v', type=int, default=384, help='lexical input dim')
        # 添加命令行参数：音频模型嵌入大小
        parser.add_argument('--embd_size_a', default=128, type=int, help='audio model embedding size')
        # 添加命令行参数：文本模型嵌入大小
        parser.add_argument('--embd_size_l', default=128, type=int, help='text model embedding size')
        # 添加命令行参数：视觉模型嵌入大小
        parser.add_argument('--embd_size_v', default=128, type=int, help='visual model embedding size')
        # 添加命令行参数：音频嵌入方法，可选'last', 'maxpool', 'attention'
        parser.add_argument('--embd_method_a', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'],
                            help='audio embedding method,last,mean or atten')
        # 添加命令行参数：视觉嵌入方法，可选'last', 'maxpool', 'attention'
        parser.add_argument('--embd_method_v', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'],
                            help='visual embedding method,last,mean or atten')
        # 添加命令行参数：分类层节点数，如'128,128'表示两层分别有128和128个节点
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
        """初始化LSTM自动编码器类，参数为实验选项类，需要是BaseOptions的子类
        参数:
            opt (Option class)-- 存储所有实验标志
        """
        # 调用父类（super类）的初始化方法
        super().__init__(opt)

        # 我们的实验设置为10折，教师模型为5折，训练集应匹配
        self.loss_names = ['CE']  # 损失函数名称列表
        self.modality = opt.modality  # 输入模态（如音频、文本、视觉）
        self.model_names = ['C']  # 模型名称列表

        # 分类层的大小
        cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))

        # 输入到分类器的总特征维度
        cls_input_size = opt.embd_size_a * int("A" in self.modality) + \
                         opt.embd_size_v * int("V" in self.modality) + \
                         opt.embd_size_l * int("L" in self.modality)

        # 创建分类器
        self.netC = FcClassifier(cls_input_size, cls_layers, output_dim=opt.output_dim, dropout=opt.dropout_rate,
                                 use_bn=opt.bn)

        # 如果包含音频模态
        if 'A' in self.modality:
            self.model_names.append('A')  # 添加模型名称到列表
            self.netA = LSTMEncoder(opt.input_dim_a, opt.embd_size_a, embd_method=opt.embd_method_a)  # 创建音频模型

        # 如果包含文本模态
        if 'L' in self.modality:
            self.model_names.append('L')
            self.netL = TextCNN(opt.input_dim_l, opt.embd_size_l)  # 创建文本模型

        # 如果包含视觉模态
        if 'V' in self.modality:
            self.model_names.append('V')
            self.netV = LSTMEncoder(opt.input_dim_v, opt.embd_size_v, opt.embd_method_v)  # 创建视觉模型

        # 训练模式下
        if self.isTrain:
            self.criterion_ce = torch.nn.CrossEntropyLoss()  # 初始化交叉熵损失函数
            # 初始化优化器，将各模型参数分组
            paremeters = [{'params': getattr(self, 'net' + net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999),
                                              weight_decay=opt.weight_decay)
            self.optimizers.append(self.optimizer)  # 添加优化器到列表
            self.output_dim = opt.output_dim  # 输出维度

        # 修改保存目录
        self.save_dir = os.path.join(self.save_dir, str(opt.cvNo))
        # 检查并创建保存目录
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

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
        执行前向传播操作；由<optimize_parameters>和<test>两个函数调用。
        """

        final_embd = []  # 初始化最终嵌入向量列表

        # 如果'A'模态存在
        if 'A' in self.modality:
            self.feat_A_miss = self.netA(self.A_miss)  # 通过netA网络处理缺失的A模态特征
            final_embd.append(self.feat_A_miss)  # 将处理后的特征添加到列表中

        # 如果'L'模态存在
        if 'L' in self.modality:
            self.feat_L_miss = self.netL(self.L_miss)  # 通过netL网络处理缺失的L模态特征
            final_embd.append(self.feat_L_miss)  # 将处理后的特征添加到列表中

        # 如果'V'模态存在
        if 'V' in self.modality:
            self.feat_V_miss = self.netV(self.V_miss)  # 通过netV网络处理缺失的V模态特征
            final_embd.append(self.feat_V_miss)  # 将处理后的特征添加到列表中

        # # 如果'A'模态存在（注释掉的代码）
        # if 'A' in self.modality:
        #     self.feat_A = self.netA(self.acoustic)  # 通过netA网络处理声学特征
        #     final_embd.append(self.feat_A)

        # # 如果'L'模态存在（注释掉的代码）
        # if 'L' in self.modality:
        #     self.feat_L = self.netL(self.lexical)  # 通过netL网络处理词汇特征
        #     final_embd.append(self.feat_L)

        # # 如果'V'模态存在（注释掉的代码）
        # if 'V' in self.modality:
        #     self.feat_V = self.netV(self.visual)  # 通过netV网络处理视觉特征
        #     final_embd.append(self.feat_V)

        # 合并所有模态的特征
        self.feat = torch.cat(final_embd, dim=-1)  # 沿着最后一个维度连接所有特征
        self.logits, self.ef_fusion_feat = self.netC(self.feat)  # 通过netC网络获取分类得分和融合特征
        self.pred = F.softmax(self.logits, dim=-1)  # 对分类得分进行softmax操作，得到概率分布

    def backward(self):
        """
        计算反向传播过程中的损失值

        此函数用于计算交叉熵损失（loss_CE），并基于此损失进行反向传播。
        """
        # 计算交叉熵损失，其中logits是模型的预测输出，label是实际标签
        self.loss_CE = self.criterion_ce(self.logits, self.label)

        # 将损失值赋给变量loss，以便后续操作
        loss = self.loss_CE

        # 反向传播损失以更新权重
        loss.backward()

        # 对每个模型进行梯度裁剪，限制最大梯度范数为5.0，以防止梯度爆炸
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