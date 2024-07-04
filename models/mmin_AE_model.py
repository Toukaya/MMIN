import torch
import os
import json
from collections import OrderedDict
import torch.nn.functional as F
from models.base_model import BaseModel
from models.networks.fc import FcEncoder
from models.networks.lstm import LSTMEncoder
from models.networks.textcnn import TextCNN
from models.networks.classifier import FcClassifier
from models.networks.autoencoder import SimpleFcAE
from models.utt_fusion_model import UttFusionModel
from .utils.config import OptConfig


class MMINAEModel(BaseModel):
    @staticmethod
    # 定义一个函数，用于在命令行选项中添加参数
    def modify_commandline_options(parser, is_train=True):
        # 添加声学输入维度参数
        parser.add_argument('--input_dim_a', type=int, default=130, help='声学输入维度')

        # 添加词法输入维度参数
        parser.add_argument('--input_dim_l', type=int, default=1024, help='词法输入维度')

        # 添加视觉输入维度参数
        parser.add_argument('--input_dim_v', type=int, default=384,
                            help='词法输入维度')  # 注意：这里的描述可能有误，应与--input_dim_l保持一致

        # 添加音频模型嵌入大小参数
        parser.add_argument('--embd_size_a', default=128, type=int, help='音频模型嵌入大小')

        # 添加文本模型嵌入大小参数
        parser.add_argument('--embd_size_l', default=128, type=int, help='文本模型嵌入大小')

        # 添加视觉模型嵌入大小参数
        parser.add_argument('--embd_size_v', default=128, type=int, help='视觉模型嵌入大小')

        # 添加音频嵌入方法参数，可选'last', 'maxpool', 'attention'
        parser.add_argument('--embd_method_a', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'],
                            help='音频嵌入方法，可选last, mean或atten')

        # 添加视觉嵌入方法参数，可选'last', 'maxpool', 'attention'
        parser.add_argument('--embd_method_v', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'],
                            help='视觉嵌入方法，可选last, mean或atten')

        # 添加自编码器层节点数参数，如'128,64,32'
        parser.add_argument('--AE_layers', type=str, default='128,64,32',
                            help='自编码器层数及每层节点数，例如2层256和128节点')

        # 添加分类层节点数参数，如'128,128'
        parser.add_argument('--cls_layers', type=str, default='128,128',
                            help='分类层层数及每层节点数，例如2层256和128节点')

        # 添加Dropout比例参数
        parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout比例')

        # 添加是否使用批量归一化层标志
        parser.add_argument('--bn', action='store_true', help='如果指定，则在全连接层中使用批量归一化层')

        # 添加预训练模型路径参数
        parser.add_argument('--pretrained_path', type=str, help='加载预训练编码器网络的路径')

        # 添加交叉熵损失权重参数
        parser.add_argument('--ce_weight', type=float, default=1.0, help='交叉熵损失权重')

        # 添加均方误差损失权重参数
        parser.add_argument('--mse_weight', type=float, default=1.0, help='均方误差损失权重')

        # 返回解析器对象
        return parser

    # 初始化LSTM自动编码器类
    def __init__(self, opt):
        """初始化LSTM自动编码器类，接收实验参数选项类opt，需要是BaseOptions的子类
        参数:
            opt (Option class)-- 存储所有实验标志
        """
        # 调用父类（即BaseModel）的初始化方法
        super().__init__(opt)

        # 获取数据集模式
        self.dataset = opt.dataset_mode.split('_')[0]

        # 定义损失函数名称
        self.loss_names = ['ce', 'recon']

        # 定义模型名称
        self.model_names = ['A', 'AA', 'V', 'VV', 'L', 'LL', 'C', 'AE']

        # 将字符串形式的分类层尺寸转换为整数列表
        cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))

        # 计算AE输入维度
        AE_input_dim = opt.embd_size_a + opt.embd_size_v + opt.embd_size_l

        # 音频模型
        self.netA = FcClassifier(opt.input_dim_a, cls_layers, output_dim=opt.embd_size_a, dropout=opt.dropout_rate,
                                 use_bn=opt.bn)
        self.netAA = FcClassifier(AE_input_dim, cls_layers, output_dim=opt.input_dim_a, dropout=opt.dropout_rate,
                                  use_bn=opt.bn)

        # 词汇模型
        self.netL = FcClassifier(opt.input_dim_l, cls_layers, output_dim=opt.embd_size_l, dropout=opt.dropout_rate,
                                 use_bn=opt.bn)
        self.netLL = FcClassifier(AE_input_dim, cls_layers, output_dim=opt.input_dim_l, dropout=opt.dropout_rate,
                                  use_bn=opt.bn)

        # 视觉模型
        self.netV = FcClassifier(opt.input_dim_v, cls_layers, output_dim=opt.embd_size_v, dropout=opt.dropout_rate,
                                 use_bn=opt.bn)
        self.netVV = FcClassifier(AE_input_dim, cls_layers, output_dim=opt.input_dim_v, dropout=opt.dropout_rate,
                                  use_bn=opt.bn)

        # 自动编码器模型
        AE_layers = list(map(lambda x: int(x), opt.AE_layers.split(',')))
        self.netAE = SimpleFcAE(AE_layers, AE_input_dim, dropout=0, use_bn=False)
        # cls_input_size = AE_layers[-1]
        self.netC = FcClassifier(AE_input_dim, cls_layers, output_dim=opt.output_dim, dropout=opt.dropout_rate,
                                 use_bn=opt.bn)

        # 如果在训练阶段
        if self.isTrain:
            # 初始化优化器
            paremeters = [{'params': getattr(self, 'net' + net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
            # 输出维度
            self.output_dim = opt.output_dim
            # 对应交叉熵损失的权重
            self.ce_weight = opt.ce_weight
            # 对应均方误差损失的权重
            self.mse_weight = opt.mse_weight

        # 修改保存目录
        self.save_dir = os.path.join(self.save_dir, str(opt.cvNo))
        # 如果保存目录不存在，则创建
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
        # 将标签移动到设备上
        self.label = input['label'].to(self.device)
        # 将缺失索引转换为长整型并移动到设备上
        self.missing_index = input['missing_index'].long().to(self.device)

        # A 模态
        # 缺失索引向量，形状为 (batch_size, 1)
        self.A_miss_index = self.missing_index[:, 0].unsqueeze(1)
        # 缺失值为 0，存在值为 1 的特征
        self.A_miss = acoustic * self.A_miss_index
        # 存在值为 0，缺失值为 1 的特征
        self.A_reverse = acoustic * -1 * (self.A_miss_index - 1)
        # 完整的声学特征
        self.A_full = acoustic

        # L 模态
        # 缺失索引向量，形状为 (batch_size, 1)
        self.L_miss_index = self.missing_index[:, 2].unsqueeze(1)
        # 缺失值为 0，存在值为 1 的特征
        self.L_miss = lexical * self.L_miss_index
        # 存在值为 0，缺失值为 1 的特征
        self.L_reverse = lexical * -1 * (self.L_miss_index - 1)
        # 完整的词汇特征
        self.L_full = lexical

        # V 模态
        # 缺失索引向量，形状为 (batch_size, 1)
        self.V_miss_index = self.missing_index[:, 1].unsqueeze(1)
        # 缺失值为 0，存在值为 1 的特征
        self.V_miss = visual * self.V_miss_index
        # 存在值为 0，缺失值为 1 的特征
        self.V_reverse = visual * -1 * (self.V_miss_index - 1)
        # 完整的视觉特征
        self.V_full = visual

    def forward(self):
        """
        执行前向传播；由<optimize_parameters>和<test>函数调用。
        """

        # 重建部分
        self.feat_A_miss, _ = self.netA(self.A_miss)  # 获取缺失A特征
        self.feat_L_miss, _ = self.netL(self.L_miss)  # 获取缺失L特征
        self.feat_V_miss, _ = self.netV(self.V_miss)  # 获取缺失V特征
        self.feat_fusion_miss = torch.cat([self.feat_A_miss, self.feat_L_miss, self.feat_V_miss], dim=-1)  # 合并缺失特征
        self.recon_fusion, self.latent = self.netAE(self.feat_fusion_miss)  # 通过自编码器得到重建融合特征和潜在表示
        self.A_rec, _ = self.netAA(self.recon_fusion)  # 从重建融合特征中恢复A特征
        self.L_rec, _ = self.netLL(self.recon_fusion)  # 从重建融合特征中恢复L特征
        self.V_rec, _ = self.netVV(self.recon_fusion)  # 从重建融合特征中恢复V特征

        # 分类器
        self.hiddens = self.recon_fusion  # 使用重建融合特征作为隐藏层输入
        self.logits, _ = self.netC(self.recon_fusion)  # 得到分类的logits
        self.logits = self.logits.squeeze()  # 压缩logits维度
        self.pred = self.logits  # 预测标签

        # 计算分类损失
        if self.dataset in ['cmumosi', 'cmumosei']:  # 使用均方误差损失
            criterion_ce = torch.nn.MSELoss()
        if self.dataset in ['boxoflies', 'iemocapfour', 'iemocapsix']:  # 使用交叉熵损失
            criterion_ce = torch.nn.CrossEntropyLoss()
        self.loss_ce = criterion_ce(self.logits, self.label)  # 计算分类损失

        # 计算重建损失（仅在数据缺失时计算）
        recon_loss = torch.nn.MSELoss(reduction='none')  # 初始化均方误差损失
        loss_recon1 = recon_loss(self.A_rec, self.A_full) * -1 * (self.A_miss_index - 1)  # A特征的重建损失
        loss_recon2 = recon_loss(self.L_rec, self.L_full) * -1 * (self.L_miss_index - 1)  # L特征的重建损失
        loss_recon3 = recon_loss(self.V_rec, self.V_full) * -1 * (self.V_miss_index - 1)  # V特征的重建损失
        loss_recon1 = torch.sum(loss_recon1) / self.A_full.shape[1]  # 每一维的平均损失
        loss_recon2 = torch.sum(loss_recon2) / self.L_full.shape[1]  # 每一维的平均损失
        loss_recon3 = torch.sum(loss_recon3) / self.V_full.shape[1]  # 每一维的平均损失
        self.loss_recon = loss_recon1 + loss_recon2 + loss_recon3  # 总重建损失

        # 合并所有损失
        self.loss = self.ce_weight * self.loss_ce + self.mse_weight * self.loss_recon  # 综合分类损失和重建损失

    def backward(self):
        """进行反向传播计算损失"""
        # 计算损失函数的反向传播
        self.loss.backward()

        # 遍历模型名称列表
        for model in self.model_names:
            # 对每个模型的参数进行梯度裁剪，限制最大梯度范数为5
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net' + model).parameters(), 5)

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