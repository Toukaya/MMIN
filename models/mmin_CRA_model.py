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
from models.networks.autoencoder import ResidualAE
from models.utt_fusion_model import UttFusionModel
from .utils.config import OptConfig


class MMINCRAModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # 添加命令行参数：声学输入维度
        parser.add_argument('--input_dim_a', type=int, default=130, help='acoustic input dim')

        # 添加命令行参数：词汇输入维度
        parser.add_argument('--input_dim_l', type=int, default=1024, help='lexical input dim')

        # 添加命令行参数：视觉输入维度（与词汇输入维度相同）
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

        # 添加命令行参数：自编码器层的节点数，如'128,64,32'
        parser.add_argument('--AE_layers', type=str, default='128,64,32',
                            help='256,128 for 2 layers with 256, 128 nodes respectively')

        # 添加命令行参数：分类层的节点数，如'128,128'
        parser.add_argument('--cls_layers', type=str, default='128,128',
                            help='256,128 for 2 layers with 256, 128 nodes respectively')

        # 添加命令行参数：Dropout比例
        parser.add_argument('--dropout_rate', type=float, default=0.3, help='rate of dropout')

        # 添加命令行参数：是否使用批量归一化层
        parser.add_argument('--bn', action='store_true', help='if specified, use bn layers in FC')

        # 添加命令行参数：预训练模型路径
        parser.add_argument('--pretrained_path', type=str, help='where to load pretrained encoder network')

        # 添加命令行参数：交叉熵损失权重
        parser.add_argument('--ce_weight', type=float, default=1.0, help='weight of ce loss')

        # 添加命令行参数：均方误差损失权重
        parser.add_argument('--mse_weight', type=float, default=1.0, help='weight of mse loss')

        # 添加命令行参数：自编码器块的数量
        parser.add_argument('--n_blocks', type=int, default=3, help='number of AE blocks')

        # 返回解析器对象
        return parser

    # 初始化LSTM自动编码器类
    def __init__(self, opt):
        """初始化LSTM自动编码器类
        参数:
            opt (Option class)-- 存储所有实验标志，需要是BaseOptions的子类
        """
        # 调用父类的初始化方法
        super().__init__(opt)

        # 获取数据集模式
        self.dataset = opt.dataset_mode.split('_')[0]

        # 定义损失函数名称
        self.loss_names = ['ce', 'recon']

        # 定义模型名称
        self.model_names = ['A', 'AA', 'V', 'VV', 'L', 'LL', 'C', 'AE']

        # 将字符串列表转换为整数列表
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
        self.netAE = ResidualAE(AE_layers, opt.n_blocks, AE_input_dim, dropout=0, use_bn=False)
        # cls_input_size = AE_layers[-1] * opt.n_blocks
        self.netC = FcClassifier(AE_input_dim, cls_layers, output_dim=opt.output_dim, dropout=opt.dropout_rate,
                                 use_bn=opt.bn)

        # 如果在训练模式下
        if self.isTrain:
            # 初始化优化器
            paremeters = [{'params': getattr(self, 'net' + net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)

            # 输出维度
            self.output_dim = opt.output_dim
            # 损失权重
            self.ce_weight = opt.ce_weight
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
        # 创建一个表示 A 模态缺失位置的张量
        self.A_miss_index = self.missing_index[:, 0].unsqueeze(1)
        # 根据缺失位置设置 A 模态的缺失值
        self.A_miss = acoustic * self.A_miss_index
        # 计算 A 模态的反向填充值
        self.A_reverse = acoustic * -1 * (self.A_miss_index - 1)
        # 保留完整的 A 模态特征
        self.A_full = acoustic

        # L 模态
        # 创建一个表示 L 模态缺失位置的张量
        self.L_miss_index = self.missing_index[:, 2].unsqueeze(1)
        # 根据缺失位置设置 L 模态的缺失值
        self.L_miss = lexical * self.L_miss_index
        # 计算 L 模态的反向填充值
        self.L_reverse = lexical * -1 * (self.L_miss_index - 1)
        # 保留完整的 L 模态特征
        self.L_full = lexical

        # V 模态
        # 创建一个表示 V 模态缺失位置的张量
        self.V_miss_index = self.missing_index[:, 1].unsqueeze(1)
        # 根据缺失位置设置 V 模态的缺失值
        self.V_miss = visual * self.V_miss_index
        # 计算 V 模态的反向填充值
        self.V_reverse = visual * -1 * (self.V_miss_index - 1)
        # 保留完整的 V 模态特征
        self.V_full = visual

    def forward(self):
        """
        运行前向传播过程；由<optimize_parameters>和<test>两个函数调用。
        """

        # 重建部分
        # 通过网络A处理缺失的A特征
        self.feat_A_miss, _ = self.netA(self.A_miss)
        # 通过网络L处理缺失的L特征
        self.feat_L_miss, _ = self.netL(self.L_miss)
        # 通过网络V处理缺失的V特征
        self.feat_V_miss, _ = self.netV(self.V_miss)
        # 合并三个特征
        self.feat_fusion_miss = torch.cat([self.feat_A_miss, self.feat_L_miss, self.feat_V_miss], dim=-1)
        # 通过融合网络得到重建特征和潜在向量
        self.recon_fusion, self.latent = self.netAE(self.feat_fusion_miss)
        # 通过网络AA进行A特征的重建
        self.A_rec, _ = self.netAA(self.recon_fusion)
        # 通过网络LL进行L特征的重建
        self.L_rec, _ = self.netLL(self.recon_fusion)
        # 通过网络VV进行V特征的重建
        self.V_rec, _ = self.netVV(self.recon_fusion)

        # 分类器部分
        # 使用重建特征作为隐藏层输入
        self.hiddens = self.recon_fusion
        # 通过分类网络得到logits和_
        self.logits, _ = self.netC(self.recon_fusion)
        # 将logits挤压为一维
        self.logits = self.logits.squeeze()
        # 预测值
        self.pred = self.logits

        # 计算分类损失
        if self.dataset in ['cmumosi', 'cmumosei']:
            criterion_ce = torch.nn.MSELoss()  # 使用均方误差损失
        elif self.dataset in ['boxoflies', 'iemocapfour', 'iemocapsix']:
            criterion_ce = torch.nn.CrossEntropyLoss()  # 使用交叉熵损失
        self.loss_ce = criterion_ce(self.logits, self.label)

        # 计算重建损失（仅在特征缺失时计算）
        recon_loss = torch.nn.MSELoss(reduction='none')
        loss_recon1 = recon_loss(self.A_rec, self.A_full) * -1 * (self.A_miss_index - 1)  # 1（存在），0（缺失）[batch, featdim]
        loss_recon2 = recon_loss(self.L_rec, self.L_full) * -1 * (self.L_miss_index - 1)  # 1（存在），0（缺失）
        loss_recon3 = recon_loss(self.V_rec, self.V_full) * -1 * (self.V_miss_index - 1)  # 1（存在），0（缺失）
        # 按维度求和损失，除以特征维度数
        loss_recon1 = torch.sum(loss_recon1) / self.A_full.shape[1]  # 每个维度的差值
        loss_recon2 = torch.sum(loss_recon2) / self.L_full.shape[1]  # 每个维度的差值
        loss_recon3 = torch.sum(loss_recon3) / self.V_full.shape[1]  # 每个维度的差值
        # 计算总重建损失
        self.loss_recon = loss_recon1 + loss_recon2 + loss_recon3

        # 合并所有损失
        self.loss = self.ce_weight * self.loss_ce + self.mse_weight * self.loss_recon

    def backward(self):
        """进行反向传播计算损失"""
        # 计算损失函数的梯度
        self.loss.backward()
        # 遍历模型名称列表
        for model in self.model_names:
            # 对对应模型的参数进行梯度裁剪，限制最大范数为5，以防止梯度爆炸
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