
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


class MMINNoCycleModel(BaseModel):
    @staticmethod
    # 定义一个函数，用于在命令行参数解析器中添加训练或测试所需的参数
    def modify_commandline_options(parser, is_train=True):
        # 添加声学输入维度参数
        parser.add_argument('--input_dim_a', type=int, default=130, help='声学输入维度')

        # 添加词汇输入维度参数
        parser.add_argument('--input_dim_l', type=int, default=1024, help='词汇输入维度')

        # 添加视觉输入维度参数
        parser.add_argument('--input_dim_v', type=int, default=384, help='视觉输入维度')

        # 添加音频模型嵌入大小参数
        parser.add_argument('--embd_size_a', default=128, type=int, help='音频模型嵌入大小')

        # 添加文本模型嵌入大小参数
        parser.add_argument('--embd_size_l', default=128, type=int, help='文本模型嵌入大小')

        # 添加视觉模型嵌入大小参数
        parser.add_argument('--embd_size_v', default=128, type=int, help='视觉模型嵌入大小')

        # 添加音频嵌入方法参数，可选'last', 'maxpool', 'attention'
        parser.add_argument('--embd_method_a', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'],
                            help='音频嵌入方法，last,mean或atten')

        # 添加视觉嵌入方法参数，可选'last', 'maxpool', 'attention'
        parser.add_argument('--embd_method_v', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'],
                            help='视觉嵌入方法，last,mean或atten')

        # 添加自编码器层节点数参数，如'128,64,32'
        parser.add_argument('--AE_layers', type=str, default='128,64,32',
                            help='自编码器层节点数，例如2层分别为256, 128个节点')

        # 添加自编码器块的数量
        parser.add_argument('--n_blocks', type=int, default=3, help='自编码器块的数量')

        # 添加分类层节点数参数，如'128,128'
        parser.add_argument('--cls_layers', type=str, default='128,128',
                            help='分类层节点数，例如2层分别为256, 128个节点')

        # 添加Dropout比例参数
        parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout比例')

        # 添加是否使用批量归一化层的标志
        parser.add_argument('--bn', action='store_true', help='如果指定，使用批量归一化层')

        # 添加预训练模型路径参数
        parser.add_argument('--pretrained_path', type=str, help='加载预训练编码器网络的路径')

        # 添加交叉熵损失权重参数
        parser.add_argument('--ce_weight', type=float, default=1.0, help='交叉熵损失权重')

        # 添加均方误差损失权重参数
        parser.add_argument('--mse_weight', type=float, default=1.0, help='均方误差损失权重')

        # 添加循环损失权重参数
        parser.add_argument('--cycle_weight', type=float, default=1.0, help='循环损失权重')

        # 返回解析器对象
        return parser

    # 初始化LSTM自动编码器类
    def __init__(self, opt):
        """初始化LSTM自动编码器类，接收一个存储实验标志的Option类，需要是BaseOptions的子类
        参数:
            opt (Option class)-- 存储所有实验标志
        """
        # 调用父类（即BaseModel）的初始化方法
        super().__init__(opt)

        # 我们的实验设置为10折，教师模型设置为5折，训练集应匹配
        self.loss_names = ['CE', 'mse']  # 损失函数名称：交叉熵和均方误差
        self.model_names = ['A', 'V', 'L', 'C', 'AE']  # 模型名称：Acoustic, Visual, Lexical, Classifier 和 AutoEncoder

        # 音频模型
        self.netA = LSTMEncoder(opt.input_dim_a, opt.embd_size_a, embd_method=opt.embd_method_a)
        # 词汇模型
        self.netL = TextCNN(opt.input_dim_l, opt.embd_size_l)
        # 视觉模型
        self.netV = LSTMEncoder(opt.input_dim_v, opt.embd_size_v, opt.embd_method_v)
        # 自动编码器模型
        AE_layers = list(map(lambda x: int(x), opt.AE_layers.split(',')))  # 解析自动编码器层数
        AE_input_dim = opt.embd_size_a + opt.embd_size_v + opt.embd_size_l  # 输入维度
        self.netAE = ResidualAE(AE_layers, opt.n_blocks, AE_input_dim, dropout=0, use_bn=False)
        self.netAE_cycle = ResidualAE(AE_layers, opt.n_blocks, AE_input_dim, dropout=0, use_bn=False)
        # 分类器模型
        cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))  # 解析分类器层数
        cls_input_size = AE_layers[-1] * opt.n_blocks  # 分类器输入大小
        self.netC = FcClassifier(cls_input_size, cls_layers, output_dim=opt.output_dim, dropout=opt.dropout_rate,
                                 use_bn=opt.bn)

        # 如果是在训练模式下
        if self.isTrain:
            # 加载预训练的编码器
            self.load_pretrained_encoder(opt)
            # 定义损失函数
            self.criterion_ce = torch.nn.CrossEntropyLoss()  # 交叉熵损失
            self.criterion_mse = torch.nn.MSELoss()  # 均方误差损失
            # 初始化优化器；调度器将在BaseModel.setup函数中自动创建
            paremeters = [{'params': getattr(self, 'net' + net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999))  # 使用Adam优化器
            self.optimizers.append(self.optimizer)  # 添加到优化器列表
            self.output_dim = opt.output_dim  # 输出维度
            self.ce_weight = opt.ce_weight  # 交叉熵损失权重
            self.mse_weight = opt.mse_weight  # 均方误差损失权重
            self.cycle_weight = opt.cycle_weight  # 循环损失权重

        # 修改保存目录
        self.save_dir = os.path.join(self.save_dir, str(opt.cvNo))
        # 如果保存目录不存在，则创建
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    # 初始化预训练模型参数
    def load_pretrained_encoder(self, opt):
        # 打印预训练路径信息
        print('Init parameter from {}'.format(opt.pretrained_path))

        # 根据预训练路径和cv编号构建完整路径
        pretrained_path = os.path.join(opt.pretrained_path, str(opt.cvNo))

        # 构建预训练配置文件路径
        pretrained_config_path = os.path.join(opt.pretrained_path, 'train_opt.conf')

        # 从配置文件加载预训练模型的配置
        pretrained_config = self.load_from_opt_record(pretrained_config_path)

        # 设置预训练模型为测试模式，因为是教师模型
        pretrained_config.isTrain = False

        # 设置GPU ID与当前模型相同
        pretrained_config.gpu_ids = opt.gpu_ids

        # 创建预训练的UttFusionModel实例
        self.pretrained_encoder = UttFusionModel(pretrained_config)

        # 加载预训练模型的CV阶段网络参数
        self.pretrained_encoder.load_networks_cv(pretrained_path)

        # 将模型转移到GPU上
        self.pretrained_encoder.cuda()

        # 设置模型为评估模式
        self.pretrained_encoder.eval()

    def post_process(self):
        # 在模型.setup()方法调用后被调用
        def transform_key_for_parallel(state_dict):
            # 将state_dict中的键添加'module.'前缀，以适应数据并行处理
            return OrderedDict([('module.' + key, value) for key, value in state_dict.items()])

        # 打印信息：从预训练的编码器网络加载参数
        print('[ Init ] Load parameters from pretrained encoder network')

        # 定义一个函数f，将state_dict中的键进行转换
        f = lambda x: transform_key_for_parallel(x)

        # 加载预训练的编码器网络参数到对应的网络模块
        self.netA.load_state_dict(f(self.pretrained_encoder.netA.state_dict()))
        self.netV.load_state_dict(f(self.pretrained_encoder.netV.state_dict()))
        self.netL.load_state_dict(f(self.pretrained_encoder.netL.state_dict()))

    def load_from_opt_record(self, file_path):
        # 从指定文件路径中加载JSON内容
        opt_content = json.load(open(file_path, 'r'))

        # 初始化OptConfig对象
        opt = OptConfig()

        # 将加载的内容加载到OptConfig对象中
        opt.load(opt_content)

        # 返回配置对象
        return opt

    def set_input(self, input):
        """
        解包从数据加载器获取的输入数据，并执行必要的预处理步骤。
        参数:
            input (dict): 包含数据本身及其元数据信息。
        """
        # 将声学特征转换为浮点数并移动到设备上
        self.acoustic = input['A_feat'].float().to(self.device)

        # 将词汇特征转换为浮点数并移动到设备上
        self.lexical = input['L_feat'].float().to(self.device)

        # 将视觉特征转换为浮点数并移动到设备上
        self.visual = input['V_feat'].float().to(self.device)

        # 将缺失索引转换为长整型并移动到设备上
        self.missing_index = input['missing_index'].long().to(self.device)

        # 将标签转换并移动到设备上
        self.label = input['label'].to(self.device)

    def forward(self):
        """
        执行前向传播；由<optimize_parameters>和<test>两个函数调用。
        """

        # A 模态
        self.A_miss_index = self.missing_index[:, 0].unsqueeze(1).unsqueeze(2)  # 缺失索引，用于A模态
        self.feat_A_miss = self.netA(self.acoustic * self.A_miss_index)  # 应用netA到声学特征并考虑缺失信息

        # L 模态
        self.L_miss_index = self.missing_index[:, 2].unsqueeze(1).unsqueeze(2)  # 缺失索引，用于L模态
        self.feat_L_miss = self.netL(self.lexical * self.L_miss_index)  # 应用netL到词汇特征并考虑缺失信息

        # V 模态
        self.V_miss_index = self.missing_index[:, 1].unsqueeze(1).unsqueeze(2)  # 缺失索引，用于V模态
        self.feat_V_miss = self.netV(self.visual * self.V_miss_index)  # 应用netV到视觉特征并考虑缺失信息

        # 合并缺失模态
        self.feat_fusion_miss = torch.cat([self.feat_A_miss, self.feat_L_miss, self.feat_V_miss],
                                          dim=-1)  # 沿着最后一维合并所有模态特征

        # 计算教师输出的重构
        self.recon_fusion, self.latent = self.netAE(self.feat_fusion_miss)  # 应用netAE得到重构和潜在表示

        # 获取缺失模态的融合输出
        self.logits, _ = self.netC(self.latent)  # 应用netC到潜在表示得到logits
        self.pred = F.softmax(self.logits, dim=-1)  # 对logits进行softmax激活，得到预测概率

        # 训练部分
        if self.isTrain:  # 如果在训练模式下
            with torch.no_grad():  # 在无梯度计算环境下
                self.T_embd_A = self.pretrained_encoder.netA(self.acoustic)  # 预训练编码器的A模态嵌入
                self.T_embd_L = self.pretrained_encoder.netL(self.lexical)  # 预训练编码器的L模态嵌入
                self.T_embd_V = self.pretrained_encoder.netV(self.visual)  # 预训练编码器的V模态嵌入
                self.T_embds = torch.cat([self.T_embd_A, self.T_embd_L, self.T_embd_V], dim=-1)  # 沿着最后一维合并所有预训练嵌入

    def backward(self):
        """
        计算反向传播过程中的损失值
        """
        # 计算交叉熵损失，乘以权重self.ce_weight
        self.loss_CE = self.ce_weight * self.criterion_ce(self.logits, self.label)

        # 计算均方误差损失，乘以权重self.mse_weight
        self.loss_mse = self.mse_weight * self.criterion_mse(self.T_embds, self.recon_fusion)

        # 汇总两种损失
        loss = self.loss_CE + self.loss_mse

        # 反向传播计算梯度
        loss.backward()

        # 对每个模型的参数进行梯度裁剪，最大范数为5
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net' + model).parameters(), 5)

    def optimize_parameters(self, epoch):
        """
        计算损失、梯度，并更新网络权重；在每个训练迭代中被调用
        """
        # 前向传播
        self.forward()
        # 反向传播
        self.optimizer.zero_grad()  # 清零优化器的梯度
        self.backward()
        # 使用优化器根据梯度更新网络参数
        self.optimizer.step()

    # 定义一个函数transform_key_for_parallel，用于处理state_dict中的键值对
    def transform_key_for_parallel(state_dict):
        # 使用OrderedDict来保持键的顺序，遍历state_dict中的所有键值对
        # 在每个键前添加'module.'，然后与对应的值一起构成新的键值对
        return OrderedDict([('module.' + key, value) for key, value in state_dict.items()])
