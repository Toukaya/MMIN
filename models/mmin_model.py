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


class MMINModel(BaseModel):
    @staticmethod
    # 定义一个函数，用于在命令行参数中添加训练或测试所需的选项
    def modify_commandline_options(parser, is_train=True):
        # 添加声学输入维度参数，默认值为130
        parser.add_argument('--input_dim_a', type=int, default=130, help='acoustic input dim')

        # 添加词汇输入维度参数，默认值为1024
        parser.add_argument('--input_dim_l', type=int, default=1024, help='lexical input dim')

        # 添加视觉输入维度参数，默认值为384
        parser.add_argument('--input_dim_v', type=int, default=384, help='lexical input dim')

        # 添加音频模型嵌入大小参数，默认值为128
        parser.add_argument('--embd_size_a', default=128, type=int, help='audio model embedding size')

        # 添加文本模型嵌入大小参数，默认值为128
        parser.add_argument('--embd_size_l', default=128, type=int, help='text model embedding size')

        # 添加视觉模型嵌入大小参数，默认值为128
        parser.add_argument('--embd_size_v', default=128, type=int, help='visual model embedding size')

        # 添加音频嵌入方法参数，可选'last', 'maxpool', 'attention'，默认为'maxpool'
        parser.add_argument('--embd_method_a', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'],
                            help='audio embedding method,last,mean or atten')

        # 添加视觉嵌入方法参数，可选'last', 'maxpool', 'attention'，默认为'maxpool'
        parser.add_argument('--embd_method_v', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'],
                            help='visual embedding method,last,mean or atten')

        # 添加自编码器层的节点数配置，默认为'128,64,32'
        parser.add_argument('--AE_layers', type=str, default='128,64,32',
                            help='256,128 for 2 layers with 256, 128 nodes respectively')

        # 添加自编码器块的数量，默认为3
        parser.add_argument('--n_blocks', type=int, default=3, help='number of AE blocks')

        # 添加分类层的节点数配置，默认为'128,128'
        parser.add_argument('--cls_layers', type=str, default='128,128',
                            help='256,128 for 2 layers with 256, 128 nodes respectively')

        # 添加Dropout比例，默认为0.3
        parser.add_argument('--dropout_rate', type=float, default=0.3, help='rate of dropout')

        # 添加是否使用批量归一化层的标志，默认不使用
        parser.add_argument('--bn', action='store_true', help='if specified, use bn layers in FC')

        # 添加预训练模型路径参数
        parser.add_argument('--pretrained_path', type=str, help='where to load pretrained encoder network')

        # 添加交叉熵损失权重，默认为1.0
        parser.add_argument('--ce_weight', type=float, default=1.0, help='weight of ce loss')

        # 添加均方误差损失权重，默认为1.0
        parser.add_argument('--mse_weight', type=float, default=1.0, help='weight of mse loss')

        # 添加循环损失权重，默认为1.0
        parser.add_argument('--cycle_weight', type=float, default=1.0, help='weight of cycle loss')

        # 返回带有参数的解析器对象
        return parser

    # 初始化LSTM自动编码器类
    def __init__(self, opt):
        """初始化LSTM自动编码器类
        参数:
            opt (Option类) -- 存储所有实验标志，需要是BaseOptions的子类
        """
        super().__init__(opt)
        # 我们的实验设置为10折，教师设置为5折，训练集应匹配
        self.loss_names = ['CE', 'mse', 'cycle']  # 损失函数名称
        self.model_names = ['A', 'V', 'L', 'C', 'AE', 'AE_cycle']  # 每一层的名称

        # 分类层的索引
        cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))

        # 音频模型
        #############################
        # 原始代码使用LSTMEncoder，这里用FcClassifier替代
        self.netA = FcClassifier(opt.input_dim_a, cls_layers, output_dim=opt.embd_size_a, dropout=opt.dropout_rate,
                                 use_bn=opt.bn)
        #############################

        # 词汇模型
        #############################
        # 原始代码使用TextCNN，这里用FcClassifier替代
        self.netL = FcClassifier(opt.input_dim_l, cls_layers, output_dim=opt.embd_size_l, dropout=opt.dropout_rate,
                                 use_bn=opt.bn)
        #############################

        # 视觉模型
        #############################
        # 原始代码使用LSTMEncoder，这里用FcClassifier替代
        self.netV = FcClassifier(opt.input_dim_v, cls_layers, output_dim=opt.embd_size_v, dropout=opt.dropout_rate,
                                 use_bn=opt.bn)
        #############################

        # 自动编码器模型
        AE_layers = list(map(lambda x: int(x), opt.AE_layers.split(',')))
        AE_input_dim = opt.embd_size_a + opt.embd_size_v + opt.embd_size_l
        self.netAE = ResidualAE(AE_layers, opt.n_blocks, AE_input_dim, dropout=0, use_bn=False)
        self.netAE_cycle = ResidualAE(AE_layers, opt.n_blocks, AE_input_dim, dropout=0, use_bn=False)
        cls_input_size = AE_layers[-1] * opt.n_blocks
        self.netC = FcClassifier(cls_input_size, cls_layers, output_dim=opt.output_dim, dropout=opt.dropout_rate,
                                 use_bn=opt.bn)

        if self.isTrain:
            # 加载预训练的编码器
            self.load_pretrained_encoder(opt)
            #############################
            # 根据数据集类型选择损失函数
            dataset = opt.dataset_mode.split('_')[0]
            if dataset in ['cmumosi', 'cmumosei']:   self.criterion_ce = torch.nn.MSELoss()
            if dataset in ['boxoflies', 'iemocapfour', 'iemocapsix']: self.criterion_ce = torch.nn.CrossEntropyLoss()
            #############################
            self.criterion_mse = torch.nn.MSELoss()
            # 初始化优化器；调度器将由BaseModel.setup函数自动创建
            paremeters = [{'params': getattr(self, 'net' + net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
            self.output_dim = opt.output_dim
            self.ce_weight = opt.ce_weight
            self.mse_weight = opt.mse_weight
            self.cycle_weight = opt.cycle_weight

        # 修改保存目录
        self.save_dir = os.path.join(self.save_dir, str(opt.cvNo))
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    # 初始化预训练编码器函数
    def load_pretrained_encoder(self, opt):
        # 打印从哪个路径加载参数
        print('Init parameter from {}'.format(opt.pretrained_path))

        # 根据预训练路径和cv编号构建完整路径
        pretrained_path = os.path.join(opt.pretrained_path, str(opt.cvNo))

        # 获取预训练配置文件路径
        pretrained_config_path = os.path.join(opt.pretrained_path, 'train_opt.conf')

        # 从配置文件加载预训练模型的配置
        pretrained_config = self.load_from_opt_record(pretrained_config_path)

        # 设置预训练模型为测试模式
        pretrained_config.isTrain = False

        # 设置预训练模型的GPU与当前模型相同
        pretrained_config.gpu_ids = opt.gpu_ids

        # 创建预训练编码器实例
        self.pretrained_encoder = UttFusionModel(pretrained_config)

        # 加载预训练模型的cv版本网络参数
        self.pretrained_encoder.load_networks_cv(pretrained_path)

        # 将预训练模型移动到GPU上
        self.pretrained_encoder.cuda()

        # 设置预训练模型为评估模式
        self.pretrained_encoder.eval()

    def post_process(self):
        # 在模型完成setup()方法后被调用
        def transform_key_for_parallel(state_dict):
            # 将state_dict中的键添加'module.'前缀，以适应数据并行处理
            return OrderedDict([('module.' + key, value) for key, value in state_dict.items()])

        # 如果是在训练模式下
        if self.isTrain:
            # 打印信息：从预训练的编码器网络加载参数
            print('[ Init ] Load parameters from pretrained encoder network')

            # 定义转换函数f，将state_dict的键进行转换
            f = lambda x: transform_key_for_parallel(x)

            # 加载预训练编码器的网络A的参数
            self.netA.load_state_dict(f(self.pretrained_encoder.netA.state_dict()))

            # 加载预训练编码器的网络V的参数
            self.netV.load_state_dict(f(self.pretrained_encoder.netV.state_dict()))

            # 加载预训练编码器的网络L的参数
            self.netL.load_state_dict(f(self.pretrained_encoder.netL.state_dict()))

    def load_from_opt_record(self, file_path):
        # 从给定的文件路径中读取json内容
        opt_content = json.load(open(file_path, 'r'))

        # 初始化OptConfig对象
        opt = OptConfig()

        # 加载json内容到OptConfig对象中
        opt.load(opt_content)

        # 返回加载后的OptConfig对象
        return opt

    def set_input(self, input):
        """
        解包输入数据，从数据加载器中获取，并执行必要的预处理步骤。
        参数:
            input (dict): 包含数据本身及其元数据信息。
        """
        # 将声学特征转换为浮点数并将其移动到设备上
        acoustic = input['A_feat'].float().to(self.device)  # [256, 512]
        # 将词汇特征转换为浮点数并将其移动到设备上
        lexical = input['L_feat'].float().to(self.device)  # [256, 1024]
        # 将视觉特征转换为浮点数并将其移动到设备上
        visual = input['V_feat'].float().to(self.device)  # [256, 1024]
        # 将标签移动到设备上
        self.label = input['label'].to(self.device)  # [256]
        # 将缺失索引转换为长整型并移动到设备上
        self.missing_index = input['missing_index'].long().to(self.device)  # [256, 3]

        # A 模态处理
        #############################################
        # 创建一个形状为 [256, 1] 的缺失索引向量
        self.A_miss_index = self.missing_index[:, 0].unsqueeze(1)
        # 根据缺失索引计算缺失的声学特征
        self.A_miss = acoustic * self.A_miss_index
        # 计算非缺失的声学特征
        self.A_reverse = acoustic * -1 * (self.A_miss_index - 1)

        # L 模态处理
        self.L_miss_index = self.missing_index[:, 2].unsqueeze(1)
        self.L_miss = lexical * self.L_miss_index
        self.L_reverse = lexical * -1 * (self.L_miss_index - 1)

        # V 模态处理
        self.V_miss_index = self.missing_index[:, 1].unsqueeze(1)
        self.V_miss = visual * self.V_miss_index
        self.V_reverse = visual * -1 * (self.V_miss_index - 1)

    def forward(self):
        """
        执行前向传播；由<optimize_parameters>和<test>两个函数调用。
        """

        # 获取 utterance 级别的表示
        ###############################################
        # 使用 netA 计算缺失模态 A 的特征
        self.feat_A_miss, _ = self.netA(self.A_miss)
        # 使用 netL 计算缺失模态 L 的特征
        self.feat_L_miss, _ = self.netL(self.L_miss)
        # 使用 netV 计算缺失模态 V 的特征
        self.feat_V_miss, _ = self.netV(self.V_miss)
        ###############################################

        # 合并缺失模态的特征
        self.feat_fusion_miss = torch.cat([self.feat_A_miss, self.feat_L_miss, self.feat_V_miss], dim=-1)

        # 计算教师输出的重构
        self.recon_fusion, self.latent = self.netAE(self.feat_fusion_miss)
        # 循环重构
        self.recon_cycle, self.latent_cycle = self.netAE_cycle(self.recon_fusion)

        # 获取用于缺失模态的融合输出
        self.hiddens = self.latent
        self.logits, _ = self.netC(self.latent)

        #############################
        # 压缩 logits 并将其转换为预测概率
        self.logits = self.logits.squeeze()
        self.pred = self.logits

        # 初始化重构损失为零
        self.loss_recon = torch.zeros(1)
        #############################

        # 如果处于训练模式
        if self.isTrain:
            with torch.no_grad():
                ###############################################
                # 使用预训练编码器计算反向模态 A 的特征
                self.T_embd_A, _ = self.pretrained_encoder.netA(self.A_reverse)
                # 使用预训练编码器计算反向模态 L 的特征
                self.T_embd_L, _ = self.pretrained_encoder.netL(self.L_reverse)
                # 使用预训练编码器计算反向模态 V 的特征
                self.T_embd_V, _ = self.pretrained_encoder.netV(self.V_reverse)
                ###############################################

                # 合并反向模态的特征
                self.T_embds = torch.cat([self.T_embd_A, self.T_embd_L, self.T_embd_V], dim=-1)

    def backward(self):
        """
        计算反向传播过程中的损失
        """
        # 计算交叉熵损失，乘以权重self.ce_weight
        self.loss_CE = self.ce_weight * self.criterion_ce(self.logits, self.label)

        # 计算均方误差损失，乘以权重self.mse_weight
        self.loss_mse = self.mse_weight * self.criterion_mse(self.T_embds, self.recon_fusion)

        # 计算循环损失，乘以权重self.cycle_weight
        self.loss_cycle = self.cycle_weight * self.criterion_mse(self.feat_fusion_miss.detach(), self.recon_cycle)

        # 汇总所有损失
        loss = self.loss_CE + self.loss_mse + self.loss_cycle

        # 反向传播计算梯度
        loss.backward()

        # 对每个模型的参数进行梯度裁剪，最大值为5
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
        self.backward()  # 计算损失和梯度
        self.optimizer.step()  # 根据梯度更新网络参数

    # 定义一个函数transform_key_for_parallel，用于处理state_dict中的键值对
    def transform_key_for_parallel(state_dict):
        # 使用OrderedDict来保持键的顺序，遍历state_dict中的所有键值对
        return OrderedDict([
            # 将每个键前加上'module.'，并保留原值，构造新的键值对
            ('module.' + key, value)
            for key, value in state_dict.items()
        ])