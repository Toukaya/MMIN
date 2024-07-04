
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
from models.networks.autoencoder import ResidualAE_test
from models.utt_fusion_model import UttFusionModel
from .utils.config import OptConfig


class MMINAblationModel(BaseModel):
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

        # 添加命令行参数：自编码器块的数量
        parser.add_argument('--n_blocks', type=int, default=3, help='number of AE blocks')

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

        # 添加命令行参数：循环损失权重
        parser.add_argument('--cycle_weight', type=float, default=1.0, help='weight of cycle loss')

        # 添加命令行参数：实验设置，如'raw'
        parser.add_argument('--case', type=str, default='raw', help='weight of cycle loss')

        # 返回解析后的参数对象
        return parser

    # 初始化LSTM自动编码器类
    def __init__(self, opt):
        """初始化LSTM自动编码器类
        参数:
            opt (Option class)-- 存储所有实验标志; 需要是BaseOptions的子类
        """
        # 调用父类构造函数
        super().__init__(opt)

        # 我们的实验设置为10折，教师模型设置为5折，训练集应匹配
        self.loss_names = ['CE', 'mse', 'cycle']  # 损失名称列表
        self.model_names = ['A', 'V', 'L', 'C', 'AE', 'AE_cycle']  # 模型名称列表

        # 音频模型
        self.netA = LSTMEncoder(opt.input_dim_a, opt.embd_size_a, embd_method=opt.embd_method_a)
        # 词汇模型
        self.netL = TextCNN(opt.input_dim_l, opt.embd_size_l)
        # 视觉模型
        self.netV = LSTMEncoder(opt.input_dim_v, opt.embd_size_v, opt.embd_method_v)
        # 自动编码器模型
        AE_layers = list(map(lambda x: int(x), opt.AE_layers.split(',')))  # 自动编码器层数列表
        AE_input_dim = opt.embd_size_a + opt.embd_size_v + opt.embd_size_l  # 输入维度
        self.netAE = ResidualAE_test(AE_layers, opt.n_blocks, AE_input_dim, dropout=0, use_bn=False, case=opt.case)
        self.netAE_cycle = ResidualAE_test(AE_layers, opt.n_blocks, AE_input_dim, dropout=0, use_bn=False,
                                           case=opt.case)
        # 分类器模型
        cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))  # 分类器层数列表
        cls_input_size = AE_layers[-1] * opt.n_blocks  # 分类器输入大小
        self.netC = FcClassifier(cls_input_size, cls_layers, output_dim=opt.output_dim, dropout=opt.dropout_rate,
                                 use_bn=opt.bn)

        # 如果在训练模式下
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
            # 输出维度和权重
            self.output_dim = opt.output_dim
            self.ce_weight = opt.ce_weight
            self.mse_weight = opt.mse_weight
            self.cycle_weight = opt.cycle_weight

        # 修改保存目录
        self.save_dir = os.path.join(self.save_dir, str(opt.cvNo))
        # 如果保存目录不存在，则创建
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    # 定义加载预训练编码器的方法
    def load_pretrained_encoder(self, opt):
        # 打印信息，显示从哪个路径初始化参数
        print('Init parameter from {}'.format(opt.pretrained_path))

        # 根据预训练路径和cv编号构建完整路径
        pretrained_path = os.path.join(opt.pretrained_path, str(opt.cvNo))

        # 构建预训练配置文件路径
        pretrained_config_path = os.path.join(opt.pretrained_path, 'train_opt.conf')

        # 从配置文件加载预训练模型的配置
        pretrained_config = self.load_from_opt_record(pretrained_config_path)

        # 设置预训练模型为测试模式，因为教师模型不应在训练模式下运行
        pretrained_config.isTrain = False

        # 设置预训练模型的GPU ID与当前模型相同
        pretrained_config.gpu_ids = opt.gpu_ids

        # 初始化预训练编码器对象
        self.pretrained_encoder = UttFusionModel(pretrained_config)

        # 加载预训练模型的CV阶段网络权重
        self.pretrained_encoder.load_networks_cv(pretrained_path)

        # 将模型转移到GPU上
        self.pretrained_encoder.cuda()

        # 设置模型为评估模式
        self.pretrained_encoder.eval()

    def post_process(self):
        # 在模型完成setup()方法后被调用
        def transform_key_for_parallel(state_dict):
            # 定义一个函数，用于将state_dict中的键添加'module.'前缀，以适应数据并行处理
            return OrderedDict([('module.' + key, value) for key, value in state_dict.items()])

        # 如果是在训练模式下
        if self.isTrain:
            # 打印信息，初始化时从预训练的编码器网络加载参数
            print('[ Init ] Load parameters from pretrained encoder network')

            # 使用lambda函数f，将预训练的编码器网络的state_dict转换
            f = lambda x: transform_key_for_parallel(x)

            # 加载预训练的编码器网络的参数到对应的网络模块
            self.netA.load_state_dict(f(self.pretrained_encoder.netA.state_dict()))
            self.netV.load_state_dict(f(self.pretrained_encoder.netV.state_dict()))
            self.netL.load_state_dict(f(self.pretrained_encoder.netL.state_dict()))

    def load_from_opt_record(self, file_path):  # 加载优化器配置记录文件
        opt_content = json.load(open(file_path, 'r'))  # 从指定路径的文件中读取JSON格式的内容
        opt = OptConfig()  # 创建一个OptConfig实例
        opt.load(opt_content)  # 将读取到的JSON内容加载到OptConfig对象中
        return opt  # 返回加载完成的OptConfig对象

    def set_input(self, input):
        """
        解包数据加载器提供的输入数据，并执行必要的预处理步骤。
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
            # 将缺失索引移动到设备上
            self.missing_index = input['missing_index'].long().to(self.device)

            # 声学模态
            self.A_miss_index = self.missing_index[:, 0].unsqueeze(1).unsqueeze(2)
            # 计算缺失声学特征
            self.A_miss = acoustic * self.A_miss_index
            # 计算反向声学特征
            self.A_reverse = acoustic * -1 * (self.A_miss_index - 1)

            # 词汇模态
            self.L_miss_index = self.missing_index[:, 2].unsqueeze(1).unsqueeze(2)
            # 计算缺失词汇特征
            self.L_miss = lexical * self.L_miss_index
            # 计算反向词汇特征
            self.L_reverse = lexical * -1 * (self.L_miss_index - 1)

            # 视觉模态
            self.V_miss_index = self.missing_index[:, 1].unsqueeze(1).unsqueeze(2)
            # 计算缺失视觉特征
            self.V_miss = visual * self.V_miss_index
            # 计算反向视觉特征
            self.V_reverse = visual * -1 * (self.V_miss_index - 1)
        # 如果处于验证或测试模式
        else:
            # 直接将声学特征设为缺失
            self.A_miss = acoustic
            # 直接将视觉特征设为缺失
            self.V_miss = visual
            # 直接将词汇特征设为缺失
            self.L_miss = lexical

    def forward(self):
        """
        执行前向传播；由<optimize_parameters>和<test>两个函数调用。
        """

        # 获取 utterance 级别的表示
        self.feat_A_miss = self.netA(self.A_miss)  # 使用 netA 计算 A 模态的缺失特征
        self.feat_L_miss = self.netL(self.L_miss)  # 使用 netL 计算 L 模态的缺失特征
        self.feat_V_miss = self.netV(self.V_miss)  # 使用 netV 计算 V 模态的缺失特征

        # 合并缺失模态的特征
        self.feat_fusion_miss = torch.cat([self.feat_A_miss, self.feat_L_miss, self.feat_V_miss], dim=-1)

        # 计算教师模型输出的重构
        self.recon_fusion, self.latent = self.netAE(self.feat_fusion_miss)  # 通过 netAE 重构融合特征，得到潜在向量
        self.recon_cycle, self.latent_cycle = self.netAE_cycle(self.recon_fusion)  # 通过 netAE_cycle 进行循环重构

        # 获取缺失模态的融合输出
        self.logits, _ = self.netC(self.latent)  # 通过 netC 得到分类的 logit 值
        self.pred = F.softmax(self.logits, dim=-1)  # 对 logit 应用 softmax 函数，得到预测概率

        # 如果处于训练模式
        if self.isTrain:
            # 不更新梯度地获取教师模型的嵌入
            with torch.no_grad():
                self.T_embd_A = self.pretrained_encoder.netA(self.A_reverse)  # 教师模型的 A 模态嵌入
                self.T_embd_L = self.pretrained_encoder.netL(self.L_reverse)  # 教师模型的 L 模态嵌入
                self.T_embd_V = self.pretrained_encoder.netV(self.V_reverse)  # 教师模型的 V 模态嵌入
                self.T_embds = torch.cat([self.T_embd_A, self.T_embd_L, self.T_embd_V], dim=-1)  # 合并所有模态的教师模型嵌入

    def backward(self):
        """
        计算反向传播过程中的损失值
        """
        # 计算交叉熵损失，乘以权重self.ce_weight
        self.loss_CE = self.ce_weight * self.criterion_ce(self.logits, self.label)

        # 计算均方误差损失，乘以权重self.mse_weight
        self.loss_mse = self.mse_weight * self.criterion_mse(self.T_embds, self.recon_fusion)

        # 计算循环一致性损失，乘以权重self.cycle_weight
        self.loss_cycle = self.cycle_weight * self.criterion_mse(self.feat_fusion_miss.detach(), self.recon_cycle)

        # 汇总所有损失
        loss = self.loss_CE + self.loss_mse + self.loss_cycle

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
        self.backward()  # 计算损失和梯度
        self.optimizer.step()  # 根据梯度更新网络参数

    # 定义一个函数transform_key_for_parallel，用于处理state_dict
    def transform_key_for_parallel(state_dict):
        # 使用OrderedDict创建一个新的字典，其中每个键值对通过在原key前添加'module.'来转换
        return OrderedDict([('module.' + key, value) for key, value in state_dict.items()])
