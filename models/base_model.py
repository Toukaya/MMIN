import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from .networks import tools


class BaseModel(ABC):
    """这个类是模型的一个抽象基类（ABC）。
    要创建一个子类，你需要实现以下五个函数：
        -- <__init__>:                      初始化类；首先调用 BaseModel.__init__(self, opt)。
        -- <set_input>:                     从数据集中解包数据并应用预处理。
        -- <forward>:                       产生中间结果。
        -- <optimize_parameters>:           计算损失、梯度并更新网络权重。
        -- <modify_commandline_options>:    （可选）添加模型特定的选项并设置默认选项。
    """

    # 初始化 BaseModel 类
    def __init__(self, opt):
        """初始化 BaseModel 类，需要传入一个 Option 类型的参数 opt，用于存储所有实验标志。
        opt 必须是 BaseOptions 的子类。

        在创建自定义类时，你需要实现自己的初始化方法。
        在这个函数中，首先调用 <BaseModel.__init__(self, opt)>
        然后，定义四个列表：
            - self.loss_names (str 列表): 指定你想要绘制和保存的训练损失。
            - self.model_names (str 列表): 指定你想要显示和保存的图像。
            - self.visual_names (str 列表): 定义训练中使用的网络。
            - self.optimizers (optimizer 列表): 定义并初始化优化器。你可以为每个网络定义一个优化器。如果两个网络同时更新，你可以使用 itertools.chain 来组合它们。参考 cycle_gan_model.py 文件中的示例。
        """
        # 保存 opt 参数
        self.opt = opt
        # 获取 GPU ID 列表
        self.gpu_ids = opt.gpu_ids
        # 判断是否处于训练模式
        self.isTrain = opt.isTrain
        # 根据 GPU ID 获取设备名称（CPU 或 GPU）
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        # 创建保存所有检查点的目录
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        # 如果启用 CUDA 性能基准测试
        if opt.cuda_benchmark:
            # 注意：当使用 [scale_width] 时，输入图像可能有不同的大小，这会影响 cudnn.benchmark 的性能
            torch.backends.cudnn.benchmark = True
        # 初始化损失名称列表
        self.loss_names = []
        # 初始化模型名称列表
        self.model_names = []
        # 初始化优化器列表
        self.optimizers = []
        # 初始化评估指标为 0
        self.metric = 0

    def modify_commandline_options(parser, is_train):
        """向命令行选项中添加特定于模型的新选项，并重写现有选项的默认值。

        参数:
            parser          -- 原始选项解析器
            is_train (bool) -- 表示当前是训练阶段还是测试阶段。你可以根据这个标志添加训练特定或测试特定的选项。

        返回:
            修改后的解析器。
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """将数据加载器提供的输入数据解包，并执行必要的预处理步骤。

        参数:
            input (dict): 包含数据本身及其元数据信息。
        """
        # 这里需要实现具体的输入数据处理逻辑，但原始代码中使用了pass，表示该方法目前为空
        pass

    @abstractmethod
    def forward(self):
        """执行前向传播；在<optimize_parameters>和<test>两个函数中被调用。"""
        # 这里需要实现前向传播的逻辑，但原始代码使用了pass，表示该方法目前为空
        pass

    @abstractmethod
    def optimize_parameters(self):
        """
        计算损失值、梯度，并更新网络权重；在每个训练迭代中都会被调用
        """
        # 实现计算损失、梯度和更新网络参数的逻辑
        pass

    def setup(self, opt):
        """
        加载并打印网络；创建调度器

        参数:
            opt (Option类) -- 存储所有实验标志，需要是BaseOptions的子类
        """
        if self.isTrain:  # 如果在训练阶段
            self.schedulers = [tools.get_scheduler(optimizer, opt) for optimizer in self.optimizers]  # 创建Adam优化器调度器列表
            for name in self.model_names:  # 模型名称，例如：['A', 'V', 'L', 'C', 'AE', 'AE_cycle'] 或 ['C', 'A', 'L', 'V']
                net = getattr(self, 'net' + name)  # 获取对应名称的网络
                net = tools.init_net(net, opt.init_type, opt.init_gain, opt.gpu_ids)  # 初始化网络，根据opt参数
                setattr(self, 'net' + name, net)  # 将初始化后的网络重新赋值给self对象
        else:  # 如果在测试阶段
            self.eval()  # 设置模型为评估模式

        self.print_networks(opt.verbose)  # 打印网络结构，如果opt.verbose为True
        self.post_process()  # 进行后期处理

    def cuda(self):
        # 检查CUDA（GPU）是否可用
        assert (torch.cuda.is_available())

        # 遍历模型名称列表
        for name in self.model_names:
            # 获取对应名称的网络模型
            net = getattr(self, 'net' + name)

            # 将网络模型移动到第一个GPU设备上
            net.to(self.gpu_ids[0])

            # 使用DataParallel将模型分发到指定的GPU IDs上，以实现并行计算
            net = torch.nn.DataParallel(net, self.gpu_ids)

    def eval(self):
        """在测试时切换模型到评估模式"""
        self.isTrain = False  # 设置训练状态为False，表示进入评估模式
        for name in self.model_names:  # 遍历模型名称列表 ['A', 'V', 'L', 'C']
            if isinstance(name, str):  # 检查名称是否为字符串类型
                net = getattr(self, 'net' + name)  # 获取对应的网络模型（如：netA, netV等）
                net.eval()  # 将网络模型设置为评估模式

    def train(self):
        """在测试后将模型恢复为训练模式"""
        self.isTrain = True  # 设置训练状态为True
        for name in self.model_names:  # 遍历模型名称列表
            if isinstance(name, str):  # 检查名称是否为字符串
                net = getattr(self, 'net' + name)  # 获取对应名称的网络模型
                net.train()  # 将网络模型设置为训练模式

    def test(self):
        """
        在测试时使用的前向传播函数。

        此函数使用no_grad()上下文管理器，确保在计算过程中不保存中间步骤，以进行反向传播。
        它还会调用<compute_visuals>来生成额外的可视化结果。
        """
        with torch.no_grad():  # 在无梯度计算模式下运行，避免保存中间计算梯度
            self.forward()  # 执行模型的前向传播过程

    def compute_visuals(self):
        """计算用于visdom和HTML可视化展示的额外输出图像"""
        # 这个方法将执行计算，生成用于可视化工具的图像数据
        pass

    def update_learning_rate(self, logger):
        """更新所有网络的学习率；在每个epoch结束时调用"""
        for scheduler in self.schedulers:
            # 如果学习率策略为'plateau'，根据评估指标调整学习率
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                # 否则，按照固定步长更新学习率
                scheduler.step()

        # 获取第一个优化器的第一个参数组的学习率
        lr = self.optimizers[0].param_groups[0]['lr']
        # 打印当前学习率（不执行，仅用于注释）
        # print('learning rate = %.7f' % lr)
        # 使用logger记录当前学习率
        logger.info('learning rate = %.7f' % lr)

    def get_current_visuals(self):
        """返回可视化图像。在train.py中，这些图像将通过visdom显示，并保存到HTML文件中"""
        visual_ret = OrderedDict()  # 创建一个有序字典来存储可视化图像
        for name in self.visual_names:  # 遍历所有可视化名称
            if isinstance(name, str):  # 检查名称是否为字符串类型
                visual_ret[name] = getattr(self, name)  # 将对应的可视化图像添加到字典中
        return visual_ret  # 返回包含所有可视化图像的字典

    def get_current_losses(self):
        """
        返回训练过程中的损失值/错误。这些错误将在train.py中打印到控制台，并保存到文件中。
        """
        errors_ret = OrderedDict()  # 创建一个有序字典来存储损失值

        for name in self.loss_names:  # 遍历损失名称列表，如['CE', 'mse', 'cycle']
            if isinstance(name, str):  # 检查名称是否为字符串
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # 获取对应的损失值，转化为浮点数，适用于张量和浮点数
        return errors_ret  # 返回包含所有损失值的有序字典

    def save_networks(self, epoch):
        """保存所有网络到磁盘。

        参数:
            epoch (int) -- 当前的周期，用于文件名 '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            # 如果名称是字符串类型
            if isinstance(name, str):
                # 构建保存文件的名称，格式为：%s_net_%s.pth (epoch, name)
                save_filename = '%s_net_%s.pth' % (epoch, name)
                # 拼接保存路径
                save_path = os.path.join(self.save_dir, save_filename)
                # 获取网络对象，如 self.netG, self.netD 等
                net = getattr(self, 'net' + name)

                # 如果有多个GPU并且可用
                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    # 将模型移动到CPU并保存其状态字典，然后返回到第一个GPU
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    # 如果没有GPU或GPU不可用，直接在CPU上保存模型的状态字典
                    torch.save(net.cpu().state_dict(), save_path)

    def load_networks(self, epoch):
        """加载所有网络模型从磁盘。

        参数:
            epoch (int) -- 当前的周期数；用于文件名 '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            # 如果名称是字符串类型
            if isinstance(name, str):
                # 构建加载文件的名称，包含周期数和网络名称
                load_filename = '%s_net_%s.pth' % (epoch, name)
                # 拼接保存目录和加载文件名
                load_path = os.path.join(self.save_dir, load_filename)
                # 获取网络对象，如 'net' + 'A' -> netA
                net = getattr(self, 'net' + name)
                # 如果网络是 DataParallel 类型，取其 module 属性
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                # 打印加载模型的路径
                print('loading the model from %s' % load_path)
                # 从磁盘加载状态字典
                state_dict = torch.load(load_path, map_location=self.device)
                # 如果状态字典有 '_metadata' 属性，删除它
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                # 加载状态字典到网络模型
                net.load_state_dict(state_dict)

    def load_networks_cv(self, folder_path):
        """从cv文件夹加载所有网络。

        参数:
            epoch (int) -- 当前的周期；用于文件名 '%s_net_%s.pth' % (epoch, name)
        """
        # 获取folder_path目录下以.pth结尾的文件
        checkpoints = list(filter(lambda x: x.endswith('.pth'), os.listdir(folder_path)))

        # 遍历self.model_names中的每个网络名称
        for name in self.model_names:
            # 如果名称是字符串类型
            if isinstance(name, str):
                # 过滤出以'net_' + name 结尾的.pth文件
                load_filename = list(filter(lambda x: x.split('.')[0].endswith('net_' + name), checkpoints))
                # 确保只找到一个匹配的文件
                assert len(load_filename) == 1, '在文件夹：{}，存在文件{}'.format(folder_path, load_filename)
                load_filename = load_filename[0]
                # 构建完整路径
                load_path = os.path.join(folder_path, load_filename)

                # 获取网络对象
                net = getattr(self, 'net' + name)
                # 如果网络是DataParallel类型，取其module
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module

                # 打印加载模型的路径
                print('正在从{}加载模型'.format(load_path))

                # 加载.pth文件中的状态字典
                state_dict = torch.load(load_path, map_location=self.device)
                # 如果state_dict有_metadata属性，删除它
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # 将状态字典加载到网络中
                net.load_state_dict(state_dict)

    def print_networks(self, verbose):
        """打印网络的总参数数量，如果verbose为真，则打印网络架构

        参数:
            verbose (bool) -- 如果verbose为真：打印网络结构
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            # 检查name是否为字符串类型
            if isinstance(name, str):
                # 获取网络对象
                net = getattr(self, 'net' + name)
                # 计算网络参数总数
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                # 如果verbose为真，打印网络结构
                if verbose:
                    print(net)
                # 打印网络名称和参数总数（以百万为单位）
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """
        设置网络参数是否需要计算梯度，以避免不必要的计算。
        参数：
            nets (network list)   -- 网络列表
            requires_grad (bool)  -- 是否需要计算梯度，默认为False
        """
        # 如果nets不是列表类型，将其转换为列表
        if not isinstance(nets, list):
            nets = [nets]

        # 遍历每个网络
        for net in nets:
            # 如果网络不为空
            if net is not None:
                # 遍历网络中的所有参数
                for param in net.parameters():
                    # 设置参数的requires_grad属性为传入的requires_grad值
                    param.requires_grad = requires_grad

    def post_process(self):
        """
        这个方法定义了一个名为post_process的函数，它没有具体的实现（pass关键字表示空函数）。
        通常，这个函数可能用于在某些操作之后进行后期处理或清理工作。
        """
        pass