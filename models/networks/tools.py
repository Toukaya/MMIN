import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import functools
from torch.optim import lr_scheduler

class Identity(nn.Module):
    def forward(self, x):
        """
        向前传播方法，用于计算模型在给定输入x时的输出。
        输入：
        - self：当前对象，通常是指调用该方法的类实例。
        - x：输入数据，可以是单个样本或一批样本。

        输出：
        - x：返回原始输入x，因为这个简单的示例中没有进行任何变换或计算。
        """
        return x


def get_norm_layer(norm_type='instance'):
    """返回一个规范化层

    参数:
        norm_type (str) -- 规范化层的名称：batch | instance | none

    对于BatchNorm，我们使用可学习的仿射参数并追踪运行统计信息（均值/标准差）。
    对于InstanceNorm，我们不使用可学习的仿射参数。我们不追踪运行统计信息。
    """
    if norm_type == 'batch':
        # 如果norm_type为'batch'，则创建一个带有学习性仿射参数和运行统计跟踪的nn.BatchNorm2d实例
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        # 如果norm_type为'instance'，则创建一个不带学习性仿射参数且不追踪运行统计的nn.InstanceNorm2d实例
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'layer':
        # 如果norm_type为'layer'，则创建一个带有eps=1e-6和元素级仿射的nn.LayerNorm实例
        norm_layer = functools.partial(nn.LayerNorm, eps=1e-6, elementwise_affine=True)
    elif norm_type == 'none':
        # 如果norm_type为'none'，则返回一个恒等函数，即不应用规范化
        norm_layer = lambda x: Identity()
    else:
        # 如果输入的norm_type不在预定义选项中，抛出异常
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    # 返回创建的规范化层
    return norm_layer


def get_scheduler(optimizer, opt):
    """返回一个学习率调度器

    参数:
        optimizer          -- 网络的优化器
        opt (option类) -- 存储所有实验标志；需要是BaseOptions的子类。
                          opt.lr_policy 是学习率策略的名称：linear | step | plateau | cosine

    对于'linear'，在前<opt.niter>个周期保持相同的学习率，
    并在接下来的<opt.niter_decay>个周期线性地将学习率衰减到零。
    对于其他调度器（step, plateau, 和 cosine），我们使用PyTorch的默认调度器。
    有关更多详细信息，请参阅https://pytorch.org/docs/stable/optim.html。
    """
    if opt.lr_policy == 'linear':
        # 定义线性衰减规则
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        # 使用LambdaLR调度器
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        # 使用StepLR调度器
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        # 使用ReduceLROnPlateau调度器
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        # 使用CosineAnnealingLR调度器
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        # 如果未实现的学习率策略
        return NotImplementedError('学习率策略 [%s] 未实现', opt.lr_policy)
    return scheduler


# 初始化网络权重函数
def init_weights(net, init_type='normal', init_gain=0.02):
    """
    初始化网络的权重。

    参数:
        net (network)   -- 需要初始化的网络
        init_type (str) -- 初始化方法的名称：normal | xavier | kaiming | orthogonal
        init_gain (float)    -- 对于normal、xavier和orthogonal初始化的缩放因子

    原始的pix2pix和CycleGAN论文中使用了'normal'。但在某些应用中，xavier和kaiming可能效果更好。
    欢迎尝试不同的初始化方法。
    """

    def init_func(m):  # 定义初始化函数
        # 获取模块的类名
        classname = m.__class__.__name__
        # 如果模块有weight属性，并且是卷积层或线性层
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            # 根据初始化类型进行相应的权重初始化
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            # 如果模块有bias属性并且不为None，初始化bias为0.0
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        # 如果是BatchNorm2d层，只用正态分布初始化weight，bias初始化为0.0
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    # 输出所使用的初始化方法
    print('initialize network with %s' % init_type)
    # 应用初始化函数到网络的所有子模块
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """
    初始化网络：1. 注册CPU/GPU设备（支持多GPU）；2. 初始化网络权重
    参数：
        net (网络)          -- 需要初始化的网络
        init_type (字符串)   -- 初始化方法的名称：normal | xavier | kaiming | orthogonal
        init_gain (浮点数)  -- 对于normal, xavier和orthogonal初始化的缩放因子
        gpu_ids (整数列表)  -- 网络运行的GPU ID：例如，0,1,2

    返回一个已初始化的网络。
    """
    if len(gpu_ids) > 0:
        # 检查是否有可用的GPU
        assert (torch.cuda.is_available())
        # 将网络移动到第一个GPU上
        net.to(gpu_ids[0])
        # 使用多GPU数据并行
        net = torch.nn.DataParallel(net, gpu_ids)
    # 初始化网络权重
    init_weights(net, init_type, init_gain=init_gain)
    # 返回初始化后的网络
    return net


def diagnose_network(net, name='network'):
    """计算并打印平均绝对梯度的均值

    参数:
        net (torch network) -- PyTorch 网络模型
        name (str) -- 网络的名称，默认为 'network'
    """
    mean = 0.0  # 初始化平均值为0
    count = 0  # 初始化计数器为0，用于记录有梯度参数的数量

    # 遍历网络的所有参数
    for param in net.parameters():
        # 如果参数有梯度信息
        if param.grad is not None:
            # 计算并累加该参数梯度数据的绝对值的平均值
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1  # 参数计数器加1

    # 如果有至少一个参数有梯度
    if count > 0:
        # 计算平均值
        mean = mean / count
    # 打印网络名称
    print(name)
    # 打印平均绝对梯度的均值
    print(mean)

class MidLayerFeatureExtractor(object):
    def __init__(self, layer):  # 初始化方法，接收一个layer对象
        self.layer = layer  # 将传入的layer对象保存为实例属性
        self.feature = None  # 初始化特征变量为None，用于存储层的输出特征
        self.layer.register_forward_hook(self.hook)  # 注册前向传播钩子函数hook，以便在layer执行后捕获其输出
        self.device = None  # 初始化设备变量为None，可能用于指定计算设备（如GPU或CPU）

    def hook(self, module, input, output):
        """
        在此函数中，我们定义了一个hook，用于捕获并处理模块的输出。

        参数:
        - module (torch.nn.Module): 被挂钩的PyTorch模块
        - input (tuple or Tensor): 模块的输入数据
        - output (Tensor): 模块处理后的输出数据

        在这个hook中：
        1. 初始化一个标志，表示特征是否为空，默认为True。
        2. 使用clone()方法复制并保存输出张量作为特征。
        3. 更新标志，将is_empty设为False，表明已经获取了特征信息。
        """
        # 判断特征是否为空，默认设置为True
        self.is_empty = True
        # 克隆输出张量以保存特征信息
        self.feature = output.clone()
        # 将特征为空的标志设置为False，表示已保存特征
        self.is_empty = False


    def extract(self):
        # 断言当前对象不为空，如果为空则抛出异常
        # 异常信息：Synic 错误在 MidLayerFeatureExtractor 中，
        # 这可能是由于在挂钩的模块执行forward方法之前调用了extract方法
        assert not self.is_empty, 'Synic Error in MidLayerFeatureExtractor, \
                this may caused by calling extract method before the hooked module has execute forward method'
        # 返回提取到的特征
        return self.feature


class MultiLayerFeatureExtractor(object):
    def __init__(self, net, layers):
        """
        初始化方法，用于创建一个中间层特征提取器实例。

        参数:
        -----------------
        net: torch.nn.Modules
            PyTorch模型，包含需要提取特征的网络结构。
        layers: str, something like "C.fc[0], module[1]"
            用逗号分隔的字符串，指定要从net中提取特征的中间层名称。
            例如，'C.fc[0]'和'module[1]'分别表示提取net.C.fc[0]和net.module[1]的特征。
        """
        self.net = net  # 存储输入的网络模型
        self.layer_names = layers.strip().split(',')  # 分割并存储层名称列表
        self.layers = [self.str2layer(layer_name) for layer_name in self.layer_names]  # 将字符串形式的层名转换为实际的网络层
        self.extractors = [MidLayerFeatureExtractor(layer) for layer in self.layers]  # 创建中间层特征提取器列表，每个元素对应一个指定的网络层


    def str2layer(self, name):  # 根据字符串表示转换为网络层
        modules = name.split('.')  # 按照点号分隔字符串，获取模块层级关系
        layer = self.net  # 初始化为网络对象

        for module in modules:  # 遍历每个模块名
            if '[' and ']' in module:  # 如果模块名包含方括号
                sequential_name = module[:module.find('[')]  # 提取序列化模块名
                target_module_num = int(module[module.find('[') + 1:module.find(']')])  # 获取目标模块索引
                layer = getattr(layer, sequential_name)  # 获取序列化模块对象
                layer = layer[target_module_num]  # 根据索引获取具体层
            else:
                layer = getattr(layer, module)  # 如果没有方括号，直接获取对应属性的层

        return layer  # 返回找到的网络层

    def extract(self):
        """
        此函数用于从多个extractors中提取信息。

        遍历self.extractors列表中的每个extractor对象，
        调用它们的extract()方法来获取数据，
        然后将所有提取到的数据收集到一个列表中。

        返回值:
        ans - 包含所有extractor提取结果的列表
        """
        ans = [extractor.extract() for extractor in self.extractors]
        return ans
   
        
    

    



