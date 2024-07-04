"""此包含了与目标函数、优化以及网络架构相关的模块。

要添加一个名为'dummy'的自定义模型类，您需要添加一个名为'dummy_model.py'的文件，并定义一个从BaseModel继承的子类DummyModel。
您需要实现以下五个函数：
    -- <__init__>:                      初始化类；首先调用BaseModel.__init__(self, opt)。
    -- <set_input>:                     从数据集中解包数据并应用预处理。
    -- <forward>:                       产生中间结果。
    -- <optimize_parameters>:           计算损失、梯度并更新网络权重。
    -- <modify_commandline_options>:    （可选）添加模型特定的选项并设置默认选项。

在<__init__>函数中，您需要定义四个列表：
    -- self.loss_names (str list):          指定您想要绘制和保存的训练损失。
    -- self.model_names (str list):         定义我们训练中使用的网络。
    -- self.visual_names (str list):        指定您想要显示和保存的图像。
    -- self.optimizers (optimizer list):    定义并初始化优化器。您可以为每个网络定义一个优化器。如果两个网络同时更新，您可以使用itertools.chain将它们组合起来。参见cycle_gan_model.py中的用法。

现在，您可以通过指定标志'--model dummy'来使用模型类。
更多详情请参见我们的模板模型类'template_model.py'。
"""

import importlib
from models.base_model import BaseModel


def find_model_using_name(model_name):
    """导入名为'models/[model_name]_model.py'的模块。

    在该文件中，将实例化一个名为DatasetNameModel()的类，
    它必须是BaseModel的子类，并且类名不区分大小写。
    """
    model_filename = "models." + model_name + "_model"  # 例如：'models.mmin_model'
    modellib = importlib.import_module(model_filename)  # 导入模块

    model = None
    target_model_name = model_name.replace('_', '') + 'model'  # 目标模型名称（不包含下划线）

    # 遍历模块字典，查找与目标模型名称匹配且为BaseModel子类的类
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() and issubclass(cls, BaseModel):
            model = cls

    # 如果未找到匹配的模型类
    if model is None:
        print("在{}中，应存在一个基类为BaseModel且小写类名匹配{}的子类。".format(model_filename, target_model_name))
        exit(0)  # 结束程序

    # 返回找到的模型类
    return model

def get_option_setter(model_name):
    """
    根据模型名称返回模型类的静态方法<modify_commandline_options>。

    参数:
    model_name (str): 模型的名称

    返回:
    function: 模型类中用于修改命令行选项的方法
    """
    model_class = find_model_using_name(model_name)  # 查找并获取名为model_name的模型类
    return model_class.modify_commandline_options  # 返回该模型类的modify_commandline_options静态方法

def create_model(opt):
    """根据给定的选项创建模型。

    此函数包装了CustomDatasetDataLoader类。
    这是该包与'train.py'/'test.py'之间的主要接口。

    示例：
        >>> from models import create_model
        >>> model = create_model(opt)
    """
    # 根据模型名称找到相应的模型类
    model = find_model_using_name(opt.model)
    # 实例化模型
    instance = model(opt)
    # 打印创建的模型类型
    print("model [%s] was created" % type(instance).__name__)
    # 返回模型实例
    return instance
