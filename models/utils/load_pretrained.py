import os
import json
from .config import OptConfig


# 定义一个函数load_from_opt_record，参数为文件路径file_path
def load_from_opt_record(file_path):
    # 使用json库加载文件_path指定的文件内容，并以字典形式存储在opt_content中
    opt_content = json.load(open(file_path, 'r'))
    # 初始化OptConfig类的实例，命名为opt
    opt = OptConfig()
    # 调用opt实例的load方法，传入opt_content的内容进行加载
    opt.load(opt_content)
    # 返回加载后的opt实例
    return opt


# 定义加载预训练模型的函数
def load_pretrained_model(model_class, checkpoints_dir, cv, gpu_ids):
    # 构建路径，根据cv值获取对应检查点目录
    path = os.path.join(checkpoints_dir, str(cv))

    # 加载配置文件路径
    config_path = os.path.join(checkpoints_dir, 'train_opt.conf')

    # 从配置文件中加载参数
    config = load_from_opt_record(config_path)

    # 设置模型为测试模式，因为是教师模型
    config.isTrain = False

    # 设置GPU使用列表
    config.gpu_ids = gpu_ids

    # 初始化指定类别的模型，传入配置参数
    model = model_class(config)

    # 将模型移动到GPU上
    model.cuda()

    # 根据路径加载模型的检查点
    model.load_networks_cv(path)

    # 设置模型为评估模式
    model.eval()

    # 返回加载好的模型
    return model
