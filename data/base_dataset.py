"""此模块实现了一个抽象基类（ABC）'BaseDataset'用于数据集。

它还包括了通用的转换函数（例如，get_transform, __scale_width），这些可以在子类中使用。"""
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod

class BaseDataset(data.Dataset, ABC):
    """此类是一个用于数据集的抽象基类（ABC）。

    要创建子类，你需要实现以下四个函数：
    -- <__init__>:                      初始化类，首先调用 BaseDataset.__init__(self, opt)。
    -- <__len__>:                       返回数据集的大小。
    -- <__getitem__>:                    获取一个数据点。
    -- <modify_commandline_options>:     （可选）添加数据集特定的选项并设置默认选项。
    """

    def __init__(self, opt):
        """初始化类；在类中保存选项

        参数：
            opt (Option class)-- 存储所有实验标志；需要是 BaseOptions 的子类
        """
        self.opt = opt
        self.manual_collate_fn = False
        # self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """添加新的数据集特定选项，并重写现有选项的默认值。

        参数：
            parser          -- 原始选项解析器
            is_train (bool) -- 是否训练阶段或测试阶段。你可以使用这个标志来添加训练特定或测试特定的选项。

        返回：
            修改后的解析器。
        """
        return parser

    @abstractmethod
    def __len__(self):
        """返回数据集中图像的总数。"""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """返回一个数据点及其元数据信息。

        参数：
            index - - 数据索引的随机整数

        返回：
            包含数据及其名称的字典。它通常包含数据本身及其元数据信息。
        """
        pass

# 下面是一些转换函数的实现，它们将被用于数据集的子类中。

def get_params(opt, size):
    """获取转换参数，比如裁剪位置和是否翻转"""
    w, h = size
    new_h = h
    new_w = w
    # 根据预处理选项调整图像大小
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    # 随机选择裁剪位置
    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    # 随机决定是否翻转图像
    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}

def get_transform(opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
    """获取转换操作列表"""
    transform_list = []
    # 如果需要灰度图像，则添加灰度转换
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    # 根据预处理选项添加调整大小或缩放宽度的转换
    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, method)))

    # 如果需要裁剪，则添加裁剪转换
    if 'crop' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

    # 如果预处理选项为none，则添加使图像尺寸成为2的幂的转换
    if opt.preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    # 如果不禁止翻转，则添加随机水平翻转的转换
    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    # 如果需要转换为张量，则添加转换为张量的转换，并进行归一化
    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

# 下面是一些辅助函数的实现，用于图像尺寸的调整和转换。

def __make_power_2(img, base, method=Image.BICUBIC):
    """将图像尺寸调整为2的幂次"""
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)

def __scale_width(img, target_width, method=Image.BICUBIC):
    """根据目标宽度调整图像尺寸"""
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)

def __crop(img, pos, size):
    """裁剪图像到指定位置和大小"""
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img

def __flip(img, flip):
    """如果需要，翻转图像"""
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def __print_size_warning(ow, oh, w, h):
    """打印关于图像尺寸的警告信息（仅打印一次）"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("图像尺寸需要是4的倍数。"
              "加载的图像尺寸为(%d, %d)，因此被调整为"
              "(%d, %d)。所有尺寸不是4的倍数的图像都将进行此调整" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
