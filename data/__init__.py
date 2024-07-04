"""此包含了与数据加载和预处理相关的所有模块

要添加一个名为'dummy'的自定义数据集类，您需要添加一个名为'dummy_dataset.py'的文件，并定义一个从BaseDataset继承的子类'DummyDataset'。
您需要实现四个函数：
    -- <__init__>:                      初始化类，首先调用BaseDataset.__init__(self, opt)。
    -- <__len__>:                       返回数据集的大小。
    -- <__getitem__>:                   从数据加载器获取一个数据点。
    -- <modify_commandline_options>:    （可选）添加特定于数据集的选项并设置默认选项。

现在，您可以通过指定标志'--dataset_mode dummy'来使用数据集类。
更多详情请参见我们的模板数据集类'template_dataset.py'。
"""
import importlib
import torch.utils.data
from data.base_dataset import BaseDataset


def find_dataset_using_name(dataset_name):
    """根据给定的名称导入相应的数据集模块。

    导入的模块格式应为"data/[dataset_name]_dataset.py"。
    在该模块中，会实例化一个名为DatasetNameDataset的类，
    这个类必须是BaseDataset的子类，且类名匹配是不区分大小写的。

    参数:
    dataset_name (str): 数据集的名称。

    返回:
    dataset: 类型为BaseDataset子类的实例，表示对应的数据集。
    """
    # 构建数据集模块的文件名
    dataset_filename = "data." + dataset_name + "_dataset"

    # 使用importlib导入模块
    datasetlib = importlib.import_module(dataset_filename)

    # 初始化数据集变量为None
    dataset = None

    # 将dataset_name转换为目标类名，移除下划线
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'

    # 遍历模块中的所有名称和类
    for name, cls in datasetlib.__dict__.items():
        # 如果类名小写后与目标类名匹配，并且是BaseDataset的子类
        if name.lower() == target_dataset_name.lower() and issubclass(cls, BaseDataset):
            # 设置数据集为找到的类的实例
            dataset = cls

    # 如果没有找到匹配的类，抛出异常
    if dataset is None:
        raise NotImplementedError(
            "在{}中，应该有一个子类化了BaseDataset的类，其类名在小写后匹配{}".format(dataset_filename,
                                                                                   target_dataset_name))

    # 返回找到的数据集实例
    return dataset


def get_option_setter(dataset_name):
    """
    根据给定的dataset名称，返回该数据集类的静态方法<modify_commandline_options>。

    参数:
    dataset_name (str): 数据集的名称

    返回:
    function: 数据集类中用于修改命令行选项的方法
    """
    # 查找并获取名为dataset_name的数据集类
    dataset_class = find_dataset_using_name(dataset_name)

    # 返回该数据集类的modify_commandline_options静态方法
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    """根据给定的选项创建数据加载器。

    此函数包装了类 CustomDatasetDataLoader。
    这是该包与 'train.py'/'test.py' 之间的主要接口。

    示例用法:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    # 创建一个CustomDatasetDataLoader实例，传入参数opt
    data_loader = CustomDatasetDataLoader(opt)
    # 返回创建的数据加载器
    return data_loader


def create_dataset_with_args(opt, **kwargs):
    """
    创建两个数据加载器，根据选项和可能的额外参数。
    此函数包装了类 CustomDatasetDataLoader。
        这是此包与 'train.py'/'test.py' 之间的主要接口。
    示例：
        >>> from data import create_split_dataset
        >>> dataset = create_dataset_with_args(opt, set_name=['trn', 'val', 'tst'])
        这将创建3个数据集，每个数据集都会为特定数据集类的特定参数提供不同的参数，
        '__init__' 函数必须接受一个参数，例如：dataset.__init__(self, set_name='trn'): ....

    """
    _kwargs = []
    for key in kwargs:  # 'set_name'
        value = kwargs[key]  # 'trn', 'val', 'tst'
        if not isinstance(value, (list, tuple)):
            value = [value]
        lens = len(value)  # lens = 3
        _kwargs += list(map(lambda x: {}, range(lens))) if len(_kwargs) == 0 else []
        for i, v in enumerate(value):
            _kwargs[i][key] = v

    # _kwargs: [{'set_name': 'trn'}, {'set_name': 'val'}, {'set_name': 'tst'}]
    dataloaders = tuple(map(lambda x: CustomDatasetDataLoader(opt, **x), _kwargs))
    # 如果数据加载器数量大于1，则返回元组，否则返回元组中的第一个元素
    return dataloaders if len(dataloaders) > 1 else dataloaders[0]

class CustomDatasetDataLoader():
    """数据集类的包装器类，执行多线程数据加载"""
    ## kwargs: [{'set_name': 'trn'}, {'set_name': 'val'}, {'set_name': 'tst'}]
    def __init__(self, opt, **kwargs):
        """初始化类
        步骤1: 根据[dataset_mode]创建数据集实例
        步骤2: 创建多线程数据加载器
        """
        self.opt = opt  # 保存选项参数
        dataset_class = find_dataset_using_name(
            opt.dataset_mode)  # 获取指定名称的数据集类，例如：'data.multimodal_dataset.MultimodalDataset'
        self.dataset = dataset_class(opt, **kwargs)  # 初始化数据集对象
        # print("已创建数据集 [%s]" % type(self.dataset).__name__)  # 输出创建的数据集类型

        """判断是否使用在dataset.collate_fn中定义的手动合并函数"""
        if self.dataset.manual_collate_fn:
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,  # 使用数据集对象
                batch_size=opt.batch_size,  # 批次大小
                shuffle=not opt.serial_batches,  # 是否打乱批次（非序列批次）
                num_workers=int(opt.num_threads),  # 工作线程数
                drop_last=False,  # 不丢弃最后一个批次
                collate_fn=self.dataset.collate_fn  # 使用自定义的合并函数
            )
        else:
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,  # 使用数据集对象
                batch_size=opt.batch_size,  # 批次大小
                shuffle=not opt.serial_batches,  # 是否打乱批次（非序列批次）
                num_workers=int(opt.num_threads),  # 工作线程数
                drop_last=False  # 不丢弃最后一个批次
            )


    def __len__(self):
        """返回数据集中数据的数量"""
        # 计算数据集中的元素数量，但不超过最大数据集大小限制
        return min(len(self.dataset), self.opt.max_dataset_size)


    def __iter__(self):
        """
        返回一个数据批次

        迭代内部数据加载器，按顺序获取每个数据批次。
        当达到最大数据集大小限制时，停止迭代。
        """
        for i, data in enumerate(self.dataloader):
            # 检查是否已超过最大数据集大小限制
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            # 生成并返回当前批次的数据
            yield data
