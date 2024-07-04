import sys

class OptConfig(object):
    def __init__(self):
        # 初始化方法，空实现，不做任何操作
        pass

    def load(self, config_dict):  # 定义load方法，用于加载配置字典
        if sys.version > '3':  # 判断Python版本是否大于3
            for key, value in config_dict.items():  # 遍历配置字典中的键值对
                if not isinstance(value, dict):  # 如果值不是字典类型
                    setattr(self, key, value)  # 将键值对设置为当前对象的属性
                else:
                    self.load(value)  # 若值是字典，则递归调用load方法加载子字典
        else:  # Python版本小于或等于3的情况
            for key, value in config_dict.iteritems():  # 使用iteritems遍历配置字典（Python2中无items方法）
                if not isinstance(value, dict):  # 如果值不是字典类型
                    setattr(self, key, value)  # 将键值对设置为当前对象的属性
                else:
                    self.load(value)  # 若值是字典，则递归调用load方法加载子字典