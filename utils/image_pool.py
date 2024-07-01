import random
import torch

class ImagePool():
    """此类实现了一个图像缓冲区，用于存储先前生成的图像。

    该缓冲区使我们能够使用生成图像的历史记录来更新鉴别器，
    而不是使用最新生成器产生的图像。
    """

    def __init__(self, pool_size):
        """初始化ImagePool类

        Parameters:
            pool_size (int) -- 图像缓冲区的大小，如果pool_size=0，则不会创建缓冲区
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # 创建一个空的缓冲区
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """从缓冲区返回一个图像。

        Parameters:
            images: 来自生成器的最新生成的图像

        返回缓冲区中的图像。

        有50%的几率，缓冲区将返回输入图像。
        另外50%的几率，缓冲区将返回之前存储在缓冲区中的图像，
        并将当前图像插入缓冲区。
        """
        if self.pool_size == 0:  # 如果缓冲区大小为0，什么也不做
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # 如果缓冲区未满；继续将当前图像插入缓冲区
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # 有50%的几率，缓冲区将返回一个之前存储的图像，并把当前图像插入缓冲区
                    random_id = random.randint(0, self.pool_size - 1)  # randint是包含的
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # 有另外50%的几率，缓冲区将返回当前图像
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # 收集所有图像并返回
        return return_images