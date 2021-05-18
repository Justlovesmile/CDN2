import random
import torch
import torchvision
from PIL import ImageEnhance,Image

from torchvision.transforms import functional as F

class Compose(object):
    """组合多个transform函数"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    """将PIL图像转为Tensor"""
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlip(object):
    """随机水平翻转图像以及bboxes"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)  # 水平翻转图片
            bbox = target["boxes"]
            # bbox: xmin, ymin, xmax, ymax
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]  # 翻转对应bbox坐标信息
            target["boxes"] = bbox
        return image, target

class RandomVerticalFlip(object):
    """随机竖直翻转图像以及bboxes"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(0)  # 竖直翻转图片
            bbox = target["boxes"]
            # bbox: xmin, ymin, xmax, ymax
            bbox[:, [1, 3]] = height - bbox[:, [3, 1]]  # 翻转对应bbox坐标信息
            target["boxes"] = bbox
        return image, target

class RandomScaleImage(object):
    """对大图做随机缩放图像以及bboxes"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        width, height = image.size
        if height>2000 and width>2000 and random.random() <self.prob:
            n=min(height,width)/1000
            scale = torchvision.transforms.Resize(1000)
            image = scale(image)  # 缩小图片
            bbox = target["boxes"]
            # bbox: xmin, ymin, xmax, ymax
            bbox[:,[0,1,2,3]]=bbox[:,[0,1,2,3]]/n  # 缩小对应bbox坐标信息
            target["boxes"] = bbox
        return image, target

class PILbrightness(object):
    """亮度增加"""
    def __init__(self, factor=5):
        self.factor = factor if factor>=0 else 5

    def __call__(self, image, target=None):
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(self.factor)
        if target==None:
            return image
        return image, target

class RandomRotate(object):
    """奇顺偶逆旋转90度"""
    def __init__(self, idx):
        self.idx = idx

    def __call__(self, image, target):
        if self.idx%2==0:
            if image is not None :
                width, height = image.size
                image = image.rotate(90,expand=True) # 逆时针90度
            else:
                width=target['width']
                height=target['height']
            bbox = target["boxes"]
            # bbox: xmin, ymin, xmax, ymax
            # 旋转对应bbox坐标信息
            temp=bbox[:,[0]]
            bbox[:,[0]]=bbox[:,[1]]
            bbox[:,[1]]=width-bbox[:,[2]]
            bbox[:, [2]]=bbox[:,[3]]
            bbox[:,[3]]=width-temp
            target["boxes"] = bbox
        else:
            if image is not None :
                width, height = image.size
                image = image.rotate(270,expand=True) # 顺时针90度
            else:
                width=target['width']
                height=target['height']
            bbox = target["boxes"]
            # bbox: xmin, ymin, xmax, ymax
            temp=bbox[:,[0]]                # 旋转对应bbox坐标信息
            bbox[:,[0]]=height-bbox[:,[3]]
            bbox[:,[3]]=bbox[:,[2]]
            bbox[:, [2]]=height-bbox[:,[1]]
            bbox[:,[1]]=temp
            target["boxes"] = bbox
        return image, target