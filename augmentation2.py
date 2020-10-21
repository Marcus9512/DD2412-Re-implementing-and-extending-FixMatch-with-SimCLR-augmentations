
# normaliseringar
# sparande av bilder
# hur anropa flera SimCLR anrop


import torch
import torchvision
# from randaugment import RandAugment
import random
import cv2
import numpy as np

import random

import numpy as np
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)


class Wrapper:
    def __init__(self, transform1, transform2):
        self.transform1 = torchvision.transforms.Compose([transform1,
                                                          torchvision.transforms.functional.to_tensor,
                                                          torchvision.transforms.Normalize(cifar10_mean, cifar10_std)
                                                          ])
        self.transform2 = torchvision.transforms.Compose([transform2,
                                                          torchvision.transforms.functional.to_tensor,
                                                          torchvision.transforms.Normalize(cifar10_mean, cifar10_std)
                                                          ])

    def __call__(self, item):
        return item, self.transform1(item), self.transform2(item)


def weak_augment(batch):
    # torchvision.utils.save_image(batch[0], "img_weak_1.png")

    weak_transform = get_weak_transform()

    for i in range(len(batch)):
        batch[i] = weak_transform(batch[i].cpu())

    # torchvision.utils.save_image(batch[0], "img_weak_aug.png")
    return batch


def get_weak_transform():
    weak_transform = torchvision.transforms.Compose([
        # torchvision.transforms.Normalize((-0.5/0.5, -0.5/0.5, -0.5/0.5), (1/0.5, 1/0.5, 1/0.5)),
        # torchvision.transforms.functional.to_pil_image,
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        # torchvision.transforms.RandomAffine(0, translate=(0.0625, 0.0625)),
        torchvision.transforms.RandomCrop(size=32,
                                          padding=int(32 * 0.125),
                                          padding_mode='reflect'),
        # torchvision.transforms.functional.to_tensor,
        # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return weak_transform


def strong_augment(batch, dataset_name):
    # torchvision.utils.save_image(batch[0], "img_strog_1.png")

    strong_transform = get_strong_transform(dataset_name)

    for i in range(len(batch)):
        batch[i] = strong_transform(batch[i].cpu())

        # cutout(batch[i], cutout_height, cutout_width)

    # torchvision.utils.save_image(batch[0], "img_strog_aug.png")
    return batch


def get_strong_transform(dataset_name):
    strong_transform = torchvision.transforms.Compose([
        # torchvision.transforms.Normalize((-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5), (1 / 0.5, 1 / 0.5, 1 / 0.5)),
        # torchvision.transforms.functional.to_pil_image,
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomCrop(size=32,
                                          padding=int(32 * 0.125),
                                          padding_mode='reflect'),
        RandAugment(2, 10),
        # cutout_transform(dataset_name),
        # torchvision.transforms.functional.to_tensor
        # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return strong_transform


PARAMETER_MAX = 10


def AutoContrast(img, **kwarg):
    return PIL.ImageOps.autocontrast(img)


def Brightness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Color(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Color(img).enhance(v)


def Contrast(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Cutout(img, v, max_v, bias=0):
    if v == 0:
        return img
    v = _float_parameter(v, max_v) + bias
    v = int(v * min(img.size))
    return CutoutAbs(img, v)


def CutoutAbs(img, v, **kwarg):
    w, h = img.size
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    xy = (x0, y0, x1, y1)
    # gray
    color = (127, 127, 127)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def Equalize(img, **kwarg):
    return PIL.ImageOps.equalize(img)


def Identity(img, **kwarg):
    return img


def Invert(img, **kwarg):
    return PIL.ImageOps.invert(img)


def Posterize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.posterize(img, v)


def Rotate(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.rotate(v)


def Sharpness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def Solarize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.solarize(img, 256 - v)


def SolarizeAdd(img, v, max_v, bias=0, threshold=128):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    img_np = np.array(img).astype(np.int)
    img_np = img_np + v
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def TranslateX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[0])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[1])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX


def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)


def fixmatch_augment_pool():
    # FixMatch paper
    augs = [(AutoContrast, None, None),
            (Brightness, 0.9, 0.05),
            (Color, 0.9, 0.05),
            (Contrast, 0.9, 0.05),
            (Equalize, None, None),
            (Identity, None, None),
            (Posterize, 4, 4),
            (Rotate, 30, 0),
            (Sharpness, 0.9, 0.05),
            (ShearX, 0.3, 0),
            (ShearY, 0.3, 0),
            (Solarize, 256, 0),
            (TranslateX, 0.3, 0),
            (TranslateY, 0.3, 0)]
    return augs


class RandAugment(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = fixmatch_augment_pool()

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            prob = np.random.uniform(0.2, 0.8)
            if random.random() + prob >= 1:
                img = op(img, v=self.m, max_v=max_v, bias=bias)
        img = CutoutAbs(img, 16)
        return img


def SimCLR_augmentation(batch, transforms_to_do,
                        dataset_name):  # transforms_to_do = ["crop", "cutout", "colour", "sobel", "noise", "blur", "rotate"]
    torchvision.utils.save_image(batch[0], "img_SimCLR.png")

    for i in range(1):  # range(len(batch)):
        img = batch[i].cpu()

        for current_transform in transforms_to_do:
            if current_transform == "crop":
                current_transform = crop_transform(img.shape[1])
                img = current_transform(img)

            elif current_transform == "cutout":
                current_transform = cutout_transform(dataset_name)
                img = current_transform(img)

            elif current_transform == "colour":
                current_transform = colour_transform(1)
                img = current_transform(img)

            elif current_transform == "rotate":
                current_transform = rotate_transform()
                img = current_transform(img)

            elif current_transform == "sobel":
                current_transform = sobel_transform()
                img = current_transform(img)

            elif current_transform == "noise":
                current_transform = noise_transform()
                img = current_transform(img)

            elif current_transform == "blur":
                current_transform = blur_transform()
                img = current_transform(img)

        last_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # img = last_transforms(img)

        batch[i] = img

    torchvision.utils.save_image(batch[0], "img_SimCLR_aug.png")
    return batch


class cutout_transform(object):
    def __init__(self, dataset_name):
        if dataset_name == "CIFAR10":
            self.cutout_height = 16
            self.cutout_width = 16
        elif dataset_name == "CIFAR100":
            self.cutout_height = 8
            self.cutout_width = 8

    def __call__(self, img):
        # img.save("cut.png")
        # img = torchvision.transforms.functional.to_tensor(img)
        # img_height = img.shape[1]
        # img_width = img.shape[2]

        img.load()
        img_height, img_width = img.size

        cut_start_height = random.randrange(img_height - self.cutout_height)
        cut_start_width = random.randrange(img_width - self.cutout_width)

        # for layer in img:
        for x in range(cut_start_height, cut_start_height + self.cutout_height):
            for y in range(cut_start_width, cut_start_width + self.cutout_width):
                img.putpixel((x, y), (0, 0, 0))  # PIL
                # layer[x][y] = 0
        # img = torchvision.transforms.functional.to_pil_image(img)
        # img.save("cut_aug.png")
        return img


class crop_transform(object):
    def __init__(self, img_shape):
        self.img_shape = img_shape

    def __call__(self, img):
        crop = torchvision.transforms.Compose([
            # torchvision.transforms.functional.to_pil_image,
            torchvision.transforms.RandomResizedCrop(self.img_shape, scale=(0.08, 1.0),
                                                     ratio=(0.75, 1.3333333333333333),
                                                     interpolation=2),  # default settings
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            # improved perfomance according to SimCLR paper, Apendix A
            # torchvision.transforms.functional.to_tensor
        ])
        return crop(img)


class colour_transform(object):
    def __init__(self, s):
        self.s = s

    def __call__(self, img):
        # s = 1 #colout distorition strength sugested by the SimCLR paper
        colour = torchvision.transforms.Compose([
            # torchvision.transforms.functional.to_pil_image,
            torchvision.transforms.RandomApply(
                [torchvision.transforms.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)], p=0.8),
            torchvision.transforms.RandomGrayscale(p=0.2),
            # torchvision.transforms.functional.to_tensor
        ])
        return colour(img)


# def rotate_function(img):
class rotate_transform(object):
    def __init__(self):
        pass

    def __call__(self, img):
        rotate_angle = random.randint(1, 3) * 90
        # img = torchvision.transforms.functional.to_pil_image(img)
        img = torchvision.transforms.functional.rotate(img=img, angle=rotate_angle)
        # img = torchvision.transforms.functional.to_tensor(img)
        return img


"""
def rotate_transform():
    rotater = torchvision.transforms.Compose([
        rotate_function
        ])
    return rotater
"""


# def sobel_function(img):
class sobel_transform(object):
    def __init__(self):
        pass

    def __call__(self, img):
        # https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html
        img = torchvision.transforms.functional.to_tensor(img)
        img = tensor_to_np(img)
        img = img * 255

        scale = 1
        delta = 0
        ddepth = cv2.CV_16S
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        img = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

        img = np_to_tensor(img / 255)
        img = torchvision.transforms.functional.to_pil_image(img)
        return img


"""     
def sobel_transform():
    sobel_transform = torchvision.transforms.Compose([
        sobel_function
        ])
    return sobel_transform
"""


# def noise_function(img):
class noise_transform(object):
    def __init__(self):
        pass

    def __call__(self, img):
        # https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
        img = torchvision.transforms.functional.to_tensor(img)
        row, col, ch = img.shape
        mean = 0
        sigma = 0.07  # 0.13
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        img = img + gauss

        img = img.type(torch.float32)
        img = torchvision.transforms.functional.to_pil_image(img)
        return img


"""
def noise_transform():
    noise_transform = torchvision.transforms.Compose([
        noise_function
        ])
    return noise_transform
"""


# def blur_function(img):
class blur_transform(object):
    def __init__(self):
        pass

    def __call__(self, img):
        random_float = random.uniform(0, 1)
        if random_float < 0.5:
            img = torchvision.transforms.functional.to_tensor(img)
            img = tensor_to_np(img) * 255

            sigma = random.uniform(0.1, 2)
            kernal_size_height = int(round(img.shape[1] * 0.1))
            kernal_size_width = int(round(img.shape[2] * 0.1))
            img = cv2.GaussianBlur(img, (kernal_size_height, kernal_size_width), sigma)

            img = np_to_tensor(img / 255)
            img = torchvision.transforms.functional.to_pil_image(img)
        return img


"""
def blur_transform():
    blur_transform = torchvision.transforms.Compose([
        blur_function
        ])
    return blur_transform
"""


def tensor_to_np(img):
    img = img.detach().cpu().numpy()
    img = np.einsum('ijk->jki', img)
    return img


def np_to_tensor(img):
    img = torchvision.transforms.functional.to_tensor(img)
    img = img.type(torch.float32)
    return img

