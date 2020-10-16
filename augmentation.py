#normaliseringar
#sparande av bilder
#elif


import torch
import torchvision
from randaugment import RandAugment
import random
import cv2
import numpy as np





class Wrapper:
    def __init__(self, transform1, transform2):
        self.transform1 = transform1
        self.transform2 = transform2

    def __call__(self, item):
        return self.transform1(item), self.transform2(item)





def weak_augment(batch):
    #torchvision.utils.save_image(batch[0], "img_weak_1.png")

    weak_transform = get_weak_transform()

    for i in range(len(batch)):
        batch[i] = weak_transform(batch[i].cpu())

    #torchvision.utils.save_image(batch[0], "img_weak_aug.png")
    return batch

def get_weak_transform():
    weak_transform = torchvision.transforms.Compose([
        #torchvision.transforms.Normalize((-0.5/0.5, -0.5/0.5, -0.5/0.5), (1/0.5, 1/0.5, 1/0.5)),
        #torchvision.transforms.functional.to_pil_image,
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomAffine(0, translate=(0.0625, 0.0625)),
        torchvision.transforms.functional.to_tensor,
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    return weak_transform





def strong_augment(batch, dataset_name):
    #torchvision.utils.save_image(batch[0], "img_strog_1.png")

    strong_transform = get_strong_transform(dataset_name)

    for i in range(len(batch)):
        batch[i] = strong_transform(batch[i].cpu())

        #cutout(batch[i], cutout_hight, cutout_width)

    #torchvision.utils.save_image(batch[0], "img_strog_aug.png")
    return batch
    
def get_strong_transform(dataset_name):
    strong_transform = torchvision.transforms.Compose([
        #torchvision.transforms.Normalize((-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5), (1 / 0.5, 1 / 0.5, 1 / 0.5)),
        #torchvision.transforms.functional.to_pil_image,
        RandAugment(),
        torchvision.transforms.functional.to_tensor,
        cutout_transform(dataset_name),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    return strong_transform





def SimCLR_augmentation(batch, transforms_to_do, dataset_name): #transforms_to_do = ["crop", "cutout", "colour", "sobel", "noise", "blur", "rotate"]
    torchvision.utils.save_image(batch[0], "img_SimCLR.png")

    for i in range(1):#range(len(batch)):
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
        #img = last_transforms(img)

        batch[i] = img

    torchvision.utils.save_image(batch[0], "img_SimCLR_aug.png")
    return batch



class cutout_transform(object):
    def __init__(self, dataset_name):
        if dataset_name == "CIFAR10":
            self.cutout_hight = 16
            self.cutout_width = 16
        elif dataset_name == "CIFAR100":
            self.cutout_hight = 8
            self.cutout_width = 8

    def __call__(self, img):
        img_hight = img.shape[1]
        img_width = img.shape[2]

        cut_start_hight = random.randrange(img_hight-self.cutout_hight)
        cut_start_width = random.randrange(img_width-self.cutout_width)

        for layer in img:
            for x in range(cut_start_hight, cut_start_hight + self.cutout_hight):
                for y in range(cut_start_width, cut_start_width + self.cutout_width):
                    layer[x][y] = 0
        return img


class crop_transform(object):
    def __init__(self, img_shape):
        self.img_shape = img_shape

    def __call__(self, img):
        crop = torchvision.transforms.Compose([
            torchvision.transforms.functional.to_pil_image,
            torchvision.transforms.RandomResizedCrop(self.img_shape, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333),
                                                    interpolation=2),  # default settings
            torchvision.transforms.RandomHorizontalFlip(p=0.5),  # improved perfomance according to SimCLR paper, Apendix A
            torchvision.transforms.functional.to_tensor
            ])
        return crop(img)

class colour_transform(object):
    def __init__(self, s):
        self.s = s

    def __call__(self, img):
        #s = 1 #colout distorition strength sugested by the SimCLR paper
        colour = torchvision.transforms.Compose([
            torchvision.transforms.functional.to_pil_image,
            torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)], p=0.8),
            torchvision.transforms.RandomGrayscale(p=0.2),
            torchvision.transforms.functional.to_tensor
            ])
        return colour(img)

def rotate_function(img):
    rotate_angle = random.randint(1, 3) * 90
    img = torchvision.transforms.functional.to_pil_image(img)
    img = torchvision.transforms.functional.rotate(img = img, angle = rotate_angle)
    img = torchvision.transforms.functional.to_tensor(img)
    return img
def rotate_transform():
    rotater = torchvision.transforms.Compose([
        rotate_function
        ])
    return rotater

def sobel_function(img):
    # https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html
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

    img = np_to_tensor(img/255)
    return img
def sobel_transform():
    sobel_transform = torchvision.transforms.Compose([
        sobel_function
        ])
    return sobel_transform

def noise_function(img):
    img = tensor_to_np(img)
    #https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
    row, col, ch = img.shape
    mean = 0
    sigma = 0.13
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    img = img + gauss

    img = np_to_tensor(img)
    return img
def noise_transform():
    noise_transform = torchvision.transforms.Compose([
        noise_function
        ])
    return noise_transform

def blur_function(img):
    random_float = random.uniform(0, 1)
    if random_float < 0.5:
        img = tensor_to_np(img)*255

        sigma = random.uniform(0.1, 2)
        kernal_size_hight = int(round(img.shape[1] * 0.1))
        kernal_size_width = int(round(img.shape[2] * 0.1))
        img = cv2.GaussianBlur(img, (kernal_size_hight, kernal_size_width), sigma)

        img = np_to_tensor(img/255)
    return img
def blur_transform():
    blur_transform = torchvision.transforms.Compose([
        blur_function
        ])
    return blur_transform


def tensor_to_np(img):
    img = img.detach().cpu().numpy()
    img = np.einsum('ijk->jki', img)
    return img

def np_to_tensor(img):
    img = torchvision.transforms.functional.to_tensor(img)
    img = img.type(torch.float32)
    return img