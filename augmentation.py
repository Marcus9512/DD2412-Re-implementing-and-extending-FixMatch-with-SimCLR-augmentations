#normaliseringar
#sparande av bilder
#hur anropa flera SimCLR anrop


import torch
import torchvision
from randaugment import RandAugment
import random
import cv2
import numpy as np


cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)



class Wrapper:
    def __init__(self, transform1, transform2, dataset):
        if dataset == "CIFAR10":
            mean = cifar10_mean
            std = cifar10_std
        elif dataset == "CIFAR100":
            mean = cifar100_mean
            std = cifar100_std
        else:
            print("WRONG PARAMETER AT WRAPPER")
            exit()

        self.transform1 = torchvision.transforms.Compose([transform1,
                                                        torchvision.transforms.functional.to_tensor,
                                                        torchvision.transforms.Normalize(mean, std)
                                                        ])
        self.transform2 = torchvision.transforms.Compose([transform2,
                                                        torchvision.transforms.functional.to_tensor,
                                                        torchvision.transforms.Normalize(mean, std)
                                                        ])

    def __call__(self, item):
        return self.transform1(item), self.transform2(item)


def select_strong_augment(experiment_name, dataset_name, augment1=None, augment2=None):
    if experiment_name == "experiment1":
        print("EXPERIMENT1")
        return get_strong_transform(dataset_name)

    elif experiment_name == "experiment3":
        print("EXPERIMENT3")
        return get_strong_transform_two_randaugment(dataset_name)

    elif experiment_name == "experiment2":
        print("EXPERIMENT2")
        print("A1 ",augment1)
        print("A2 ",augment2)
        return get_sim_clr_augmentations(dataset_name, augment1, augment2)

    else:
        print("NO expperiment slected")
        exit()

def weak_augment(batch):
    #torchvision.utils.save_image(batch[0], "img_weak_1.png")

    weak_transform = get_weak_transform()

    for i in range(len(batch)):
        batch[i] = weak_transform(batch[i].cpu())

    #torchvision.utils.save_image(batch[0], "img_weak_aug.png")
    return batch

def get_weak_transform():
    weak_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomAffine(0, translate=(0.0625, 0.0625)),
        ])
    return weak_transform

def strong_augment(batch, dataset_name):

    strong_transform = get_strong_transform(dataset_name)

    for i in range(len(batch)):
        batch[i] = strong_transform(batch[i].cpu())

    return batch
    
def get_strong_transform(dataset_name):
    strong_transform = torchvision.transforms.Compose([
        RandAugment(),
        cutout_transform(dataset_name),
        ])
    return strong_transform

def get_strong_transform_two_randaugment(dataset_name):
    strong_transform = torchvision.transforms.Compose([
        RandAugment(),
        RandAugment(),
        cutout_transform(dataset_name),
        ])
    return strong_transform

def get_sim_clr_augmentations(dataset_name, augment1, augment2):
    if augment1 == "color":
        a1 = colour_transform(1)
    elif augment1 == "sobel":
        a1 = sobel_transform()
    elif augment1 == "cutout":
        a1 = cutout_transform(dataset_name)
    elif augment1 == "crop":
        a1 = crop_transform([32,32])
    else:
        print("NO VAILD a1 transform")
        exit()

    if augment2 == "color":
        a2 = colour_transform(1)
    elif augment2 == "sobel":
        a2 = sobel_transform()
    elif augment2 == "cutout":
        a2 = cutout_transform(dataset_name)
    elif augment2 == "crop":
        a2 = crop_transform([32,32])
    else:
        print("NO VAILD a1 transform")
        exit()

    sim_clr_transform = torchvision.transforms.Compose([
        a1,
        a2
    ])

    return sim_clr_transform



def SimCLR_augmentation(batch, transforms_to_do, dataset_name): #transforms_to_do = ["crop", "cutout", "colour", "sobel", "noise", "blur", "rotate"]
    '''
    DEPRECATED, not used anymore
    :param batch:
    :param transforms_to_do:
    :param dataset_name:
    :return:
    '''
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
            self.cutout_height = 16
            self.cutout_width = 16
        elif dataset_name == "CIFAR100":
            self.cutout_height = 8
            self.cutout_width = 8

    def __call__(self, img):

        img.load()
        img_height, img_width= img.size

        cut_start_height = random.randrange(img_height-self.cutout_height)
        cut_start_width = random.randrange(img_width-self.cutout_width)

        #for layer in img:
        for x in range(cut_start_height, cut_start_height + self.cutout_height):
            for y in range(cut_start_width, cut_start_width + self.cutout_width):
                img.putpixel((x, y), (0, 0, 0)) #PIL
        return img
    


class crop_transform(object):
    def __init__(self, img_shape):
        self.img_shape = img_shape

    def __call__(self, img):
        crop = torchvision.transforms.Compose([
            #torchvision.transforms.functional.to_pil_image,
            torchvision.transforms.RandomResizedCrop(self.img_shape, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333),
                                                    interpolation=2),  # default settings
            torchvision.transforms.RandomHorizontalFlip(p=0.5),  # improved perfomance according to SimCLR paper, Apendix A
            #torchvision.transforms.functional.to_tensor
            ])
        return crop(img)

class colour_transform(object):
    def __init__(self, s):
        self.s = s

    def __call__(self, img):
        #s = 1 #colout distorition strength sugested by the SimCLR paper
        colour = torchvision.transforms.Compose([
            #torchvision.transforms.functional.to_pil_image,
            torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)], p=0.8),
            torchvision.transforms.RandomGrayscale(p=0.2),
            #torchvision.transforms.functional.to_tensor
            ])
        return colour(img)

#def rotate_function(img):
class rotate_transform(object):
    def __init__(self):
        pass
    def __call__(self, img):
        rotate_angle = random.randint(1, 3) * 90
        #img = torchvision.transforms.functional.to_pil_image(img)
        img = torchvision.transforms.functional.rotate(img = img, angle = rotate_angle)
        #img = torchvision.transforms.functional.to_tensor(img)
        return img
"""
def rotate_transform():
    rotater = torchvision.transforms.Compose([
        rotate_function
        ])
    return rotater
"""

#def sobel_function(img):
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

        img = np_to_tensor(img/255)
        img = torchvision.transforms.functional.to_pil_image(img).convert('RGB')
        return img
"""     
def sobel_transform():
    sobel_transform = torchvision.transforms.Compose([
        sobel_function
        ])
    return sobel_transform
"""

#def noise_function(img):
class noise_transform(object):
    def __init__(self):
        pass
    def __call__(self, img):
        #https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
        img = torchvision.transforms.functional.to_tensor(img)
        row, col, ch = img.shape
        mean = 0
        sigma = 0.07#0.13
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
#def blur_function(img):
class blur_transform(object):
    def __init__(self):
        pass
    def __call__(self, img):
        random_float = random.uniform(0, 1)
        if random_float < 0.5:
            img = torchvision.transforms.functional.to_tensor(img)
            img = tensor_to_np(img)*255

            sigma = random.uniform(0.1, 2)
            kernal_size_height = int(round(img.shape[1] * 0.1))
            kernal_size_width = int(round(img.shape[2] * 0.1))
            img = cv2.GaussianBlur(img, (kernal_size_height, kernal_size_width), sigma)

            img = np_to_tensor(img/255)
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