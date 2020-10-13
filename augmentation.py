import torchvision
from randaugment import RandAugment
import random

def weak_augment(batch):
    torchvision.utils.save_image(batch[0], "img_weak_1.png")

    weak_transform = torchvision.transforms.Compose([
        #torchvision.transforms.Normalize((-0.5/0.5, -0.5/0.5, -0.5/0.5), (1/0.5, 1/0.5, 1/0.5)),
        torchvision.transforms.functional.to_pil_image,
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomAffine(0, translate=(0.0625, 0.0625)),
        torchvision.transforms.functional.to_tensor,
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    for i in range(len(batch)):
        batch[i] = weak_transform(batch[i].cpu())

    torchvision.utils.save_image(batch[0], "img_weak_aug.png")
    return batch


def strong_augment(batch):
    torchvision.utils.save_image(batch[0], "img_strog_1.png")

    strong_transform = torchvision.transforms.Compose([
        #torchvision.transforms.Normalize((-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5), (1 / 0.5, 1 / 0.5, 1 / 0.5)),
        torchvision.transforms.functional.to_pil_image,
        RandAugment(),
        torchvision.transforms.functional.to_tensor,
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    for i in range(len(batch)):
        batch[i] = strong_transform(batch[i].cpu())

        batch[i] = cutout(batch[i], 16, 16)


    torchvision.utils.save_image(batch[0], "img_strog_aug.png")
    return batch

def cutout(img, cut_hight, cut_width):
    img_hight = img.shape[1]
    img_width = img.shape[2]

    cut_start_hight = random.randrange(img_hight-cut_hight)
    cut_start_width = random.randrange(img_width-cut_width)

    for layer in img:
        for x in range(cut_start_hight, cut_start_hight + cut_hight):
            for y in range(cut_start_width, cut_start_width + cut_width):
                layer[x][y] = 0

    return img


