import torchvision
from randaugment import RandAugment

def weak_augment(batch):
    #torchvision.utils.save_image(batch[0], "img_weak_1.png")

    weak_transform = torchvision.transforms.Compose([
        #torchvision.transforms.Normalize((-0.5/0.5, -0.5/0.5, -0.5/0.5), (1/0.5, 1/0.5, 1/0.5)),
        torchvision.transforms.functional.to_pil_image,
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomAffine(0, translate=(0.0625, 0.0625)),
        torchvision.transforms.functional.to_tensor,
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    for i in range(len(batch)):
        batch[i] = weak_transform(batch[i])

    #torchvision.utils.save_image(batch[0], "img_weak_aug.png")
    return batch


def strong_augment(batch):
    #torchvision.utils.save_image(batch[0], "img_strog_1.png")

    strong_transform = torchvision.transforms.Compose([
        #torchvision.transforms.Normalize((-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5), (1 / 0.5, 1 / 0.5, 1 / 0.5)),
        torchvision.transforms.functional.to_pil_image,
        RandAugment(),
        torchvision.transforms.functional.to_tensor,
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    for i in range(len(batch)):
        batch[i] = strong_transform(batch[i])

    #torchvision.utils.save_image(batch[0], "img_strog_aug.png")
    return batch



