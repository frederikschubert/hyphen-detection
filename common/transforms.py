import torch
from torchvision import transforms
import kornia.augmentation as K


def transforms_val(
    mean=0.5,
    std=0.5,
):
    aug = K.AugmentationSequential(
        K.Normalize(mean, std),
        data_keys=[
            "input",
            "mask",
        ],
        keepdim=True,
    )
    return None, aug


def transforms_train(
    mean=0.5,
    std=0.5,
):
    aug = K.AugmentationSequential(
        K.ColorJitter(),
        K.RandomBoxBlur(),
        K.RandomElasticTransform(),
        K.RandomGrayscale(),
        K.RandomGaussianNoise(),
        K.RandomHorizontalFlip(),
        K.RandomPosterize(),
        K.RandomAffine(180),
        K.RandomSharpness(),
        K.RandomSolarize(),
        K.RandomVerticalFlip(),
        data_keys=[
            "input",
            "mask",
        ],
        same_on_batch=False,
        random_apply=(2, 5),
        keepdim=True,
    )

    aug_norm = K.AugmentationSequential(
        K.Normalize(mean, std), data_keys=["input", "mask"], keepdim=True
    )
    return aug, aug_norm
