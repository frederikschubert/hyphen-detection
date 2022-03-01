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
        K.ColorJitter(0.4, 0.4, 0.4, 0.4),
        K.RandomBoxBlur(),
        K.RandomElasticTransform(),
        K.RandomGrayscale(),
        K.RandomHorizontalFlip(),
        K.RandomPosterize(),
        K.RandomAffine(
            degrees=360,
            translate=(0.2, 0.2),
            scale=(0.8, 0.9),
            padding_mode="reflection",
        ),
        K.RandomSharpness(),
        K.RandomSolarize(),
        K.RandomVerticalFlip(),
        data_keys=[
            "input",
            "mask",
        ],
        same_on_batch=False,
        random_apply=(3, 8),
        keepdim=True,
    )

    aug_norm = K.AugmentationSequential(
        K.Normalize(mean, std), data_keys=["input", "mask"], keepdim=True
    )
    return aug, aug_norm
