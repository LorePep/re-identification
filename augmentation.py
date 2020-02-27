import torchvision.transforms as transforms


def get_train_augmentation(img_size, mean, std):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(degrees=(-5, 5)),
            transforms.ColorJitter(brightness=0.1, hue=0.1),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def get_val_augmentation(img_size, mean, std):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
