from augmentation import get_train_augmentation, get_val_augmentation

from torch.utils.data import DataLoader
from datasets import TripletNetworkDataset, SingleImageDataset


DEFAULT_RESNET_MEAN = [0.485, 0.456, 0.406]
DEFAULT_RESNET_STD = [0.229, 0.224, 0.225]


def get_dataloaders(df_train, df_val, train_path, img_size=224, mean=DEFAULT_RESNET_MEAN, std=DEFAULT_RESNET_STD, batch_size=16):
    labels_to_idx = {label: i for i, label in enumerate(sorted(df_train["Id"].unique()))}

    train_tranforms = get_train_augmentation(img_size, mean, std)
    val_transforms = get_val_augmentation(img_size, mean, std)

    train_dataset = TripletNetworkDataset(
        path=train_path,
        df=df_train,
        transforms=train_tranforms)


    single_train_dataset = SingleImageDataset(
        path=train_path,
        df=df_train,
        labels_to_idx=labels_to_idx,
        transforms=val_transforms)

    single_val_dataset = SingleImageDataset(
        path=train_path,
        df=df_val,
        labels_to_idx=labels_to_idx,
        transforms=val_transforms)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        num_workers=4,
        batch_size=batch_size
    )

    single_train_dataloader = DataLoader(
        single_train_dataset,
        shuffle=False,
        num_workers=4,
        batch_size=batch_size
    )

    single_val_dataloader = DataLoader(
        single_val_dataset,
        shuffle=False,
        num_workers=4,
        batch_size=batch_size
    )

    return train_dataloader, single_train_dataloader, single_val_dataloader
