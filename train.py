import click
import os
import pandas as pd
import logging

import torchvision
import torchvision.datasets as dset

from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
import numpy as np
import random
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchvision import models
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist as scipy_pdist
from sklearn.metrics import accuracy_score

from tqdm.autonotebook import tqdm
from torch.optim.lr_scheduler import _LRScheduler
import matplotlib.pyplot as plt

from models import TripletNetwork
from loss import TripletLoss

TRAIN_PATH = "../input/humpback-whale-identification/train"
TRAIN_LABELS_PATH = "../input/humpback-whale-identification/train.csv"
BOUNDING_BOXES_PATH = "../input/fixed/bounding_boxes_fixed.csv"

logger = logging.getLogger("re-identification")
logger.setLevel(logging.INFO)

np.random.seed(42)


@click.command()
@click.option("--train-path", type=str, help="Path to training images", required=True)
@click.option("--labels", type=str, help="Path to label file.", required=True)
@click.option("--boxes", type=str, help="Path to bounding boxes file.", required=True)
@click.option("--verbose", is_flag=True, help="Verbose execution.")
@click.option("--output-dir", trype=str, help="Output dir.", required=True)
def train(train_path, labels, boxes, output_dir, verbose):
    df_train, df_val = _get_toy_dataset(train_path, boxes)
    if verbose:
        logger.info("Train size: {}, validation size: {}".format(len(df_train), len(df_val)))
    
    train_dl, val_dl, single_train_dl = get_dataloaders(df_train, df_train)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if verbose:
        logger.info("Using device {}".format(device))
    

    net = TripletNetwork(embedding_size=128).to(device)
    criterion = TripletLoss(margin=1)
    optimizer = optim.Adam(net.parameters(), lr = 1e-4)

    epochs = 40
    net = _train(
        model=net, 
        optimizer=optimizer, 
        criterion=criterion, 
        train_dataloader=train_dl, 
        single_train_dataloader=single_train_dl, 
        val_dataloader=val_dl,
        num_epochs=epochs, 
        save_path=output_dir, 
        device=device)


def _train(model, optimizer, criterion, train_dataloader, single_train_dataloader, val_dataloader, num_epochs, save_path, device, patience=5):    
    best_accuracy = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for data in tqdm(train_dataloader):
            images, _ = data
            images = tuple(img.to(device) for img in images)

            optimizer.zero_grad()
            anchor_embeddings, positive_embeddings, negative_embeddings = model(images)
            loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
            running_loss += loss.item() * images.shape[0]
            loss.backward()
            optimizer.step()

        accuracy = _eval_model(model, single_train_dataloader, val_dataloader)

        if accuracy >= best_accuracy:
            torch.save(model.state_dict(), os.path.join(save_path, "best.pth"))
            best_accuracy = accuracy
        else:
            patience -= 1
        
        logger.info(f"Epoch {epoch}")
        logger.info(f"Training Loss: {running_loss / len(train_dataloader.dataset)}")
        logger.info(f"Validation Accuracy: {accuracy}, Best Accuracy: {best_accuracy}")
        
        if patience == 0:
            model.load_state_dict(torch.load(os.path.join(save_path, "best.pth")))
            return model



def _extract_embeddings(dataloader, model, embedding_sz=128, device="cuda"):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), embedding_sz))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            images = images.to(device)
            embeddings[k:k+len(images)] = model.get_embeddings(images).cpu().numpy()
            labels[k:k+len(images)] = [t for t in target]
            k += len(images)
    return embeddings, labels


def _eval_model(net, train_dataloader, val_dataloader):
    train_embeddings, train_labels = _extract_embeddings(train_dataloader, net)
    val_embeddings, val_labels = _extract_embeddings(val_dataloader, net)
    nneigh = NearestNeighbors(n_neighbors=1)
    nneigh.fit(train_embeddings)
    _, neighbors_trn = nneigh.kneighbors(val_embeddings)
    pred_labels = train_labels[neighbors_trn]
    
    return accuracy_score(val_labels, pred_labels)


def _get_toy_dataset(labels_file_path, bounding_boxes_path, num_classes=10, val_size=0.2):
    labels = pd.read_csv(labels_file_path)
    bboxes = pd.read_csv(bounding_boxes_path)
    df = pd.merge(labels, bboxes, on=["Image"])
    df = df[df["Id"] != "new_whale"]
    samples_per_class_count = df.Id.value_counts()
    df = df[df["Id"].isin(samples_per_class_count.head(num_classes).index)]
    df_train, df_val, _, _ = train_test_split(df, df["Id"], test_size=val_size)
    df_train = df_train.reset_index()
    df_val = df_val.reset_index()
    return df_train, df_val


if __name__ == "__main__":
    train()
