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
from sklearn.metrics import accuracy_score

from tqdm.autonotebook import tqdm
from torch.optim.lr_scheduler import _LRScheduler
import matplotlib.pyplot as plt

from models import TripletNetwork
from losses import TripletLoss
from dataloaders import get_dataloaders

COLOR_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

TRAIN_PATH = "../input/humpback-whale-identification/train"
TRAIN_LABELS_PATH = "../input/humpback-whale-identification/train.csv"
BOUNDING_BOXES_PATH = "../input/fixed/bounding_boxes_fixed.csv"

logging.getLogger().setLevel(logging.INFO)
np.random.seed(42)

@click.command()
@click.option("--train-path", type=str, help="Path to training images", required=True)
@click.option("--labels", type=str, help="Path to label file.", required=True)
@click.option("--boxes", type=str, help="Path to bounding boxes file.", required=True)
@click.option("--verbose", "-v", is_flag=True, help="Verbose execution.")
@click.option("--output-dir", type=str, help="Output dir.", required=True)
@click.option("--num-epochs", "-e", type=int, help="Output dir.", default=100)
def train(train_path, labels, boxes, output_dir, num_epochs, verbose):
    df_train, df_val = _get_toy_dataset(labels, boxes)
    if verbose:
        logging.info("Train size: {}, validation size: {}".format(len(df_train), len(df_val)))
    
    train_dl, single_train_dl, val_dl  = get_dataloaders(df_train, df_val, train_path)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if verbose:
        logging.info("Using device {}".format(device))
    

    net = TripletNetwork(embedding_size=128).to(device)
    criterion = TripletLoss(margin=1)
    optimizer = optim.Adam(net.parameters(), lr = 1e-4)

    net, history = _train(model=net, optimizer=optimizer, criterion=criterion, 
    train_dataloader=train_dl, single_train_dataloader=single_train_dl, 
    val_dataloader=val_dl, num_epochs=num_epochs, save_path=output_dir, device=device)
    _plot_history(history)


def _train(model, optimizer, criterion, train_dataloader, single_train_dataloader, val_dataloader, num_epochs, save_path, device, patience=10):    
    best_accuracy = 0.0
    history_emb_norm_50 = []
    history_emb_norm_95 = []

    history_emb_dist_50 = []
    history_emb_dist_95 = []
    
    patience_counter = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for data in tqdm(train_dataloader):
            images, _ = data
            images = tuple(img.to(device) for img in images)

            optimizer.zero_grad()
            anchor_embeddings, positive_embeddings, negative_embeddings = model(images)
            norm = np.linalg.norm(anchor_embeddings.detach().cpu().numpy(), axis=1)
            history_emb_norm_50.append(np.percentile(norm, 50))
            history_emb_norm_95.append(np.percentile(norm, 95))
            
            dists = _pdist(anchor_embeddings, anchor_embeddings).detach().cpu().numpy()
            history_emb_dist_50.append(np.percentile(dists, 50))
            history_emb_dist_95.append(np.percentile(dists, 95))

            loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
            running_loss += loss.item() * images[0].shape[0]
            loss.backward()
            optimizer.step()

        accuracy = _eval_model(model, single_train_dataloader, val_dataloader, epoch)
        if accuracy >= best_accuracy:
            torch.save(model.state_dict(), os.path.join(save_path, "./best.pth"))
            best_accuracy = accuracy
            patience_counter = 0
        else:
            patience += 1
        print(f"Epoch {epoch}")
        print(f"Training Loss: {running_loss / len(train_dataloader.dataset)}")
        print(f"Validation Accuracy: {accuracy}, Best Accuracy: {best_accuracy}")
        
        if patience == patience_counter:
            model.load_state_dict(torch.load(os.path.join(save_path, "./best.pth")))
            return model, dict(history_emb_norm_50=history_emb_norm_50, history_emb_norm_95=history_emb_norm_95, 
            history_emb_dist_50=history_emb_dist_50, history_emb_dist_95=history_emb_dist_95)

    return model, dict(history_emb_norm_50=history_emb_norm_50, history_emb_norm_95=history_emb_norm_95, 
            history_emb_dist_50=history_emb_dist_50, history_emb_dist_95=history_emb_dist_95)


def _pdist(emb1, emb2):
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx


def _extract_embeddings(dataloader, model, embedding_sz=128, device="cuda"):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), embedding_sz))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            images = images.to(device)
            output = model.get_embeddings(images).cpu().numpy()
            embeddings[k:k+len(images)] = output
            labels[k:k+len(images)] = [t for t in target]
            k += len(images)
    return embeddings, labels


def _eval_model(net, train_dataloader, val_dataloader, epoch):
    train_embeddings, train_labels = _extract_embeddings(train_dataloader, net)
    labels_set = set(train_labels)
    _plot_embeddings(train_embeddings, train_labels, labels_set)
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.savefig("{}.png".format(epoch))
    plt.close()

    val_embeddings, val_labels = _extract_embeddings(val_dataloader, net)
    nneigh = NearestNeighbors(n_neighbors=1)
    nneigh.fit(train_embeddings)
    distances_trn, neighbors_trn = nneigh.kneighbors(val_embeddings)
    pred_labels = train_labels[neighbors_trn]
    return accuracy_score(val_labels, pred_labels)


def _get_toy_dataset(labels_file_path, box_file_path, num_classes=10, val_size=0.2):
    labels = pd.read_csv(labels_file_path)
    bboxes = pd.read_csv(box_file_path)
    df = pd.merge(labels, bboxes, on=["Image"])
    df = df[df["Id"] != "new_whale"]
    samples_per_class_count = df.Id.value_counts()
    df = df[df["Id"].isin(samples_per_class_count.head(num_classes).index)]
    df_train, df_val, _, _ = train_test_split(df, df["Id"], test_size=val_size)
    df_train = df_train.reset_index()
    df_val = df_val.reset_index()
    return df_train, df_val


def _plot_history(history):
    plt.figure(figsize=(10, 10))
    plt.subplot(211)
    plt.plot(history["history_emb_norm_50"], label="p50")
    plt.plot(history["history_emb_norm_95"], label="P95")
    plt.xlabel("batches")
    plt.title("L2 Norm of embeddings")
    plt.legend()
    plt.subplot(212)
    plt.plot(history["history_emb_dist_50"], label="p50")
    plt.plot(history["history_emb_dist_95"], label="P95")
    plt.title("Distance between embeddings")
    plt.xlabel("batches")
    plt.legend()
    plt.savefig("history.png")


def _plot_embeddings(embeddings, targets, idxs):
    plt.figure(figsize=(10,10))
    for idx in idxs:
        inds = np.where(targets == idx)[0]
        plt.scatter(embeddings[inds, 0], embeddings[inds, 1], alpha=0.5, color=COLOR_PALETTE[int(idx)%10])
    plt.legend(idxs)


if __name__ == "__main__":
    train()
