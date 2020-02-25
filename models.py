import torch.nn as nn
from torchvision import models


def unfreeze_all(model_params):
    for param in model_params:
        param.requires_grad = True
        
        
class TripletNetwork(nn.Module):
    def __init__(self, embedding_size, return_triplets=False):
        super(TripletNetwork, self).__init__()
        self.model = models.resnet34(pretrained=True)
        self.model.fc = nn.Sequential(
            nn.Linear(512, embedding_size),
        )
        unfreeze_all(self.model.parameters())
        
    def get_embeddings(self, x):
        return self.model(x)

    def forward(self, x):
        assert len(x) == 3
        anchor_embedding = self.model(x[0])
        positive_embedding = self.model(x[1])
        negative_embedding = self.model(x[2])
        return anchor_embedding, positive_embedding, negative_embedding
