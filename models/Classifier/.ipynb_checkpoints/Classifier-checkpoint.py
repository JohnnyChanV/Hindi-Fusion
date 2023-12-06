import torch
import torch.nn as nn
from transformers import AutoModel

class Classifier(nn.Module):
    def __init__(self,max_len,num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(max_len,num_classes)

    def forward(self,x):
        x = (self.fc(x))
        proba = torch.softmax(x,-1)
        return x,proba