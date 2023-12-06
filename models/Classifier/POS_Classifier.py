import torch
import torch.nn as nn
from transformers import AutoModel

class POSClassifier(nn.Module):
    ## 上一层不做pooling!!!!! 因为是词语级别的分类
    def __init__(self,pre_hidden,num_classes):
        super(POSClassifier, self).__init__()
        self.fc = nn.Linear(pre_hidden,num_classes)
        self.fc1 = nn.ReLU()

    def forward(self,x):
        x = self.fc1(self.fc(x))
        proba = torch.softmax(x,-1)
        return x,proba