import torch
import torch.nn as nn
from transformers import AutoModel
from .Fusion.ModelFusion import ModelFusion as MF
from .Fusion.FeatureFusion import FeatureFusion as FF
from .Fusion.DecisionFusion import DecisionFusion as DF
from .Classifier.POS_Classifier import POSClassifier
from .SinglePLM.Devanagari import DevanagariFusion as dev_PLM
from .SinglePLM.Romanized import RomanizedFusion as rom_PLM

class PosModel(nn.Module):
    def __init__(self,config,num_classes):
        super(PosModel, self).__init__()
        self.config = config

        ##pos任务不降维Pooling
        if self.config.encoder_name == 'MF':
            self.Encoder = MF(pooling_mode=None,core_model=config.encoder_config['core_model'],usePLM=config.usePLM)
        elif self.config.encoder_name=='FF':
            self.Encoder = FF(pooling_mode=None,fusion_mode=config.encoder_config['fusion_mode'],usePLM=config.usePLM,useCLS=config.useCLS)
        elif self.config.encoder_name=='DF':
            self.Encoder = DF(max_length=config.max_length,pooling_mode=config.encoder_config['pooling_mode'],core_model=config.encoder_config['core_model'],usePLM=config.usePLM,useCLS=config.useCLS)
        elif self.config.encoder_name=='dev_PLM':
            self.Encoder = dev_PLM(max_length=config.max_length,pooling_mode=config.encoder_config['pooling_mode'],core_model=config.encoder_config['core_model'],usePLM=config.usePLM,useCLS=config.useCLS)
        elif self.config.encoder_name=='rom_PLM':
            self.Encoder = rom_PLM(max_length=config.max_length,pooling_mode=config.encoder_config['pooling_mode'],core_model=config.encoder_config['core_model'],usePLM=config.usePLM,useCLS=config.useCLS)

        self.dropout = nn.Dropout(0.5)

        self.classifier = POSClassifier(self.Encoder.sizeBeforePool,num_classes)



    def forward(self,dev_id,dev_mask,rom_id,rom_mask):
        out = self.Encoder(dev_id,dev_mask,rom_id,rom_mask)
        out = self.dropout(out)
        x,proba = self.classifier(out)
        return x