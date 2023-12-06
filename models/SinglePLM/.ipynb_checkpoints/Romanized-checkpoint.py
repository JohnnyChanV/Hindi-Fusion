import torch
import torch.nn as nn
from transformers import AutoModel

from models.Fusion.TextCNN import TextCNN


class RomanizedFusion(nn.Module):
    def __init__(self,max_length,pooling_mode='mean',core_model=None,embedding_dim=300,usePLM=False,useCLS=False):
        super(RomanizedFusion, self).__init__()
        self.usePLM = usePLM
        self.useCLS = useCLS
        if usePLM:
            self.rom_model = AutoModel.from_pretrained("GKLMIP/roberta-hindi-romanized")
        else:
            self.rom_word2vec = nn.Embedding(num_embeddings=32000, embedding_dim=embedding_dim, padding_idx=1)

        self.pooling_mode = pooling_mode #mean sum max
        self.core_model = core_model #gate bilstm

        if useCLS==True and usePLM==True:
            self.PLM_hidden_size = 1
        elif usePLM==True or (type(usePLM)!=bool and "combine" in usePLM): #not use CLS but use PLM, or combine two methods
            try:
                self.PLM_hidden_size = self.dev_model.config.hidden_size
            except:
                self.PLM_hidden_size = self.rom_model.config.hidden_size
        else:
            self.PLM_hidden_size = embedding_dim
            
        self.lstm_h = 64
        if self.core_model=='bilstm':
            self.core = nn.LSTM(self.PLM_hidden_size,64,batch_first=True,bidirectional=True)
        elif self.core_model=='cnn':
            self.core = TextCNN(self.PLM_hidden_size, max_length)

        self.sizeBeforePool = self.lstm_h * 2 if self.core_model == 'bilstm' else self.PLM_hidden_size

        self.fusionfc = nn.Sequential(
            nn.Linear(self.sizeBeforePool, self.sizeBeforePool),
            nn.ReLU()
        )

    def core_m(self,rom_embed):
        if self.core_model not in ['bilstm','cnn',None]:
            return Exception(f"Core {self.core_model} not supported..[mean,sum,max]")

        if self.core_model == None:
            return rom_embed
        elif self.core_model == 'bilstm':
            x,_ = self.core(rom_embed)
            # x = self.fusionfc(x)
            return x
        elif self.core_model=='cnn':
            x = self.core(rom_embed)
            # x = self.fusionfc(x)
            return x
    def pooling(self,x,pooling_mode='mean'):
        if pooling_mode not in ['mean','sum','max',None]:
            return Exception(f"Pooling {pooling_mode} not supported..[mean,sum,max]")

        if pooling_mode=='mean':
            return x.mean(2)
        elif pooling_mode=='sum':
            return x.sum(2)
        elif pooling_mode=='max':
            return x.max(2).values
        elif pooling_mode==None:
            return x

    def forward(self,dev_id,dev_mask,rom_id,rom_mask):
        if self.usePLM:
            rom_embed = self.rom_model(input_ids=rom_id,attention_mask=rom_mask).last_hidden_state
            if self.useCLS:
                rom_embed = rom_embed[:,:,0].unsqueeze(-1)
        else:
            rom_embed = self.rom_word2vec(rom_id)
        fusion_feats = self.core_m(rom_embed)
        pooled_feats = self.pooling(fusion_feats,self.pooling_mode)
        return pooled_feats
