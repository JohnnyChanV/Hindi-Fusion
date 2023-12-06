import torch
import torch.nn as nn
from transformers import AutoModel

from models.Fusion.TextCNN import TextCNN


class ModelFusion(nn.Module):
    def __init__(self,pooling_mode='mean',core_model='gate',embedding_dim=300,usePLM=True):
        super(ModelFusion, self).__init__()
        self.usePLM = usePLM
        if usePLM:
            self.dev_model = AutoModel.from_pretrained("GKLMIP/roberta-hindi-devanagari")
            self.rom_model = AutoModel.from_pretrained("GKLMIP/roberta-hindi-romanized")
            self.dev_model.train()
            self.rom_model.train()
        else:
            self.dev_word2vec = nn.Embedding(num_embeddings=32000,embedding_dim=embedding_dim,padding_idx=1)
            self.rom_word2vec = nn.Embedding(num_embeddings=32000,embedding_dim=embedding_dim,padding_idx=1)

        self.pooling_mode = pooling_mode #mean sum max
        self.core_model = core_model #gate bilstm

        self.PLM_hidden_size = self.dev_model.config.hidden_size if usePLM else embedding_dim

        self.sizeBeforePool = None

        if core_model=='gate':
            self.core = nn.Sequential(
                nn.Linear(self.PLM_hidden_size, self.PLM_hidden_size * 3),
                nn.Tanh(),
                nn.Linear(self.PLM_hidden_size * 3, self.PLM_hidden_size),
                nn.ReLU()
            )
            self.sizeBeforePool = self.PLM_hidden_size
            self.fusionfc = nn.Linear(self.PLM_hidden_size,self.PLM_hidden_size)
        elif core_model=='bilstm':
            self.lstm_h = 256
            self.core = nn.LSTM(768,self.lstm_h,batch_first=True,bidirectional=True)
            self.sizeBeforePool = self.lstm_h * 2 + self.PLM_hidden_size
            self.fusionfc = nn.Linear(self.lstm_h * 2 + self.PLM_hidden_size,self.lstm_h * 2 + self.PLM_hidden_size)

        elif core_model=='cnn':
            self.core = TextCNN(embed_size=300, num_filters=8, kernel_sizes=[4,8,16])



    def fusion(self,dev_embed,rom_embed):
        if self.core_model not in ['gate','bilstm']:
            return Exception(f"Fusion {self.core_model} not supported..[gate,bilstm]")

        if self.core_model=='gate':
            x = self.core(dev_embed) * rom_embed
            x = self.fusionfc(x)
            return x

        elif self.core_model=='bilstm': ##以拼接方式链接
            out,_ = self.core(dev_embed)
            out = torch.cat([out,rom_embed],2)
            out = self.fusionfc(out)
            return out

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
            dev_embed = self.dev_model(input_ids=dev_id,attention_mask=dev_mask).last_hidden_state
            rom_embed = self.rom_model(input_ids=rom_id,attention_mask=rom_mask).last_hidden_state
        else:
            dev_embed = self.dev_word2vec(dev_id)
            rom_embed = self.rom_word2vec(rom_id)

        fusion_feats = self.fusion(dev_embed,rom_embed)
        pooled_feats = self.pooling(fusion_feats,self.pooling_mode)
        return pooled_feats
