import torch
import torch.nn as nn
from transformers import AutoModel

from models.Fusion.TextCNN import TextCNN


class DecisionFusion(nn.Module):
    def __init__(self,max_length,pooling_mode='mean',core_model='cnn',embedding_dim=300,usePLM=True,useCLS=False):
        super(DecisionFusion, self).__init__()
        self.useCLS = useCLS
        self.usePLM = usePLM
        self.core_model = core_model
        if usePLM=='combine_dev':
            print(f"[INFO] :Combine PLM and Word2Vec..(dev)")
            self.dev_model = AutoModel.from_pretrained("GKLMIP/roberta-hindi-devanagari")
            self.dev_word2vec = nn.Embedding(num_embeddings=32000,embedding_dim=self.dev_model.config.hidden_size,padding_idx=1)
        elif usePLM=='combine_rom':
            print(f"[INFO] :Combine PLM and Word2Vec..(rom)")
            self.rom_model = AutoModel.from_pretrained("GKLMIP/roberta-hindi-romanized")
            self.rom_word2vec = nn.Embedding(num_embeddings=32000,embedding_dim=self.rom_model.config.hidden_size,padding_idx=1)
        elif usePLM==True:
            print(f"[INFO] :Use PLM..")
            self.dev_model = AutoModel.from_pretrained("GKLMIP/roberta-hindi-devanagari")
            self.rom_model = AutoModel.from_pretrained("GKLMIP/roberta-hindi-romanized")
            self.dev_model.train()
            self.rom_model.train()
        elif usePLM==False:
            print(f"[INFO] :Not Using PLM..")
            self.dev_word2vec = nn.Embedding(num_embeddings=32000,embedding_dim=embedding_dim,padding_idx=1)
            self.rom_word2vec = nn.Embedding(num_embeddings=32000,embedding_dim=embedding_dim,padding_idx=1)

        self.pooling_mode = pooling_mode #mean sum max

        if useCLS and usePLM==True:
            self.PLM_hidden_size = 1
        elif usePLM==True or (type(usePLM)!=bool and "combine" in usePLM): #not use CLS but use PLM, or combine two methods
            try:
                self.PLM_hidden_size = self.dev_model.config.hidden_size
            except:
                self.PLM_hidden_size = self.rom_model.config.hidden_size
        elif usePLM==False:
            self.PLM_hidden_size = embedding_dim


        self.lstm_h = 128
        if core_model=='bilstm':
            self.core = nn.LSTM(self.PLM_hidden_size*2,self.lstm_h,batch_first=True,bidirectional=True)
        elif core_model=='cnn':
            self.core = TextCNN(self.PLM_hidden_size*2,max_length)
        elif core_model=='mix':
            self.core1 = TextCNN(self.PLM_hidden_size*2,max_length)
            self.core2 = nn.LSTM(self.PLM_hidden_size*2,self.lstm_h,batch_first=True,bidirectional=True)
            
        self.sizeBeforePool = self.lstm_h * 2 if core_model=='bilstm' else 1 if core_model!='mix' else (self.lstm_h*2 + 1)

        self.fusionfc = nn.Sequential(
            nn.Linear(self.sizeBeforePool, self.sizeBeforePool),
            nn.ReLU()
        )

    def core_m(self,dev_embed,rom_embed):
        if self.core_model != 'mix':
            x = torch.cat([dev_embed,rom_embed],2)
            x = self.core(x)
            if type(x) == tuple:
                x = x[0]
            # x = self.fusionfc(x)
            return x
        elif self.core_model == 'mix':
            x = torch.cat([dev_embed,rom_embed],2)
            x1 = self.core1(x)
            x2 = self.core2(x)[0]
            x = torch.cat([x1,x2],-1)
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
        if self.usePLM==True:
            dev_embed = self.dev_model(input_ids=dev_id,attention_mask=dev_mask).last_hidden_state
            rom_embed = self.rom_model(input_ids=rom_id,attention_mask=rom_mask).last_hidden_state
            if self.useCLS:
                # self.pooling_mode = None
                dev_embed = dev_embed[:,:,0].unsqueeze(-1)
                rom_embed = rom_embed[:,:,0].unsqueeze(-1)
        elif self.usePLM==False:
            dev_embed = self.dev_word2vec(dev_id)
            rom_embed = self.rom_word2vec(rom_id)
        elif self.usePLM=="combine_dev":
            dev_embed = self.dev_model(input_ids=dev_id,attention_mask=dev_mask).last_hidden_state
            rom_embed = self.dev_word2vec(dev_id)
        elif self.usePLM=="combine_rom":
            dev_embed = self.rom_model(input_ids=rom_id,attention_mask=rom_mask).last_hidden_state
            rom_embed = self.rom_word2vec(rom_id)
            
        fusion_feats = self.core_m(dev_embed,rom_embed)
        pooled_feats = self.pooling(fusion_feats,self.pooling_mode)
        return pooled_feats
