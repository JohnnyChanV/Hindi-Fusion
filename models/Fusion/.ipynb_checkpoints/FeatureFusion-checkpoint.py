import torch
import torch.nn as nn
from transformers import AutoModel

class FeatureFusion(nn.Module):
    def __init__(self,pooling_mode='mean',fusion_mode='cat',embedding_dim=300,usePLM=True,useCLS=False):
        super(FeatureFusion, self).__init__()
        self.useCLS = useCLS
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
        self.fusion_mode = fusion_mode #cat mul

        if useCLS and usePLM:
            self.PLM_hidden_size = 1
        elif usePLM:
            self.PLM_hidden_size = self.dev_model.config.hidden_size
        else:
            self.PLM_hidden_size = embedding_dim
        self.sizeBeforePool = None

        if fusion_mode=='gate':
            self.sizeBeforePool = self.PLM_hidden_size
            self.gate = nn.Sequential(
                nn.Linear(self.PLM_hidden_size, self.PLM_hidden_size * 3),
                nn.Tanh(),
                nn.Linear(self.PLM_hidden_size * 3, self.PLM_hidden_size),
                nn.ReLU()
            )
            self.fusion_fc = nn.Sequential(
                nn.Linear(self.sizeBeforePool,self.sizeBeforePool),
                # nn.ReLU()
            )
        elif fusion_mode=='cat':
            self.sizeBeforePool = self.PLM_hidden_size * 2
            self.fusion_fc = nn.Sequential(
                nn.Linear(self.sizeBeforePool,self.sizeBeforePool*4),
                nn.Linear(self.sizeBeforePool*4,self.sizeBeforePool*2),
                nn.Linear(self.sizeBeforePool*2,self.sizeBeforePool),
                nn.ReLU()
            )
        elif fusion_mode == 'mul':
            self.sizeBeforePool = self.PLM_hidden_size
            self.fusion_fc = nn.Sequential(
                nn.Linear(self.sizeBeforePool,self.sizeBeforePool),
                # nn.ReLU()
            )


    def fusion(self,dev_embed,rom_embed,fusion_mode='cat'):
        if fusion_mode not in ['cat','mul','gate']:
            Exception(f"Fusion {fusion_mode} not supported..[cat,mul,gate]")

        if fusion_mode=='cat':
            x = torch.cat([dev_embed, rom_embed], 2) #(batch,len,2)
            # x = self.fusion_fc(x)
            return x
        elif fusion_mode=='mul':
            x = dev_embed * rom_embed #(batch,len)
            # x = self.fusion_fc(x)
            return x
        elif fusion_mode=='gate':
            x = self.gate(dev_embed) * rom_embed
            # x = self.fusion_fc(x)
            return x

    def pooling(self,x,pooling_mode='mean'):
        if pooling_mode not in ['mean','sum','max',None]:
            Exception(f"Pooling {pooling_mode} not supported..[mean,sum,max]")

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
            if self.useCLS:
                # self.pooling_mode = None
                dev_embed = dev_embed[:,:,0].unsqueeze(-1)
                rom_embed = rom_embed[:,:,0].unsqueeze(-1)
        else:
            dev_embed = self.dev_word2vec(dev_id)
            rom_embed = self.rom_word2vec(rom_id)

        fusion_feats = self.fusion(dev_embed,rom_embed,self.fusion_mode)
        pooled_feats = self.pooling(fusion_feats,self.pooling_mode)
        return pooled_feats

