import os
import json
import torch
import pickle
import numpy as np
import torch.utils.data as data
from tqdm import tqdm
from transformers import AutoTokenizer
from .utils import *
from config import config
import concurrent.futures

class ClsDataset(data.Dataset):
##Data Preprocess for Classification

    def __init__(self, config, datatype='test'):
        ##dev 天城体 ##rom 罗马体
        super().__init__()
        self.config = config
        self.datapath = f"{config.datadir}/{datatype}.txt"
        self.label2id = self.init_POS()
        self.data=self.process_raw_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        full_data = self.data[index]
        rom_id=     torch.tensor(np.array([full_data['rom_id']]),dtype=torch.long)
        rom_mask=   torch.tensor(np.array([full_data['rom_mask']]),dtype=torch.long)
        dev_id=     torch.tensor(np.array([full_data['dev_id']]),dtype=torch.long)
        dev_mask=   torch.tensor(np.array([full_data['dev_mask']]),dtype=torch.long)
        length=     torch.tensor(np.array([full_data['length']]),dtype=torch.long)
        labels=      torch.tensor(np.array([full_data['label']]),dtype=torch.float)


        return rom_id,rom_mask,dev_id,dev_mask,labels,length

    def dump_line_data(self,line,dev_tk,rom_tk):
        sent, label = line.strip().split('\t')
        label = self.label2id[label]

        line_data = {'rom_sent':devanagari_to_latin(sent),
                     'rom_id':None,
                     'rom_mask':None,

                     'dev_sent': sent,
                     'dev_id':None,
                     'dev_mask':None,

                     'label': self.label_to_onehot(label,num_class=self.get_class_num()),
                     # 'label': label,
                     'length':len(sent)}
        if self.config.useCLS:
            temp_rom_data = rom_tk.encode_plus(f"{rom_tk.cls_token} " + line_data['rom_sent'] + f" {rom_tk.sep_token}", padding='max_length', max_length=self.config.max_length, truncation=True)
            temp_dev_data = dev_tk.encode_plus(f"{dev_tk.cls_token} " + line_data['dev_sent'] + f" {dev_tk.sep_token}", padding='max_length', max_length=self.config.max_length, truncation=True)
        else:
            temp_rom_data = rom_tk.encode_plus(line_data['rom_sent'], padding='max_length', max_length=self.config.max_length, truncation=True)
            temp_dev_data = dev_tk.encode_plus(line_data['dev_sent'], padding='max_length', max_length=self.config.max_length, truncation=True)
        line_data['rom_id'], line_data['rom_mask'] = temp_rom_data['input_ids'] ,  temp_rom_data['attention_mask']
        line_data['dev_id'], line_data['dev_mask'] = temp_dev_data['input_ids'] ,  temp_dev_data['attention_mask']
        return line_data


    def label_to_onehot(self,label,num_class):

        # 创建一个全零的数组，形状为 (max_length, num_classes)
        onehot = np.zeros(num_class)

        # 遍历标签列表，将对应位置的元素设置为1
        onehot[label] = 1

        return onehot

    def process_raw_data(self):
        try:
            print(f"[INFO]: Loading Processed data...{self.datapath.split('.')[0]}")
            data = pickle.load(open(self.datapath.split('.')[0]+'.pkl','rb'))
        except:
            dev_tokenizer = AutoTokenizer.from_pretrained("GKLMIP/roberta-hindi-devanagari")
            rom_tokenizer = AutoTokenizer.from_pretrained("GKLMIP/roberta-hindi-romanized")
            data = []
            futures = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    with open(self.datapath, 'r', encoding='utf-8') as f:
                        print(f"[INFO]: Processing raw data...{self.datapath.split('.')[0]}")
                        for item in tqdm(f.readlines()):
                            future = executor.submit(self.dump_line_data, item,dev_tokenizer,rom_tokenizer)
                            futures.append(future)

                        for future in tqdm(concurrent.futures.as_completed(futures),total=len(futures)):
                            result = future.result()
                            data.append(result)
            pickle.dump(data,open(self.datapath.split('.')[0]+'.pkl','wb'))
            print("[INFO]: Processed data saved.")
        return data

    def init_POS(self):
        return json.load(open(f"{self.config.datadir}/dict.json"))

    def get_class_num(self):
        return len(self.label2id)


def Cls_collate_fn(X):
    X = list(zip(*X))
    rom_id, rom_mask, dev_id, dev_mask, labels, length = X

    rom_id = torch.cat(rom_id, 0)
    rom_mask = torch.cat(rom_mask, 0)
    dev_id = torch.cat(dev_id, 0)
    dev_mask = torch.cat(dev_mask, 0)
    labels = torch.cat(labels, 0)
    length = torch.cat(length, 0)

    return rom_id, rom_mask, dev_id, dev_mask, labels, length


def Cls_data_loader(config,mode='train',shuffle=True,num_workers=4):
    dataset = ClsDataset(config,datatype=mode)
    loader = data.DataLoader(dataset=dataset,
                             batch_size=config.batch_size,
                             shuffle=shuffle,
                             pin_memory=False,
                             num_workers=num_workers,
                             collate_fn=Cls_collate_fn)
    return loader
