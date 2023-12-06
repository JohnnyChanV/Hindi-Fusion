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

class NERDataset(data.Dataset):
##Data Preprocess for POS

    def __init__(self, config, datatype='test'):
        ##dev 天城体 ##rom 罗马体 且数据均为天城体
        super().__init__()
        self.config = config
        self.datapath = f"{config.datadir}/{datatype}.json"
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
        poses=      torch.tensor(np.array([full_data['poses']]),dtype=torch.float)


        return rom_id,rom_mask,dev_id,dev_mask,poses,length

    def dump_line_data(self,line,dev_tk,rom_tk):
        words, poses = [i[1] for i in line], [self.label2id[i[-1]] for i in line]

        while len(poses) < self.config.max_length:
            poses+=[-100]

        line_data = {'rom_sent':devanagari_to_latin(' '.join(words)),
                     'rom_id':None,
                     'rom_mask':None,

                     'dev_sent': ' '.join(words),
                     'dev_id':None,
                     'dev_mask':None,

                     'words': words,
                     'poses': self.labels_to_onehot(poses[:self.config.max_length],max_length=self.config.max_length,num_class=self.get_class_num()),
                     # 'poses': poses[:self.config.max_length],

                     'length':len(poses)}

        if self.config.useCLS:
            temp_rom_data = rom_tk.encode_plus(f"{rom_tk.cls_token} " + line_data['rom_sent'] + f" {rom_tk.sep_token}", padding='max_length', max_length=self.config.max_length, truncation=True)
            temp_dev_data = dev_tk.encode_plus(f"{dev_tk.cls_token} " + line_data['dev_sent'] + f" {dev_tk.sep_token}", padding='max_length', max_length=self.config.max_length, truncation=True)
        else:
            temp_rom_data = rom_tk.encode_plus(line_data['rom_sent'], padding='max_length', max_length=self.config.max_length, truncation=True)
            temp_dev_data = dev_tk.encode_plus(line_data['dev_sent'], padding='max_length', max_length=self.config.max_length, truncation=True)
        line_data['rom_id'], line_data['rom_mask'] = temp_rom_data['input_ids'] ,  temp_rom_data['attention_mask']
        line_data['dev_id'], line_data['dev_mask'] = temp_dev_data['input_ids'] ,  temp_dev_data['attention_mask']
        # print(line_data)
        return line_data

    # def dump_line_data(self, line, dev_tk, rom_tk):
    #     words, poses = [i[1] for i in line], [self.label2id[i[-1]] for i in line]
    #
    #     while len(poses) < self.config.max_length:
    #         poses += [0]
    #
    #     rom_inputs = rom_tk(words, padding='max_length', max_length=self.config.max_length, truncation=True)
    #     dev_inputs = dev_tk(words, padding='max_length', max_length=self.config.max_length, truncation=True)
    #
    #     line_data = {
    #         'rom_sent': devanagari_to_latin(' '.join(words)),
    #         'rom_id': rom_inputs['input_ids'],
    #         'rom_mask': rom_inputs['attention_mask'],
    #
    #         'dev_sent': ' '.join(words),
    #         'dev_id': dev_inputs['input_ids'],
    #         'dev_mask': dev_inputs['attention_mask'],
    #
    #         'words': words,
    #         'poses': self.labels_to_onehot(poses[:self.config.max_length], max_length=self.config.max_length, num_class=self.get_class_num()),
    #         'length': len(words)
    #     }
    #
    #     return line_data

    def dump_line_data_(self,line,dev_tk,rom_tk):
        words, poses = [i[1] for i in line], [self.label2id[i[-1]] for i in line]

        while len(poses) < self.config.max_length:
            poses+=[0]

        line_data = {'rom_sent':[devanagari_to_latin(dev) for dev in words],
                     'rom_id':None,
                     'rom_mask':None,

                     'dev_sent': words,
                     'dev_id':None,
                     'dev_mask':None,

                     'words': words,
                     'poses': self.labels_to_onehot(poses[:self.config.max_length],max_length=self.config.max_length,num_class=self.get_class_num()),
                     # 'poses': poses[:self.config.max_length],

                     'length':len(words)}

        temp_dev_ids = dev_tk.convert_tokens_to_ids(line_data['dev_sent'])
        temp_rom_ids = rom_tk.convert_tokens_to_ids(line_data['rom_sent'])
        temp_dev_att = [1 for i in range(len(temp_dev_ids))]
        temp_rom_att = [1 for i in range(len(temp_rom_ids))]
        while len(temp_dev_ids) < self.config.max_length:
            temp_dev_ids += [1]
            temp_dev_att += [0]
        while len(temp_rom_ids) < self.config.max_length:
            temp_rom_ids += [1]
            temp_rom_att += [0]
        # print(temp_rom_ids)
        line_data['rom_id'], line_data['rom_mask'] = temp_rom_ids[:self.config.max_length],temp_rom_att[:self.config.max_length]
        line_data['dev_id'], line_data['dev_mask'] = temp_dev_ids[:self.config.max_length],temp_dev_att[:self.config.max_length]
        # print(len(temp_rom_ids),len(temp_rom_att),len(temp_dev_ids),len(temp_dev_att))
        return line_data
    def labels_to_onehot(self,labels,max_length,num_class):

        # 创建一个全零的数组，形状为 (max_length, num_classes)
        onehot = np.zeros((max_length, num_class))

        # 遍历标签列表，将对应位置的元素设置为1
        for i, label in enumerate(labels):
            onehot[i, label] = 1

        return onehot

    def process_raw_data(self):
        try:
            print(f"[INFO]: Loading Processed data...{self.datapath.split('.')[0]}")
            data = pickle.load(open(self.datapath.split('.')[0]+f'_{self.config.max_length}.pkl','rb'))
        except:
            dev_tokenizer = AutoTokenizer.from_pretrained("GKLMIP/roberta-hindi-devanagari",mirror='tuna')
            rom_tokenizer = AutoTokenizer.from_pretrained("GKLMIP/roberta-hindi-romanized",mirror='tuna')
            data = []
            f = json.load(open(self.datapath))
            print("[INFO]: Processing raw data...")
            current_index = f[0][0]
            sentences = []
            sentence = []
            for i in f:
                if i[0] == current_index:
                    sentence.append(i)
                else:
                    sentences.append(sentence)
                    current_index = i[0]
                    sentence = [i]
            sentences.append(sentence)
            for line in tqdm(sentences):
                data.append(self.dump_line_data(line,dev_tokenizer,rom_tokenizer))

        pickle.dump(data, open(self.datapath.split('.')[0] + f'_{self.config.max_length}.pkl', 'wb'))
        print("[INFO]: Processed data saved.")
        return data

    def init_POS(self):
        return json.load(open(f"{self.config.datadir}/dict.json"))

    def get_class_num(self):
        return len(self.label2id)


def NER_collate_fn(X):
    X = list(zip(*X))
    rom_id, rom_mask, dev_id, dev_mask, poses, length = X

    rom_id = torch.cat(rom_id, 0)
    rom_mask = torch.cat(rom_mask, 0)
    dev_id = torch.cat(dev_id, 0)
    dev_mask = torch.cat(dev_mask, 0)
    poses = torch.cat(poses, 0)
    length = torch.cat(length, 0)

    return rom_id, rom_mask, dev_id, dev_mask, poses, length


def NER_data_loader(config,mode='train',shuffle=True,num_workers=4):
    dataset = NERDataset(config,datatype=mode)
    loader = data.DataLoader(dataset=dataset,
                             batch_size=config.batch_size,
                             shuffle=shuffle,
                             pin_memory=False,
                             num_workers=num_workers,
                             collate_fn=NER_collate_fn)
    return loader
