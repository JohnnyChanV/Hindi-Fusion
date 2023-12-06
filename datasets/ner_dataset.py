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
        poses=      full_data['poses']
        return rom_id,rom_mask,dev_id,dev_mask,poses,length

    def dump_line_data(self,line,dev_tk,rom_tk):
        line_data = list(zip([i[1] for i in line], [self.label2id[i[-1]] for i in line]))

        poses = []
        dev_sentence_ids = []
        rom_sentence_ids = []

        for word,pos in line_data:
            word_dev_tokens = dev_tk.encode(word)
            word_rom_tokens = rom_tk.encode(devanagari_to_latin(word))
            word_poses = []

            word_poses.append(pos)
            while len(word_poses) < len(word_dev_tokens): ## dev as dominant
                word_poses.append(-100)
            poses += word_poses

            dev_sentence_ids += word_dev_tokens
            rom_sentence_ids += word_rom_tokens

        dev_sentence_ids = dev_sentence_ids[:self.config.max_length]
        rom_sentence_ids = rom_sentence_ids[:self.config.max_length]
        poses = poses[:self.config.max_length]
        dev_sentence_atts = [1 for i in range(len(dev_sentence_ids))]
        rom_sentence_atts = [1 for i in range(len(rom_sentence_ids))]

        while len(dev_sentence_atts) < self.config.max_length:
            dev_sentence_atts += [0]
            dev_sentence_ids += [dev_tk.pad_token_id]

        while len(rom_sentence_atts) < self.config.max_length:
            rom_sentence_atts += [0]
            rom_sentence_ids += [rom_tk.pad_token_id]

        while len(poses) < self.config.max_length:
            poses+=[-100]

        line_data = {'rom_sent':rom_sentence_ids,
                     'rom_id':rom_sentence_ids,
                     'rom_mask':rom_sentence_atts,

                     'dev_sent': dev_sentence_ids,
                     'dev_id':dev_sentence_ids,
                     'dev_mask':dev_sentence_atts,

                     'words': None,
                     'poses': self.labels_to_onehot(poses,max_length=self.config.max_length,num_class=self.get_class_num()),
                     # 'poses': poses[:self.config.max_length],

                     'length':len(dev_sentence_ids)}
        return line_data

    def labels_to_onehot(self,labels,max_length,num_class):

        return torch.tensor(labels).unsqueeze(0)

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

def NER_data_crossval_loader(config, num_folds=5, num_workers=4):
    dataset = NERDataset(config, datatype='train')
    total_data_length = len(dataset)
    fold_size = total_data_length // num_folds

    fold_loaders = []

    for fold in range(num_folds):
        start_idx = fold * fold_size
        end_idx = (fold + 1) * fold_size if fold < num_folds - 1 else total_data_length

        train_indices = list(range(0, start_idx)) + list(range(end_idx, total_data_length))
        test_indices = list(range(start_idx, end_idx))

        train_sampler = data.sampler.SubsetRandomSampler(train_indices)
        test_sampler = data.sampler.SubsetRandomSampler(test_indices)

        train_loader = data.DataLoader(dataset=dataset,
                                       batch_size=config.batch_size,
                                       sampler=train_sampler,
                                       pin_memory=False,
                                       num_workers=num_workers,
                                       collate_fn=NER_collate_fn,
                                       shuffle=True)

        test_loader = data.DataLoader(dataset=dataset,
                                      batch_size=config.batch_size,
                                      sampler=test_sampler,
                                      pin_memory=False,
                                      num_workers=num_workers,
                                      collate_fn=NER_collate_fn,
                                      shuffle=False)

        fold_loaders.append((train_loader, test_loader))

    return fold_loaders

