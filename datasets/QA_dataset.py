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

class QADataset(data.Dataset):
##Data Preprocess for QA

    def __init__(self, config, datatype='test'):
        ##dev 天城体 ##rom 罗马体 且数据均为天城体
        super().__init__()
        self.config = config
        self.datapath = f"{config.datadir}/{datatype}.txt"
        self.data=self.process_raw_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        full_data = self.data[index]
        rom_id=  full_data['rom_ids']
        rom_mask=full_data['rom_masks']
        dev_id=  full_data['dev_ids']
        dev_mask=full_data['dev_masks']
        length=  torch.tensor([self.config.max_length])
        answer= full_data['answer']


        return rom_id,rom_mask,dev_id,dev_mask,answer,length


    def dump_line_data(self,instance, dev_tokenizer,rom_tokenizer,max_context_length=256, stride=128):
        def locate_answer(context_ids, answer_ids):
            answer_start = 0
            answer_end = 0

            for i in range(len(context_ids) - len(answer_ids) + 1):
                if context_ids[i:i + len(answer_ids)] == answer_ids:
                    answer_start = i
                    answer_end = i + len(answer_ids) - 1
                    # print(context_ids[i:i + len(answer_ids)])
                    # print(answer_ids)
                    break

            return answer_start, answer_end

        instance = eval(instance)

        ##raw instance, convert to latin
        dev_question = instance['question'].strip()
        dev_context = instance['context'].strip()
        dev_answer = instance['answer'].strip()
        rom_question = devanagari_to_latin(dev_question)
        rom_context = devanagari_to_latin(dev_context)

        ##tokenize
        # 编码question和context
        encoded_dev_inputs = dev_tokenizer("<s> " + dev_question + " </s>", dev_context + " </s>",
                                           truncation="only_second",
                                           max_length=max_context_length,
                                           return_overflowing_tokens=True,
                                           stride=stride,
                                           padding='max_length',
                                           return_tensors='pt')

        encoded_rom_inputs = rom_tokenizer("<s> " + rom_question + " </s>", rom_context + " </s>",
                                           truncation="only_second",
                                           max_length=max_context_length,
                                           return_overflowing_tokens=True,
                                           stride=stride,
                                           padding='max_length',
                                           return_tensors='pt')

        encoded_dev_answer = dev_tokenizer.encode(dev_answer)

        dev_ids = encoded_dev_inputs['input_ids']  # [NumOfOverflow, MaxLength]
        dev_masks = encoded_dev_inputs['attention_mask']
        dev_answers = []
        for item in dev_ids:
            dev_ans_start, dev_ans_end = locate_answer(list(item.numpy()), encoded_dev_answer)
            dev_ans_tensor = torch.zeros((2, max_context_length))
            dev_ans_tensor[0][dev_ans_start] = 1
            dev_ans_tensor[1][dev_ans_end] = 1
            dev_answers.append(dev_ans_tensor.unsqueeze(0))
        dev_answer = torch.cat(dev_answers)

        rom_ids = encoded_rom_inputs['input_ids']  # [NumOfOverflow, MaxLength]
        rom_masks = encoded_rom_inputs['attention_mask']
        
        truncate_trunk = min(dev_ids.shape[0],rom_ids.shape[0])

        data = {
            'dev_ids': dev_ids[:truncate_trunk],
            'dev_masks': dev_masks[:truncate_trunk],
            'answer': dev_answer[:truncate_trunk],  # (batch,start+end,length)

            'rom_ids': rom_ids[:truncate_trunk],
            'rom_masks': rom_masks[:truncate_trunk],
        }
        
        # print(data['dev_ids'].shape)
        # {'answer': torch.Size([5, 2, 256]),
        #  'dev_ids': torch.Size([5, 256]),
        #  'dev_masks': torch.Size([5, 256]),
        #  'rom_ids': torch.Size([5, 256]),
        #  'rom_masks': torch.Size([5, 256])}

        return data


    def process_raw_data(self):
        dev_tokenizer = AutoTokenizer.from_pretrained("GKLMIP/roberta-hindi-devanagari")
        rom_tokenizer = AutoTokenizer.from_pretrained("GKLMIP/roberta-hindi-romanized")
        data = []
        with open(self.datapath, 'r', encoding='utf-8') as f:
            print("[INFO]: Processing raw data...")
            for line in tqdm(f.readlines()):
                data.append(self.dump_line_data(line,dev_tokenizer,rom_tokenizer,max_context_length=self.config.max_length,stride=self.config.stride))
        return data

    def get_class_num(self):
        return 2


def QA_collate_fn(X):
    X = list(zip(*X))
    rom_id,rom_mask,dev_id,dev_mask,answer,length = X

    rom_id = torch.cat(rom_id, 0)
    rom_mask = torch.cat(rom_mask, 0)
    dev_id = torch.cat(dev_id, 0)
    dev_mask = torch.cat(dev_mask, 0)
    answer = torch.cat(answer, 0)
    length = torch.cat(length, 0)

    return rom_id, rom_mask, dev_id, dev_mask, answer, length


def QA_data_loader(config,mode='train',shuffle=True,num_workers=4):
    dataset = QADataset(config,datatype=mode)
    loader = data.DataLoader(dataset=dataset,
                             batch_size=config.batch_size,
                             shuffle=shuffle,
                             pin_memory=False,
                             num_workers=num_workers,
                             collate_fn=QA_collate_fn)
    return loader


