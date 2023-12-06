import json

import matplotlib.pyplot as plt
from transformers import AutoTokenizer

dev_tokenizer = AutoTokenizer.from_pretrained("GKLMIP/roberta-hindi-devanagari", mirror='tuna')
rom_tokenizer = AutoTokenizer.from_pretrained("GKLMIP/roberta-hindi-romanized", mirror='tuna')

train_data = json.load(open('train.json'))


train_tokens = []
pre = 0
this_tokens = []
for item in train_data:
    if item[0]== pre:
        this_tokens.append(item[1])
    else:
        train_tokens.append(dev_tokenizer.encode(' '.join(this_tokens)))
        this_tokens = []
        pre+=1
train_tokens.append(this_tokens)
lens = [len(item) for item in train_tokens]
lens.sort()
plt.scatter([i for i in range(len(lens))],lens)
plt.show()