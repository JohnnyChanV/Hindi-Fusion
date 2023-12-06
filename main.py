import numpy as np
import torch
from torch import nn
from config import config
from datasets.classification_dataset import Cls_data_loader
from datasets.ner_dataset import NER_data_loader
from datasets.pos_dataset import Pos_data_loader
from datasets.QA_dataset import QA_data_loader
from datasets.Infer_dataset import Infer_data_loader
from models.ClassificationModel import ClsModel
from models.PosModel import PosModel
from models.InferModel import InferModel
from models.QAModel import QAModel
from sklearn import metrics
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score, precision_score, \
    recall_score
from tqdm import tqdm
from pprint import pprint
import datetime
import copy

from torch.utils.tensorboard import SummaryWriter

import os

os.environ["http_proxy"] = "http://192.168.235.34:7890"
os.environ["https_proxy"] = "http://192.168.235.34:7890"


def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    y_true = []
    y_pred = []
    print(f'[INFO]: Training..{epoch}')
    loader_ = tqdm(loader)
    for index, data in enumerate(loader_):
        rom_id, rom_mask, dev_id, dev_mask, poses, length = data
        optimizer.zero_grad()
        logits = model(dev_id.to(device), dev_mask.to(device), rom_id.to(device), rom_mask.to(device))
        proba = torch.softmax(logits, -1)
        poses_ = poses.argmax(-1)
        # print(logits.shape,poses_.shape)
        loss = criterion(logits.view(-1, loader.dataset.get_class_num()), poses_.view(-1).to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        y_true += poses.argmax(dim=-1).flatten().tolist()
        y_pred += proba.argmax(dim=-1).flatten().tolist()

        ##show info
        loader_.set_postfix(loss=loss.item())
        # if index%10==0 and index!=0:
        #     log_writer.add_scalar(f"train/{run_name}-loss",scalar_value=float(total_loss)/10,global_step=index+len(loader_)*epoch)
        #     log_writer.close()
        #     total_loss = 0

    # report = classification_report(y_true, y_pred)
    # print(report)
    return total_loss / len(loader)


def valid(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    y_true = []
    y_pred = []
    probas = []
    with torch.no_grad():
        print('[INFO]: Validating..')
        loader_ = tqdm(loader)
        for index, data in enumerate(loader_):
            rom_id, rom_mask, dev_id, dev_mask, poses, length = data
            logits = model(dev_id.to(device), dev_mask.to(device), rom_id.to(device), rom_mask.to(device))
            poses_ = poses.argmax(-1)
            proba = torch.softmax(logits, -1)
            loss = criterion(logits.view(-1, loader.dataset.get_class_num()), poses_.view(-1).to(device))
            total_loss += loss.item()

            y_true += poses.argmax(dim=-1).flatten().tolist()
            y_pred += proba.argmax(dim=-1).flatten().tolist()
            probas += proba.unsqueeze(0).detach().cpu()
            ##show info
            loader_.set_postfix(loss=loss.item())
        if config.task == "pos" or config.task == "ner":
            filtered_y_true = []
            filtered_y_pred = []
            for true_label, pred_label in zip(y_true, y_pred):
                if true_label != 0:
                    filtered_y_true.append(true_label)
                    filtered_y_pred.append(pred_label)
            acc_no_zero = accuracy_score(filtered_y_true, filtered_y_pred)
            f1_score_ = f1_score(filtered_y_true, filtered_y_pred, average='micro')
            pr_ = precision_score(filtered_y_true, filtered_y_pred, average='micro')
            recall_score_ = recall_score(filtered_y_true, filtered_y_pred, average='micro')

            report = classification_report(filtered_y_true, filtered_y_pred, digits=5)
            auc = None
            # print('Accuracy (without label 0):', acc_no_zero)
        else:
            report = classification_report(y_true, y_pred, digits=5)
            acc_no_zero = accuracy_score(y_true, y_pred)
            f1_score_ = f1_score(y_true, y_pred, average='macro')
            auc = roc_auc_score(y_true, torch.cat(probas, 0), multi_class='ovr')
            # print('Accuracy:', acc_no_zero)
        # log_writer.add_scalar(f"val/{run_name}-acc", scalar_value=acc_no_zero,global_step=epoch//config.ValidRound)
        # log_writer.close()
        # print(report)
        return total_loss / len(loader), acc_no_zero, f1_score_, auc


def test(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    probas = []
    with torch.no_grad():
        print('[INFO]: Testing..')
        for index, data in enumerate(tqdm(loader)):
            rom_id, rom_mask, dev_id, dev_mask, poses, length = data
            logits = model(dev_id.to(device), dev_mask.to(device), rom_id.to(device), rom_mask.to(device))
            proba = torch.softmax(logits, -1)
            y_true += poses.argmax(dim=-1).flatten().tolist()
            y_pred += proba.argmax(dim=-1).flatten().tolist()
            probas += proba.unsqueeze(0).detach().cpu()
        auc, acc_no_zero, recall_score_, f1_score_ = None, None, None, None
        if config.task == "pos" or config.task == "ner":
            filtered_y_true = []
            filtered_y_pred = []
            for true_label, pred_label in zip(y_true, y_pred):
                if true_label != 0:
                    filtered_y_true.append(true_label)
                    filtered_y_pred.append(pred_label)
            acc_no_zero = accuracy_score(filtered_y_true, filtered_y_pred)
            f1_score_ = f1_score(filtered_y_true, filtered_y_pred, average='micro')
            pr_ = precision_score(filtered_y_true, filtered_y_pred, average='micro')
            recall_score_ = recall_score(filtered_y_true, filtered_y_pred, average='micro')
            report = classification_report(filtered_y_true, filtered_y_pred, digits=5,labels=train_loader.dataset.label2id.keys())
            auc = None
            print('Accuracy (without label 0):', acc_no_zero)
        else:
            auc = roc_auc_score(y_true, torch.cat(probas, 0), multi_class='ovr')
            report = classification_report(y_true, y_pred, digits=5)
            acc_no_zero = accuracy_score(y_true, y_pred)
            print(f"AUC:{auc}, ACC:{acc_no_zero}")
        log_writer.add_text(f"{run_name}-result", report)
        log_writer.close()
        open(f"results/{run_name}.txt", 'w').write(str(report))
        open(f"results/{run_name}.txt", 'a').write(f"\nAUC\tmi-ACC\tmi-RECALL\tmi-PR\tmi-F1\n")
        open(f"results/{run_name}.txt", 'a').write(f"{auc}\t{acc_no_zero}\t{recall_score_}\t{pr_}\t{f1_score_}")
        print(report)
        return


if __name__ == "__main__":
    start_time = start = datetime.datetime.now()
    config = config()
    run_name = config.dataset_name + f'-{config.max_length}-' + config.encoder_name + '-' + str(
        config.args.fusion_mode) + '-' + str(config.args.core_model) + "-plm_on" if config.args.use_plm else \
        config.dataset_name + '-' + config.encoder_name + '-' + str(config.args.fusion_mode) + '-' + str(
            config.args.core_model) + "-plm_off"
    log_writer = SummaryWriter(f'logs/{run_name}')

    print(f"[INFO]: This run: {run_name}")
    torch.manual_seed(config.seed)
    if config.task == 'pos':
        train_loader = Pos_data_loader(config, mode='train')
        valid_loader = Pos_data_loader(config, mode='val', shuffle=False)
        test_loader = Pos_data_loader(config, mode='test', shuffle=False)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        model = PosModel(config, train_loader.dataset.get_class_num())
    elif config.task == 'ner':
        train_loader = NER_data_loader(config, mode='train')
        valid_loader = NER_data_loader(config, mode='val', shuffle=False)
        test_loader = NER_data_loader(config, mode='test', shuffle=False)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        model = PosModel(config, train_loader.dataset.get_class_num())
    elif config.task == 'classification':
        train_loader = Cls_data_loader(config, mode='train')
        valid_loader = Cls_data_loader(config, mode='val', shuffle=False)
        test_loader = Cls_data_loader(config, mode='test', shuffle=False)
        criterion = nn.CrossEntropyLoss()
        model = ClsModel(config, train_loader.dataset.get_class_num())
    elif config.task == 'QA':
        train_loader = QA_data_loader(config, mode='train')
        valid_loader = QA_data_loader(config, mode='val', shuffle=False)
        test_loader = QA_data_loader(config, mode='test', shuffle=False)
        criterion = nn.CrossEntropyLoss()
        model = QAModel(config, train_loader.dataset.get_class_num())
    elif config.task == 'Infer':
        train_loader = Infer_data_loader(config, mode='train')
        valid_loader = Infer_data_loader(config, mode='val', shuffle=False)
        test_loader = Infer_data_loader(config, mode='test', shuffle=False)
        criterion = nn.CrossEntropyLoss()
        model = InferModel(config, train_loader.dataset.get_class_num())

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    bestloss = 1e9
    bestacc = -1e9
    bestf1 = -1e9
    bestauc = -1e9
    notbetter_count = 0
    bestmodel = model
    # for epoch in range(config.epochs):
    #     if notbetter_count > config.earlystop:
    #         print("[INFO]: Not better, Early stop.")
    #         break
    #     # print(f"---------Epoch:{epoch}---------")
    #     train(model, train_loader, optimizer, criterion, device)
    #     if epoch%config.ValidRound==0:
    #         notbetter_count += 1
    #         thisloss,thisacc,thisf1,thisauc = valid(model, valid_loader, criterion, device)
    #         if thisloss < bestloss:
    #             bestloss = thisloss
    #             notbetter_count=0
    #             torch.save(model,f"best_models/{run_name}.ckpt")
    #             bestmodel = copy.deepcopy(model)
    #             print("BEST MODEL!!")
    #         print(f"Model Acc:{thisacc}, AvgLoss:{thisloss}, Macro F1:{thisf1}, Auc:{thisauc}")
    model = torch.load(f"best_models/{run_name}.ckpt")
    end_time = start = datetime.datetime.now()
    # model = bestmodel
    test(model, test_loader, device)
    # torch.save(model, f"best_models/{run_name}.ckpt")
    print(f"[INFO]: This run: {run_name}")
    # open(f"results/{run_name}.txt",'a').write(f'\nTotal train time: {end_time - start_time}')

    print(f'Total train time: {end_time - start_time}')
    print(f"-------------------------------------------------------------------------")

