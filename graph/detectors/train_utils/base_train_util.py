import abc
import os
import json
from time import time
from typing import List, Dict
import torch
import torch.nn as nn
from sklearn import metrics
from graph.detectors.train_utils.dataloader import get_dataloader
from tqdm import tqdm
import shutil
import numpy as np
from gensim.models import Word2Vec
from torch_geometric.data import Batch, Data
from torch.optim import Adam
import sys

class BaseTrainUtil:
    def __init__(self, w2v_model: Word2Vec, gnnNets, train_args):
        self.w2v_model: Word2Vec = w2v_model
        self.gnnNets = gnnNets
        self.device = train_args.device
        self.gnnNets.to(self.device)

        self.train_positive: List[Dict] = json.load(open(os.path.join(train_args.dataset_dir, "function/train_vul.json"),
                                                           'r', encoding='utf-8'))
        self.train_negative: List[Dict] = json.load(open(os.path.join(train_args.dataset_dir, "function/train_nor.json"),
                                                           'r', encoding='utf-8'))
        self.val_positive: List[Dict] = json.load(open(os.path.join(train_args.dataset_dir, "function/val_vul.json"),
                                                         'r', encoding='utf-8'))
        self.val_negative: List[Dict] = json.load(open(os.path.join(train_args.dataset_dir, "function/val_nor.json"),
                                                         'r', encoding='utf-8'))
        self.test_positive: List[Dict] = json.load(open(os.path.join(train_args.dataset_dir, "function/test_vul.json"),
                                                          'r', encoding='utf-8'))
        self.test_negative: List[Dict] = json.load(open(os.path.join(train_args.dataset_dir, "function/test_nor.json"),
                                                          'r', encoding='utf-8'))

        self.train_args = train_args

    def save_best(self, epoch, eval_f1, is_best):
        print('saving....')
        self.gnnNets.to('cpu')
        state = {
            'net': self.gnnNets.state_dict(),
            'epoch': epoch,
            'f1': eval_f1
        }
        pth_name = f"{self.train_args.model_dir}/{self.train_args.detector}_latest.pth"
        best_pth_name = f'{self.train_args.model_dir}/{self.train_args.detector}_best.pth'
        ckpt_path = os.path.join(self.train_args.model_dir, pth_name)
        torch.save(state, ckpt_path)
        if is_best:
            shutil.copy(ckpt_path, os.path.join(self.train_args.model_dir, best_pth_name))
        self.gnnNets.to(self.device)

    @abc.abstractmethod
    def generate_features(self, sample):
        pass

    @abc.abstractmethod
    def generate_graph_from_feature(self, feature, device) -> Data:
        pass

    def evaluate(self, eval_dataloader, num_batch):
        acc = []
        recall = []
        precision = []
        f1 = []
        self.gnnNets.eval()
        with torch.no_grad():
            for batch_features in tqdm(eval_dataloader, total=num_batch, desc="evaluating process", file=sys.stdout):
                batch_data: List[Data] = [self.generate_graph_from_feature(feature, self.device) for feature in batch_features]
                batch: Batch = Batch.from_data_list(batch_data)
                batch.to(self.device)
                logits = self.gnnNets(data=batch)

                ## record
                _, prediction = torch.max(logits, -1)
                acc.append(metrics.accuracy_score(batch.y.cpu().numpy(), prediction.cpu().numpy()))
                recall.append(metrics.recall_score(batch.y.cpu().numpy(), prediction.cpu().numpy(), zero_division=0))
                precision.append(metrics.precision_score(batch.y.cpu().numpy(), prediction.cpu().numpy(), zero_division=0))
                f1.append(metrics.f1_score(batch.y.cpu().numpy(), prediction.cpu().numpy(), zero_division=0))

            eval_state = {
                'acc': np.average(acc),
                'recall': np.average(recall),
                'precision': np.average(precision),
                'f1':np.average(f1)
            }

        return eval_state

    def test(self):
        # load model
        checkpoint = torch.load(os.path.join(self.train_args.model_dir, f'{self.train_args.detector}_best.pth'))
        self.gnnNets.load_state_dict(checkpoint['net'])

        # embedding data
        print("start embedding data=================")
        start_time = time()
        test_vul_features = [self.generate_features(sample) for sample in
                            tqdm(self.test_positive, desc="generating feature for vul sample", file=sys.stdout)]
        test_nor_features = [self.generate_features(sample) for sample in
                            tqdm(self.test_negative, desc="generating feature for normal sample", file=sys.stdout)]
        end_time = time()
        spent_time = end_time - start_time
        print(f"embedding spent time: {spent_time // 3600}h-{(spent_time % 3600) // 60}m-{(spent_time % 3600) % 60}s")

        labels = []
        predictions = []

        test_dataloader = get_dataloader(test_vul_features, test_nor_features, self.train_args.batch_size)
        self.gnnNets.eval()
        print('start testing model==================')
        test_start_time = time()
        num_batch = len(test_vul_features + test_nor_features) // self.train_args.batch_size
        with torch.no_grad():
            for batch_features in tqdm(test_dataloader, total=num_batch, desc="testing process",
                                       file=sys.stdout):
                batch_data: List[Data] = [self.generate_graph_from_feature(feature, self.device) for feature in
                              batch_features]
                batch: Batch = Batch.from_data_list(batch_data)
                batch.to(self.device)
                probs = self.gnnNets(data=batch)

                # record
                _, prediction = torch.max(probs, -1)
                labels.extend(batch.y.cpu().tolist())
                predictions.extend(prediction.cpu().tolist())

        cm = metrics.confusion_matrix(labels, predictions)
        TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        FNR = FN / (TP + FN)
        FPR = FP / (FP + TN)

        test_end_time = time()
        test_spent_time = test_end_time - test_start_time
        print(
            f"training spent time: {test_spent_time // 3600}h-{(test_spent_time % 3600) // 60}m-{(test_spent_time % 3600) % 60}s")

        print(f"Acc: {metrics.accuracy_score(labels, predictions):.3f} |"
              f"Recall: {metrics.recall_score(labels, predictions, zero_division=0):.3f} |"
              f"Precision: {metrics.precision_score(labels, predictions, zero_division=0):.3f} |"
              f"F1: {metrics.f1_score(labels, predictions, zero_division=0):.3f} |"
              f"FNR: {FNR:.3f} |"
              f"FPR: {FPR:.3f} |"
              f"confusion matrix: {cm}")

        test_state = {
            'acc': metrics.accuracy_score(labels, predictions),
            'recall': metrics.recall_score(labels, predictions, zero_division=0),
            'precision': metrics.precision_score(labels, predictions, zero_division=0),
            'f1': metrics.f1_score(labels, predictions, zero_division=0),
            'cm': cm
        }

        return test_state


    def train(self):
        print("start embedding data=================")
        start_time = time()
        train_vul_features = [self.generate_features(sample) for sample in
                                 tqdm(self.train_positive, desc="generating feature for train vul sample",
                                      file=sys.stdout)]
        train_nor_features = [self.generate_features(sample) for sample in
                                 tqdm(self.train_negative, desc="generating feature for train normal sample",
                                      file=sys.stdout)]
        val_vul_features = [self.generate_features(sample) for sample in
                                 tqdm(self.val_positive, desc="generating feature for val vul sample",
                                      file=sys.stdout)]
        val_nor_features = [self.generate_features(sample) for sample in
                                 tqdm(self.val_negative, desc="generating feature for val normal sample",
                                      file=sys.stdout)]
        end_time = time()
        spent_time = end_time - start_time
        print(f"embedding spent time: {spent_time // 3600}h-{(spent_time % 3600) // 60}m-{(spent_time % 3600) % 60}s")

        path = os.path.join(self.train_args.model_dir, f'{self.train_args.detector}_best.pth')
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.gnnNets.load_state_dict(checkpoint['net'])

        print('start training model==================')
        rate = len(self.train_negative) / len(self.train_positive)
        weight_ce = torch.FloatTensor([1, rate]).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=weight_ce)
        optimizer = Adam(self.gnnNets.parameters(), lr=self.train_args.learning_rate, weight_decay=self.train_args.weight_decay,
                         betas=(0.9, 0.999))
        # save path for model
        best_f1 = 0
        early_stop_count = 0
        train_start_time = time()
        num_batch = len(train_vul_features + train_nor_features) // self.train_args.batch_size
        num_batch_val = len(val_vul_features + val_nor_features) // self.train_args.batch_size
        for epoch in range(self.train_args.max_epochs):
            print(f"epoch: {epoch} start===================")
            epoch_start_time = time()
            acc = []
            recall = []
            precision = []
            f1 = []
            loss_list = []
            self.gnnNets.train()
            train_dataloader = get_dataloader(train_vul_features, train_nor_features, self.train_args.batch_size)
            for batch_features in tqdm(train_dataloader, total=num_batch, desc="training process",
                                       file=sys.stdout):
                batch_data: List[Data] = [self.generate_graph_from_feature(feature, self.device) for feature in
                              batch_features]

                batch: Batch = Batch.from_data_list(batch_data)
                batch.to(self.device)
                logits = self.gnnNets(data=batch)
                loss = criterion(logits, batch.y)

                # optimization
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.gnnNets.parameters(), clip_value=2.0)
                optimizer.step()

                ## record
                _, prediction = torch.max(logits, -1)
                loss_list.append(loss.item())
                acc.append(metrics.accuracy_score(batch.y.cpu().numpy(), prediction.cpu().numpy()))
                recall.append(metrics.recall_score(batch.y.cpu().numpy(), prediction.cpu().numpy(), zero_division=0))
                precision.append(
                    metrics.precision_score(batch.y.cpu().numpy(), prediction.cpu().numpy(), zero_division=0))
                f1.append(
                    metrics.f1_score(batch.y.cpu().numpy(), prediction.cpu().numpy(), zero_division=0)
                )
            # report train msg
            print(f"Train Epoch:{epoch}  |Loss: {np.average(loss_list):.3f} | "
                  f"Acc: {np.average(acc):.3f} |"
                  f"Recall: {np.average(recall):.3f} |"
                  f"Precision: {np.average(precision):.3f} |"
                  f"F1: {np.average(f1):.3f}")

            # report eval msg
            batch_dataloader = get_dataloader(val_vul_features, val_nor_features,
                                                   batch_size=self.train_args.batch_size)
            eval_state = self.evaluate(batch_dataloader, num_batch_val)
            print(
                f"Eval Epoch: {epoch} | Acc: {eval_state['acc']:.3f} | Recall: {eval_state['recall']:.3f} "
                f"| Precision: {eval_state['precision']:.3f} | F1: {eval_state['f1']:.3f}")

            # only save the best model
            is_best = (eval_state['f1'] > best_f1)

            if eval_state['f1'] > best_f1:
                early_stop_count = 0
            else:
                early_stop_count += 1

            if early_stop_count > self.train_args.early_stopping:
                break

            if is_best:
                best_f1 = eval_state['f1']
                early_stop_count = 0
            if is_best or epoch % self.train_args.save_epoch == 0:
                self.save_best(epoch, eval_state['f1'], is_best)

            epoch_end_time = time()
            epoch_spent_time = epoch_end_time - epoch_start_time
            print(
                f"epoch spent time: {epoch_spent_time // 3600}h-{(epoch_spent_time % 3600) // 60}m-{(epoch_spent_time % 3600) % 60}s")

        train_end_time = time()
        train_spent_time = train_end_time - train_start_time
        print(f"training spent time: {train_spent_time // 3600}h-{(train_spent_time % 3600) // 60}m-{(train_spent_time % 3600) % 60}s")
        print(f"The best validation f1 is {best_f1}.")