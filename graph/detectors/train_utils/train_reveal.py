from graph.detectors.models.reveal import ClassifyModel, model_args
from graph.detectors.detector_utils.reveal_util import RevealUtil
from graph.detectors.train_utils.dataloader import get_dataloader

from typing import List, Dict
from time import time
from tqdm import tqdm
import sys
import os
import json
import shutil
from sklearn import metrics
import numpy as np
from gensim.models import Word2Vec

import torch
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Batch, Data


class TrainUtil(object):
    def __init__(self, w2v_model: Word2Vec, gnnNets: ClassifyModel, device: str,
                 dataset_dir: str, train_args):
        self.w2v_model: Word2Vec = w2v_model
        self.gnnNets: ClassifyModel = gnnNets
        self.reveal_util: RevealUtil = RevealUtil(w2v_model, gnnNets, device)
        self.device: str = device
        self.train_args = train_args


        self.train_positive: List[Dict] = json.load(
            open(os.path.join(dataset_dir, "function/train_vuls.json"),
                                                    'r', encoding='utf-8'))
        self.train_negative: List[Dict] = json.load(
            open(os.path.join(dataset_dir, "function/train_nors.json"),
                 'r', encoding='utf-8'))
        self.val_positive: List[Dict] = json.load(
            open(os.path.join(dataset_dir, "function/val_vuls.json"),
                 'r', encoding='utf-8'))
        self.val_negative: List[Dict] = json.load(
            open(os.path.join(dataset_dir, "function/val_nors.json"),
                 'r', encoding='utf-8'))
        self.test_positive: List[Dict] = json.load(
            open(os.path.join(dataset_dir, "function/test_vuls.json"),
                 'r', encoding='utf-8'))
        self.test_negative: List[Dict] = json.load(
            open(os.path.join(dataset_dir, "function/test_nors.json"),
                 'r', encoding='utf-8'))

    def save_best(self, epoch, eval_f1, is_best):
        print('saving....')
        self.gnnNets.to('cpu')
        state = {
            'net': self.gnnNets.state_dict(),
            'epoch': epoch,
            'f1': eval_f1
        }
        pth_name = f"{model_args.model_dir}/{model_args.model_name}_{model_args.detector}_latest.pth"
        best_pth_name = f'{model_args.model_dir}/{model_args.model_name}_{model_args.detector}_best.pth'
        ckpt_path = os.path.join(model_args.model_dir, pth_name)
        torch.save(state, ckpt_path)
        if is_best:
            shutil.copy(ckpt_path, os.path.join(model_args.model_dir, best_pth_name))
        self.gnnNets.to(model_args.device)

    def evaluate(self, eval_dataloader):
        '''
        :param eval_dataloader: 数据集
        :param gnnNets:分类模型
        :return:
        '''
        acc = []
        recall = []
        precision = []
        f1 = []
        self.gnnNets.eval()
        with torch.no_grad():
            for batch_graph_info in eval_dataloader:
                batch_data = [self.reveal_util.generate_initial_graph_embedding(graph_info) for graph_info in batch_graph_info]
                batch = Batch.from_data_list(batch_data)
                batch.to(self.device)
                logits = self.gnnNets(data = batch)

                ## record
                _, prediction = torch.max(logits, -1)
                acc.append(metrics.accuracy_score(batch.y.cpu().numpy(), prediction.cpu().numpy()))
                recall.append(metrics.recall_score(batch.y.cpu().numpy(), prediction.cpu().numpy(), zero_division=0))
                precision.append(metrics.precision_score(batch.y.cpu().numpy(), prediction.cpu().numpy(),zero_division=0))
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
        checkpoint = torch.load(os.path.join(model_args.model_dir, f'{model_args.model_name}_{model_args.detector}_best.pth'))
        self.gnnNets.load_state_dict(checkpoint['net'])

        # embedding data
        print("start embedding data=================")
        start_time = time()
        test_vul_graph_infos = [self.reveal_util.generate_initial_embedding(sample) for sample in
                               tqdm(self.test_positive, desc="generating feature for vul sample", file=sys.stdout)]
        test_nor_graph_infos = [self.reveal_util.generate_initial_embedding(sample) for sample in
                               tqdm(self.test_negative, desc="generating feature for normal sample", file=sys.stdout)]
        end_time = time()
        spent_time = end_time - start_time
        print(f"embedding spent time: {spent_time // 3600}h-{(spent_time % 3600) // 60}m-{(spent_time % 3600) % 60}s")

        labels = []
        predictions = []
        test_dataloader = get_dataloader(test_vul_graph_infos, test_nor_graph_infos, self.train_args.batch_size)
        self.gnnNets.eval()
        print('start testing model==================')
        test_start_time = time()
        with torch.no_grad():
            for batch_graph_info in test_dataloader:
                batch_data = [self.reveal_util.generate_initial_graph_embedding(graph_info) for graph_info in
                              batch_graph_info]
                batch = Batch.from_data_list(batch_data)
                batch.to(model_args.device)
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
              )

    def train(self):
        print("start embedding data=================")
        start_time = time()
        train_vul_graph_infos = [self.reveal_util.generate_initial_embedding(sample) for sample in
                                 tqdm(self.train_positive, desc="generating feature for train vul sample",
                                      file=sys.stdout)]
        train_nor_graph_infos = [self.reveal_util.generate_initial_embedding(sample) for sample in
                                 tqdm(self.train_negative, desc="generating feature for train normal sample",
                                      file=sys.stdout)]
        val_vul_graph_infos = [self.reveal_util.generate_initial_embedding(sample) for sample in
                                 tqdm(self.val_positive, desc="generating feature for val vul sample",
                                      file=sys.stdout)]
        val_nor_graph_infos = [self.reveal_util.generate_initial_embedding(sample) for sample in
                                 tqdm(self.val_negative, desc="generating feature for val normal sample",
                                      file=sys.stdout)]
        end_time = time()
        spent_time = end_time - start_time
        print(f"embedding spent time: {spent_time // 3600}h-{(spent_time % 3600) // 60}m-{(spent_time % 3600) % 60}s")

        path = os.path.join(model_args.model_dir, f'{model_args.model_name}_{model_args.detector}_best.pth')
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.gnnNets.load_state_dict(checkpoint['net'])

        print('start training model==================')
        rate = len(self.train_negative) / len(self.train_positive)
        weight_ce = torch.FloatTensor([1, rate]).to(model_args.device)
        criterion = nn.CrossEntropyLoss(weight=weight_ce)
        optimizer = Adam(self.gnnNets.parameters(), lr=self.train_args.learning_rate, weight_decay=self.train_args.weight_decay,
                         betas=(0.9, 0.999))
        # save path for model
        best_f1 = 0
        early_stop_count = 0

        train_start_time = time()

        for epoch in range(self.train_args.max_epochs):
            print(f"epoch: {epoch} start===================")
            epoch_start_time = time()
            acc = []
            recall = []
            precision = []
            f1 = []
            loss_list = []
            self.gnnNets.train()
            train_dataloader = get_dataloader(train_vul_graph_infos, train_nor_graph_infos, self.train_args.batch_size)
            for i, batch_graph_info in enumerate(train_dataloader):
                batch_data = [self.reveal_util.generate_initial_graph_embedding(graph_info) for graph_info in
                              batch_graph_info]

                batch = Batch.from_data_list(batch_data)
                batch.to(model_args.device)
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
            batch_dataloader = get_dataloader(val_vul_graph_infos, val_nor_graph_infos,
                                                   batch_size=self.train_args.batch_size)
            eval_state = self.evaluate(batch_dataloader)
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




if __name__ == '__main__':
    pretrain_model = Word2Vec.load(model_args.pretrain_word2vec_model)
    reveal_model: ClassifyModel = ClassifyModel()
    reveal_model.to(model_args.device)
    train_util = TrainUtil(pretrain_model, reveal_model)