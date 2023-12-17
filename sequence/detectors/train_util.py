from gensim.models.word2vec import Word2Vec
from sequence.detectors.sequence_util import SequenceUtil
from sequence.detectors.models.tokenlstm import model_args as tl_model_args
from sequence.detectors.models.sysyer import model_args as syse_model_args
from sequence.detectors.models.vuldeepecker import model_args as vdp_model_args

import numpy as np

import random
import json
import os
from typing import List, Dict
from collections import Counter

import keras

masking_len = {
    "tokenlstm": tl_model_args.maxLen,
    "vuldeepecker": vdp_model_args.maxLen,
    "sysevr": syse_model_args.maxLen
}

def load_data(dataset_dir: str, file_name: str , label: int):
    datas = json.load(open(os.path.join(dataset_dir, file_name), 'r', encoding='utf-8'))
    for data in datas:
        data["target"] = label
    return datas


class SequenceTrainUtil:
    def __init__(self, w2v_model: Word2Vec, sequence_model, train_args, model_path):
        self.w2v_model: Word2Vec = w2v_model
        self.sequence_model = sequence_model
        self.device = train_args.device
        self.model_path = model_path

        self.train_positive: List[Dict] = load_data(train_args.dataset_dir, "train_vul.json", 1)
        self.train_negative: List[Dict] = load_data(train_args.dataset_dir, "train_normal.json", 0)
        self.val_positive: List[Dict] = load_data(train_args.dataset_dir, "eval_vul.json", 1)
        self.val_negative: List[Dict] = load_data(train_args.dataset_dir, "eval_normal.json", 0)
        self.test_positive: List[Dict] = load_data(train_args.dataset_dir, "test_vul.json", 1)
        self.test_negative: List[Dict] = load_data(train_args.dataset_dir, "test_normal.json", 0)

        self.train_args = train_args
        self.util: SequenceUtil = SequenceUtil(w2v_model, self.device, masking_len[train_args.detector])

    def generator_of_data(self, datas):
        iter_num = int(len(datas) / self.train_args.batchSize)
        i = 0

        while iter_num:
            batchdata = datas[i:i + self.train_args.batchSize]
            batch_idxs = [self.util.tokenize_data(data) for data in batchdata]
            batch_vectors = np.array([self.util.embeddings_matrix[idx] for idx in batch_idxs])
            batched_labels = []
            for data in batchdata:
                if data["label"][0] == 0:
                    batched_labels.append(0)
                else:
                    batched_labels.append(1)

            yield ([batch_vectors], batched_labels)
            i = i + self.train_args.batchSize

            iter_num -= 1
            if iter_num == 0:
                iter_num = int(len(datas) / self.train_args.batchSize)
                i = 0

    def train(self):
        weight = len(self.train_negative) / len(self.train_positive)
        weight_dict = {0: 1, 1: weight}
        callback = keras.callbacks.EarlyStopping(monitor='loss', patience=2)  # 使用loss作为监测数据，轮数设置为1
        all_datas = self.train_positive + self.train_negative

        random.shuffle(all_datas)
        train_generator = self.generator_of_data(all_datas)
        self.sequence_model.fit_generator(train_generator, steps_per_epoch=int(len(all_datas) / self.train_args.batchSize), epochs=20,
                            callbacks=[callback], class_weight=weight_dict)
        self.sequence_model.save(self.model_path)


    def test(self):
        all_datas = self.test_positive + self.test_negative
        test_dataloader = self.generator_of_data(all_datas)
        batch_num = len(all_datas) // self.train_args.batchSize if len(all_datas) % self.train_args.batchSize == 0 \
            else len(all_datas) // self.train_args.batchSize + 1
        TN = TP = FP = FN = 0

        for i, data in enumerate(test_dataloader):
            if i >= batch_num:
                break
            print(f"{i}/{batch_num}")
            vectors = data[0]
            batch_labels = np.array(data[1])

            pred = self.sequence_model.predict(vectors).reshape(batch_labels.shape[0], )
            result = (pred > 0.5).astype(np.int_)

            diff1 = result + batch_labels  # 2表示TP, 0表示TN
            diff2 = result - batch_labels  # 1表示FP，-1表示FN
            counts_1 = Counter(diff1)
            counts_2 = Counter(diff2)

            TP += counts_1[2]
            TN += counts_1[0]
            FP += counts_2[1]
            FN += counts_2[-1]

        FPR = FP / (FP + TN)
        FNR = FN / (TP + FN)
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2 * precision * recall / (precision + recall)

        print(f"FPR: {FPR:.3f} | FNR: {FNR:.3f} | \n accuracy: {accuracy:.3f} | "
              f"precision: {precision:.3f} | recall: {recall:.3f} | F1: {F1:.3f}")