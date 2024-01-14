import nltk
import numpy as np
import time

from Explainer import *
import shap
import lime
from GradientExplainer.GradientExplainer_masking import *
from Explainer import Explainer

from gensim.models.word2vec import Word2Vec
from sequence.detectors.sequence_util import SequenceUtil
from sequence.detectors.models.tokenlstm import model_args as tl_model_args
from sequence.detectors.models.sysyer import model_args as syse_model_args
from sequence.detectors.models.vuldeepecker import model_args as vdp_model_args

import random
import json
import os
from typing import List, Dict
from tqdm import tqdm
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


class SequenceExplainUtil:
    def __init__(self, w2v_model: Word2Vec, sequence_model, train_args):
        self.w2v_model: Word2Vec = w2v_model
        self.sequence_model = sequence_model
        self.device = train_args.device
        self.explainer = Explainer(self.sequence_model, train_args.explainer)

        self.test_positive: List[Dict] = load_data(train_args.dataset_dir, "test_vul.json", 1)
        self.util: SequenceUtil = SequenceUtil(w2v_model, self.device, masking_len[train_args.detector])

    def explain(self):
        for data in tqdm(self.test_positive):
            idxs = self.util.tokenize_data(data)
            input_vectors = np.array(self.util.embeddings_matrix[idxs])
            pred = self.sequence_model.predict(input_vectors)
            result = int(pred > 0.5)
            if result == 0:
                continue
            self.explainer.explain(input_vectors, 1)