import nltk
import numpy as np
import time

from Explainer import *
import shap
import lime
from GradientExplainer.GradientExplainer_masking import *

from gensim.models.word2vec import Word2Vec
from sequence.detectors.sequence_util import SequenceUtil
from sequence.detectors.models.tokenlstm import model_args as tl_model_args
from sequence.detectors.models.sysyer import model_args as syse_model_args
from sequence.detectors.models.vuldeepecker import model_args as vdp_model_args

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


class SequenceExplaintil:
    def __init__(self, w2v_model: Word2Vec, sequence_model, train_args, model_path):
        self.w2v_model: Word2Vec = w2v_model
        self.sequence_model = sequence_model
        self.device = train_args.device
        self.model_path = model_path

        self.test_positive: List[Dict] = load_data(train_args.dataset_dir, "test_vul.json", 1)
        self.util: SequenceUtil = SequenceUtil(w2v_model, self.device, masking_len[train_args.detector])

