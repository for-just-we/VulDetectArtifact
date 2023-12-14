from gensim.models import Word2Vec
import numpy as np
import json
from typing import Dict, List, Tuple

from sequence.detectors.models.tokenlstm import build_model as token_lstm_build

class TokenLSTMUtil(object):
    def __init__(self, pretrain_model: Word2Vec, device: str):
        self.pretrain_model = pretrain_model
        self.device = device