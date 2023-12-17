from gensim.models import Word2Vec
import numpy as np
import nltk
import json
from typing import Dict, List, Tuple

def build_initial_embedding(pretrained_model):
    embeddings_matrix = np.zeros((len(pretrained_model.wv.vocab.items()) + 2, pretrained_model.vector_size))
    word2idx = {'PAD': len(embeddings_matrix) - 1, 'Other': len(embeddings_matrix) - 2}
    embeddings_matrix[-1] = np.ones(pretrained_model.vector_size, )
    print(embeddings_matrix.shape)

    vocab_list = [(k, pretrained_model.wv[k]) for k, v in pretrained_model.wv.vocab.items()]
    # word2idx 字典
    for i in range(len(vocab_list)):
        word = vocab_list[i][0]
        word2idx[word] = i
        embeddings_matrix[i] = vocab_list[i][1]

    return word2idx, embeddings_matrix

# 将一个data转化为向量
class SequenceUtil(object):
    def __init__(self, pretrain_model: Word2Vec, device: str, masking_len: int):
        self.pretrain_model = pretrain_model
        self.device = device
        self.mask_len = masking_len
        items = build_initial_embedding(pretrain_model)
        self.word2idx: dict = items[0]
        self.embeddings_matrix = items[1]

    def tokenize_data(self, data: dict):
        token_seq = []
        for line_content in data["node-line-content"]:
            token_seq.extend(nltk.word_tokenize(line_content))

        idx_seq = list(map(lambda token: self.word2idx.get(token, self.word2idx["Other"]), token_seq))
        return idx_seq