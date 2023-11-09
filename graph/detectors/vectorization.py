from typing import Dict
import json

import numpy as np
import torch
from torch_geometric.data import Data


class VectorizationUtil:
    pass

class DeepWuKongVectorization(VectorizationUtil):
    def __init__(self, w2v_model, device, vector_size):
        self.w2v_model = w2v_model
        self.device = device
        self.vector_size = vector_size

    def generate_initial_training_datas(self, data: Dict) -> Data:
        token_seqs = [json.loads(node_info)["contents"][0][1] for node_info in data["line-nodes"]]
        n_vs = [np.array([self.w2v_model[word] if word in self.w2v_model.wv.vocab else
                          np.zeros(self.vector_size) for word in token_seq.split(" ")]).mean(axis=0) for token_seq
                in token_seqs]
        t_vs = [torch.FloatTensor(n_v).to(self.device) for n_v in n_vs]
        vector = torch.stack(t_vs)
        edges = [json.loads(edge) for edge in data["data-dependences"] + data["control-dependences"]]
        edge_index = torch.LongTensor(edges).to(self.device).t()

        return Data(x=vector, edge_index=edge_index, y=data["target"])