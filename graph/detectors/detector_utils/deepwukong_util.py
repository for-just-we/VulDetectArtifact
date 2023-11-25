import torch
from torch_geometric.data import Data
import numpy as np
from gensim.models import Word2Vec
import json
from typing import Dict
from graph.detectors.models.deepwukong import model_args as dwk_model_args

def vectorize_xfg(w2v_model: Word2Vec, data: Dict, device: str) -> Data:
    token_seqs = data["line-contents"]
    # token_seqs = [json.loads(node_info)["contents"][0][1] for node_info in data["line-nodes"]]
    n_vs = [np.array([w2v_model[word] if word in w2v_model.wv.vocab else
                      np.zeros(dwk_model_args.vector_size) for word in token_seq.split(" ")]).mean(axis=0) for token_seq
            in token_seqs]
    t_vs = [torch.FloatTensor(n_v).to(device) for n_v in n_vs]
    vector = torch.stack(t_vs)
    edges = [json.loads(edge) for edge in data["data-dependences"] + data["control-dependences"]]
    edge_index = torch.LongTensor(edges).to(device).t()

    return Data(x=vector, edge_index=edge_index, y=data["target"])
