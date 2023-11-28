
import abc
import os
import json
from time import time
from typing import List, Dict, Set
import torch
import torch.nn as nn
from tqdm import tqdm
from gensim.models import Word2Vec
from torch_geometric.data import Batch, Data
import sys

def load_data(dataset_dir: str, file_name: str , label: int, is_empty_data = None):
    datas = json.load(open(os.path.join(dataset_dir, file_name), 'r', encoding='utf-8'))
    if is_empty_data:
        datas = list(filter(is_empty_data, datas))
    for data in datas:
        data["target"] = label
    return datas

class BaseExplainerUtil:
    def __init__(self, w2v_model: Word2Vec, gnnNets: nn.Module, args, explainer: nn.Module):
        self.w2v_model: Word2Vec = w2v_model
        self.gnnNets = gnnNets
        self.device = args.device
        self.gnnNets.to(self.device)

        is_not_empty_data = lambda data: len(data["nodes"]) > 0 and (len(data["cdgEdges"]) + len(data["ddgEdges"]) > 0)
        self.test_positive: List[Dict] = load_data(args.dataset_dir, "test_vul.json", 1, is_not_empty_data)
        self.args = args
        self.explainer = explainer

    @abc.abstractmethod
    def generate_graph_data(self, sample: Dict) -> Data:
        pass

    def get_explanation_result(self, graph_data: Data) -> Set[int]:
        pass

    def test(self):
        # load model
        checkpoint = torch.load(os.path.join(self.args.model_dir, f'{self.args.detector}_best.pth'))
        self.gnnNets.load_state_dict(checkpoint['net'])

        # embedding data
        print("start embedding data=================")
        start_time = time()

        with torch.no_grad():
            for sample in tqdm(self.test_positive, desc="evaluating explainers", file=sys.stdout):
                graph_data: Data = self.generate_graph_data(sample)
                prob, node_emb = self.gnnNets(Batch.from_data_list([graph_data]).to(self.device))
                _, prediction = torch.max(prob, -1)
                # 必须是true positive
                if prediction.cpu() == 0:
                    continue
                res = self.explainer(graph_data.x, graph_data.edge_index, node_emb)