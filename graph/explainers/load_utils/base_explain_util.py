
import abc
import os
import json

from typing import List, Dict
import torch
import torch.nn as nn
from tqdm import tqdm
from gensim.models import Word2Vec
from torch_geometric.data import Batch, Data
import sys

from graph.explainers.approaches.pgexplainer import PGExplainer
from graph.explainers.approaches.gnnexplainer import GNNExplainer
from graph.explainers.approaches.gradcam import GradCAM
from graph.explainers.approaches.deeplift import DeepLIFT
from graph.explainers.approaches.gnnlrp import GNN_LRP

explainer_classes = {
    "pgexplainer": PGExplainer,
    "gnnexplainer": GNNExplainer,
    "gradcam": GradCAM,
    "deeplift": DeepLIFT,
    "gnnlrp": GNN_LRP
}

def load_data(dataset_dir: str, file_name: str , label: int, is_empty_data = None):
    datas = json.load(open(os.path.join(dataset_dir, file_name), 'r', encoding='utf-8'))
    if is_empty_data:
        datas = list(filter(is_empty_data, datas))
    for data in datas:
        data["target"] = label
    return datas

class BaseExplainerUtil:
    def __init__(self, w2v_model: Word2Vec, gnnNets: nn.Module, args, explainer, k: int):
        self.w2v_model: Word2Vec = w2v_model
        self.gnnNets = gnnNets
        self.device = args.device
        self.gnnNets.to(self.device)

        is_not_empty_data = lambda data: len(data["nodes"]) > 0 and (len(data["cdgEdges"]) + len(data["ddgEdges"]) > 0)
        self.test_positive: List[Dict] = load_data(args.dataset_dir, "test_vul.json", 1, is_not_empty_data)
        self.args = args
        self.explainer = self.construct_explainer(explainer)
        self.k = k
        self.sparsity = 0.5

    def construct_explainer(self, explainer_name: str):
        cls = explainer_classes[explainer_name]
        explainer = cls(self.gnnNets)
        return explainer


    @abc.abstractmethod
    def generate_graph_data(self, sample: Dict) -> Data:
        pass

    def sample_nodes(self, sorted_indices, flag, node_num, edge_index) -> List[int]:
        # flag为true表示node-level explanation，false表示edge-level
        if flag:
            selected_node_num = min(self.k, int(node_num * self.sparsity))
            return sorted_indices.tolist()[:selected_node_num]
        else:
            # top_k: max_node_num
            max_node_num = min(int(self.sparsity * node_num), self.k)
            nodes_set = set()
            index = 0

            while len(nodes_set) < max_node_num:
                if index >= edge_index.size(1):
                    break
                e_index = sorted_indices[index]
                start_node, end_node = edge_index[:, e_index].tolist()
                nodes_set.add(start_node)
                if len(nodes_set) >= max_node_num:
                    break
                nodes_set.add(end_node)
                index += 1

            return list(nodes_set)


    def test(self):
        # load model
        checkpoint = torch.load(os.path.join(self.args.model_dir, f'{self.args.detector}_best.pth'))
        self.gnnNets.load_state_dict(checkpoint['net'])

        with torch.no_grad():
            for sample in tqdm(self.test_positive, desc="evaluating explainers", file=sys.stdout):
                graph_data: Data = self.generate_graph_data(sample)
                prob, node_emb = self.gnnNets(Batch.from_data_list([graph_data]).to(self.device))
                _, prediction = torch.max(prob, -1)
                # 必须是true positive
                if prediction.cpu() == 0:
                    continue
                sorted_indices, flag = self.explainer.explain(graph_data.x, graph_data.edge_index)
                nodes: List[int] = self.sample_nodes(sorted_indices, flag, len(graph_data.x), graph_data.edge_index)