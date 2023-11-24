from graph.detectors.models.reveal import ClassifyModel
from graph.detectors.detector_utils.reveal_util import RevealUtil
from graph.detectors.models.ivdetect import IVDetectModel
from graph.detectors.detector_utils.ivdetect_util import IVDetectUtil
from graph.detectors.models.devign import DevignModel
from graph.detectors.detector_utils.devign_util import DevignUtil
from graph.detectors.models.deepwukong import DeepWuKongModel, model_args as dwk_model_args

from typing import List, Dict, Union, Tuple
from gensim.models import Word2Vec
import json

import torch
from torch_geometric.data import Data
import numpy as np

from graph.detectors.train_utils.base_train_util import BaseTrainUtil


class RevealTrainUtil(BaseTrainUtil):
    def __init__(self, w2v_model: Word2Vec, gnnNets: ClassifyModel, device: str, dataset_dir: str, train_args):
        super().__init__(w2v_model, gnnNets, device, dataset_dir, train_args)
        self.reveal_util: RevealUtil = RevealUtil(w2v_model, gnnNets, device)

    def generate_features(self, sample: Dict[str, Union[str, List[int], List[str]]]) \
            -> Tuple[int, List[Data], torch.LongTensor]:
        return self.reveal_util.generate_initial_embedding(sample)


    def generate_graph_from_feature(self, feature: Tuple[int, List[Data], torch.LongTensor], device: str) -> Data:
        return self.reveal_util.generate_initial_graph_embedding(feature)


class IVDetectTrainUtil(BaseTrainUtil):
    def __init__(self, w2v_model: Word2Vec, gnnNets: IVDetectModel, device: str, dataset_dir: str, train_args):
        super().__init__(w2v_model, gnnNets, device, dataset_dir, train_args)
        self.ivdetect_util: IVDetectUtil = IVDetectUtil(w2v_model, device)

    def generate_features(self, sample: Dict[str, Union[str, List[int], List[str]]]) \
            -> Tuple[List[torch.Tensor], List[Tuple], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], torch.LongTensor, int]:
        return self.ivdetect_util.generate_all_features(sample)

    def generate_graph_from_feature(self, feature: Tuple[List[torch.Tensor], List[Tuple], List[torch.Tensor], List[torch.Tensor],
                                                         List[torch.Tensor], torch.LongTensor, int],
                                            device: str) -> Data:
        return self.gnnNets.vectorize_graph(feature, device)


class DevignTrainUtil(BaseTrainUtil):
    def __init__(self, w2v_model: Word2Vec, gnnNets: DevignModel, device: str, dataset_dir: str, train_args):
        super().__init__(w2v_model, gnnNets, device, dataset_dir, train_args)
        self.devign_util: DevignUtil = DevignUtil(w2v_model, gnnNets, device)

    def generate_features(self, sample: Dict[str, Union[str, List[int], List[str]]]) \
            -> Tuple[int, List[Data], torch.LongTensor]:
        return self.devign_util.generate_initial_embedding(sample)


    def generate_graph_from_feature(self, feature: Tuple[int, List[Data], torch.LongTensor], device: str) -> Data:
        return self.devign_util.generate_initial_graph_embedding(feature)


class DeepWuKongTrainUtil(BaseTrainUtil):
    def __init__(self, w2v_model: Word2Vec, gnnNets: DeepWuKongModel, device: str, dataset_dir: str, train_args):
        super().__init__(w2v_model, gnnNets, device, dataset_dir, train_args)

    def generate_features(self, data: Dict) -> Data:
        token_seqs = data["line-contents"]
        # token_seqs = [json.loads(node_info)["contents"][0][1] for node_info in data["line-nodes"]]
        n_vs = [np.array([self.w2v_model[word] if word in self.w2v_model.wv.vocab else
                          np.zeros(dwk_model_args.vector_size) for word in token_seq.split(" ")]).mean(axis=0) for token_seq
                in token_seqs]
        t_vs = [torch.FloatTensor(n_v).to(self.device) for n_v in n_vs]
        vector = torch.stack(t_vs)
        edges = [json.loads(edge) for edge in data["data-dependences"] + data["control-dependences"]]
        edge_index = torch.LongTensor(edges).to(self.device).t()

        return Data(x=vector, edge_index=edge_index, y=data["target"])

    def generate_graph_from_feature(self, feature: Data, device: str) -> Data:
        return feature