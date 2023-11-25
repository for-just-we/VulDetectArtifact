from graph.detectors.models.reveal import ClassifyModel
from graph.detectors.detector_utils.reveal_util import RevealUtil
from graph.detectors.models.ivdetect import IVDetectModel
from graph.detectors.detector_utils.ivdetect_util import IVDetectUtil
from graph.detectors.models.devign import DevignModel
from graph.detectors.detector_utils.devign_util import DevignUtil
from graph.detectors.models.deepwukong import DeepWuKongModel
from graph.detectors.detector_utils.deepwukong_util import vectorize_xfg

from typing import List, Dict, Union, Tuple
from gensim.models import Word2Vec

import torch
from torch_geometric.data import Data

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
        return vectorize_xfg(self.w2v_model, data, self.device)

    def generate_graph_from_feature(self, feature: Data, device: str) -> Data:
        return feature