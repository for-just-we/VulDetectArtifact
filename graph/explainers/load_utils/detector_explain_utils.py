from graph.detectors.models.reveal import ClassifyModel
from graph.detectors.detector_utils.reveal_util import RevealUtil
from graph.detectors.models.ivdetect import IVDetectModel
from graph.detectors.detector_utils.ivdetect_util import IVDetectUtil
from graph.detectors.models.devign import DevignModel
from graph.detectors.detector_utils.devign_util import DevignUtil
from graph.detectors.models.deepwukong import DeepWuKongModel
from graph.detectors.detector_utils.deepwukong_util import vectorize_xfg

from typing import List, Dict, Tuple
from gensim.models import Word2Vec

import torch
from torch_geometric.data import Data

from graph.explainers.load_utils.base_explain_util import BaseExplainerUtil


class RevealExplainerUtil(BaseExplainerUtil):
    def __init__(self, w2v_model: Word2Vec, gnnNets: ClassifyModel, args, explainer_name, k: int):
        super().__init__(w2v_model, gnnNets, args, explainer_name, k)
        self.reveal_util: RevealUtil = RevealUtil(w2v_model, gnnNets, args.device)

    def generate_graph_data(self, sample: Dict) -> Data:
        feature: Tuple[int, List[Data], torch.LongTensor] = self.reveal_util.generate_initial_embedding(sample)
        return self.reveal_util.generate_initial_graph_embedding(feature)


class IVDetectExplainerUtil(BaseExplainerUtil):
    def __init__(self, w2v_model: Word2Vec, gnnNets: IVDetectModel, args, explainer, k: int):
        super().__init__(w2v_model, gnnNets, args, explainer, k)
        self.ivdetect_util: IVDetectUtil = IVDetectUtil(w2v_model, args.device)


    def generate_graph_data(self, sample: Dict) -> Data:
        feature: Tuple[List[torch.Tensor], List[Tuple], List[torch.Tensor], List[torch.Tensor],
                       List[torch.Tensor], torch.LongTensor, int] = self.ivdetect_util.generate_all_features(sample)
        return self.gnnNets.vectorize_graph(feature, self.device)


class DevignExplainerUtil(BaseExplainerUtil):
    def __init__(self, w2v_model: Word2Vec, gnnNets: DevignModel, args, explainer, k: int):
        super().__init__(w2v_model, gnnNets, args, explainer, k)
        self.devign_util: DevignUtil = DevignUtil(w2v_model, gnnNets, args.device)

    def generate_graph_data(self, sample: Dict) -> Data:
        feature: Tuple[int, List[Data], torch.LongTensor] = self.devign_util.generate_initial_embedding(sample)
        return self.devign_util.generate_initial_graph_embedding(feature)


class DeepWuKongExplainerUtil(BaseExplainerUtil):
    def __init__(self, w2v_model: Word2Vec, gnnNets: DeepWuKongModel, args, explainer, k: int):
        super().__init__(w2v_model, gnnNets, args, explainer, k)

    def generate_graph_data(self, sample: Dict) -> Data:
        return vectorize_xfg(self.w2v_model, sample, self.device)