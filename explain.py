import torch
from gensim.models.word2vec import Word2Vec
import argparse

from graph.detectors.train_utils.detectors_train_utils import DevignTrainUtil, \
    RevealTrainUtil, IVDetectTrainUtil, DeepWuKongTrainUtil
from graph.detectors.train_utils.base_train_util import BaseTrainUtil
from graph.detectors.models.deepwukong import DeepWuKongModel
from graph.detectors.models.devign import DevignModel
from graph.detectors.models.reveal import ClassifyModel
from graph.detectors.models.ivdetect import IVDetectModel

from graph.explainers.approaches.pgexplainer import PGExplainer
from graph.explainers.approaches.gnnexplainer import GNNExplainer
from graph.explainers.approaches.gradcam import GradCAM
from graph.explainers.approaches.deeplift import DeepLIFT
from graph.explainers.approaches.gnnlrp import GNN_LRP

dwk: str = "deepwukong"
reveal: str = "reveal"
ivdetect: str = "ivdetect"
devign: str = "devign"
tokenlstm: str = "tokenlstm"
vdp: str = "vuldeepecker"
sysevr: str = "sysevr"

graph_detector_train_utils = {dwk: DeepWuKongTrainUtil,
                              reveal: RevealTrainUtil,
                              ivdetect: IVDetectTrainUtil,
                              devign: DevignTrainUtil}
graph_detector_models = {dwk: DeepWuKongModel,
                              reveal: ClassifyModel,
                              ivdetect: IVDetectModel,
                              devign: DevignModel}

sequence_detectors = ["tokenlstm", "vuldeepecker", "sysevr"]
graph_detectors = [name for name in graph_detector_models.keys()]


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Command-line tool to explain results.")
    parser.add_argument("--dataset_dir", type=str, required=True, help='specify dataset dir, '
                                                                       'should contain test_vul.json')
    default_device: str = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, help="specify device, cuda or cpu", default=default_device)
    parser.add_argument("--model_dir", type=str, required=True, help="specify where to store GNN or RNN models")
    parser.add_argument("--w2v_model_path", type=str, required=True, help="path to word2vec model")
    parser.add_argument("--detector", type=str, required=True, help="the detector name here.", choices=graph_detectors + sequence_detectors)

    return parser

def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    w2v_model: Word2Vec = Word2Vec.load(args.w2v_model_path)
    # 训练graph detectors
    if args.detector in graph_detector_models.keys():
        model_cls = graph_detector_models[args.detector]
        model = model_cls()

if __name__ == '__main__':
    main()