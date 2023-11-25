import torch
from gensim.models.word2vec import Word2Vec
from enum import Enum

from graph.detectors.train_utils.detectors_train_utils import DevignTrainUtil, \
    RevealTrainUtil, IVDetectTrainUtil, DeepWuKongTrainUtil
from graph.detectors.train_utils.base_train_util import BaseTrainUtil
from graph.detectors.models.deepwukong import DeepWuKongModel
from graph.detectors.models.devign import DevignModel
from graph.detectors.models.reveal import ClassifyModel
from graph.detectors.models.ivdetect import IVDetectModel

import argparse

class DetectorNames(Enum):
    dwk = "deepwukong",
    reveal = "reveal",
    ivdetect = "ivdetect",
    devign = "devign",
    tokenlstm = "tokenlstm",
    vdp = "vuldeepecker",
    sysevr = "sysevr"

graph_detector_train_utils = {DetectorNames.dwk.value: DeepWuKongTrainUtil,
                              DetectorNames.reveal.value: RevealTrainUtil,
                              DetectorNames.ivdetect.value: IVDetectTrainUtil,
                              DetectorNames.devign.value: DevignTrainUtil}
graph_detector_models = {DetectorNames.dwk.value: DeepWuKongModel,
                              DetectorNames.reveal.value: ClassifyModel,
                              DetectorNames.ivdetect.value: IVDetectModel,
                              DetectorNames.devign.value: DevignModel}

sequence_detectors = ["tokenlstm", "vuldeepecker", "sysevr"]

def build_arg_parser():
    parser = argparse.ArgumentParser(description="Command-line tool to detect vulnerable code fragments.")
    parser.add_argument("--dataset_dir", type=str, required=True, help='specify dataset dir, '
                                                                       'should contain 6 json files including train_vul.json')
    default_device: str = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, help="specify device, cuda or cpu", default=default_device)
    parser.add_argument("--model_dir", type=str, required=True, help="specify where to store GNN or RNN models")
    parser.add_argument("--w2v_model_path", type=str, required=True, help="path to word2vec model")
    parser.add_argument("--detector", type=str, required=True, help="the detector name here.", choices=list(graph_detector_models.keys()) + sequence_detectors)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=64)
    parser.add_argument("--learning_rate", type=float, help="learning rate", default=0.0001)
    parser.add_argument("--weight_decay", type=float, help="weight decay", default=1.3e-6)
    parser.add_argument("--early_stopping", type=int, default=5)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--save_epoch", type=int, default=5)
    parser.add_argument("--train", type=bool, action="store_true", default=False)
    parser.add_argument("--test", type=bool, action="store_true", default=False)

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    w2v_model: Word2Vec = Word2Vec.load(args.w2v_model_path)
    # 训练graph detectors
    if args.detector in graph_detector_models.keys():
        model_cls = graph_detector_models[args.detector]
        train_util_cls = graph_detector_train_utils[args.detector]
        model = model_cls()
        train_util: BaseTrainUtil = train_util_cls(w2v_model, model, args)
        if args.train:
            print("training {} start:".format(args.detector))
            train_util.train()
        if args.test:
            print("testing {} start:".format(args.detector))
            train_util.test()

if __name__ == '__main__':
    main()