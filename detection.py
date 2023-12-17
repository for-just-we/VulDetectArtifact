import torch
from gensim.models.word2vec import Word2Vec

from graph.detectors.train_utils.detectors_train_utils import DevignTrainUtil, \
    RevealTrainUtil, IVDetectTrainUtil, DeepWuKongTrainUtil
from graph.detectors.train_utils.base_train_util import BaseTrainUtil
from graph.detectors.models.deepwukong import DeepWuKongModel
from graph.detectors.models.devign import DevignModel
from graph.detectors.models.reveal import ClassifyModel
from graph.detectors.models.ivdetect import IVDetectModel

from sequence.detectors.models.tokenlstm import build_model as tl_build_model
from sequence.detectors.models.vuldeepecker import build_model as vdp_build_model
from sequence.detectors.models.sysyer import build_model as syse_build_model
from sequence.detectors.train_util import SequenceTrainUtil

from keras.models import load_model

import os
import argparse

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

sequence_detector_models = {
    tokenlstm: tl_build_model,
    vdp: vdp_build_model,
    sysevr: syse_build_model
}

sequence_detectors = [name for name in sequence_detector_models.keys()]
graph_detectors = [name for name in graph_detector_models.keys()]

def build_arg_parser():
    parser = argparse.ArgumentParser(description="Command-line tool to detect vulnerable code fragments.")
    parser.add_argument("--dataset_dir", type=str, required=True, help='specify dataset dir, '
                                                                       'should contain 6 json files including train_vul.json')
    default_device: str = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, help="specify device, cuda or cpu", default=default_device)
    parser.add_argument("--model_dir", type=str, required=True, help="specify where to store GNN or RNN models")
    parser.add_argument("--w2v_model_path", type=str, required=True, help="path to word2vec model")
    parser.add_argument("--detector", type=str, required=True, help="the detector name here.", choices=graph_detectors + sequence_detectors)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=64)
    parser.add_argument("--learning_rate", type=float, help="learning rate", default=0.0001)
    parser.add_argument("--weight_decay", type=float, help="weight decay", default=1.3e-6)
    parser.add_argument("--early_stopping", type=int, default=5)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--save_epoch", type=int, default=5)
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)

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

    # 训练sequence detectors
    elif args.detector in sequence_detectors:
        # 如果已经存在detector文件，则加载
        model_path = f"{args.model_dir}/{args.detector}.h5"
        if os.path.exists(model_path):
            model = load_model(model_path)
        else:
            model = sequence_detector_models[args.detector]()
        train_util: SequenceTrainUtil = SequenceTrainUtil(w2v_model, model, args, model_path)
        if args.train:
            print("training {} start:".format(args.detector))
            train_util.train()
        if args.test:
            print("testing {} start:".format(args.detector))
            train_util.test()

if __name__ == '__main__':
    main()