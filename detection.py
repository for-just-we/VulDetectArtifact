import os
import torch
from gensim.models.word2vec import Word2Vec

from graph.detectors.train_utils.detectors_train_utils import DevignTrainUtil, \
    RevealTrainUtil, IVDetectTrainUtil, DeepWuKongTrainUtil
from graph.detectors.models.deepwukong import DeepWuKongModel
from graph.detectors.models.devign import DevignModel
from graph.detectors.models.reveal import ClassifyModel
from graph.detectors.models.ivdetect import IVDetectModel

import argparse

graph_detectors = ["deepwukong", "reveal", "ivdetect", "devign"]
sequence_detectors = ["tokenlstm", "vuldeepecker", "sysevr"]

def build_arg_parser():
    parser = argparse.ArgumentParser(description="Command-line tool to detect vulnerable code fragments.")
    parser.add_argument("--dataset_dir", type=str, required=True, help='specify dataset dir, '
                                                                       'should contain 6 json files including train_vuls.json')
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

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    w2v_model: Word2Vec = Word2Vec.load(args.w2v_model_path)
    # 训练graph detectors
