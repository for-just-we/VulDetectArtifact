import torch
from gensim.models.word2vec import Word2Vec
import argparse
import os

from graph.detectors.models.deepwukong import DeepWuKongModel
from graph.detectors.models.devign import DevignModel
from graph.detectors.models.reveal import ClassifyModel
from graph.detectors.models.ivdetect import IVDetectModel

from graph.explainers.load_utils.detector_explain_utils import RevealExplainerUtil, \
    IVDetectExplainerUtil, DevignExplainerUtil, DeepWuKongExplainerUtil
from graph.explainers.load_utils.base_explain_util import BaseExplainerUtil



dwk: str = "deepwukong"
reveal: str = "reveal"
ivdetect: str = "ivdetect"
devign: str = "devign"
tokenlstm: str = "tokenlstm"
vdp: str = "vuldeepecker"
sysevr: str = "sysevr"

graph_detector_explain_utils = {dwk: DeepWuKongExplainerUtil,
                              reveal: RevealExplainerUtil,
                              ivdetect: IVDetectExplainerUtil,
                              devign: DevignExplainerUtil}
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

    parser.add_argument("--explainer", type=str, required=True, choices={"gnnexplainer", "pgexplainer", "gradcam", "deeplift", "gnnlrp"})
    parser.add_argument("--k", type=int, default=5, help="max_node num in explanation results")
    return parser

def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    w2v_model: Word2Vec = Word2Vec.load(args.w2v_model_path)
    # 训练graph detectors
    if args.detector in graph_detector_models.keys():
        model_cls = graph_detector_models[args.detector]
        need_node_emb_flag = (args.explainer == "pgexplainer")
        model = model_cls(need_node_emb=need_node_emb_flag)
        model_path = os.path.join(args.model_dir, f"{args.detector}_best.pth")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['net'])
        model.to(args.device)
        model.eval()
        explainer_name = args.explainer
        explainer_util_cls = graph_detector_explain_utils[args.detector]
        explainer_util: BaseExplainerUtil = explainer_util_cls(w2v_model, model, args, explainer_name, args.k)
        explainer_util.test()


if __name__ == '__main__':
    main()