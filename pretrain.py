import os

from gensim.models.word2vec import Word2Vec
import nltk
import json
from typing import List, Dict
import sys

from graph.detectors.models.deepwukong import model_args as dwk_model_args
from graph.detectors.models.devign import model_args as devign_model_args
from graph.detectors.models.reveal import model_args as reveal_model_args
from graph.detectors.models.ivdetect import model_args as ivdetect_model_args
from graph.detectors.detector_utils.ivdetect_util import lexical_parse

from sequence.detectors.models.tokenlstm import model_args as tl_model_args

# pretrain embedding models for detectors
window_size = 10

w2v_sizes = {
    "deepwukong": dwk_model_args.vector_size,
    "devign": devign_model_args.vector_size,
    "reveal": reveal_model_args.vector_size,
    "ivdetect": ivdetect_model_args.feature_representation_size,

    "tokenlstm": tl_model_args.vector_size
}

class Sentences:
    def __init__(self, datas: List[Dict], detector: str):
        self.datas: List[Dict] = datas
        self.detector: str = detector

    def __iter__(self):
        for graph_data in self.datas:
            if self.detector in {"devign", "reveal", "ivdetect"}:
                json_contents = graph_data["nodes"]
                for json_content in json_contents:
                    graph_data: Dict = json.loads(json_content)
                    if self.detector == "ivdetect":
                        statement_after_split = lexical_parse(graph_data["contents"][0][1])
                    else:
                        statement_after_split = nltk.word_tokenize(graph_data["contents"][0][1])
                    yield statement_after_split
            # deepwukong
            elif self.detector == "deepwukong":
                contents = graph_data["nodes-line-sym"]
                for statement in contents:
                    statement_after_split = nltk.word_tokenize(statement)
                    yield statement_after_split

            elif self == "tokenlstm":
                contents: List[str] = list()
                json_contents = graph_data["nodes"]
                for json_content in json_contents:
                    graph_data: Dict = json.loads(json_content)
                    statement_after_split = nltk.word_tokenize(graph_data["contents"][0][1])
                    contents.extend(statement_after_split)
                yield contents

            else:
                raise RuntimeError("unsupported detector")


if __name__ == '__main__':
    # name of detector, choice is deepwukong, ivdetect, reveal, devign
    detector_name = sys.argv[1]
    # path to dataset, should include train_vul.json and train_normal.json
    train_dataset_path = sys.argv[2]
    # path to save
    pretrain_doc2vec_model_path = sys.argv[3]

    train_jsons: List[Dict] = json.load(open(os.path.join(train_dataset_path,
                                                          "train_vul.json"), 'r', encoding='utf-8')) + \
                              json.load(open(os.path.join(train_dataset_path,
                                                          "train_normal.json"), 'r', encoding='utf-8'))
    sentences = Sentences(train_jsons, detector_name)
    model = Word2Vec(sentences, size=w2v_sizes[detector_name], window=window_size, hs=1, min_count=1, iter=20)
    model.save(pretrain_doc2vec_model_path)