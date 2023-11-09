import sys

from gensim.models import Word2Vec
import numpy as np
import torch
from torch_geometric.data import Data, Batch

import json
from typing import Dict, List, Tuple, Set

from graph.detectors.models.devign import type_map, model_args, DevignModel

class DevignUtil(object):
    def __init__(self, pretrain_model: Word2Vec, devign_model: DevignModel, device):
        self.pretrain_model = pretrain_model
        self.devign_model = devign_model
        self.arrays = np.eye(80)
        self.device = device

    # 生成图中每个ASTNode的初始embedding，在训练阶段这个函数只执行一次
    # nodeContent[0] 为type, nodeContent[1] 为token sequence
    def generate_initial_astNode_embedding(self, nodeContent: List[str]) -> np.array:
        # type vector
        n_c = self.arrays[type_map[nodeContent[0]]]
        # token sequence
        token_seq: List[str] = nodeContent[1].split(' ')
        n_v = np.array([self.pretrain_model[word] if word in self.pretrain_model.wv.vocab else
                           np.zeros(model_args.vector_size) for word in token_seq]).mean(axis=0)

        v = np.concatenate([n_c, n_v])
        return v

    # 生成每个AST初始结点的信息
    # 与Reveal相比,它还需要在AST终端结点添加NCS边
    def generate_initial_node_info(self, ast: Dict) -> Data:
        astEmbedding: np.array = np.array([self.generate_initial_astNode_embedding(node_info) for node_info in ast["contents"]])
        x: torch.FloatTensor = torch.FloatTensor(astEmbedding)
        edges: List[List[int]] = [[edge[1], edge[0]] for edge in ast["edges"]]
        # 添加NCS边
        # 先找出图中所有的AST终端结点，按照索引顺序升序排序
        # 找出非终端结点的索引
        parent_idxs: Set[int] = set()
        terminal_idxs: List[int] = list()
        for edge in edges:
            parent_idxs.add(edge[1])
        # 找出终端结点的索引
        for i in range(len(x)):
            if i not in parent_idxs:
                terminal_idxs.append(i)
        # 添加NCS边
        for i in range(1, len(terminal_idxs)):
            edges.append([terminal_idxs[i - 1], terminal_idxs[i]])

        edge_index: torch.LongTensor = torch.LongTensor(edges).t()
        return Batch.from_data_list([Data(x=x, edge_index=edge_index)]).to(device=self.device)


    # 预处理训练数据， 训练过程只调用1次
    def generate_initial_embedding(self, data: Dict) -> Tuple[int, List[Data], torch.LongTensor]:
        # label, List[ASTNode], edge
        label: int = 0
        if "target" in data.keys():
            label = data["target"]

        # edges
        cfgEdges = [json.loads(edge)[:2] for edge in data["cfgEdges"]]
        ddgEdges = [json.loads(edge)[:2] for edge in data["ddgEdges"]]
        edges = cfgEdges + ddgEdges
        edge_index: torch.LongTensor = torch.LongTensor(edges).t()

        # nodes
        nodes_info: List[Dict] = [json.loads(node_infos) for node_infos in data["nodes"]]
        graph_data_for_each_nodes: List[Data] = [self.generate_initial_node_info(node_info) for node_info in nodes_info]

        return (label, graph_data_for_each_nodes, edge_index)

    # 生成图初始向量， 每个epoch会调用1次
    def generate_initial_graph_embedding(self, graph_info: Tuple[int, List[Data], torch.LongTensor]) -> Data:
        # self.reveal_model.embed_graph(data)[0] return graph_embedding for initial CPG node
        initial_embeddings: List[torch.FloatTensor] = [self.devign_model.embed_graph(data.x, data.edge_index, None)[0].reshape(-1,)
                                                       # 某些AST子树可能没有子结点，直接取其值作为node embedding
                                                       if len(data.edge_index) > 0 else data.x[0]
                                                       for data in graph_info[1]]
        X: torch.FloatTensor = torch.stack(initial_embeddings)
        return Data(x=X, edge_index=graph_info[2], y=torch.tensor([graph_info[0]], dtype=torch.long))


if __name__ == '__main__':
    device = "cuda"
    w2v_model_path = sys.argv[1]
    pretrain_model = Word2Vec.load(w2v_model_path)
    devign_model: DevignModel = DevignModel().to(device)
    devign_util = DevignUtil(pretrain_model, devign_model, device)

    sample = {
		"fileName":"CWE121_Stack_Based_Buffer_Overflow__char_type_overrun_memcpy_01.c",
		"cfgEdges":[
			"[0,1,\"\"]",
			"[1,2,\"\"]",
			"[2,3,\"\"]",
			"[3,4,\"\"]",
			"[4,5,\"\"]",
			"[5,6,\"\"]"
		],
		"nodes":[
			"{\"line\":37,\"edges\":[[0,1],[1,2],[1,3]],\"contents\":[[\"IdentifierDeclStatement\",\"charVoid structCharVoid ;\",\"\"],[\"IdentifierDecl\",\"structCharVoid\",\"\"],[\"IdentifierDeclType\",\"charVoid\",\"\"],[\"Identifier\",\"structCharVoid\",\"\"]]}",
			"{\"line\":38,\"edges\":[[0,1],[1,2],[1,3],[2,4],[2,5],[3,6],[3,7]],\"contents\":[[\"ExpressionStatement\",\"structCharVoid . voidSecond = ( void * ) SRC_STR ;\",\"\"],[\"AssignmentExpr\",\"structCharVoid . voidSecond = ( void * ) SRC_STR\",\"=\"],[\"MemberAccess\",\"structCharVoid . voidSecond\",\"\"],[\"CastExpression\",\"( void * ) SRC_STR\",\"\"],[\"Identifier\",\"structCharVoid\",\"\"],[\"Identifier\",\"voidSecond\",\"\"],[\"CastTarget\",\"void *\",\"\"],[\"Identifier\",\"SRC_STR\",\"\"]]}",
			"{\"line\":40,\"edges\":[[0,1],[1,2],[1,3],[2,4],[3,5],[5,6],[6,7],[6,8],[8,9],[8,10]],\"contents\":[[\"ExpressionStatement\",\"printLine ( ( char * ) structCharVoid . voidSecond ) ;\",\"\"],[\"CallExpression\",\"printLine ( ( char * ) structCharVoid . voidSecond )\",\"\"],[\"Callee\",\"printLine\",\"\"],[\"ArgumentList\",\"( char * ) structCharVoid . voidSecond\",\"\"],[\"Identifier\",\"printLine\",\"\"],[\"Argument\",\"( char * ) structCharVoid . voidSecond\",\"\"],[\"CastExpression\",\"( char * ) structCharVoid . voidSecond\",\"\"],[\"CastTarget\",\"char *\",\"\"],[\"MemberAccess\",\"structCharVoid . voidSecond\",\"\"],[\"Identifier\",\"structCharVoid\",\"\"],[\"Identifier\",\"voidSecond\",\"\"]]}",
			"{\"line\":42,\"edges\":[[0,1],[1,2],[1,3],[2,4],[3,5],[3,6],[3,7],[5,8],[6,9],[7,10],[8,11],[8,12],[10,13],[10,14]],\"contents\":[[\"ExpressionStatement\",\"memcpy ( structCharVoid . charFirst , SRC_STR , sizeof ( structCharVoid ) ) ;\",\"\"],[\"CallExpression\",\"memcpy ( structCharVoid . charFirst , SRC_STR , sizeof ( structCharVoid ) )\",\"\"],[\"Callee\",\"memcpy\",\"\"],[\"ArgumentList\",\"structCharVoid . charFirst , SRC_STR , sizeof ( structCharVoid )\",\"\"],[\"Identifier\",\"memcpy\",\"\"],[\"Argument\",\"structCharVoid . charFirst\",\"\"],[\"Argument\",\"SRC_STR\",\"\"],[\"Argument\",\"sizeof ( structCharVoid )\",\"\"],[\"MemberAccess\",\"structCharVoid . charFirst\",\"\"],[\"Identifier\",\"SRC_STR\",\"\"],[\"SizeofExpr\",\"sizeof ( structCharVoid )\",\"\"],[\"Identifier\",\"structCharVoid\",\"\"],[\"Identifier\",\"charFirst\",\"\"],[\"Sizeof\",\"sizeof\",\"\"],[\"Identifier\",\"structCharVoid\",\"\"]]}",
			"{\"line\":43,\"edges\":[[0,1],[1,2],[1,3],[2,4],[2,5],[4,6],[4,7],[5,8],[5,9],[8,10],[8,11],[10,12],[10,13],[11,14],[11,15],[13,16],[13,17]],\"contents\":[[\"ExpressionStatement\",\"structCharVoid . charFirst [ ( sizeof ( structCharVoid . charFirst ) / sizeof ( char ) ) - 1 ] = '\\\\0' ;\",\"\"],[\"AssignmentExpr\",\"structCharVoid . charFirst [ ( sizeof ( structCharVoid . charFirst ) / sizeof ( char ) ) - 1 ] = '\\\\0'\",\"=\"],[\"ArrayIndexing\",\"structCharVoid . charFirst [ ( sizeof ( structCharVoid . charFirst ) / sizeof ( char ) ) - 1 ]\",\"\"],[\"CharExpression\",\"'\\\\0'\",\"\"],[\"MemberAccess\",\"structCharVoid . charFirst\",\"\"],[\"AdditiveExpression\",\"( sizeof ( structCharVoid . charFirst ) / sizeof ( char ) ) - 1\",\"-\"],[\"Identifier\",\"structCharVoid\",\"\"],[\"Identifier\",\"charFirst\",\"\"],[\"MultiplicativeExpression\",\"sizeof ( structCharVoid . charFirst ) / sizeof ( char )\",\"/\"],[\"IntegerExpression\",\"1\",\"\"],[\"SizeofExpr\",\"sizeof ( structCharVoid . charFirst )\",\"\"],[\"SizeofExpr\",\"sizeof ( char )\",\"\"],[\"Sizeof\",\"sizeof\",\"\"],[\"MemberAccess\",\"structCharVoid . charFirst\",\"\"],[\"Sizeof\",\"sizeof\",\"\"],[\"SizeofOperand\",\"char\",\"\"],[\"Identifier\",\"structCharVoid\",\"\"],[\"Identifier\",\"charFirst\",\"\"]]}",
			"{\"line\":44,\"edges\":[[0,1],[1,2],[1,3],[2,4],[3,5],[5,6],[6,7],[6,8],[8,9],[8,10]],\"contents\":[[\"ExpressionStatement\",\"printLine ( ( char * ) structCharVoid . charFirst ) ;\",\"\"],[\"CallExpression\",\"printLine ( ( char * ) structCharVoid . charFirst )\",\"\"],[\"Callee\",\"printLine\",\"\"],[\"ArgumentList\",\"( char * ) structCharVoid . charFirst\",\"\"],[\"Identifier\",\"printLine\",\"\"],[\"Argument\",\"( char * ) structCharVoid . charFirst\",\"\"],[\"CastExpression\",\"( char * ) structCharVoid . charFirst\",\"\"],[\"CastTarget\",\"char *\",\"\"],[\"MemberAccess\",\"structCharVoid . charFirst\",\"\"],[\"Identifier\",\"structCharVoid\",\"\"],[\"Identifier\",\"charFirst\",\"\"]]}",
			"{\"line\":45,\"edges\":[[0,1],[1,2],[1,3],[2,4],[3,5],[5,6],[6,7],[6,8],[8,9],[8,10]],\"contents\":[[\"ExpressionStatement\",\"printLine ( ( char * ) structCharVoid . voidSecond ) ;\",\"\"],[\"CallExpression\",\"printLine ( ( char * ) structCharVoid . voidSecond )\",\"\"],[\"Callee\",\"printLine\",\"\"],[\"ArgumentList\",\"( char * ) structCharVoid . voidSecond\",\"\"],[\"Identifier\",\"printLine\",\"\"],[\"Argument\",\"( char * ) structCharVoid . voidSecond\",\"\"],[\"CastExpression\",\"( char * ) structCharVoid . voidSecond\",\"\"],[\"CastTarget\",\"char *\",\"\"],[\"MemberAccess\",\"structCharVoid . voidSecond\",\"\"],[\"Identifier\",\"structCharVoid\",\"\"],[\"Identifier\",\"voidSecond\",\"\"]]}"
		],
		"cdgEdges":[],
		"ddgEdges":[
			"[0,1,\"structCharVoid\"]",
			"[1,2,\"structCharVoid\"]",
			"[1,3,\"structCharVoid\"]",
			"[1,4,\"structCharVoid\"]",
			"[1,5,\"structCharVoid\"]",
			"[1,6,\"structCharVoid\"]",
			"[1,2,\"structCharVoid . voidSecond\"]",
			"[1,6,\"structCharVoid . voidSecond\"]"
		],
		"functionName":"CWE121_Stack_Based_Buffer_Overflow__char_type_overrun_memcpy_01_bad",
        "target": 1
	}

    graph_info: Tuple[int, List[Data], torch.LongTensor] = devign_util.generate_initial_embedding(sample)
    data: Data = devign_util.generate_initial_graph_embedding(graph_info)
    print(data)
    probs = devign_model(data=Batch.from_data_list([data]).to(device))
    _, prediction = torch.max(probs, -1)
    ori_prob = torch.softmax(probs, dim=1)
    print(prediction)
    print(ori_prob)