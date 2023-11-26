import csv
import json
import sys
import os
from typing import Dict, List, Union

NODE_FILE = "nodes.csv"
EDGE_FILE = "edges.csv"

# AST
AST_EDGE_TYPE = "IS_AST_PARENT"
# control dependence
CDG_EDGE_TYPE= "CONTROLS"
# data dependence
DDG_EDGE_TYPE = "REACHES"
# control flow
CFG_EDGE_TYPE = "FLOWS_TO"

FILE_NODE = "File"
FUNCTION_NODE = "Function"
FUNCDEF_NODE= "FunctionDef"
CFG_ENTRY = "CFGEntryNode"

'''
每一个CFG node dump到json里的格式为
{   
    "line":37,
    "edges":[[0,1],[1,2],[1,3]], # edges will be placed in pre-order of AST
    "contents":[
        ["IdentifierDeclStatement","charVoid structCharVoid ;"],
        ["IdentifierDecl","structCharVoid"],
        ["IdentifierDeclType","charVoid"],
        ["Identifier","structCharVoid"]
    ]
}
'''
class CFGNode:
    def __init__(self, line: int):
        self.line = line
        self.ast_nodes: List[List[str]] = list()
        self.ast_edges: List[List[int]] = list()
        self.idx_transform: Dict[int, int] = dict()

    def ast_node_size(self):
        return len(self.ast_nodes)

    def to_json(self):
        return {
            "line": self.line,
            "edges": self.ast_edges,
            "contents": self.ast_nodes
        }

'''
每一个graph dump为json后为: 
{
    "fileName": "CWE124_Buffer_Underwrite__malloc_wchar_t_ncpy_15.c",
    "cfgEdges": [
      ... # list of cfg edges
    ],
    "nodes": [
      ... # json format of each cfg nodes
    ],
    "cdgEdges": [
      ... # list of cdg edges
    ],
    "ddgEdges": [
      ... # list of ddg edges
    ],
    "functionName": "CWE124_Buffer_Underwrite__malloc_wchar_t_ncpy_15_bad",
  }
'''
class FunctionGraph:
    def __init__(self, func_name, filename):
        self.func_name: str = func_name
        self.file_name: str = filename
        self.cfg_nodes: List[CFGNode] = list()
        # control flow
        self.cfg_edges: List[List[int]] = list()
        # control dependence
        self.cdg_edges: List[List[int]] = list()
        # data dependence
        self.ddg_edges: List[List[int]] = list()

        self.idx_transform: Dict[int, int] = dict()

    def node_size(self):
        return len(self.cfg_nodes)

    def to_json(self):
        edge_set_to_json = lambda edge_set: [json.dumps(edge) for edge in edge_set]
        return {
            "fileName": self.file_name,
            "functionName": self.func_name,
            "nodes": [json.dumps(node.to_json()) for node in self.cfg_nodes],
            "cfgEdges": edge_set_to_json(self.cfg_edges),
            "cdgEdges": edge_set_to_json(self.cdg_edges),
            "ddgEdges": edge_set_to_json(self.ddg_edges)
        }

class CSVAnalyzer:
    def __init__(self, path):
        self.filename = path.split(os.sep)[-1]
        self.nodes_csv_path = os.sep.join([path, NODE_FILE])
        self.edges_csv_path = os.sep.join([path, EDGE_FILE])

        self.cfg_edges: List[List[int]] = list()
        self.cdg_edges: List[List[int]] = list()
        self.ddg_edges: List[List[int]] = list()
        self.raw_ast_parent: Dict[int, int] = dict()

        self.func_graphs: List[FunctionGraph] = list()
        self.enter_func: bool = False
        self.cfg_idx_2_func_idx: Dict[int, int] = dict()

    def analyze_edges_files(self):
        csv_edge_reader = csv.DictReader(open(self.edges_csv_path, 'r', encoding='utf-8'), delimiter='\t')
        for row in csv_edge_reader:
            start_node_idx = int(row['start'])
            end_node_idx = int(row['end'])
            edge_type = row['type']

            if edge_type == AST_EDGE_TYPE:
                self.raw_ast_parent[end_node_idx] = start_node_idx
            elif edge_type == CFG_EDGE_TYPE:
                self.cfg_edges.append([start_node_idx, end_node_idx])
            elif edge_type == CDG_EDGE_TYPE:
                self.cdg_edges.append([start_node_idx, end_node_idx])
            elif edge_type == DDG_EDGE_TYPE:
                self.ddg_edges.append([start_node_idx, end_node_idx])

    def analyze_nodes_files(self):
        csv_node_reader = csv.DictReader(open(self.nodes_csv_path, 'r', encoding='utf-8'), delimiter='\t')
        for row in csv_node_reader:
            idx = int(row['key'])
            node_type = row['type']
            # start of a function
            if node_type == FUNCTION_NODE:
                func_name = row['code']
                self.func_graphs.append(FunctionGraph(func_name, self.filename))
                self.enter_func = True
                continue
            # end of a function:
            elif node_type == CFG_ENTRY:
                self.enter_func = False
                continue
            elif node_type in {FUNCDEF_NODE, FILE_NODE}:
                continue

            if not self.enter_func:
                continue
            # 当前处理的graph data
            cur_graph = self.func_graphs[len(self.func_graphs) - 1]
            is_cfg_node: bool = (row['isCFGNode'] == "True")
            # 如果当前node是CFG node
            if is_cfg_node:
                location: str = row['location']
                line: int = int(location.split(':')[0])
                cfg_node = CFGNode(line)
                cur_graph.cfg_nodes.append(cfg_node)
                cur_graph.idx_transform[idx] = cur_graph.node_size() - 1
                self.cfg_idx_2_func_idx[idx] = len(self.func_graphs) - 1
            if cur_graph.node_size() == 0:
                continue
            cur_cfg_node = cur_graph.cfg_nodes[cur_graph.node_size() -1]
            code: str = row['code']
            ast_node_idx = cur_cfg_node.ast_node_size()
            if is_cfg_node:
                assert ast_node_idx == 0
            # 如果不是CFG Node，需要添加AST关系
            else:
                parent_raw_idx = self.raw_ast_parent[idx]
                # 当前ASTNode不属于这个CFGNode之下
                if parent_raw_idx not in cur_cfg_node.idx_transform.keys():
                    continue
                parent_node_idx = cur_cfg_node.idx_transform[parent_raw_idx]
                cur_cfg_node.ast_edges.append([parent_node_idx, ast_node_idx])
            cur_cfg_node.idx_transform[idx] = ast_node_idx
            cur_cfg_node.ast_nodes.append([node_type, code])

    def add_cfg_node_edges(self):
        def process(label):
            key_2_edge_set = {
                "cfg": self.cfg_edges,
                "cdg": self.cdg_edges,
                "ddg": self.ddg_edges
            }
            for edge in key_2_edge_set[label]:
                start_raw_idx, end_raw_idx = edge
                if start_raw_idx not in self.cfg_idx_2_func_idx.keys() or end_raw_idx not in self.cfg_idx_2_func_idx.keys():
                    continue
                graph_idx = self.cfg_idx_2_func_idx[start_raw_idx]
                cur_graph = self.func_graphs[graph_idx]
                start_idx = cur_graph.idx_transform[start_raw_idx]
                end_idx = cur_graph.idx_transform[end_raw_idx]
                edge_label_2_set = {
                    "cfg": cur_graph.cfg_edges,
                    "cdg": cur_graph.cdg_edges,
                    "ddg": cur_graph.ddg_edges
                }
                edge_label_2_set[label].append([start_idx, end_idx])

        process("cfg")
        process("cdg")
        process("ddg")


def recurse_folder(folder_path, output_file):
    json_graphs: List[Dict[str, Union[str, List[str]]]] = list()
    for root, dirs, files in os.walk(folder_path):
        if len(dirs) == 0 and NODE_FILE in files and EDGE_FILE in files:
            cur_path = root[len(folder_path) + 1:]
            unify_path = '/'.join(cur_path.split(os.sep))
            tmp = cur_path.split(os.sep)
            testcase_name = tmp[0]
            testcase_path = '/'.join(tmp[2:])
            print("testcase_name: {}".format(testcase_name))
            print("testcase_path: {}".format(testcase_path))
            csv_analyzer = CSVAnalyzer(root)
            csv_analyzer.analyze_edges_files()
            csv_analyzer.analyze_nodes_files()
            csv_analyzer.add_cfg_node_edges()
            for graph_data in csv_analyzer.func_graphs:
                func_graph: Dict[str, Union[str, List[str]]] = graph_data.to_json()
                func_graph["testcase-path"] = unify_path
                json_graphs.append(func_graph)
            print("======================")

    # make sure there is not empty graph datas
    is_not_empty_data = lambda data: len(data["nodes"]) > 0 and (len(data["cdgEdges"]) + len(data["ddgEdges"]) > 0)
    json_graphs = list(filter(is_not_empty_data, json_graphs))
    json.dump(json_graphs, open(output_file, 'w', encoding='utf8'), indent=2)



if __name__ == '__main__':
    csv_dir = sys.argv[1]
    output_file = sys.argv[2]
    recurse_folder(csv_dir, output_file)