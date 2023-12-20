import json
from tqdm import tqdm
import sys
from typing import List, Dict, Union, DefaultDict, Set
from collections import defaultdict
from utils.ast_def import ASTNode, json2astNode
from utils.ast_analyzer import CallTargetChecker, SyVCChecker

def group_by_testcase(datas: List[Dict[str, Union[str, List[str]]]]) -> \
        DefaultDict[str, List[Dict[str, Union[str, List[str]]]]]:
    '''
    :param datas: 所有的graph data
    :return: 为1个dict，list每个元素为同一个testcase对应的1组function对应的cpg
    '''
    group_datas: DefaultDict[str, List[Dict[str, Union[str, List[str]]]]] = defaultdict(list)
    for cpg in datas:
        testcase_id = cpg["testcase-path"]
        testcase_id = '/'.join(testcase_id.split('/')[:5])
        group_datas[testcase_id].append(cpg)
    return group_datas

class SlicingTool:
    def __init__(self, vul_funcs: Set[str], datas: List[Dict[str, Union[str, List[str]]]]):
        self.vul_funcs: Set[str] = vul_funcs
        self.func_names: Set[str] = set([data["functionName"] for data in datas])
        self.func_name2data: Dict[str, Dict[str, Union[str, List[str]]]] = dict()

        # 遍历每个函数的参数node索引
        self.func_name2param_idx: DefaultDict[str, Set[int]] = defaultdict(set)
        # 每个函数对应的调用目标以及调用目标索引
        # func_name --> call target name --> node idx
        self.func_name2call_target: DefaultDict[str, Dict[str, int]] = defaultdict(dict)
        # func_name --> node idx --> call target names
        self.func_name2call_target_reverse: DefaultDict[str, DefaultDict[int, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.key_line_set: DefaultDict[str, Set[int]] = defaultdict(set)
        self.file_name2idx: Dict[str, int] = dict()

        for data in datas:
            self.build_key_query_structure(data)

    def build_key_query_structure(self, data: Dict[str, Union[str, List[str]]]):
        syse_checker = SyVCChecker(self.vul_funcs)
        func_name: str = data["functionName"]
        if func_name in self.func_name2param_idx.keys():
            return
        self.func_name2data[func_name] = data
        testcase_path = data["testcase-path"]
        if testcase_path not in self.file_name2idx.keys():
            self.file_name2idx[testcase_path] = len(self.file_name2idx)

        json_nodes: List[str] = data["nodes"]
        for i, json_node_str in enumerate(json_nodes):
            json_node: Dict[str, Union[int, list]] = json.loads(json_node_str)
            ast_node: ASTNode = json2astNode(json_node)

            if ast_node.type == "Parameter":
                self.func_name2param_idx[func_name].add(i)
                continue

            call_checker = CallTargetChecker(self.func_names)
            flag = syse_checker.visit(ast_node)
            call_checker.visit(ast_node)

            if flag:
                self.key_line_set[func_name].add(i)

            for called_func_name in reversed(call_checker.called_func_names):
                self.func_name2call_target[func_name][called_func_name] = i
                self.func_name2call_target_reverse[func_name][i].append(called_func_name)

    def run_slice(self):
        for func_name, key_lines in self.key_line_set.items():
            # 对每个key-line进行切片
            for key_line in key_lines:
                cur_slice_lines: List[str] = list()
                cur_loc_idxs: List[str] = list()
                self.extract_backward_slice(func_name, key_line, cur_slice_lines, cur_loc_idxs)
                self.extract_forward_slice(func_name, key_line, cur_slice_lines, cur_loc_idxs)

    # conduct backward program slice, where func_name and line is the start of current location,
    # cur_slice_lines and cur_loc_idxs stores
    def extract_backward_slice(self, func_name: str, line: int,
                               cur_slice_lines: List[str], cur_loc_idxs: List[str]):
        pass

    def extract_forward_slice(self, func_name: str, line: int,
                              cur_slice_lines: List[str], cur_loc_idxs: List[str], funcs: Set[str]):
        if func_name in funcs:
            return
        cur_data: Dict[str, Union[str, List[str]]] = self.func_name2data[func_name]
        # only use data dependence
        ddg_edges_text: List[str] = cur_data["ddgEdges"]
        ddg_edges: List[List[int]] = [json.loads(edge_text) for edge_text in ddg_edges_text]

        # append cur line to slice
        cur_line_data: dict = json.loads(cur_data["nodes"][line])
        file_line = cur_line_data['line']
        line_content = cur_line_data['contents'][0][1]
        # cur_slice_lines.append(line_content)
        # cur_loc_idxs.append(str(self.file_name2idx[cur_data["testcase-path"]]) + "-" + str(file_line))

        # 如果调用了其它函数
        call_targets: List[str] = self.func_name2call_target_reverse.get(func_name, {}).get(line, [])


        ddg_edges_from_cur_line: List[List[int]] = list(filter(lambda edge: edge[0] == line, ddg_edges))
        for edge in ddg_edges_from_cur_line:
            pass


def main():
    json_data_path = sys.argv[1]
    sensiAPIs: Set[str] = set(open("sensiAPI.txt", 'r', encoding='utf-8').read().split(','))
    datas: List[Dict[str, Union[str, List[str]]]] = json.load(open(json_data_path, 'r', encoding='utf-8'))
    group_datas: DefaultDict[str, List[Dict[str, Union[str, List[str]]]]] = group_by_testcase(datas)

    for testcase_id, test_case_datas in tqdm(group_datas.items(), desc="processing datas"):
        slicing_tool = SlicingTool(sensiAPIs, test_case_datas)
        slicing_tool.run_slice()
        pass


if __name__ == '__main__':
    main()