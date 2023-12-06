import json
import os
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
        group_datas[testcase_id].append(cpg)
    return group_datas

class SlicingTool:
    def __init__(self, vul_funcs: Set[str], datas: List[Dict[str, Union[str, List[str]]]]):
        self.vul_funcs: Set[str] = vul_funcs
        self.func_names: Set[str] = set([data["functionName"] for data in datas])

        # 遍历每个函数的参数node索引
        self.func_name2param_idx: DefaultDict[str, Set[int]] = defaultdict(set)
        # 每个函数对应的调用目标以及调用目标索引
        # func_name --> call target name --> node idx
        self.func_name2call_target: DefaultDict[str, Dict[str, int]] = defaultdict(dict)
        # func_name --> node idx --> call target names
        self.func_name2call_target_reverse: DefaultDict[str, DefaultDict[int, List[str]]] = defaultdict(defaultdict(list))
        self.key_line_set: DefaultDict[str, Set[int]] = defaultdict(set)

        for data in datas:
            self.build_key_query_structure(data)

    def build_key_query_structure(self, data: Dict[str, Union[str, List[str]]]):
        syse_checker = SyVCChecker(self.vul_funcs)
        func_name: str = data["functionName"]
        assert func_name not in self.func_name2param_idx.keys()
        json_nodes: List[str] = data["nodes"]
        for i, json_node_str in json_nodes:
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


def main():
    json_data_path = sys.argv[1]
    datas: List[Dict[str, Union[str, List[str]]]] = json.load(open(json_data_path, 'r', encoding='utf-8'))
    group_datas: DefaultDict[str, List[Dict[str, Union[str, List[str]]]]] = group_by_testcase(datas)
    print()

if __name__ == '__main__':
    main()