import json
from tqdm import tqdm
import sys
from typing import List, Dict, Union, DefaultDict, Set, Tuple
from collections import defaultdict
from utils.ast_def import ASTNode, json2astNode
from utils.ast_analyzer import CallTargetChecker, SyVCChecker
from symbolizing_tool import SymbolizingTool

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


class SySeSlice(object):
    def __init__(self, symbolized_lines: List[str], loc_idxs: List[str]):
        # slice覆盖到的行号以及每一行所在的文件
        self.lineNumbers: List[List[int]] = list()
        # slice中每个语句对应的token序列
        self.lineContents: List[str] = symbolized_lines
        # 文件id对文件名
        self.id2file: Dict[int, str] = dict()
        self.build(loc_idxs)

    def build(self, loc_idxs: List[str]):
        testcase_paths2idx: Dict[str, int] = dict()
        for loc_idx in loc_idxs:
            testcase_path, line = loc_idx.split('=')
            line = int(line)
            if testcase_path not in testcase_paths2idx.keys():
                idx = len(testcase_paths2idx)
                self.id2file[idx] = testcase_path
                testcase_paths2idx[testcase_path] = idx
            else:
                idx = testcase_paths2idx[testcase_path]
            self.lineNumbers.append([idx, line])


    def __hash__(self):
        return hash(json.dumps(self.lineContents))

    def toJson(self) -> Dict:
        return {
            "id2file": self.id2file,
            "line-Nos": self.lineNumbers,
            "line-contents": self.lineContents
        }

systemDefinedVars: Set[str] = {"argc", "argv", "stdin", "stdout", "cin", "cout", "SOCKET_ERROR"}

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
        # func_name --> callers, line idx
        self.func_name2callers: DefaultDict[str, Set[Tuple[str, int]]] = defaultdict(set)
        self.key_line_set: DefaultDict[str, Set[int]] = defaultdict(set)
        self.file_name2idx: Dict[str, int] = dict()

        for data in datas:
            self.build_key_query_structure(data)

        self.symbolizing_tool = SymbolizingTool(systemDefinedVars, self.func_names)
        self.symbolizing_tool.getVarFuncNamesInFile(list(self.func_name2data.values()))
        self.syses: Set[SySeSlice] = set()

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
                self.func_name2callers[called_func_name].add((func_name, i))

    def run_slice(self):
        for func_name, key_lines in self.key_line_set.items():
            # 对每个key-line进行切片
            for key_line in key_lines:
                backward_funcs = set()
                backward_slice_lines: List[str] = list()
                backward_loc_idxs: List[str] = list()

                forward_funcs = set()
                forward_slice_lines: List[str] = list()
                forward_loc_idxs: List[str] = list()

                self.extract_backward_slice(func_name, key_line, backward_slice_lines, backward_loc_idxs, backward_funcs)
                self.extract_forward_slice(func_name, {key_line}, forward_slice_lines, forward_loc_idxs, forward_funcs)

                total_slice_lines: List[str] = backward_slice_lines[:-1] + forward_slice_lines
                total_loc_idxs: List[str] = backward_loc_idxs[:-1] + forward_loc_idxs
                total_slice_lines_symbolized: List[str] = [self.symbolizing_tool.symbolize(line)
                                                           for line in total_slice_lines]

                syse_slice: SySeSlice = SySeSlice(total_slice_lines_symbolized, total_loc_idxs)
                self.syses.add(syse_slice)


    # conduct backward program slice, where func_name and line is the start of current location,
    # cur_slice_lines and cur_loc_idxs stores
    def extract_backward_slice(self, func_name: str, line: int,
                               cur_slice_lines: List[str], cur_loc_idxs: List[str], funcs: Set[str]):
        if func_name in funcs:
            return
        funcs.add(func_name)

        cur_data: Dict[str, Union[str, List[str]]] = self.func_name2data[func_name]
        #use control dependence and data dependence
        pdg_edges_text: List[str] = cur_data["cdgEdges"] + cur_data["ddgEdges"]
        pdg_edges: List[List[int]] = [json.loads(edge_text) for edge_text in pdg_edges_text]

        slices_in_cur_funcs: List[int] = [line]
        queue: List[int] = [line]
        visited: Set[int] = set()

        # BFS遍历pdg_edges
        while len(queue) > 0:
            cur_line: int = queue.pop(0)
            if cur_line in visited:
                continue
            visited.add(cur_line)
            pdg_edges_from_cur_line: List[List[int]] = list(filter(lambda edge: edge[1] == cur_line, pdg_edges))
            for edge in pdg_edges_from_cur_line:
                if edge[0] not in slices_in_cur_funcs:
                    slices_in_cur_funcs.append(edge[0])
                queue.append(edge[0])

        # slices_in_cur_funcs去重
        slices_in_cur_funcs = list(set(slices_in_cur_funcs))
        # slices_in_cur_funcs按行号排序
        slices_in_cur_funcs.sort()

        param_idxs: Set[int] = self.func_name2param_idx.get(func_name, set())
        checked_params = False
        # 遍历slices_in_cur_funcs的每一行
        for line in reversed(slices_in_cur_funcs):
            cur_line_data: dict = json.loads(cur_data["nodes"][line])
            line_content = cur_line_data['contents'][0][1]
            file_line = cur_line_data['line']
            cur_slice_lines.append(line_content)
            cur_loc_idxs.append(cur_data["testcase-path"] + "=" + str(file_line))
            # 如果追溯到param
            if line in param_idxs and not checked_params:
                checked_params = True
                callers_idxs: Set[Tuple[str, int]] = self.func_name2callers.get(func_name, set())
                for caller, idx in callers_idxs:
                    self.extract_backward_slice(caller, idx, cur_slice_lines, cur_loc_idxs, funcs)



    def extract_forward_slice(self, func_name: str, lines: Set[int],
                              cur_slice_lines: List[str], cur_loc_idxs: List[str], funcs: Set[str]):
        if func_name in funcs:
            return
        if len(lines) == 0:
            return
        funcs.add(func_name)
        cur_data: Dict[str, Union[str, List[str]]] = self.func_name2data[func_name]
        # only use data dependence
        ddg_edges_text: List[str] = cur_data["ddgEdges"]
        ddg_edges: List[List[int]] = [json.loads(edge_text) for edge_text in ddg_edges_text]

        slices_in_cur_funcs: List[int] = list(lines.copy())
        queue: List[int] = list(lines.copy())
        visited: Set[int] = set()
        # BFS遍历ddg_edges
        while len(queue) > 0:
            cur_line: int = queue.pop(0)
            if cur_line in visited:
                continue
            visited.add(cur_line)
            ddg_edges_from_cur_line: List[List[int]] = list(filter(lambda edge: edge[0] == cur_line, ddg_edges))
            for edge in ddg_edges_from_cur_line:
                if edge[1] not in slices_in_cur_funcs:
                    slices_in_cur_funcs.append(edge[1])
                queue.append(edge[1])

        # slices_in_cur_funcs去重
        slices_in_cur_funcs = list(set(slices_in_cur_funcs))
        # slices_in_cur_funcs按行号排序
        slices_in_cur_funcs.sort()

        # 遍历slices_in_cur_funcs的每一行
        for line in slices_in_cur_funcs:
            cur_line_data: dict = json.loads(cur_data["nodes"][line])
            line_content = cur_line_data['contents'][0][1]
            file_line = cur_line_data['line']
            cur_slice_lines.append(line_content)
            cur_loc_idxs.append(cur_data["testcase-path"] + "=" + str(file_line))
            # 如果调用了其它函数
            call_targets: List[str] = self.func_name2call_target_reverse.get(func_name, {}).get(line, [])
            for call_target in call_targets:
                # call target的每个参数加入lines
                param_idxs: Set[int] = self.func_name2param_idx[call_target]
                self.extract_forward_slice(call_target, param_idxs, cur_slice_lines, cur_loc_idxs, funcs)



def main():
    json_data_path = sys.argv[1]
    sensiAPIs: Set[str] = set(open("sensiAPI.txt", 'r', encoding='utf-8').read().split(','))
    datas: List[Dict[str, Union[str, List[str]]]] = json.load(open(json_data_path, 'r', encoding='utf-8'))
    group_datas: DefaultDict[str, List[Dict[str, Union[str, List[str]]]]] = group_by_testcase(datas)
    total_syses: Set[SySeSlice] = set()

    for testcase_id, test_case_datas in tqdm(group_datas.items(), desc="processing datas"):
        slicing_tool = SlicingTool(sensiAPIs, test_case_datas)
        slicing_tool.run_slice()
        total_syses.update(slicing_tool.syses)

    output_path = sys.argv[2]
    json.dump(list(total_syses), open(output_path, 'w', encoding='utf-8'), indent=2)


if __name__ == '__main__':
    main()