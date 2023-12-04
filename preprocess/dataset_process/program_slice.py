import json
import os
import sys
from typing import List, Dict, Union, DefaultDict, Set
from collections import defaultdict

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


def main():
    json_data_path = sys.argv[1]
    datas: List[Dict[str, Union[str, List[str]]]] = json.load(open(json_data_path, 'r', encoding='utf-8'))
    group_datas: DefaultDict[str, List[Dict[str, Union[str, List[str]]]]] = group_by_testcase(datas)
    print()

if __name__ == '__main__':
    main()