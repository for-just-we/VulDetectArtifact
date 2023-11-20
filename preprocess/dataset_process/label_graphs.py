import json
import os
import sys
import random
from tqdm import tqdm
from typing import Dict, List, Union
import hashlib


def getMD5(s: str):
    '''
    得到字符串s的md5加密后的值

    :param s:
    :return:
    '''
    hl = hashlib.md5()
    hl.update(s.encode("utf-8"))
    return hl.hexdigest()

def get_func_md5(cpg_graph: Dict[str, Union[str, List[str], List[int]]]):
    line_contents: List[str] = [
        json.loads(json_node)["contents"][0][1] for json_node in cpg_graph["nodes"]
    ]
    md5_value = getMD5(json.dumps(line_contents))
    return md5_value

def process_func_datas(label_info: Dict[str, List[int]],
                  cpg_graphs: List[Dict[str, Union[str, List[str], List[int]]]]):
    vul_samples: List[Dict[str, Union[str, List[str]]]] = list()
    normal_samples: List[Dict[str, Union[str, List[str]]]] = list()

    vul_md5_set = set()
    normal_md5_set = set()

    # 标注每一个graph
    for cpg_graph in tqdm(cpg_graphs, desc="labeling function graphs"):
        testcase_path = cpg_graph["testcase-path"]
        md5_value = get_func_md5(cpg_graph)
        # vulnerable line中没有这个文件的标注，则是白样本
        if testcase_path not in label_info.keys():
            if md5_value not in normal_md5_set:
                normal_md5_set.add(md5_value)
                normal_samples.append(cpg_graph)
            continue
        # 漏洞行号
        vul_line_nos: List[int] = label_info[testcase_path]
        line_nos: List[int] = [json.loads(json_node)["line"] for json_node in cpg_graph["nodes"]]
        vul_idxs: List[int] = list()
        for i, line_no in enumerate(line_nos):
            if line_no in vul_line_nos:
                vul_idxs.append(i)

        # 如果这段代码片段不包含漏洞行
        if len(vul_idxs) == 0:
            if md5_value not in normal_md5_set:
                normal_md5_set.add(md5_value)
                normal_samples.append(cpg_graph)
        # 漏洞函数
        else:
            if md5_value not in vul_md5_set:
                vul_md5_set.add(md5_value)
                cpg_graph["tri_line_idxs"] = vul_idxs
                vul_samples.append(cpg_graph)

    # 处理黑白样本交叉部分，
    both_md5_set = vul_md5_set & normal_md5_set
    final_normal_samples = list()
    for cpg_graph in tqdm(normal_samples, desc="removing noise normal samples"):
        md5_value = get_func_md5(cpg_graph)
        if md5_value in both_md5_set:
            continue
        final_normal_samples.append(cpg_graph)

    print("labeling graphs done. vulnerable function num: {}, "
                 "normal function num: {}".format(len(vul_samples), len(final_normal_samples)))
    return vul_samples, final_normal_samples


def dump_to_output_dir(vul_samples: List[Dict], normal_samples: List[Dict], output_json_dir: str):
    random.shuffle(vul_samples)
    random.shuffle(normal_samples)

    train_vul_idx = int(0.8 * len(vul_samples))
    test_vul_idx = int(0.9 * len(vul_samples))

    train_normal_idx = int(0.8 * len(normal_samples))
    test_normal_idx = int(0.9 * len(normal_samples))

    # train_vul, test_vul, eval_vul, train_normal, test_normal, eval_normal
    output_datas: List[List[Dict]] = [
        vul_samples[:train_vul_idx], vul_samples[train_vul_idx: test_vul_idx], vul_samples[test_vul_idx:],
        normal_samples[:train_normal_idx], normal_samples[train_normal_idx: test_normal_idx], normal_samples[test_normal_idx:]
    ]

    output_file_names = [
        "train_vul.json", "test_vul.json", "eval_vul.json",
        "train_normal.json", "test_normal.json", "eval_normal.json"
    ]

    for datas, file_name in zip(output_datas, output_file_names):
        json.dump(datas, open(os.path.join(output_json_dir, file_name), 'w', encoding='utf-8'), indent=2)


if __name__ == '__main__':
    level = sys.argv[1]
    label_file = sys.argv[2]
    json_data_file = sys.argv[3]
    output_dir = sys.argv[4]
    label_info: Dict[str, List[int]] = json.load(open(label_file, 'r', encoding='utf-8'))
    if level == "function":
        cpg_graphs: List[Dict[str, Union[str, List[str], List[int]]]] = \
            json.load(open(json_data_file, 'r', encoding='utf-8'))
        vul_samples, normal_samples = process_func_datas(label_info, cpg_graphs)
        dump_to_output_dir(vul_samples, normal_samples, output_dir)