
We need to preprocess the dataset first, our objective is to extract code snippets into a json file, each element is a code snippet with labeled vul line.

# 1.Extract Vulnerable line

In sard, vulnerable line are labeled in xml file. `extract_label.py` can be used to extract labeled vul line in json format.

For example, use `python extract_label.py xxx/cwe119-source-code xxx/label.json` command to dump all vulnerable line info into label.json.

The line info is like, each element denotes a file with corresponding vulnerable line:

```json
{
  "119-4900-c/testcases/000/077/704/CWE127_Buffer_Underread__CWE839_rand_15.c": [
    47
  ],
  "119-4900-c/testcases/000/077/705/CWE127_Buffer_Underread__CWE839_rand_16.c": [
    41
  ]
}
```

# 2.Parsing Source files

Our main pipeline for extract code graphs is done by [Joern](https://github.com/joernio/joern).
specifically, we use old version of [Joern](https://github.com/octopus-platform/joern) because we find the new version is too hard to use, the old version will dump csv files, so our preprocess is mainly parsing the csv files.

Also, we recently notice another tool [cpg](https://github.com/Fraunhofer-AISEC/cpg), we found it much easier to use than Joern. 
It is based on eclipse CDT to parse, produce less syntax error than old version of Joern.
It can also produce inter-procedural data-flow graph in a single file.
We provide another parsing tool based-on cpg named [CodeGraphAnalyzer](https://github.com/for-just-we/CodeGraphAnalyzer).

## 2.1.graph data generation

For SVF dumped cpg, we follow [DeepWuKong](https://github.com/jumormt/DeepWukong) by directly [download](https://bupteducn-my.sharepoint.com/:u:/g/personal/jackiecheng_bupt_edu_cn/EalnVAYC8zZDgwhPmGJ034cBYNZ8zB7-mNSNm-a7oYXkcw?e=eRUc50) their xfg.

Steps:

1.For source codes in a directory, for example `xxx/cwe125-source-code`, run `joern-parse outputDirectory xxx/cwe125-source-code` to get the csv files.

2.Run `extract_func_graph.py` under `graph_gen`, `python extract_func_graph.py outputDirectory output_json_file` to dump all graph data of functions into a json file.

- `output_json_file` is path to the output_file

- `outputDirectory` is root path of all csv files produced by step 1.

The content of json format function data is like, we transform the `dict` and `list` value into `str` to avoid it been split into multiple lines for better view:

```json
{
    "fileName": "CWE126_Buffer_Overread__CWE129_connect_socket_01.c",
    "functionName": "goodG2B",
    "nodes": [
      "{\"line\": 127, \"edges\": [[0, 1], [1, 2], [1, 3]], \"contents\": [[\"IdentifierDeclStatement\", \"int data ;\"], [\"IdentifierDecl\", \"data\"], [\"IdentifierDeclType\", \"int\"], [\"Identifier\", \"data\"]]}",
      "{\"line\": 129, \"edges\": [[0, 1], [1, 2], [1, 3], [3, 4], [3, 5]], \"contents\": [[\"ExpressionStatement\", \"data = - 1\"], [\"AssignmentExpression\", \"data = - 1\"], [\"Identifier\", \"data\"], [\"UnaryOperationExpression\", \"- 1\"], [\"UnaryOperator\", \"-\"], [\"PrimaryExpression\", \"1\"]]}",
      "{\"line\": 132, \"edges\": [[0, 1], [1, 2], [1, 3]], \"contents\": [[\"ExpressionStatement\", \"data = 7\"], [\"AssignmentExpression\", \"data = 7\"], [\"Identifier\", \"data\"], [\"PrimaryExpression\", \"7\"]]}",
      "{\"line\": 134, \"edges\": [[0, 1], [1, 2], [1, 3], [1, 4], [1, 5], [5, 6], [5, 7], [7, 8]], \"contents\": [[\"IdentifierDeclStatement\", \"int buffer [ 10 ] = { 0 } ;\"], [\"IdentifierDecl\", \"buffer [ 10 ] = { 0 }\"], [\"IdentifierDeclType\", \"int [ 10 ]\"], [\"Identifier\", \"buffer\"], [\"PrimaryExpression\", \"10\"], [\"AssignmentExpression\", \"buffer [ 10 ] = { 0 }\"], [\"Identifier\", \"buffer\"], [\"InitializerList\", \"0\"], [\"PrimaryExpression\", \"0\"]]}",
      "{\"line\": 137, \"edges\": [[0, 1], [1, 2], [1, 3]], \"contents\": [[\"Condition\", \"data >= 0\"], [\"RelationalExpression\", \"data >= 0\"], [\"Identifier\", \"data\"], [\"PrimaryExpression\", \"0\"]]}",
      "{\"line\": 139, \"edges\": [[0, 1], [1, 2], [2, 3], [1, 4], [4, 5], [5, 6], [6, 7], [6, 8]], \"contents\": [[\"ExpressionStatement\", \"printIntLine ( buffer [ data ] )\"], [\"CallExpression\", \"printIntLine ( buffer [ data ] )\"], [\"Callee\", \"printIntLine\"], [\"Identifier\", \"printIntLine\"], [\"ArgumentList\", \"buffer [ data ]\"], [\"Argument\", \"buffer [ data ]\"], [\"ArrayIndexing\", \"buffer [ data ]\"], [\"Identifier\", \"buffer\"], [\"Identifier\", \"data\"]]}",
      "{\"line\": 143, \"edges\": [[0, 1], [1, 2], [2, 3], [1, 4], [4, 5], [5, 6]], \"contents\": [[\"ExpressionStatement\", \"printLine ( \\\"ERROR: Array index is negative\\\" )\"], [\"CallExpression\", \"printLine ( \\\"ERROR: Array index is negative\\\" )\"], [\"Callee\", \"printLine\"], [\"Identifier\", \"printLine\"], [\"ArgumentList\", \"\\\"ERROR: Array index is negative\\\"\"], [\"Argument\", \"\\\"ERROR: Array index is negative\\\"\"], [\"PrimaryExpression\", \"\\\"ERROR: Array index is negative\\\"\"]]}"
    ],
    "cfgEdges": [
      "[0, 1]",
      "[1, 2]",
      "[2, 3]",
      "[3, 4]",
      "[4, 5]",
      "[4, 6]"
    ],
    "cdgEdges": [
      "[4, 5]",
      "[4, 6]"
    ],
    "ddgEdges": [
      "[2, 4]",
      "[2, 5]",
      "[3, 5]"
    ],
    "testcase-path": "125-c/testcases/000/075/598/CWE126_Buffer_Overread__CWE129_connect_socket_01.c"
}
```

## 2.2.Sequence representation

program_slice: run `python program_slice.py <input_json_file> <output_json_file>`. Where `<input_json_file>` is the json parsed from Joern, `<output_json_file>` is the dumped slices.



# 3.label and deduplication datas

Use `label_graphs.py` in dataset_process. Run `python label_graphs.py level info_file_path output_json_file output_json_dir`. Here:

- `level` is `function` or `slice`. Meaning labeling for function datas or slice datas.

- `info_file_path` is the `label.json` in section `Extract Vulnerable line`.

- `output_json_file` is the graph json in section `Parsing Source files`.

- `output_json_dir` is the dir for storing labeled datas. It will store `train_vul.json, test_vul.json, eval_vul.json, train_normal.json, test_normal.json, eval_normal.json`

