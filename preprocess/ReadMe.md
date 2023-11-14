
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
We will provide another pipeline driven by cpg.

For SVF dumped cpg, we follow [DeepWuKong](https://github.com/jumormt/DeepWukong) by directly [download](https://bupteducn-my.sharepoint.com/:u:/g/personal/jackiecheng_bupt_edu_cn/EalnVAYC8zZDgwhPmGJ034cBYNZ8zB7-mNSNm-a7oYXkcw?e=eRUc50) their xfg.

Steps:

1.For source codes in a directory, for example `xxx/cwe125-source-code`, run `joern-parse outputDirectory xxx/cwe125-source-code` to get the csv files.