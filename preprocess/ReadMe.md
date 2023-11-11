
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



