# VulDetectArtifact
 Artifact for TOSEM paper: [Beyond Fidelity: Explaining Vulnerability Localization of Learning-based Detectors](https://arxiv.org/abs/2401.02686).

# 1.Datasets

For SARD dataset we have uploaded to [zenodo](https://zenodo.org/records/10088191), for Fan dataset, the related information is at [MSR_20_Code_vulnerability_CSV_Dataset](https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset), the dataset csv can be downloaded from [google driver](https://drive.google.com/file/d/1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X/view?usp=sharing). We extract `func_before` and `func_after` from it.

# 2.Preprocess Pipeline

For preprocess code into graph, please refer to [preprocess/ReadMe.md](preprocess/ReadMe.md)

# 3.Pretrain embedding model

Run `python pretrain.py detector_name path2train_datas embedding_model_path`

- `detector_name`: The name of detectors, choice is `reveal`, `devign`, `ivdetect`, `deepwukong`, we will soon add remaining 3 sequence-based detectors into this pipeline.

- `path2train_datas`: The dir which stores `train_vul.json`, `train_normal.json`, `eval_vul.json`, `eval_normal.json`, `test_vul.json`, `test_normal.json`, the script will read training data from train jsons.

- `embedding_model_path`: The path to the saved embedding model.

# 4.Detection Pipeline

Run `python detection.py <args>` to train detectors. `<args>` includes:

- `--detector <detector_name>`, `<detector_name>` could be one of `["deepwukong", "reveal", "ivdetect", "devign", "tokenlstm", "vuldeepecker", "sysevr"]`

- `--w2v_model_path <model_path>`, `<model_path>` could be relative or absolute path of pretrained word2vec model.

- `--dataset_dir <dataset_dir>`, `<dataset_dir>` is path to the dir storing json datas. It should include `train_vul.json`, `train_normal.json`, `eval_vul.json`, `eval_normal.json`, `test_vul.json`, `test_normal.json`.

- `--model_dir <model_dir>`, `<model_dir>` is where the model pth file placed, it's corresponding directory. The scripts will automatically load the best model in the dir.

- `--train`, means will train model. If there exist a model in `<model_dir>`, the script will first load that model and then train.

- `--test`, means will test the model. There must be a model in `<model_dir>` first.

# 5.Explanation Pipeline

Run `python explain.py <args>`. `<args> includes`:

- `--detector <detector_name>`, `<detector_name>` could be one of `["deepwukong", "reveal", "ivdetect", "devign", "tokenlstm", "vuldeepecker", "sysevr"]`

- `--w2v_model_path <model_path>`, `<model_path>` could be relative or absolute path of pretrained word2vec model.

- `--dataset_dir <dataset_dir>`, `<dataset_dir>` is path to the dir storing json datas. It should include `test_vul.json`.

- `--model_dir <model_dir>`, `<model_dir>` is where the model pth file placed, it's corresponding directory. The scripts will automatically load the best model in the dir.

- `--explainer <explainer_name>`, `<explainer_name>` could be one of `["gnnexplainer", "pgexplainer", "gnnlrp", "gradcam", "deeplift"]` for now. We are organizing the code in sequence-based explainers into this pipeline.


# 6.Citation

```
@misc{cheng2024fidelity,
      title={Beyond Fidelity: Explaining Vulnerability Localization of Learning-based Detectors}, 
      author={Baijun Cheng and Shengming Zhao and Kailong Wang and Meizhen Wang and Guangdong Bai and Ruitao Feng and Yao Guo and Lei Ma and Haoyu Wang},
      year={2024},
      eprint={2401.02686},
      archivePrefix={arXiv},
      primaryClass={cs.CR}
}
```