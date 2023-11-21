# VulDetectArtifact
 Artifact for TOSEM. (Note: we are still organizing our code. We are gradually releasing code. ) 


# Datasets

For SARD dataset we have uploaded to [zenodo](https://zenodo.org/records/10088191), for Fan dataset, the related information is at [MSR_20_Code_vulnerability_CSV_Dataset](https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset), the dataset csv can be downloaded from [google driver](https://drive.google.com/file/d/1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X/view?usp=sharing). We extract `func_before` and `func_after` from it.

# Preprocess Pipeline

For preprocess code into graph, please refer to [preprocess/ReadMe.md](preprocess/ReadMe.md)

# Pretrain embedding model

Run `python pretrain.py detector_name path2train_datas embedding_model_path`

- `detector_name`: The name of detectors, choice is `reveal`, `devign`, `ivdetect`, `deepwukong`, we will soon add remaining 3 sequence-based detectors into this pipeline.

- `path2train_datas`: The dir which stores `train_vul.json`, `train_normal.json`, `eval_vul.json`, `eval_normal.json`, `test_vul.json`, `test_normal.json`, the script will read training data from train jsons.

- `embedding_model_path`: The path to the saved embedding model.

# Detection Pipeline
