# Vanilla NER

This project is drivied from [VanillaNER](https://github.com/LiyuanLucasLiu/Vanilla_NER), and provides a vanilla Char-LSTM-CRF model for Named Entity Recognition (train vanilla NER models w. pre-trained embedding). 


## Training

### Dependency

Our package is based on Python 3.6 and the following packages:
```
numpy
tqdm
torch-scope
torch==0.4.1
```

### Pre-processing

Please first generate the word dictionary by:
```
python pre_seq/gene_map.py -h
```

Then encode the dictionary by:
```
python pre_seq/encode_data.py -h
```

### Training

Extraction Phase:
```
./scripts/train_extraction.sh
```

Aggregation Phase:
```
./scripts/train_aggregation.sh
```

## Inference

```
./scripts/test_extraction.sh
```

## Citation

[Efficient Contextualized Representation: Language Model Pruning for Sequence Labeling](https://arxiv.org/abs/1804.07827)
```
@inproceedings{liu2018efficient,
  title = "{Efficient Contextualized Representation: Language Model Pruning for Sequence Labeling}", 
  author = {Liu, Liyuan and Ren, Xiang and Shang, Jingbo and Peng, Jian and Han, Jiawei}, 
  booktitle = {EMNLP}, 
  year = 2018, 
}
```
