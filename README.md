# ConNet

This project is for ``Learning to Contextually Aggregate Multi-Source Supervision for Sequence Labeling".

## Dependency

Our package is based on Python 3.6 and the following packages:
```
numpy
tqdm
torch-scope
torch==0.4.1
```

## Crowd-Sourcing

* Generate the word dictionary by:
```
python pre_seq/gene_map.py -h
```

* Encode the dictionary by:
```
python pre_seq/encode_data.py -h
```

* Train/Test the decoupling phase by:
```
python train_seq_decoupling.py -h
python test_seq_decoupling.py -h
```

* Train/Test the aggregation phase by:
```
python train_seq_aggregation.py -h
python test_seq_aggregation.py -h
```

## Cross-Domain

* Universal Dependencies - GUM (Zeldes, 2017) dataset is included.
* Download OntoNotes-5.0 dataset from https://github.com/yuchenlin/OntoNotes-5.0-NER-BIO/releases

* Preprocess data by: 
```
sh Cross_Domain/scripts/submit_read_data.sh
```

* Train and evaluate on UD-GUM by running commands in
```
Cross_Domain/scripts/submit_train_ud.sh
```

* Train and evaluate on OntoNotes by running commands in
```
Cross_Domain/scripts/submit_train_on.sh
```


