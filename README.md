# Consensus Network (ConNet)

Paper: [Learning to Contextually Aggregate Multi-Source Supervision for Sequence Labeling](https://arxiv.org/abs/1910.04289)

This repository contains the implementation of ConNet described in the paper.

TL;DR: Consensus Network (ConNet) conducts training with imperfect annotations from multiple sources. We evaluate the proposed framework in two practical settings of multi-source learning: learning with crowd annotations and unsupervised cross-domain model adaptation. 

![Overview of ConNet](https://github.com/INK-USC/ConNet/blob/master/images/overview.png)

Sequence labeling is a fundamental framework for various natural language processing problems including part-of-speech tagging and named entity recognition. Its performance is largely influenced by the annotation quality and quantity in supervised learning scenarios. In many cases, ground truth labels are costly and time-consuming to collect or even non-existent, while imperfect ones could be easily accessed or transferred from different domains. A typical example is crowd-sourced datasets which have multiple annotations for each sentence which may be noisy or incomplete. Additionally, predictions from multiple source models in transfer learning can be seen as a case of multi-source supervision. In this paper, we propose a novel framework named Consensus Network (ConNet) to conduct training with imperfect annotations from multiple sources. It learns the representation for every weak supervision source and dynamically aggregates them by a context-aware attention mechanism. Finally, it leads to a model reflecting the consensus among multiple sources. We evaluate the proposed framework in two practical settings of multi-source learning: learning with crowd annotations and unsupervised cross-domain model adaptation. Extensive experimental results show that our model achieves significant improvements over existing methods in both settings.

If you make use of this code or the RE-Net algorithm in your work, please cite the following paper:
```
@misc{lan2019learning,
    title={Learning to Contextually Aggregate Multi-Source Supervision for Sequence Labeling},
    author={Ouyu Lan and Xiao Huang and Bill Yuchen Lin and He Jiang and Liyuan Liu and Xiang Ren},
    year={2019},
    eprint={1910.04289},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## Quick Links
* [Dependency](##Dependency)
* [Datasets](##Datasets)
* [Training/Evaluation](##Training/Evaluation)

## Dependency

Our package is based on Python 3.6 and the following packages:
```
numpy
tqdm
torch-scope
torch==0.4.1
```

## Datasets
### Learning with crowd annotations

### Unsupervised cross-domain model adaptation
* [Universal Dependencies - GUM (Zeldes, 2017)](https://github.com/INK-USC/ConNet/tree/master/Cross_Domain/data/ud-treebanks-v2.3/UD_English-GUM) - included
* [OntoNotes-5.0](https://github.com/yuchenlin/OntoNotes-5.0-NER-BIO/releases)

## Training/Evaluation
### Learning with crowd annotations

* Generate the word dictionary by:
```
cd crowdsourcing
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

### Unsupervised cross-domain model adaptation

* Preprocess data by: 
```
cd crossdomain
sh scripts/submit_read_data.sh
```

* Train and evaluate on UD-GUM by running commands in
```
sh scripts/submit_train_ud.sh
```

* Train and evaluate on OntoNotes by running commands in
```
sh scripts/submit_train_on.sh
```


