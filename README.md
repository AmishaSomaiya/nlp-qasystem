# Question Answering (QA) System using NLP with SQuAD
## Repository Structure
```
.
├── EE562_Group3_ProjectReport.pdf
├── README.md
├── distilled_bert
│   └── Question-Answering-SQUAD-main
│       ├── data
│       │   ├── newgelu_activation_3epochs
│       │   │   ├── config.json
│       │   │   ├── special_tokens_map.json
│       │   │   ├── tokenizer.json
│       │   │   ├── tokenizer_config.json
│       │   │   ├── training_args.bin
│       │   │   └── vocab.txt
│       │   ├── oldgelu_activation_3epochs
│       │   │   ├── config.json
│       │   │   ├── special_tokens_map.json
│       │   │   ├── tokenizer.json
│       │   │   ├── tokenizer_config.json
│       │   │   ├── training_args.bin
│       │   │   └── vocab.txt
│       │   ├── predictions.json
│       │   ├── test.json
│       │   ├── test_set.json
│       │   ├── train.json
│       │   ├── training_set.json
│       │   └── val.json
│       ├── results
│       │   ├── 0.1WD.png
│       │   ├── 7epoch.png
│       │   ├── BS_8.png
│       │   ├── DefaultLR_32BS_0.01WD.png
│       │   ├── LR2e-5_32BS_0.01WD.png
│       │   ├── gelu_3epochs.png
│       │   ├── gelu_new_3epochs.png
│       │   ├── lr_5e-04_bs_32_decay_0.01.png
│       │   ├── lr_5e-05_bs32_decay_0.001.png
│       │   ├── lr_5e-05_bs_16_decay_0.01.png
│       │   ├── training_gelu.png
│       │   └── training_gelu_new.png
│       └── src
│           ├── Distilbert_Eval.ipynb
│           └── DistillBert_Train.ipynb
├── multinomial_and_randomforest
│   ├── InferSent
│   │   ├── README.md
│   │   ├── extract_features.py
│   │   ├── models.py
│   │   └── samples.txt
│   ├── __pycache__
│   │   └── models.cpython-310.pyc
│   ├── data
│   │   └── train-v1.1.json
│   ├── results
│   │   ├── multinomial.png
│   │   └── randomforest.png
│   └── src
│       ├── Traditional_ml_model.ipynb
│       ├── models.py
│       └── preprocessing.py
└── pretrained_bert
    ├── BERT_Pretrained.ipynb
```
14 directories, 46 files
