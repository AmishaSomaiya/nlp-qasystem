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


This project builds a closed-domain, extractive Question Answering (QA) System using NLP with SQuAD. The system takes 2 inputs : the passage which can be one or more paragraphs and the query which is the question posed by the user. The output is the response to the question, which can be 1 word or longer. It is a substring of the passage which is most contextually relevant to the question based on the calculated similarity score. This system is closed-domain which means the answer to the question is expected within the passage itself and incase the user poses an open-domain question, the system will still respond with an answer with the highest similarity score, but in such cases the answer may not always be correct. Examples of this are specified in the Experimental Results section. The system is extractive and factoid which means the system takes the passage as a fact and responds with the substring extracted from the passage itself based on contextual understanding. It does not generate abstractive generative responses outside of the passage. 

There are two major sub-tasks for this system, one to develop highly efficient contextual understanding of the passage and second, to find the best substring from the passage most contextually relevant to the query by calculating similarity scores between the substrings and the query. This project builds this system using 4 approaches : 2 traditional ML baseline and 2 DL approaches. The 2 traditional ML approaches are unable to correctly comprehend the complex linguistic patterns and are unable to correctly capture the contextual understanding of words and phrases leading to low scores. Thus, Multinomial Logistic Regression achieved baseline F1 : 22.42 and Random Forest F1 : 30.24, both of which are not sufficient. This motivates us to look at DL approaches such as BERT. BERT is based on the Transformer architecture, and hence it can understand context from entire input sequence and not just a fixed window size. Also, BERT is pre-trained on a large corpus using Unsupervised Learning which helps it understand complex contextual patterns easily. The third model implementation uses pre-trained BERT achieves F1 : 88.7 and EM : 85.4. This was implemented mainly for comparison to the final implementation.

The final project implementation is ‘DistilBERT backbone + additional head’. This implementation retains the first few layers of DistilBERT backbone and thus has a high contextual understanding of words, phrases and patterns resulting in very high performance as compared to traditional ML baselines. The model is further appended with an additional head with 3 layers and then trained specifically for QA task by which it achieves F1 : 84.56 and EM : 75.846 which is comparable to large model like BERT in just 3 epochs while at the same time having far lesser parameters, light weight and computationally efficient. 

##References 

[1] “Sklearn.linear model.logisticregression,” scikit, https://scikit-learn.org/
stable/modules/generated/sklearn.linear_model.LogisticRegression.
html (accessed Dec. 12, 2023).

[2] “Sklearn.ensemble.randomforestclassifier,” scikit, https://scikit-learn.org/
stable/modules/generated/sklearn.ensemble.RandomForestClassifier.
html (accessed Dec. 12, 2023).

[3] J. Devlin, BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,
Oct. 2018. doi: https://doi.org/10.48550/arXiv.1810.04805

[4] Vaswani, A et al. (2017) Attention is All You Need, Jun 2017. doi:
https://doi.org/10.48550/arXiv.1706.03762.

[5] V. Sanh, DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter, Oct. 2019.
doi:https://doi.org/10.48550/arXiv.1910.01108

[6] “Squad,” The Stanford Question Answering Dataset, https://rajpurkar.github.io/SQuADexplorer/ (accessed Dec. 12, 2023).

[7] “Simplified text processing,” TextBlob, https://textblob.readthedocs.io/en/dev/ (accessed Dec.
12, 2023).

[8] NLTK, https://www.nltk.org/api/nltk.tokenize.punkt.html (accessed Dec. 12, 2023).

[9] Facebookresearch, “Facebookresearch/infersent: Infersent sentence embeddings,” GitHub,
https://github.com/facebookresearch/InferSent (accessed Dec. 12, 2023).

[10] J. Pennington, Glove: Global vectors for word representation,
https://nlp.stanford.edu/projects/glove/ (accessed Dec. 12, 2023).

[11] “F1 score in Machine Learning: Intro & Calculation,” V7, https://www.v7labs.com/blog/f1-
score-guide (accessed Dec. 12, 2023).

[12] “Exact match - a hugging face space by evaluate-metric,” Hugging Face,
https://huggingface.co/spaces/evaluate-metric/exact match (accessed Dec. 12, 2023).

[13] “Models,” Hugging Face, https://huggingface.co/models (accessed Dec. 12, 2023).

[14] “Nlpunibo,” Hugging Face, https://huggingface.co/nlpunibo (accessed Dec. 12, 2023).

[15] “Transformers Model Doc,” Hugging Face, https://huggingface.co/docs/
transformers/model_doc/auto (accessed Dec. 12, 2023).

[16] “Transformers Main Classes,” Hugging Face, https://huggingface.co/docs/
transformers/main_classes/data_collator (accessed Dec. 12, 2023).

[17] Mgreenbe, “Squad/bert: Why max length=384 by default and not” Hugging Face Forums, https://discuss.huggingface.co/t/squad-bert-why-max-length-384-by-default-and-not-512/11693 (accessed Dec. 12, 2023).

