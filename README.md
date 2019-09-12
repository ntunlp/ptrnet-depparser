# ptrnet-depparser
This is the source code of our depedency parser proposed in paper "Hierarchical Pointer Net Parsing" accepted by EMNLP 2019.
Git Repository: https://github.com/ntunlp/ptrnet-depparser.git

# Requirements
Python 2.7, PyTorch >=0.3.0, Gensim >= 0.12.0

# Models
We have implemented the below models in this project, which can be found in ./neuronlp2/models/parsing2.py:
- HPtrNetPSTGate: In each step, decoder receives hidden states from sibling, parent and previous step. Use Gate described in the paper.
- HPtrNetPSTSGate: In each step, decoder receives hidden states from sibling, parent and previous step. Use SGate described in the paper.
- HPtrNetPSGate: In each step, decoder receives hidden states from sibling and parent. Use Gate described in the paper.

# Data Format
For CoNLL-x format, the schema is:
ID, FORM, LEMMA, CPOSTAG, POSTAG, MORPH-FEATURES, HEAD, DEPREL, PHEAD, PDEPREL

# Running Experiments
1. Update ./examples/run_HPtrNetParser.sh to select the model you want to test, for example `MODELNAME=HPtrNetPSTGate`.
2. Run command `bash ./examples/run_HPtrNetParser.sh`.

