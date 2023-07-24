# Parts-of-Speech-Tagging
This project gives hands-on experience on using HMMs on parts-of-speech tagging. 

## Dataset
Wall Street Journal section of the Penn Treebank. In the folder named `data`, there are three files: `train`, `dev`, `test`. In the files of `train` and `dev`, the sentences have human-annotated part-of-speech tags. In the file `test`, has raw sentences that you need to predict the part-of-speech tags for. The data format is that, each line contains three items separated by the tab symbol ‘\t’. The first item is the index of the word in the sentence. The second item is the word type and the third item is the corresponding part-of-speech tag. There will be a blank line at the end of one sentence.

## Tasks
The project is divided into 4 tasks:

### Task-1 Vocabulary Creation
The first task is to create a vocabulary using the training data. In HMM, one important problem when creating the vocabulary is to handle unknown words. One simple solution is to replace rare words whose occurrences are less than a threshold (e.g. 3) with a special token ‘< unk >’.

### Task-2 Model Learning
The second task is to learn an HMM from the training data. Remember that the solution of the emission and transition parameters in HMM is in the
the following formulation: 

t(s′|s) = count(s→s′ )/count(s)

e(x|s) = count(s→x)/count(s)

where t(·|·) is the transition parameter and e(·|·) is the emission parameter.

### Taks-3 Greedy Decoding with HMM

### Taks-4 Viterbi Decoding with HMM

  



