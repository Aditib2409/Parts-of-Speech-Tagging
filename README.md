# Parts-of-Speech-Tagging
This project gives hands-on experience on using HMMs on parts-of-speech tagging. 

## Dataset
Wall Street Journal section of the Penn Treebank. In the folder named `data`, there are three files: `train`, `dev`, `test`. In the files of `train` and `dev`, the sentences have human-annotated part-of-speech tags. In the file `test`, has raw sentences that you need to predict the part-of-speech tags for. The data format is that each line contains three items separated by the tab symbol ‘\t’. The first item is the index of the word in the sentence. The second item is the word type and the third item is the corresponding part-of-speech tag. There will be a blank line at the end of one sentence.

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

## About the Repo
A txt file named `vocab.txt`, contains the vocabulary created on the training data. The format of the vocabulary file is that each line contains a word type, its index, and its occurrences, separated by the tab symbol ‘\t’. (see task 1).
2. A JSON file named `hmm.json`, containing the emission and transition probabilities (see task 2).
3. Two prediction files named `greedy.out` and `viterbi.out`, contain the predictions of your model on the test data with the greedy and Viterbi decoding algorithms.

## Results and Observations
Selected Threshold for unknown words replacement = 3 Total size of the vocabulary before ‘’ replacement = 43193 Total size of the vocabulary after ‘’ replacement = 16919 Total occurrences of ‘’ after replacement = 26274
1.2 Task 2 -
Computed the transition matrix and the Emission matrices and stored the resultant parameters into a dictionary.
Number of Transition Parameters in the HMM model = 1389 Number of Emission Parameters in the HMM model = 23373
1.3 Task 3 -
1.3.1 Greedy Algorithm
Using the above computed matrices, calculated the POS tags using greedy algorithm as discussed in the lectures Accuracy on the dev data = 92.82%
Predicted on the test set and recorded the resultant predictions in the greedy.out file
1.4 Task 4 -
1.4.1 Viterbi Algorithm
Used the above computed transition and emission matrices, and calculated the POS tags taking each sentence at once and computing multiplications for each word within the sentence. This algorithm resulted in a better, improved accuracy on the test set. Accuracy on the dev data = 94.37%


  



