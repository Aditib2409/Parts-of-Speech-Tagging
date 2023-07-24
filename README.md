# Parts-of-Speech-Tagging
This project gives hands-on experience on using HMMs on parts-of-speech tagging. 

## Dataset
Wall Street Journal section of the Penn Treebank. In the folder named `data`, there are three files: `train`, `dev`, `test`. In the files of `train` and `dev`, the sentences have human-annotated part-of-speech tags. In the file `test`, has raw sentences that you need to predict the part-of-speech tags for. The data format is that, each line contains three items separated by the tab symbol ‘\t’. The first item is the index of the word in the sentence. The second item is the word type and the third item is the corresponding part-of-speech tag. There will be a blank line at the end of one sentence.
