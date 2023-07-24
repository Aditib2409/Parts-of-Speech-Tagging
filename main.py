import pandas as pd
import numpy as np
from collections import Counter
import os
from tqdm import tqdm
import json


train_data = pd.read_csv('data/train', header=None, on_bad_lines='skip', delimiter='\t')

indices = train_data[1].value_counts()
word_types = train_data[1].value_counts().index.tolist()
unique_words = np.unique(word_types)
vocab_df = pd.DataFrame(data=word_types, columns=["Word Type"])
Indexes = np.arange(1, vocab_df.shape[0]+1)
vocab_df['Index'] = np.array(Indexes)
vocab_df['Occurrences'] = np.array(indices)
nrows = vocab_df.shape[0]
vocab_df = vocab_df[vocab_df['Occurrences'] >= 3]
new_nrows = vocab_df.shape[0]
unique_words = vocab_df["Word Type"].to_list()
unique_words.insert(0, '<unk>')
unique_words =  np.array(unique_words)

# TODO: Change unk count
unknown_counts = nrows-new_nrows

print(f'Total size of vocabulary before <unk> Replacement: {nrows}')
print(f'Total size of vocabulary after <unk> Replacement: {new_nrows}')
print(f'Total Occurrences of <unk>: {unknown_counts}')

vocab = np.array(vocab_df)
new = np.array(['<unk>', 0, unknown_counts])

new_vocab = np.vstack([new, vocab])
new_vocab_df = pd.DataFrame(data=new_vocab)

with open('vocab.txt', 'w') as f:
    new_vocab_df.to_csv(f, mode='a', index=False, header=False, sep='\t')

states = train_data[2].value_counts().index.tolist()
traindata = np.array(train_data)
states_count = []
for state in states:
    states_count.append((traindata[:,-1]==state).sum())

# print(f'no. of states: {states_count}')

with open('vocab.txt', 'r') as file:
    sentences = file.readlines()
    sentences = [sentence.rstrip('\n') for sentence in sentences]
    sentences = [sentence.split('\t') for sentence in sentences]

    # ALL_TAGS = []
    ALL_WORDS = []

    for sentence in sentences:
        single_word, d, _ = sentence
        ALL_WORDS.append(single_word)

with open('data/train', 'r') as trainfile:
    trainlines = trainfile.readlines()
    trainlines = [trainline.rstrip('\n') for trainline in trainlines]
    trainlines = [trainline.split('\t') for trainline in trainlines]

def create_transitionmatrix(S, X, SC):
    start = [None, None, 'start']
    transition_matrix = np.zeros((len(S) + 1, len(S)))
    transition = {}
    # start_indx = states.index(X[0][-1])
    # transition_matrix[0, start_indx] = 1
    row_tags= [start] + X
    col_tags = X
    for line1, line2 in zip(row_tags, col_tags):
        if len(line1) == 1:
            line1 = start
        if len(line2) == 3:
            t1 = line1[-1]
            t2 = line2[-1]
            if t1 == 'start':
                row_idx = 0
            else:
                row_idx = S.index(t1) + 1
        col_idx = S.index(t2)
        transition_matrix[row_idx, col_idx] += 1
        # transition.update({f'({t1}, {t2})': transition_matrix[row_idx+1, col_idx]})

    total_transition = np.sum(transition_matrix, axis=1)
    transition_matrix = np.divide(transition_matrix, total_transition[:, np.newaxis])
    # transition_matrix = np.divide(transition_matrix,
    #                               np.sum(transition_matrix, axis=1)[:,
    #                               np.newaxis])

    return transition_matrix


def create_emissionmatrix1(X, S, AW, SC):
    emission_matrix = np.zeros((len(S), len(AW)))
    emission = {}
    for i in range(X.shape[0]):
        word = X[i][1]
        tag = X[i][-1]
        row_idx = S.index(tag)
        col_idx = AW.index(word) if word in AW else 0
        emission_matrix[row_idx, col_idx] += 1
        emission.update({f'({tag}, {word})': emission_matrix[row_idx, col_idx]})
    
    for i in range(len(SC)):
        emission_matrix[i, :] = emission_matrix[i, :]/SC[i]

    return emission_matrix, emission



def greedy_algorithm(X, EM, TM, S):
    transition_idx = -1
    accuracy = 0
    PREDICTIONS = []
    for i in range(X.shape[0]):
        word = X[i][1]
        if word in unique_words:
            emission_idx = np.where(unique_words==word)[0][0]
        else:
            emission_idx = 0
        emission_col = EM[:, emission_idx]
        transition_row = TM[transition_idx + 1, :]
        probability = emission_col * transition_row
        transition_idx = np.argmax(probability.reshape(-1,1))
        predicted_tag = S[transition_idx]
        target_tag = X[i][-1]

        PREDICTIONS.append(predicted_tag)

        if target_tag == predicted_tag:
            accuracy += 1

    return accuracy*100/X.shape[0], PREDICTIONS



devdata = np.array(pd.read_csv('data/dev', header=None, on_bad_lines='skip', delimiter='\t'))
testdata = np.array(pd.read_csv('data/test', header=None, on_bad_lines='skip', delimiter='\t'))

train_parameters = {
    'transition': {},
    'emission': {}
}

emissionmatrix, emission = create_emissionmatrix1(traindata, states, ALL_WORDS, states_count)

for i, w in enumerate(ALL_WORDS):
    for j, t in enumerate (states):
        if emissionmatrix[j, i] > 0:
            train_parameters['emission'].__setitem__(f'({t}, {w})',emissionmatrix[j, i])

transitionmatrix = create_transitionmatrix(states, trainlines, states_count)

for i, t1 in enumerate(states):
    for j, t2 in enumerate (states):
        if transitionmatrix[j, i] > 0:
            train_parameters['transition'].__setitem__(f'({t1}, {t2})',transitionmatrix[i, j])

print(f'Number of Transition Parameters: {len(train_parameters["transition"]) + 1}')
print(f'Number of Emission Parameters: {len(train_parameters["emission"])}')


with open('hmm.json', 'w') as f:
    json.dump(train_parameters, f, indent=4)

ACC, _ = greedy_algorithm(devdata, emissionmatrix, transitionmatrix, states)

print(f'********* Greedy Algorithm ***********')
print(f'Accuracy on the Dev data: {ACC:.2f}%')

_, test_predictions = greedy_algorithm(testdata, emissionmatrix, transitionmatrix, states)

# print(f'Predictions on the Test data: {test_predictions}')

with open('greedy.out', 'w') as f:
    for id, data in enumerate(testdata):
        f.write(f'{id + 1}\t{data[1]}\t{test_predictions[id]}\n')

def viterbi_algorithm(S, W, TW, EM, TM):
    
    viterbimatrix = np.zeros((len(S), len(W)))
    pointer = np.zeros((len(S), len(W)), dtype=int)

    if W[0] in TW:
        emission_idx = TW.index(W[0])
    else:
        emission_idx = 0
    
    viterbimatrix[:, 0] = TM[0, :] * EM[:, emission_idx]
    pointer[:, 0] = 0
    for i, w in enumerate(W[1:], start=1):
        if w in TW:
            emission_idx = TW.index(w)
        else:
            emission_idx = 0

        for j, t in enumerate(S):
            transition_idx = j
            probability = viterbimatrix[:, i - 1] * (TM[1:, transition_idx]) * (EM[transition_idx, emission_idx])
            viterbimatrix[transition_idx, i] = np.nanmax(probability)
    
            try:
                pointer[j, i] = np.nanargmax(probability)
            except ValueError:
                pointer[j, i] = -1000
    

    best_pointer = np.nanargmax(viterbimatrix[:, len(W) - 1])

    get_best_path = [S[best_pointer]]
    
    for i in range(pointer.shape[1] -1, 0, -1):
        get_best_path.append(S[pointer[best_pointer, i]])
        best_pointer = pointer[best_pointer, i]

    get_best_path.reverse()

    return get_best_path


def get_lines(filepath):

    with open(filepath, 'r') as testfile:
        testlines = testfile.readlines()

    testlines = [testline.rstrip('\n') for testline in testlines]
    testlines = [testline.split('\t') for testline in testlines]

    TEST = []
    TEST_UNIT = []
    TEST_TAGS = []
    TESTTAGS = []

    for tl in testlines:
        # print(tl)
        if len(tl) == 1:
            TEST.append(TEST_UNIT)
            TEST_UNIT = []
            TESTTAGS.append(TEST_TAGS)
            TEST_TAGS = []
        else:
            TEST_UNIT.append(tl[1])
            TEST_TAGS.append(tl[2])

    TEST.append(TEST_UNIT)
    TESTTAGS.append(TEST_TAGS)

    return TEST, TESTTAGS


devlines, devtags = get_lines('data/dev')

def get_test_lines(filepath):

    with open(filepath, 'r') as testfile:
        testlines = testfile.readlines()

    testlines = [testline.rstrip('\n') for testline in testlines]
    testlines = [testline.split('\t') for testline in testlines]

    TEST = []
    TEST_UNIT = []

    for tl in testlines:
        # print(tl)
        if len(tl) == 1:
            TEST.append(TEST_UNIT)
            TEST_UNIT = []
        else:
            TEST_UNIT.append(tl[1])

    TEST.append(TEST_UNIT)

    return TEST

testlines = get_test_lines('data/test')

DEV_PREDS_VITERBI = []
TEST_PREDS_VITERBI = []


for each_devline in devlines:
    dev_pred_viterbi = viterbi_algorithm(states, each_devline, ALL_WORDS, emissionmatrix, transitionmatrix)
    DEV_PREDS_VITERBI.append(dev_pred_viterbi)

for each_testline in testlines:
    test_pred_viterbi = viterbi_algorithm(states, each_testline, ALL_WORDS, emissionmatrix, transitionmatrix)
    TEST_PREDS_VITERBI.append(test_pred_viterbi)


acc_line = 0
total = 0
for i in range(len(DEV_PREDS_VITERBI)):
    each_devline_tag = devtags[i]
    each_predline = DEV_PREDS_VITERBI[i]
    
    for k in range(len(each_devline_tag)):
        target_dev_tag = each_devline_tag[k]
        pred_test_tag = each_predline[k]
        total += 1
        if target_dev_tag == pred_test_tag:
            acc_line += 1



accuracy = acc_line*100/total
print(f'******** VITERBI *********')
print(f'Accuracy on dev set: {accuracy:.2f}%')

# print(f'Preds on test set')
# print(TEST_PREDS_VITERBI)

with open('viterbi.out', 'w') as f:
    for line_id, each_line in enumerate(testlines):
        for l, w in enumerate(each_line):
            f.write(f'{l + 1}\t{w}\t{TEST_PREDS_VITERBI[line_id][l]}\n')
        f.write('\n')
