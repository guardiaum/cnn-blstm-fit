import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from nltk import word_tokenize
from constants import *


def get_train_data():
    property_name = 'state'
    train_sentences = TRAIN_EXT_DIR + "/us_county/" + property_name +".csv"

    data = pd.read_csv(train_sentences, names=['sentence', 'ner', 'entity', 'value', 'label'],
                       dtype={'sentence': str, 'entity': str, 'value': str, 'label': str},
                       converters={'ner': eval}, encoding='utf-8')
    data['property'] = property_name
    data = data[data['label'] == 't']

    print(data.keys())

    return data


def get_test_data():
    test_sentences = TEST_ARTICLES_DIR + "/test-labeled-sentences.csv"

    df_test = pd.read_csv(test_sentences,
                          dtype={'property': str, 'entity': str, 'class': str, 'value': str, 'sentence': str},
                          converters={'ner': eval}, encoding='utf-8')

    df_test.replace(["NaN"], np.nan, inplace=True)
    df_test.dropna(inplace=True)

    print(df_test.keys())

    return df_test


def get_val_data():
    val_sentences = VALIDATION_ARTICLES_DIR + "/validation-labeled-sentences.csv"

    df_val = pd.read_csv(val_sentences,
                         dtype={'property': str, 'entity': str, 'class': str, 'value': str, 'sentence': str},
                         converters={'ner': eval}, encoding='utf-8')

    df_val.replace(["NaN"], np.nan, inplace=True)
    df_val.dropna(inplace=True)

    print(df_val.keys())

    return df_val


def split_data(train_data, test_data):
    X_train = []
    Y_train = []
    for t, c, ch, l in train_data:
        X_train.append([t, c, ch])
        Y_train.append(l)

    X_test = []
    Y_test = []
    for t, c, ch, l in test_data:
        X_test.append([t, c, ch])
        Y_test.append(l)

    print("len train: ", len(X_train))
    print("len test: ", len(X_test))

    return X_train, Y_train, X_test, Y_test


def define_dicts(words):

    label2Idx = {'O': 0, 'VALUE': 1, 'PROP': 2}

    # mapping for token cases
    case2Idx = {'numeric': 0, 'allLower': 1, 'allUpper': 2, 'initialUpper': 3, 'other': 4, 'mainly_numeric': 5,
                'contains_digit': 6, 'PADDING_TOKEN': 7}
    caseEmbeddings = np.identity(len(case2Idx), dtype='float32')  # identity matrix used

    # read GLoVE word embeddings
    word2Idx = {}
    wordEmbeddings = []

    fEmbeddings = open("embeddings/glove.6B/glove.6B.50d.txt", encoding="utf-8")

    # loop through each word in embeddings
    for line in fEmbeddings:
        split = line.strip().split(" ")
        word = split[0]  # embedding word entry

        if len(word2Idx) == 0:  # add padding+unknown
            word2Idx["PADDING_TOKEN"] = len(word2Idx)
            vector = np.zeros(len(split) - 1)  # zero vector for 'PADDING' word
            wordEmbeddings.append(vector)

            word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
            vector = np.random.uniform(-0.25, 0.25, len(split) - 1)
            wordEmbeddings.append(vector)

        if word.lower() in words:
            vector = np.array([float(num) for num in split[1:]])
            wordEmbeddings.append(vector)  # word embedding vector
            word2Idx[word] = len(word2Idx)  # corresponding word dict

    wordEmbeddings = np.array(wordEmbeddings)

    # dictionary of all possible characters
    char2Idx = {"PADDING": 0, "UNKNOWN": 1}
    for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|<>":
        char2Idx[c] = len(char2Idx)

    return case2Idx, caseEmbeddings, word2Idx, wordEmbeddings, char2Idx, label2Idx


def embed(sentences):
    """Create word- and character-level embeddings"""

    labelSet = set()
    words = {}

    # unique words and labels in data
    for dataset in sentences:
        for sentence in dataset:
            for token, char, label in [sentence]:
                # token ... token, char ... list of chars, label ... BIO labels
                labelSet.add(label)
                words[token.lower()] = True

    case2Idx, caseEmbeddings, word2Idx, wordEmbeddings, char2Idx, label2Idx = define_dicts(words)

    # format: [[wordindices], [label indices], [caseindices], [padded word indices]]
    data, sentences_maxlen, words_maxlen = create_matrices(sentences, word2Idx, label2Idx, case2Idx, char2Idx)

    # idx2Word = {v: k for k, v in word2Idx.items()}
    # idx2Label = {v: k for k, v in label2Idx.items()}

    return data, case2Idx, caseEmbeddings, word2Idx, wordEmbeddings, char2Idx, label2Idx, sentences_maxlen, words_maxlen


'''
Gets characters information and adds to sentences
Returns a matrix where the row is the sentence and
each column is composed by token setence, characters information from tokens and label for token
'''
def add_char_information_in(sentences):
    for i, sentence in enumerate(sentences):
        for j, data in enumerate(sentence):
            chars = [c for c in data[0]]  # data[0] is the token
            sentences[i][j] = [data[0], chars, data[1]]  # data[1] is the annotation (label)

    return sentences


def prepare_data(data, prop='state'):
    # unique_properties = df_test['property'].unique()
    # for prop in unique_properties:
    #print("CLASS: {}\n".format(prop))
    prop_data = data.loc[data['property'] == prop]
    tagged_data = tag_data(prop_data)
    sentences = add_char_information_in(tagged_data)
    return embed(sentences)


def tag_data(test_data):
    subset = test_data[['ner', 'sentence']]
    sentences = subset['sentence'].values
    ner = subset['ner'].values

    tagged_data = []
    for i, s in enumerate(sentences):
        s_tokens = word_tokenize(s)
        if len(s_tokens) == len(ner[i]):
            tags = [''] * len(ner[i])
            for j, token in enumerate(s_tokens):
                tags[j] = [token, ner[i][j]]
            tagged_data.append(tags)
        else:
            print(">>> DIFFERENT LENGHTS FOR SENTENCE TOKENS AND NER")

    return tagged_data


def get_casing(word, caseLookup):
    casing = 'other'

    numDigits = 0
    for char in word:
        if char.isdigit():
            numDigits += 1

    digitFraction = numDigits / float(len(word))

    if word.isdigit():
        casing = 'numeric'
    elif digitFraction > 0.5:
        casing = 'mainly_numeric'
    elif word.islower():
        casing = 'allLower'
    elif word.isupper():
        casing = 'allUpper'
    elif word[0].isupper():
        casing = 'initialUpper'
    elif numDigits > 0:
        casing = 'contains_digit'

    return caseLookup[casing]


def create_matrices(sentences, word2Idx, label2Idx, case2Idx, char2Idx):
    unknownIdx = word2Idx['UNKNOWN_TOKEN']
    paddingIdx = word2Idx['PADDING_TOKEN']

    dataset = []

    sentences_maxlen = 1000
    for sentence in sentences:
        wordcount = 0
        for word, char, label in sentence:
            wordcount += 1
        sentences_maxlen = max(sentences_maxlen, wordcount)

    print("sentence maxlen: %s" % sentences_maxlen)

    '''PADDING FOR SENTENCES AND EMBED OF WORDS, WORD CASEING AND CHARACTERS'''

    for sentence in sentences:
        wordIndices = []
        caseIndices = []
        charIndices = []
        labelIndices = []

        for word, char, label in sentence:
            if word in word2Idx:
                wordIdx = word2Idx[word]
            elif word.lower() in word2Idx:
                wordIdx = word2Idx[word.lower()]
            else:
                wordIdx = unknownIdx

            charIdx = []
            for x in char:
                if x not in char2Idx:
                    charIdx.append(char2Idx[' '])
                else:
                    charIdx.append(char2Idx[x])

            wordIndices.append(wordIdx)
            caseIndices.append(get_casing(word, case2Idx))
            charIndices.append(charIdx)
            labelIndices.append(label2Idx[label])

        while len(wordIndices) < sentences_maxlen:
            wordIndices.append(paddingIdx)
        while len(caseIndices) < sentences_maxlen:
            caseIndices.append(case2Idx['PADDING_TOKEN'])
        while len(charIndices) < sentences_maxlen:
            charIndices.append([char2Idx['PADDING']])

        dataset.append([wordIndices, caseIndices, charIndices, labelIndices])

    words_maxlen = 50
    for sentence in dataset:
        char = sentence[2]
        for x in char:
            words_maxlen = max(words_maxlen, len(x))
    print("words maxlen: %s" % words_maxlen)

    '''PADDING FOR CHARACTERS'''
    for i, sentence in enumerate(dataset):
        dataset[i][2] = pad_sequences(dataset[i][2], words_maxlen, padding='post')

    return dataset