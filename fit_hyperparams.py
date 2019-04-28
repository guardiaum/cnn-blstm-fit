from keras.models import Model, load_model
from keras.layers import TimeDistributed, Conv1D, Dense, Embedding, Input, Dropout, LSTM, Bidirectional, MaxPooling1D, \
    Flatten, concatenate
from keras.initializers import RandomUniform
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from util import *


def model(X_train, Y_train, X_val, Y_val, caseEmbeddings, wordEmbeddings, label2Idx, char2Idx, sentences_maxlen, words_maxlen):

    lstm_state_size = 275

    print("sentences maxlen: %s" % sentences_maxlen)
    print("words maxlen: %s" % words_maxlen)
    print("wordEmbeddings: (%s, %s)" % wordEmbeddings.shape)
    print("caseEmbeddings: (%s, %s)" % caseEmbeddings.shape)
    print("char2Idx len: %s" % len(char2Idx))
    print("label2Idx len: %s" % len(label2Idx))

    """Model layers"""

    # character input
    character_input = Input(shape=(None, words_maxlen,), name="Character_input")

    # embedding -> Size of input dimension based on dictionary, output dimension
    embed_char_out = TimeDistributed(
        Embedding(len(char2Idx), 30, embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)),
        name="Character_embedding")(
        character_input)

    dropout = Dropout({{uniform(0, 1)}})(embed_char_out)

    # CNN
    conv1d_out = TimeDistributed(
        Conv1D(kernel_size={{choice([3, 4, 5, 6, 7])}}, filters=30, padding='same',
               activation={{choice(['tanh', 'relu', 'sigmoid'])}}, strides=1),
        name="Convolution")(dropout)
    maxpool_out = TimeDistributed(MaxPooling1D(words_maxlen), name="maxpool")(conv1d_out)
    char = TimeDistributed(Flatten(), name="Flatten")(maxpool_out)
    char = Dropout({{uniform(0, 1)}})(char)

    # word-level input
    words_input = Input(shape=(None,), dtype='int32', name='words_input')
    words = Embedding(input_dim=wordEmbeddings.shape[0], output_dim=wordEmbeddings.shape[1],
                      weights=[wordEmbeddings],
                      trainable=False)(words_input)

    # case-info input
    casing_input = Input(shape=(None,), dtype='int32', name='casing_input')
    casing = Embedding(input_dim=caseEmbeddings.shape[0], output_dim=caseEmbeddings.shape[1],
                       weights=[caseEmbeddings],
                       trainable=False)(casing_input)

    # concat & BLSTM
    output = concatenate([words, casing, char])
    output = Bidirectional(LSTM(lstm_state_size,
                                return_sequences=True,
                                dropout={{uniform(0, 1)}},  # on input to each LSTM block
                                recurrent_dropout={{uniform(0, 1)}}  # on recurrent input signal
                                ), name="BLSTM")(output)

    output = TimeDistributed(Dense(len(label2Idx), activation={{choice(['relu', 'sigmoid'])}}),
                             name="activation_layer")(output)

    # set up model
    model = Model(inputs=[words_input, casing_input, character_input], outputs=[output])

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer={{choice(['nadam', 'rmsprop', 'adam', 'sgd'])}},
                  metrics=['accuracy'])

    model.summary()

    print(len(X_train[0][1]))
    print(X_train[0][2].shape)

    model.fit(X_train, Y_train,
              batch_size={{choice([32, 64, 128, 256])}},
              epochs={{choice([10, 20, 30, 40])}},
              verbose=2,
              validation_data=(X_val, Y_val))

    score, acc = model.evaluate(X_val, Y_val, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

def data():
    val_data, case2Idx_val, caseEmbeddings_val, word2Idx_val, \
    wordEmbeddings_val, char2Idx_val, label2Idx_val, sentences_maxlen, words_maxlen = prepare_data(get_val_data())

    train_data, case2Idx, caseEmbeddings, word2Idx, wordEmbeddings, \
    char2Idx, label2Idx, sentences_maxlen, words_maxlen = prepare_data(get_train_data())

    X_train, Y_train, X_val, Y_val = split_data(train_data, val_data)
    return X_train, Y_train, X_val, Y_val, caseEmbeddings, wordEmbeddings, label2Idx, char2Idx, sentences_maxlen, words_maxlen

best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())

train_data = prepare_data(get_train_data())
test_data = prepare_data(get_test_data())

X_train, Y_train, X_test, Y_test = split_data(train_data, test_data)

print("Evalutation of best performing model:")
print(best_model.evaluate(X_test, Y_test))
