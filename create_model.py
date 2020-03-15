
from keras.models import *

from keras.layers import Input, Embedding, LSTM

from keras.layers.pooling import GlobalAveragePooling1D
from keras.regularizers import l2
from attention import *
from zeromasking import ZeroMaskedEntries

from keras import optimizers


def build_model(opts, overall_maxlen, vocab_size=0, embedd_dim=50, embedding_weights=None, verbose=True, init_mean_value=None):

    word_input = Input(shape=(overall_maxlen, ), dtype='int32', name='word_input')

    x = Embedding(output_dim=embedd_dim, input_dim=vocab_size, input_length=overall_maxlen, weights=[embedding_weights],
                  mask_zero=True, name='x')(word_input)
    x_maskedout = ZeroMaskedEntries(name='x_maskedout')(x)
    drop_x = Dropout(opts.dropout, name='drop_x')(x_maskedout)
    position_embedding = Position_Embedding(name='position_embedding')(drop_x)
    self_att = MutilHeadAttention(nb_head=8, size_per_head=16, name='self_att')(position_embedding)

    lstm = LSTM(opts.rnn_dim, return_sequences=True, name='lstm')(self_att)
    avg_pooling = GlobalAveragePooling1D(name='avg_pooling')(lstm)

    y = Dense(output_dim=1, activation='sigmoid', name='y', W_regularizer=l2(opts.l2_value))(avg_pooling)


    model = Model(input=[word_input], output=y)

    if opts.init_bias and init_mean_value:

        bias_value = (np.log(init_mean_value) - np.log(1 - init_mean_value)).astype(K.floatx())
        model.layers[-1].b.set_value(bias_value)

    if verbose:
        model.summary()

    optimize = optimizers.rmsprop(lr=0.001)

    model.compile(loss='mse', optimizer=optimize)

    return model
