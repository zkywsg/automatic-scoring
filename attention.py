from keras import backend as K
from keras.engine.topology import Layer
from keras.layers.core import *
from keras.layers import merge
import numpy as np
import theano

class Position_Embedding(Layer):

    def __init__(self, size=None, seq_len=None, mode='sum', **kwargs):
        self.supports_masking = True
        self.size = size
        self.mode = mode
        self.seq_len = seq_len
        super(Position_Embedding, self).__init__(**kwargs)

    def build(self, input_shape):
        self.size = input_shape[-1]
        if(self.seq_len == None):
            self.seq_len = input_shape[1]
        super(Position_Embedding, self).build(input_shape)

    def call(self, x, mask=None):
        # if (self.size == None) or (self.mode == 'sum'):
        #     self.size = int(x.shape[-1])
        batch_size, seq_len = K.shape(x)[0], K.shape(x)[1]
        position_j = 1. / K.pow(10000., 2 * K.arange(self.size / 2, dtype='float32') / self.size)
        position_j = K.expand_dims(position_j, 0)
        #position_i = K.sum(K.ones_like(x[:, :, 0]), 1, keepdims=True) - 1
        ini = np.transpose(np.tri(self.seq_len, self.seq_len))
        position_i = K.dot(K.ones_like(x[:,:,0]), K.variable(ini)) - 1
        position_i = K.expand_dims(position_i, 2)
        position_ij = K.dot(position_i, position_j)
        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)
        if self.mode == 'sum':
            return position_ij + x
        elif self.mode == 'concat':
            return K.concatenate([position_ij, x], 2)

    def get_output_shape_for(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2] + self.size)



class MutilHeadAttention(Layer):

    def __init__(self, nb_head, size_per_head, **kwargs):
        self.supports_masking = True
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head * size_per_head
        super(MutilHeadAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQs = []
        self.WKs = []
        self.WVs = []
        for i in range(self.nb_head):
            WQ = self.add_weight(shape=(input_shape[-1], self.size_per_head),
                                      name='WQ' + str(i),
                                      initializer='glorot_uniform',
                                      trainable=True)
            WK = self.add_weight(shape=(input_shape[-1], self.size_per_head),
                                      name='WK' + str(i),
                                      initializer='glorot_uniform',
                                      trainable=True)
            WV = self.add_weight(shape=(input_shape[-1], self.size_per_head),
                                      name='WV' + str(i),
                                      initializer='glorot_uniform',
                                      trainable=True)
            self.WQs.append(WQ)
            self.WKs.append(WK)
            self.WVs.append(WV)
        self.W = self.add_weight(shape=(self.output_dim, self.output_dim),
                                 name='W',
                                 initializer='glorot_uniform',
                                 trainable=True)

        super(MutilHeadAttention, self).build(input_shape)

    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:, 0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape) - 2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

    def call(self, x, mask=None):

        O_seq = None
        for i in range(self.nb_head):
            Q_seq = K.dot(x, self.WQs[i])
            # print(K.shape(Q_seq))
            # Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
            # Q_seq = K.permute_dimensions(Q_seq, (0,2,1,3))
            K_seq = K.dot(x, self.WKs[i])
            # K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
            # K_seq = K.permute_dimensions(K_seq, (0,2,1,3))
            V_seq = K.dot(x, self.WVs[i])
            # V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
            # V_seq = K.permute_dimensions(V_seq, (0,2,1,3))
            A = K.batch_dot(Q_seq, K_seq, axes=[2, 2]) / K.sqrt(K.shape(x)[2])
            A = K.cast(A, 'float32')
            #        K.transpose()
            # A = K.permute_dimensions(A, (0,3,2,1,4))
            #
            # A = K.reshape(A, (-1,))
            # A = K.softmax(A)
            # A = K.reshape(A, (K.shape(x)[0], K.shape(x)[1], K.shape(x)[1]))
            A = K.exp(A)
            A /= K.cast(K.sum(A, axis=1, keepdims=True) + K.epsilon(), 'float32')

            # A = Permute((0,3,2,1))(A)
            # A = K.permute_dimensions(A, (0,3,2,1,4))
            # shape = K.shape(A)
            # A = self.Mask(A, V_len, 'add')
            # A = K.permute_dimensions(A, (0,3,2,1))
            # A = K.reshape(A, (-1,))
            # A = K.softmax(A)
            # A = K.reshape(A, (shape[0], shape[1], shape[2], shape[3], shape[4]))
            A = K.batch_dot(A, V_seq, axes=[2, 1])
            if(O_seq == None):
                O_seq = A
            else:
                O_seq = merge([O_seq, A], mode='concat', name='merge')
            # O_seq = K.permute_dimensions(O_seq, (0,2,1,3, 4, 5))
            # O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
            # O_seq = self.Mask(O_seq, Q_len, 'mul')
        weighted_output = K.dot(O_seq, self.W)
        weighted_output = K.relu(weighted_output)
        # O_seq = O_seq * K.sigmoid(O_seq)
        return weighted_output

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)