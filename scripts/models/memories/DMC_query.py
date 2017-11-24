from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import numpy as np
import tensorflow as tf
import functools


def prelu_func(features, initializer=None, scope=None):
    """
    Implementation of [Parametric ReLU](https://arxiv.org/abs/1502.01852) borrowed from Keras.
    """
    with tf.variable_scope(scope, 'PReLU', initializer=initializer):
        alpha = tf.get_variable('alpha', features.get_shape().as_list()[1:])
        pos = tf.nn.relu(features)
        neg = alpha * (features - tf.abs(features)) * 0.5
        return pos + neg
prelu = functools.partial(prelu_func, initializer=tf.constant_initializer(1.0))

class DynamicMemoryCell(tf.contrib.rnn.RNNCell):
    """
    Implementation of a dynamic memory cell as a gated recurrent network.
    The cell's hidden state is divided into blocks and each block's weights are tied.
    """

    def __init__(self, num_blocks, num_units_per_block, keys, query_embedding,
                activation = prelu,
                initializer=tf.random_normal_initializer(stddev=0.1)):
        self._num_blocks = num_blocks # M
        self._num_units_per_block = num_units_per_block # d
        self._keys = keys
        self._initializer = initializer
        self._activation = activation
        self._q = query_embedding


    @property
    def state_size(self):
        "Return the total state size of the cell, across all blocks."
        return self._num_blocks * self._num_units_per_block

    @property
    def output_size(self):
        "Return the total output size of the cell, across all blocks."
        return self._num_blocks * self._num_units_per_block

    def zero_state(self, batch_size, dtype):
        """
        We initialize the memory to the key values.
        """
        zero_state = tf.concat([tf.expand_dims(key, 0) for key in self._keys], 1)
        zero_state_batch = tf.tile(zero_state, tf.stack([batch_size, 1]))
        return zero_state_batch

    def get_gate(self, state_j, key_j, inputs):
        """
        Implements the gate (scalar for each block). Equation 2:

        g_j <- \sigma(s_t^T h_j + s_t^T w_j + s_t^T q)
        """
        a = tf.reduce_sum(inputs * state_j, axis=1)
        b = tf.reduce_sum(inputs * tf.expand_dims(key_j, 0), axis=1)
        c = tf.reduce_sum(inputs * tf.squeeze(self._q), axis=1)
        return tf.nn.sigmoid(a + b + c)

    def get_candidate(self, state_j, key_j, inputs, U, V, W, b):
        """
        Represents the new memory candidate that will be weighted by the
        gate value and combined with the existing memory. Equation 3:

        h_j^~ <- \phi(U h_j + V w_j + W s_t)
        """
        key_V = tf.matmul(tf.expand_dims(key_j, 0), V)
        state_U = tf.matmul(state_j, U) + b
        inputs_W = tf.matmul(inputs, W)
        return self._activation(state_U + key_V + inputs_W)

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__, initializer=self._initializer):
            # Split the hidden state into blocks (each U, V, W are shared across blocks).

            U = tf.get_variable('U', [self._num_units_per_block, self._num_units_per_block])
            V = tf.get_variable('V', [self._num_units_per_block, self._num_units_per_block])
            W = tf.get_variable('W', [self._num_units_per_block, self._num_units_per_block])

            b = tf.get_variable('biasU',[self._num_units_per_block])
            # self._q = tf.Print(self._q, [self._q],summarize=10)
            # TODO: layer norm?


            state = tf.split(state, self._num_blocks, 1)
            next_states = []
            for j, state_j in enumerate(state): # Hidden State (j)
                key_j = self._keys[j]
                gate_j = self.get_gate(state_j, key_j, inputs)
                candidate_j = self.get_candidate(state_j, key_j, inputs, U, V, W, b)

                # Equation 4: h_j <- h_j + g_j * h_j^~
                # Perform an update of the hidden state (memory).
                state_j_next = state_j + tf.expand_dims(gate_j, -1) * candidate_j

                # # Forget previous memories by normalization.
                # Equation 5: h_j <- h_j / \norm{h_j}
                state_j_next = tf.nn.l2_normalize(state_j_next, -1) # TODO: Is epsilon necessary?


                # Forget previous memories by normalization.
                # state_j_next_norm = tf.norm(tensor=state_j_next,
                #                             ord='euclidean',
                #                             axis=-1,
                #                             keep_dims=True)
                # state_j_next_norm = tf.where(
                #     tf.greater(state_j_next_norm, 0.0),
                #     state_j_next_norm,
                #     tf.ones_like(state_j_next_norm))
                # state_j_next = state_j_next / state_j_next_norm


                next_states.append(state_j_next)
            state_next = tf.concat(next_states, 1)
        return state_next, state_next
