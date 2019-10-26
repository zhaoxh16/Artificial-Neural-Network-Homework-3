import numpy as np
import tensorflow as tf

from tensorflow.python.ops.nn import dynamic_rnn
from tensorflow.contrib.seq2seq.python.ops.loss import sequence_loss
from tensorflow.contrib.lookup.lookup_ops import MutableHashTable
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.session_bundle import exporter

# from rnn_cell import GRUCell, BasicLSTMCell, MultiRNNCell, BasicRNNCell
from rnn_cell import BasicRNNCell

PAD_ID = 0
UNK_ID = 1
_START_VOCAB = ['_PAD', '_UNK']


class RNN(object):
    def __init__(self,
                 num_symbols,
                 num_embed_units,
                 num_units,
                 num_layers,
                 num_labels,
                 embed,
                 learning_rate=0.005,
                 max_gradient_norm=5.0,
                 param_da=150,
                 param_r=10):

        self.texts = tf.placeholder(
            tf.string, (None, None), 'texts')  # shape: [batch, length]

        # todo: implement placeholders
        # shape: [batch]
        self.texts_length = tf.placeholder(tf.int64, None, 'texts_length')
        self.labels = tf.placeholder(tf.int64, None, 'labels')  # shape: [batch]

        self.symbol2index = MutableHashTable(
            key_dtype=tf.string,
            value_dtype=tf.int64,
            default_value=UNK_ID,
            shared_name="in_table",
            name="in_table",
            checkpoint=True)

        batch_size = tf.shape(self.texts)[0]
        # build the vocab table (string to index)
        # initialize the training process
        self.learning_rate = tf.Variable(float(learning_rate),
                                         trainable=False, dtype=tf.float32)
        self.global_step = tf.Variable(0, trainable=False)

        self.index_input = self.symbol2index.lookup(
            self.texts)   # shape: [batch, length]

        # build the embedding table (index to vector)
        if embed is None:
            # initialize the embedding randomly
            self.embed = tf.get_variable(
                'embed', [num_symbols, num_embed_units], tf.float32)
        else:
            # initialize the embedding by pre-trained word vectors
            self.embed = tf.get_variable(
                'embed', dtype=tf.float32, initializer=embed)

        # todo: implement embedding inputs
        # shape: [batch, length, num_embed_units]
        self.embed_input = tf.nn.embedding_lookup(self.embed, self.index_input)

        # todo: implement 3 RNNCells (BasicRNNCell, GRUCell, BasicLSTMCell) in a multi-layer setting with #num_units neurons and #num_layers layers
        cell_fw = BasicRNNCell(num_units)
        cell_bw = BasicRNNCell(num_units)

        # todo: implement bidirectional RNN
        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.embed_input, self.texts_length, dtype=tf.float32, scope="rnn")
        H = tf.concat(outputs, 2)  # shape: (batch, length, 2*num_units)
        length = tf.shape(H)[1]

        with tf.variable_scope('logits'):
            # todo: implement self-attention mechanism, feel free to add codes to calculate temporary results
            Ws1 = tf.get_variable("Ws1", [2*num_units, param_da])   # shape: (2*num_units, param_da)
            Ws2 = tf.get_variable("Ws2", [param_da, param_r])       # shape: (param_da, param_r)

            TEMP_1 = tf.tanh(tf.matmul(tf.reshape(H, (batch_size*length, 2*num_units)), Ws1)) # shape: (batch*length, param_da)
            TEMP_2 = tf.transpose(tf.reshape(tf.matmul(TEMP_1, Ws2), (batch_size, length, param_r)), (0, 2, 1))  # shape: (batch, param_r, length)
            A = tf.nn.softmax(TEMP_2)   # shape: (batch, param_r, length)
            M = tf.matmul(A, H)  # shape: [batch, param_r, 2*num_units]
            # shape: [batch, param_r*2*num_units]
            flatten_M = tf.reshape(M, shape=[batch_size, param_r*2*num_units])

            # shape: [batch, num_labels]
            logits = tf.layers.dense(
                flatten_M, num_labels, activation=None, name='projection')

        # todo: calculate additional loss, feel free to add codes to calculate temporary results
        identity = tf.reshape(tf.tile(tf.diag(tf.ones([param_r])), [batch_size, 1]), [
                              batch_size, param_r, param_r])    # shape: (batch, param_r, param_r)
        self.penalized_term = tf.reduce_mean(tf.square(tf.norm(tf.matmul(A, A, transpose_b=True) - identity, ord='fro', axis=[-2, -1])))

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.labels, logits=logits), name='loss') + 0.0001*self.penalized_term
        predict_labels = tf.argmax(logits, 1, 'predict_labels')
        self.accuracy = tf.reduce_sum(
            tf.cast(tf.equal(self.labels, predict_labels), tf.int32), name='accuracy')

        self.params = tf.trainable_variables()

        # calculate the gradient of parameters
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        gradients = tf.gradients(self.loss, self.params)
        clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients,
                                                                       max_gradient_norm)
        self.update = opt.apply_gradients(zip(clipped_gradients, self.params),
                                          global_step=self.global_step)

        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2,
                                    max_to_keep=5, pad_step_number=True)

    def print_parameters(self):
        for item in self.params:
            print('%s: %s' % (item.name, item.get_shape()))

    def train_step(self, session, data):
        input_feed = {self.texts: data['texts'],
                      self.texts_length: data['texts_length'],
                      self.labels: data['labels']}
        output_feed = [self.loss, self.accuracy,
                       self.gradient_norm, self.update]
        return session.run(output_feed, input_feed)
