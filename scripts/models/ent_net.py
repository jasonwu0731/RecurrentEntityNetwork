from __future__ import print_function
from memories.DMC import DynamicMemoryCell
import numpy as np
import tensorflow as tf
from tflearn.activations import sigmoid, softmax
from tensorflow.python.framework import ops
from functools import partial
import logging
import datetime


class EntityNetwork():
    def __init__(self, vocab_size, sent_len, sent_numb, num_blocks, embedding_size,
                 label_num,L2,clip_gradients,session=tf.Session(),
                initializer=tf.random_normal_initializer(stddev=0.1)):
        """
        Initialize an Entity Network with the necessary hyperparameters.

        :param vocabulary: Word Vocabulary for given model.
        :param sentence_len: Maximum length of a sentence.
        :param story_len: Maximum length of a story.
        """
        self.vocab_size, self.sent_len, self.sent_numb = vocab_size, sent_len, sent_numb
        self.embedding_size, self.num_blocks, self.init = embedding_size, num_blocks, initializer
        self.opt = 'Adam'
        self.clip_gradients = clip_gradients
        self.label_num,  self.L2 = label_num, L2
        ## setup placeholder
        self.S = tf.placeholder(tf.int32, shape=[None,None,self.sent_len],name="Story")
        self.Q = tf.placeholder(tf.int32, shape=[None,self.sent_len],name="Question")
        self.A = tf.placeholder(tf.int64, shape=[None],name="Answer")
        self.visualization = False

        # self.keep_prob = tf.placeholder(tf.float32, name= "dropout")
        self.learning_rate = tf.placeholder(tf.float32, [], name="learning_rate")
        self.batch_size = tf.shape(self.S)[0]

        # Setup Global, Epoch Step
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))

        # Instantiate Network Weights
        self.instantiate_weights()

        # Build Inference Pipeline
        self.logits = self.inference()

        # Build Loss Computation
        self.loss_op = self.loss()

        # Build Training Operation
        self.train_op = self.train()

        self.saver = tf.train.Saver(max_to_keep=5)

        # Create operations for computing the accuracy
        self.correct_prediction = tf.equal(tf.argmax(self.logits, 1),self.A)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32), name="Accuracy")

        
        # predict op 
        predict_op = tf.argmax(self.logits, 1, name="predict_op")
        predict_proba_op = tf.nn.softmax(self.logits, name="predict_proba_op")
        predict_log_proba_op = tf.log(predict_proba_op, name="predict_log_proba_op")

        self.predict_op = predict_op
        self.predict_proba_op = predict_proba_op
        self.predict_log_proba_op = predict_log_proba_op

        parameters = self.count_parameters()
        logging.info('Parameters: {}'.format(parameters))

        init_op = tf.global_variables_initializer()
        self._sess = session
        self._sess.run(init_op)
        self.cnt = 0

    def instantiate_weights(self):
        """
        Instantiate Network Weights, including all weights for the Input Encoder, Dynamic
        Memory Cell, as well as Output Decoder.
        """

        self.E = tf.get_variable("Embedding",[self.vocab_size, self.embedding_size], initializer=self.init)

        zero_mask = tf.constant([0 if i == 0 else 1 for i in range(self.vocab_size)],dtype=tf.float32, shape=[self.vocab_size, 1])
        self.E = self.E * zero_mask

        alpha = tf.get_variable(name='alpha',
                                shape=self.embedding_size,
                                initializer=tf.constant_initializer(1.0))
        self.activation = partial(prelu, alpha=alpha)
        # self.activation = selu

        # Create Learnable Mask
        self.story_mask = tf.get_variable("Story_Mask", [self.sent_len, self.embedding_size],
                                          initializer=tf.constant_initializer(1.0),trainable=True)
        self.query_mask = tf.get_variable("Query_Mask", [self.sent_len, self.embedding_size],
                                          initializer=tf.constant_initializer(1.0),trainable=True)

        self.keys = [tf.get_variable('key_{}'.format(j), [self.embedding_size]) for j in range(self.num_blocks)]


        self.H = tf.get_variable("H", [self.embedding_size, self.embedding_size], initializer=self.init)
        self.R = tf.get_variable("R", [self.embedding_size, self.label_num], initializer=self.init)


    def inference(self):
        """
        Build inference pipeline, going from the story and question, through the memory cells, to the
        distribution over possible answers.
        """

        # Story Input Encoder
        story_embeddings = tf.nn.embedding_lookup(self.E, self.S) # Shape: [None, story_len, sent_len, embed_sz]
        # story_embeddings = tf.nn.dropout(story_embeddings, self.keep_prob)               # Shape: [None, story_len, sent_len, embed_sz]
        story_embeddings = tf.multiply(story_embeddings, self.story_mask)
        self.story_embeddings = tf.reduce_sum(story_embeddings, axis=[2])                     # Shape: [None, story_len, embed_sz]

        # Query Input Encoder
        query_embedding = tf.nn.embedding_lookup(self.E, self.Q)  # Shape: [None, sent_len, embed_sz]
        query_embedding = tf.multiply(query_embedding, self.query_mask)                  # Shape: [None, sent_len, embed_sz]
        self.query_embedding = tf.reduce_sum(query_embedding, axis=[1])                       # Shape: [None, embed_sz]

        ## to input into a dynacmicRNN we need to specify the lenght of each sentence
        # length = tf.cast(tf.reduce_su2m(tf.sign(tf.reduce_max(tf.abs(self.S), axis=2)), axis=1), tf.int32)
       
       
        self.length = self.get_sequence_length()

        # Create Memory Cell
        self.cell = DynamicMemoryCell(self.num_blocks, self.embedding_size,
                                      self.keys, self.query_embedding)
        # self.cell =tf.contrib.rnn.DropoutWrapper(self.cell, output_keep_prob=self.keep_prob)

        # Send Story through Memory Cell
        initial_state = self.cell.zero_state(self.batch_size, dtype=tf.float32)
        self.out, memories = tf.nn.dynamic_rnn(self.cell, self.story_embeddings,
                                        sequence_length=self.length,
                                        initial_state=initial_state)

        # Output Module
        # stacked_memories = tf.stack(memories, axis=1)
        stacked_memories = tf.stack(tf.split(memories, self.num_blocks, 1), 1)


        # Generate Memory Scores
        p_scores = softmax(tf.reduce_sum(tf.multiply(stacked_memories, tf.expand_dims(self.query_embedding,1)), axis=[2])) # Shape: [None, mem_slots]
                                                      

        # Subtract max for numerical stability (softmax is shift invariant)
        p_max = tf.reduce_max(p_scores, axis=-1, keep_dims=True)
        attention = tf.nn.softmax(p_scores - p_max)
        attention = tf.expand_dims(attention, 2)                                         # Shape: [None, mem_slots, 1]


        # Weight memories by attention vectors
        u = tf.reduce_sum(tf.multiply(stacked_memories, attention), axis=1)          # Shape: [None, embed_sz]

        # Output Transformations => Logits
        hidden = self.activation(tf.matmul(u, self.H) + tf.squeeze(self.query_embedding))      # Shape: [None, embed_sz]
        logits = tf.matmul(hidden, self.R)
        return logits
    

    def loss(self):
        """
        Build loss computation - softmax cross-entropy between logits, and correct answer.
        """
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.A, name="cross_entropy")
        cross_entropy_sum = tf.reduce_sum(cross_entropy, name="cross_entropy_sum")
        if(self.L2 !=0.0):
            lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'rnn/DynamicMemoryCell/biasU:0' != v.name ])  * self.L2
            # lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in var])  * self.L2
            return cross_entropy_sum + lossL2
            # return tf.losses.sparse_softmax_cross_entropy(self.A,self.logits)+lossL2
        else:
            return cross_entropy_sum
            # return tf.losses.sparse_softmax_cross_entropy(self.A,self.logits)

    def train(self):
        """
        Build Optimizer Training Operation.
        """
        # learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
        #                                            self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_op, global_step=self.global_step,
                                                   learning_rate=self.learning_rate, optimizer=self.opt,
                                                   clip_gradients=self.clip_gradients)
        return train_op

    def get_sequence_length(self):
        """
        This is a hacky way of determining the actual length of a sequence that has been padded with zeros.
        """
        used = tf.sign(tf.reduce_max(tf.abs(self.S), axis=-1))
        length = tf.cast(tf.reduce_sum(used, axis=-1), tf.int32)
        return length

    def position_encoding(self,sentence_size, embedding_size):
        """
        Position Encoding described in section 4.1 [1]
        """
        encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
        ls = sentence_size+1
        le = embedding_size+1
        for i in range(1, le):
            for j in range(1, ls):
                encoding[i-1, j-1] = (i - (le-1)/2) * (j - (ls-1)/2)
        encoding = 1 + 4 * encoding / embedding_size / sentence_size
        return np.transpose(encoding)

    def count_parameters(self):
        "Count the number of parameters listed under TRAINABLE_VARIABLES."
        num_parameters = sum([np.prod(tvar.get_shape().as_list())
                              for tvar in tf.trainable_variables()])
        return num_parameters
    

    def batch_fit(self, stories, queries, answers, learning_rate, cand=[]):
        """Runs the training algorithm over the passed batch

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
            answers: Tensor (None, vocab_size)
        Returns:
            loss: floating-point number, the loss computed for the batch
        """

        feed_dict = {self.S: stories, self.Q: queries, self.A: answers, self.learning_rate: learning_rate}
        loss, _, acc = self._sess.run([self.loss_op, self.train_op, self.accuracy], feed_dict=feed_dict)
        return loss, acc

    def predict(self, stories, queries, answers=[], cand=[], word_in = []):
        feed_dict = {self.S: stories, self.Q: queries}
        if self.visualization:
            self.viz(stories,queries,feed_dict,word_in)
        if answers!=[]:
            feed_dict[self.A] = answers
            return self._sess.run([self.predict_op, self.loss_op], feed_dict=feed_dict)
        else:
            return self._sess.run(self.predict_op, feed_dict=feed_dict)

    def predict_proba(self, stories, queries, cand=[]):
        feed_dict = {self.S: stories, self.Q: queries}
        return self._sess.run(self.predict_proba_op, feed_dict=feed_dict)

    def predict_log_proba(self, stories, queries):
        feed_dict = {self.S: stories, self.Q: queries}
        return self._sess.run(self.predict_log_proba_op, feed_dict=feed_dict)


    def sigmoid(self,x):
        x = np.array(x, dtype=np.float128)
        return 1 / (1 + np.exp(-x))

    def viz(self,mb_x1,mb_x2,dic,word_ind):
        import seaborn as sns
        import matplotlib.pyplot as plt
        plt.rc('text', usetex=True)
        plt.rc('font', family='Times-Roman')
        sns.set_style(style='white')
        self.cnt += 1

        idx2candid = dict((i, c) for c, i in word_ind.items())
        s_s=[]
        ### mb_x1 input senteces
        for m in mb_x1[0]:
            t = []
            for e in m:
                if(e != 0):
                    if (idx2candid[e][0]=='#'):
                        temp = idx2candid[e].split('_')
                        if(temp[1][:3]=='res'):
                            t.append("[NAME"+temp[2][0]+"]")
                        else:
                            t.append("["+temp[1][:3].upper()+temp[2][0]+"]")
                    elif idx2candid[e][0]=='<':
                        t.append(idx2candid[e][1:-1])
                    elif idx2candid[e][:2]=='r_':
                        t.append(idx2candid[e][2:])
                    elif idx2candid[e][:3]=='api':
                        t.append(idx2candid[e][:3]+r'\_'+ idx2candid[e][4:])
                    else:
                        t.append(idx2candid[e])
            if(len(t) > 0):
                s_s.append(t[2:])

        s_s = [" ".join(sss) for sss in s_s]

        q_q = []
        for e2 in mb_x2[0]:
            if(e2 != 0):
                if (idx2candid[e2][0]=='#'):
                    temp = idx2candid[e2].split('_')
                    q_q.append("["+temp[1][:3]+'-'+temp[2][0]+"]")
                elif idx2candid[e2][0]=='<':
                    q_q.append(idx2candid[e2][1:-1])
                elif idx2candid[e2][:2]=='r_':
                    q_q.append(idx2candid[e2][2:])
                else:
                    q_q.append(idx2candid[e2])


        q_q = " ".join(q_q[2:])

        k,o,s,q,l,E = self._sess.run([self.keys,
                                    self.out,
                                    self.story_embeddings,
                                    self.query_embedding,
                                    self.length,
                                    self.E],feed_dict=dic)

        gs=[]
        for i in range(int(l[0])):
            temp = np.split(o[0][i], len(k))
            g =[]
            for j in range(len(k)):
                # a = np.argmax(np.matmul(E,k[j]))
                # print(idx2candid[a])
                # print(np.argmax(np.matmul(E,k[j])))
                # print(np.max(np.matmul(E,k[j])))             
                # g.append(sigmoid(np.inner(s[0][i],temp[j])+np.inner(s[0][i],k[j])+np.inner(s[0][i],q[0][0])))
                g.append(self.sigmoid(np.inner(s[0][i],temp[j])+np.inner(s[0][i],k[j])))
            gs.append(g)

        plt.figure(figsize=(5,7.5))
        ax = sns.heatmap(np.array(gs),cmap="YlGnBu",vmin=0, vmax=1,cbar=False)
        
        ax.set_yticks([i+0.5 for i in range(len(s_s))],minor=True)
        ax.set_yticklabels(s_s,rotation=0,fontsize=7)
        ax.set_xticklabels([ i+1 for i in range(len(k)) ],rotation=0 )

        plt.title(q_q,fontsize=7)
        plt.tight_layout()
        plt.subplots_adjust(left=0.75, right=0.99, top=0.96, bottom=0.4)

        plt.savefig('../data/plot/%s.pdf'%str(self.cnt), format='pdf', dpi=300)
        plt.close()

def prelu(features, alpha, scope=None):
    """
    Implementation of [Parametric ReLU](https://arxiv.org/abs/1502.01852) borrowed from Keras.
    """
    with tf.variable_scope(scope, 'PReLU'):
        pos = tf.nn.relu(features)
        neg = alpha * (features - tf.abs(features)) * 0.5
        return pos + neg

def selu(x):
    with ops.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))



