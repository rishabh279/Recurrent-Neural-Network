import tensorflow as tf
import numpy as np

from sklearn.utils import shuffle
from util import init_weight, get_robert_frost


class SimpleRNN:

    def __init__(self, d, m, v, f, session):
        self.d = d
        self.m = m
        self.v = v
        self.f = f
        self.session = session

    def set_session(self, session):
        self.session = session

    def build(self, we, wx, wh, bh, h0, wo, bo):

        self.we = tf.Variable(we)
        self.wx = tf.Variable(wx)
        self.wh = tf.Variable(wh)
        self.bh = tf.Variable(bh)
        self.h0 = tf.Variable(h0)
        self.wo = tf.Variable(wo)
        self.bo = tf .Variable(bo)
        self.params = [self.we, self.wx, self.wh, self.bh, self.h0, self.wo, self.bo]

        v = self.v
        d = self.d
        m = self.m

        self.tfx = tf.placeholder(tf.int32, shape=(None,), name='x')
        self.tfy = tf.placeholder(tf.int32, shape=(None,), name='y')

        xw = tf.nn.embedding_lookup(we, self.tfx)

        xw_wx = tf.matmul(xw, self.wx)

        def recurrence(h_t1, xw_wx_t):
            h_t1 = tf.reshape(h_t1, (1, m))
            h_t = self.f(xw_wx_t + tf.matmul(h_t1, self.wh) + self.bh)
            h_t = tf.reshape(h_t, (m,))
            return h_t

        h = tf.scan(
            fn=recurrence,
            elems=xw_wx,
            initializer=self.h0
        )

        logits = tf.matmul(h, self.wo) + self.bo
        prediction = tf.argmax(logits, 1)
        self.output_probs = tf.nn.softmax(logits)

        nce_weights = tf.transpose(self.wo, [1, 0])
        nce_biases = self.bo

        h = tf.reshape(h, (-1, m))
        labels = tf.reshape(self.tfy, (-1, 1))

        self.cost = tf.reduce_mean(
            tf.nn.sampled_softmax_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=labels,
                inputs=h,
                num_sampled=50,
                num_classes=v
            )
        )

        self.predict_op = prediction
        self.train_op = tf.train.AdamOptimizer(1e-2).minimize(self.cost)

        init = tf.global_variables_initializer()
        self.session.run(init)

    def fit(self, x, epochs=500):
        n = len(x)
        d = self.d
        m = self.m
        v = self.v

        we = init_weight(v, d).astype(np.float32)
        wx = init_weight(d, m).astype(np.float32)
        wh = init_weight(m, m).astype(np.float32)
        bh = np.zeros(m).astype(np.float32)
        h0 = np.zeros(m).astype(np.float32)
        wo = init_weight(m, v).astype(np.float32)
        bo = np.zeros(v).astype(np.float32)

        self.build(we, wx, wh, bh, h0, wo, bo)

        costs = []
        n_total = sum((len(sentence)+1) for sentence in x)
        for i in range(epochs):
            x = shuffle(x)
            n_correct = 0
            cost = 0
            for j in range(n):
                input_sentence = [0] + x[j]
                output_sentence = x[j] + [1]

                _, c, p = self.session.run(
                    (self.train_op, self.cost, self.predict_op),
                    feed_dict={self.tfx: input_sentence, self.tfy: output_sentence}
                )
                cost += c
                for pj, xj in zip(p, output_sentence):
                    if pj == xj:
                        n_correct += 1
            print('cost is {} i is {} rate is {}'.format(cost, i, (float(n_correct)/n_total)))
            costs.append(cost)

    def save(self, filename):
        actual_params = self.session.run(self.params)
        np.savez(filename, *[p for p in actual_params])

    def predict(self, prev_words):
        return self.session.run(
            self.output_probs,
            feed_dict={self.tfx: prev_words}
        )

    @staticmethod
    def load(filename, activation, session):
        npz = np.load(filename)
        we = npz['arr_0']
        wx = npz['arr_1']
        wh = npz['arr_2']
        bh = npz['arr_3']
        h0 = npz['arr_4']
        wo = npz['arr_5']
        bo = npz['arr_6']
        v, d = we.shape
        _, m = wx.shape
        rnn = SimpleRNN(d, m, v, activation, session)
        rnn.build(we, wx, wh, bh, h0, wo, bo)
        return rnn

    def generate(self, pi, word2idx):
        idx2word = {v: k for k, v in word2idx.items()}
        v = len(pi)

        n_lines = 0

        x = [np.random.choice(v, p=pi)]
        print(idx2word[x[0]])

        while n_lines < 4:
            probs1 = self.predict(x)
            probs = self.predict(x)[-1]
            word_idx = np.random.choice(v, p=probs)
            x.append(word_idx)

            if word_idx > 1:
                word = idx2word[word_idx]
                print(word, end=" ")
            elif word_idx == 1:
                n_lines += 1
                print('')
                if n_lines < 4:
                    x = [np.random.choice(v, p=pi)]
                    print(idx2word[x[0]], end=" ")


def generate_poetry(session, savefile):
    sentences, word2idx = get_robert_frost()
    rnn = SimpleRNN.load(savefile, tf.nn.relu, session)

    v = len(word2idx)
    pi = np.zeros(v)
    print(pi[263])
    print(type(sentences[0][0]))
    print(len(sentences))
    for sentence in sentences:
        if sentence:
            pi[sentence[0]] += 1
    pi /= pi.sum()

    rnn.generate(pi, word2idx)


def train_poetry(session, dims, savefile):
    sentences, word2idx = get_robert_frost()
    rnn = SimpleRNN(dims, dims, len(word2idx), tf.nn.relu, session)
    rnn.fit(sentences, epochs=17)
    rnn.save(savefile)


if __name__ == '__main__':
    dims = 50
    savefile = 'RNN_D50_M50_tf.npz'
    session = tf.InteractiveSession()
    # train_poetry(session, dims, savefile)
    generate_poetry(session, savefile)

