import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from util import init_weight, all_parity_pairs_with_sequence_labels


class SimpleRNN:
    def __init__(self, m):
        self.m = m

    def fit(self, x, y, learning_rate=1.0, mu=0.99, reg=1.0, activation=tf.tanh, epochs=100):
        n, t, d = x.shape
        k = len(set(y.flatten()))
        m = self.m
        self.f = activation

        wx = init_weight(d, m).astype(np.float32)
        wh = init_weight(m, m).astype(np.float32)
        bh = np.zeros(m, dtype=np.float32)
        ho = np.zeros(m, dtype=np.float32)
        wo = init_weight(m, k).astype(np.float32)
        bo = np.zeros(k, dtype=np.float32)

        self.wx = tf.Variable(wx)
        self.wh = tf.Variable(wh)
        self.bh = tf.Variable(bh)
        self.ho = tf.Variable(ho)
        self.wo = tf.Variable(wo)
        self.bo = tf.Variable(bo)

        tfx = tf.placeholder(tf.float32, shape=(t, d), name='x')
        tfy = tf.placeholder(tf.int32, shape=(t,), name='y')

        xwx = tf.matmul(tfx, self.wx)
        xwx_print = tf.Print(xwx, [xwx])

        def recurrence(ht1, xwt):
            ht = self.f(xwt + tf.matmul(tf.reshape(ht1, (1, m)), self.wh) + self.bh)
            return tf.reshape(ht, (m,))

        h = tf.scan(
            fn=recurrence,
            elems=xwx_print,
            initializer=self.ho
        )

        logits = tf.matmul(h, self.wo) + self.bo

        cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tfy,
                logits=logits
            )
        )

        predict_op = tf.argmax(logits, 1)
        train_op = tf.train.AdadeltaOptimizer(1e-2).minimize(cost)

        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)

            costs = []
            for i in range(epochs):
                x, y = shuffle(x, y)
                n_correct = 0
                batch_cost = 0

                for j in range(n):
                    _, c, p = session.run([train_op, cost, predict_op], feed_dict={tfx: x[j].reshape(t, d), tfy: y[j]})
                    batch_cost += c
                    if p[-1] == y[j, -1]:
                        n_correct += 1
                print('{}, {}, {}'.format(i, j, batch_cost))
                if n_correct == n:
                    break


def parity(b=4, learning_rate=1e-4, epochs=200):
    x, y = all_parity_pairs_with_sequence_labels(b)
    x = x.astype(np.float32)

    rnn = SimpleRNN(20)
    rnn.fit(x, y, learning_rate=learning_rate, epochs=epochs, activation=tf.nn.relu)

parity()

