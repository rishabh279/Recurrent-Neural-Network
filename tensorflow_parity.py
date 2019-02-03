import tensorflow as tf
import numpy as np

from tensorflow.contrib.rnn import static_rnn as get_rnn_output
from tensorflow.contrib.rnn import BasicRNNCell, GRUCell

from sklearn.utils import shuffle
from util import init_weight, all_parity_pairs_with_sequence_labels, all_parity_pairs

def x2sequence(x, t, d, batch_sz):
    x = tf.transpose(x, (1, 0, 2))
    x = tf.reshape(x, (t*batch_sz, d))
    x = tf.split(x, t)

class SimpleRNN:

    def __init__(self, m):
        self.m = m

    def fit(self, x, y, batch_sz=20, learning_rate=0.1, mu=0.1, activation=tf.nn.sigmoid, epochs=100):
        n, t, d = x.shape
        k = len(set(y.flatten()))
        m = self.m
        self.f = activation

        wo = init_weight(m, k).astype(np.float32)
        bo = np.zeros(k, dtype=np.float32)

        self.wo = tf.Variable(wo)
        self.bo = tf.Variable(bo)

        tfx = tf.placeholder(tf.float32, shape=(batch_sz, t, d), name='inputs')
        tfy = tf.placeholder(tf.int64, shape=(batch_sz, t), name='targets')

        sequencex = x2sequence(tfx, t, d, batch_sz)

        rnn_unit = BasicRNNCell(num_units=self.M, activation=self.f)

        outputs, states = get_rnn_output(rnn_unit, sequencex, dtype=tf.float32)

        outputs = tf.transpose(outputs, (1, 0, 2))
        outputs = tf.reshape(outputs, (t*batch_sz, m))

        logits = tf.matmul(outputs, self.wo) + self.bo

        outputs = tf.transpose(outputs, (1, 0, 2))
        outputs = tf.reshape(outputs, (t*batch_sz, m))

        logits = tf.matmu(outputs, self.wo) + self.bo
        predict_op = tf.argmax(logits, 1)
        targets = tf.reshape(tfy, (t*batch_sz, ))

        cost_op = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=targets
            )
        )

        train_op = tf.train.MomentumOptimizer(learning_rate, momentum=mu).minimize(cost)

        costs = []
        n_batches = n // batch_sz

        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            for i in range(epochs):
                x, y = shuffle(x, y)
                n_correct = 0
                cost = 0
                for j in range(n_batches):
                    xbatch = x[j*batch_sz:(j+1)*batch_sz]
                    ybatch = y[j*batch_sz:(j+1)*batch_sz]

                    _, c, p = session.run([train_op, cost_op, predict_op], feed_dict={tfx:xbatch, tfy:ybatch})
                    cost += c
                    for b in range(batch_sz):
                        idx = (b+1)*t -1
                        n_correct += (p[idx] == ybatch[b][-1])
                if i % 10 == 0:
                    print("i:", i, "cost:", cost, "classification rate:", (float(n_correct) / n))
                if n_correct == n:
                    print("i:", i, "cost:", cost, "classification rate:", (float(n_correct) / n))
                    break
                costs.append(cost)


def parity(b=12, learning_rate=1, epochs=1000):
    x, y = all_parity_pairs_with_sequence_labels(b)

    rnn = SimpleRNN(4)
    rnn.fit(
        x, y,
        batch_sz=len(y),
        learning_rate=learning_rate,
        epochs=epochs,
        activation=tf.nn.sigmoid,
    )

if __name__ == '__main__':
    parity()