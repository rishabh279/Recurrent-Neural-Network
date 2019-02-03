import theano
import numpy as np
import theano.tensor as T

from sklearn.utils import shuffle
from utility import init_weight, all_parity_pairs_with_sequence_labels

class SimpleRNN:

    def __init__(self, m):
        self.m = m

    def fit(self, x, y, batch_sz=20, learning_rate=1.0, mu=0.99, reg=1.0, activation=T.tanh, epochs=100):
        d = x[0].shape[1]
        k = len(set(y.flatten()))
        n = len(y)
        m = self.m
        self.f = activation

        wx = init_weight(d, m)
        wh = init_weight(m, m)
        bh = np.zeros(m)
        h0 = np.zeros(m)
        wo = init_weight(m, k)
        bo = np.zeros(k)

        self.wx = theano.shared(wx)
        self.wh = theano.shared(wh)
        self.bh = theano.shared(bh)
        self.h0 = theano.shared(h0)
        self.wo = theano.shared(wo)
        self.bo = theano.shared(bo)
        self.params = [self.wx, self.wh, self.bh, self.h0, self.wo, self.bo]

        thx = T.fmatrix('x')
        thy = T.ivector('y')

        the_start_points = T.ivector('start_points')

        xw = thx.dot(self.wx)

        def recurrence(xw_t, is_start, h_t1, h0):
            h_t = T.switch(
                T.eq(is_start, 1),
                self.f(xw_t + h0.dot(self.wh) + self.bh),
                self.f(xw_t + h_t1.dot(self.wh) + self.bh)
            )
            return h_t

        h, _ = theano.scan(
            fn=recurrence,
            outputs_info=[self.h0],
            sequences=[xw, the_start_points],
            non_sequences=[self.h0],
            n_steps=xw.shape[0],
        )

        py_x = T.nnet.softmax(h.dot(self.wo) + self.bo)
        prediction = T.argmax(py_x, axis=1)

        cost = -T.mean(T.log(py_x[T.arange(thy.shape[0]), thy]))
        grads = T.grad(cost, self.params)
        dparams = [theano.shared(p.get_value()*0) for p in self.params]

        updates =[
            (p, p + mu - learning_rate*g) for p, dp, g in zip(self.params, dparams, grads)
        ] + [
            (dp, mu*dp - learning_rate) for dp, g in zip(dparams, grads)
        ]

        self.train_op = theano.function(
            inputs=[thx, thy, the_start_points],
            outputs=[cost, prediction, py_x],
            updates=updates
        )

        costs = []
        n_batches = n // batch_sz
        sequence_length = x.shape[1]

        start_points = np.zeros(sequence_length * batch_sz, dtype=np.int32)
        for b in range(batch_sz):
            start_points[b*sequence_length] = 1
        for i in range(epochs):
            x, y = shuffle(x, y)
            n_correct = 0
            cost = 0
            for j in range(n_batches):
                xbatch = x[j*batch_sz:(j+1)*batch_sz].reshape(batch_sz*sequence_length, d)
                ybatch = y[j*batch_sz:(j+1)*batch_sz].reshape(batch_sz*sequence_length, d).astype(np.int32)
                c, p, rout = self.train_op(xbatch, ybatch, start_points)

                cost += c

                for b in range(batch_sz):
                    idx = sequence_length*(b + 1) -1
                    if p[idx] == ybatch[idx]:
                        n_correct += 1

            if i % 10 == 0:
                print("shape y:", rout.shape)
                print("i:", i, "cost:", cost, "classification rate:", (float(n_correct) / n))
            if n_correct == n:
                print("i:", i, "cost:", cost, "classification rate:", (float(n_correct) / n))
                break
            costs.append(cost)


def parity(b=4, learning_rate=1e-3, epochs=3000):
    x, y = all_parity_pairs_with_sequence_labels(b)
    rnn = SimpleRNN(4)
    rnn.fit(
        x,y,
        batch_sz=10,
        learning_rate=learning_rate,
        epochs=epochs,
        activation=T.nnet.sigmoid
    )


if __name__ == '__main__':
    parity()


