import theano
from sklearn.utils import shuffle
import numpy as np
from sklearn.utils import shuffle
from gated_recurrent_unit_theano import GRU
from long_short_term_memory_theano import LSTM
from .utility import init_weight
from brown import get_sentences_with_word2idx_limit_vocab
import theano.tensor as T
import sys

class RNN:
    def __init__(self, d, hidden_layer_sizes, v):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.d = d
        self.v = v

    def fit(self, x, learning_rate=1e-5, mu=0.99, epochs=10, show_fig=True, activation=T.nnet.relu, RecurrentUnit=GRU, normalize=True):
        d = self.d
        v = self.v
        n = len(x)

        we = init_weight(v, d)
        self.hidden_layers = []
        mi =d
        for mo in self.hidden_layer_sizes:
            ru = RecurrentUnit(mi, mo, activation)
            self.hidden_layers.append(ru)
            mi = mo

        wo = init_weight(mi, v)
        bo = np.zeros(v)

        self.we = theano.shared(we)
        self.wo = theano.shared(wo)
        self.bo = theano.shared(bo)
        self.params = [self.wo, self.bo]

        for ru in self.hidden_layers:
            self.params += ru.params

        thX = T.ivector('X')
        thY = T.ivector('Y')

        z = self.We[thX]
        for ru in self.hidden_layers:
            z = ru.output(z)

        py_x = T.nnet.softmax(z.dot(self.wo) + self.bo)

        prediction = T.argmax(py_x, axis=1)

        self.predict_op = theano.function(
            inputs=[thX],
            outputs=[py_x, prediction],
            allow_input_downcast=True
        )

        cost = -T.mean(T.log(py_x[T.arange(thY.shape[0], thY)]))
        grads = T.grad(cost, self.params)
        dparams = [theano.shared(p.get_value()*0) for p in self.params]

        dwe = theano.shared(self.we.get_value()*0)
        gwe = T.grad(cost, self.We)
        dwe_update = mu*dwe - learning_rate*gwe
        we_update = self.we + dwe_update
        if normalize:
            we_update /= we_update.norm(2)

        updates = [
            (p, p + mu*dp - learning_rate*g) for p, dp, g in zip(self.params, dparams, grads)
        ] + [
            (dp, mu*dp - learning_rate*g) for dp, g in zip(dparams, grads) ]  + [
            (self.we, we_update), (dwe, dwe_update)
        ]

        self.train_op = theano.function(
            inputs=[thX, thY],
            outputs=[cost, prediction],
            updates=updates
        )

        costs = []
        for i in range(epochs):
            x = shuffle(x)
            n_correct = 0
            n_total = 0
            cost = 0
            for j in range(n):
                if np.random.random() < 0.01 or len(x[j]) <= 1:
                    input_sequence = [0] + x[j]
                    output_sequence = x[j] + [1]
                else:
                    input_sequence = [0] + x[j][:-1]
                    output_sequence = x[j]

                n_total += len(output_sequence)

                try:
                    c, p = self.train_op(input_sequence, output_sequence)

                except Exception as e:
                    pyx, pred = self.predict_op(input_sequence)
                    print("input_sequence len:", len(input_sequence))
                    print("PYX.shape:", pyx.shape)
                    print("pred.shape:", pred.shape)
                    raise e

                cost += c

                for pj, xj in zip(p, output_sequence):
                    if pj == xj:
                        n_correct += 1

                    if j % 200 == 0:
                        sys.stdout.write("j/N: %d/%d correct rate so far: %f\r" % (j, N, float(n_correct) / n_total))
                        sys.stdout.flush()


def train_brown_corpus(we_file='word_embeddings.npy', w2i_file='wikipedia_word2idx.json', RecurrentUnit=GRU):

    sentences, word2idx = get_sentences_with_word2idx_limit_vocab()

    print('Finish Retrieving Data')
    rnn = RNN(30, [30], len(word2idx))
    rnn.fit(sentences, learning_rate=1e-5, epochs=10, activation=T.nnet.relu)


if __name__ == '__main__':
    train_brown_corpus()