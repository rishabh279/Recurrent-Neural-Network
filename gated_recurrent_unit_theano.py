import numpy as np
import theano
import theano.tensor as T

from .util import init_weight


class GRU:

    def __init__(self, mi, mo, activation):
        self.mi = mi
        self.mo = mo
        self.f = activation

        wxr = init_weight(mi, mo)
        whr = init_weight(mo, mo)
        br = init_weight(mo)
        wxz = init_weight(mi, mo)
        whz = init_weight(mo, mo)
        bz = np.zeros(mo)
        wxh = init_weight(mi, mo)
        whh = init_weight(mo, mo)
        bh = np.zeros(mo)
        h0 = np.zeros(mo)

        self.wxr = theano.shared(wxr)
        self.whr = theano.shared(whr)
        self.br = theano.shared(br)
        self.wxz = theano.shared(wxz)
        self.whz = theano.shared(whz)
        self.bz = theano.shared(bz)
        self.wxh = theano.shared(wxh)
        self.whh = theano.shared(whh)
        self.bh = theano.shared(bh)
        self.h0 = theano.shared(h0)
        self.params = [self.wxr, self.whr, self.br, self.wxz, self.whz, self.bz, self.xh, self.whh, self.bh, self.h0]

    def recurrence(self, x_t, h_t1):
        r = T.nnet.sigmoid(x_t.dot(self.wxr) + h_t1.dot(self.whr) + self.br)
        z = T.nnet.sigmoid(x_t.dot(self.wxz) + h_t1.dot(self.whz) + self.bz)
        hhat = self.f(x_t.dot(self.wxh) + (r * h_t1).dot(self.whh) + self.bh)
        h = (1 - z) * h_t1 + z * hhat
        return h

    def output(self, x):
        h, _ = theano.scan(
            fn=self.recurrence,
            sequences=x,
            outputs_info=self,
            n_steps=x.shape[0],
        )
        return h