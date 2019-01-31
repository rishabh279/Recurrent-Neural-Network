import numpy as np
import theano
import theano.tensor as T

from .util import init_weight

class LSTM:
    def __init__(self, mi, mo, activation):
        self.mi = mi
        self.mo = mo
        self.activation = activation

        # numpy init
        wxi = init_weight(mi, mo)
        whi = init_weight(mo, mo)
        wci = init_weight(mo, mo)
        bi = np.zeros(mo)
        wxf = init_weight(mi, mo)
        whf = init_weight(mo, mo)
        wcf = init_weight(mo, mo)
        bf = np.zeros(mo)
        wxc = init_weight(mi, mo)
        whc = init_weight(mo, mo)
        bc = np.zeros(mo)
        wxo = init_weight(mi, mo)
        who = init_weight(mo, mo)
        wco = init_weight(mo, mo)
        bo = np.zeros(mo)
        c0 = np.zeros(mo)
        h0 = np.zeros(mo)

        #theano vars
        self.wxi = theano.shared(wxi)
        self.whi = theano.shared(whi)
        self.wci = theano.shared(wci)
        self.bi = theano.shared(bi)
        self.wxf = theano.shared(wxf)
        self.whf = theano.shared(whf)
        self.wcf = theano.shared(wcf)
        self.bf = theano.shared(bf)
        self.wxc = theano.shared(wxc)
        self.whc = theano.shared(whc)
        self.bc = theano.shared(bc)
        self.wxo = theano.shared(wxo)
        self.who = theano.shared(who)
        self.wco = theano.shared(wco)
        self.bo = theano.shared(bo)
        self.c0 = theano.shared(c0)
        self.h0 = theano.shared(h0)

        self.params = [
            self.wxi,
            self.whi,
            self.wci,
            self.bi,
            self.wxf,
            self.whf,
            self.wcf,
            self.bf,
            self.wxc,
            self.whc,
            self.bc,
            self.wxo,
            self.who,
            self.wco,
            self.wbo,
            self.bo,
            self.c0,
            self.h0,
        ]

    def recurrence(self, x_t, h_t1, c_t1):
        i_t = T.nnet.sigmoid(x_t.dot(self.wxi) + h_t1.dot(self.whi) + c_t1.dot(self.wci) + self.bi)
        f_t = T.nnet.sigmoid(x_t.dot(self.wxf) + h_t1.dot(self.whf) + c_t1.dot(self.wcf) + self.bf)
        c_t = f_t * c_t1 + i_t * T.tanh(x_t.dot(self.wxc) + h_t1.dot(self.whc) + self.bc)
        o_t = T.nnet.sigmoid(x_t.dot(self.wxo) + h_t1.dot(self.who) + c_t.dot(self.wco) + self.bo)
        h_t = o_t * T.tanh(c_t)
        return h_t, c_t

    def output(self, x):
        [h, c], _ = theano.scan(
            fn=self.recurrence,
            sequences=x,
            outputs_info=[self.h0, self.c0],
            n_steps=x.shape[0],
        )
        return h
