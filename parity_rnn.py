# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 13:47:07 2018

@author: rishabh
"""

import theano 
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from util import init_weight,all_parity_pairs_with_sequence_labels

class RNN:
  def __init__(self,M):
    self.M=M
    
  def fit(self,X,Y,learning_rate=0.1,mu=0.99,reg=1.0,activation=T.tanh,epochs=100,show_fig=False):
    D=X[0].shape[1]
    K=len(set(Y.flatten()))
    N=len(Y)    
    M=self.M
    self.f=activation
    
    #intial weights
    Wx=init_weight(D,M)
    Wh=init_weight(M,M)
    bh=np.zeros(M)
    h0=np.zeros(M)
    Wo=init_weight(M,K)
    bo=np.zeros(K)
    
    #make them theano shared
    self.Wx=theano.shared(Wx)
    self.Wh=theano.shared(Wh)
    self.bh=theano.shared(bh)
    self.h0=theano.shared(h0)
    self.Wo=theano.shared(Wo)
    self.bo=theano.shared(bo)
    self.params=[self.Wx,self.Wh,self.bh,self.h0,self.Wo,self.bo]
  
    thX=T.fmatrix('X')
    thY=T.ivector('Y')
    
    def recurrence(x_t,h_t1):
      h_t=self.f(x_t.dot(self.Wx)+h_t1.dot(self.Wh)+self.bh)
      y_t=T.nnet.softmax(h_t.dot(self.Wo)+self.bo)
      return h_t,y_t
      
    [h,y],_=theano.scan(fn=recurrence,
                  outputs_info=[self.h0,None],
                  sequences=thX,n_steps=thX.shape[0],)
    
    py_x=y[:,0,:]
    prediction=T.argmax(py_x,axis=1)
    
    cost=-T.mean(T.log(py_x[T.arange(thY.shape[0]),thY]))
    grads=T.grad(cost,self.params)
    dparams=[theano.shared(p.get_value()*0) for p in self.params]

    updates = [
      (p, p + mu*dp - learning_rate*g) for p, dp, g in zip(self.params, dparams, grads)
    ]+[
     (dp,mu*dp-learning_rate*g) for dp,g in zip(dparams,grads)
    ]
    
    self.predict_op=theano.function(inputs=[thX],outputs=prediction)
    self.train_op=theano.function(
              inputs=[thX,thY],
              outputs=[cost,prediction,y],updates=updates)
    
    costs=[]
    for i in range(epochs):
      X,Y=shuffle(X,Y)
      n_correct=0
      cost=0
      for j in range(N):
        c,p,yout=self.train_op(X[j],Y[j])
        cost+=c
        if p[-1]==Y[j,-1]:
          n_correct+=1
        print("shape yout",yout.shape)
        print("i:",i,"cost:",cost,"classification rate:",(float(n_correct)/N))
        costs.append(cost)
        if n_correct==N:
          break
    
    if show_fig:
      plt.plot(costs)
            
if __name__=='__main__':
  X,Y=all_parity_pairs_with_sequence_labels(12)
  model=RNN(20)
  model.fit(X,Y,learning_rate=1e-4,epochs=200,activation=T.nnet.relu, show_fig=False)