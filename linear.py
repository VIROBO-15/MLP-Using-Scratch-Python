
import numpy as np
import math

class Linear():
    for_back_Linear=[]
    for_back_Linear_dw=[]
    for_back_Linear_db = []
    def __init__(self, in_feature, out_feature, weight_init_fn, bias_init_fn):

        self.W = weight_init_fn(in_feature, out_feature)
        self.b = bias_init_fn(out_feature)

        self.dW = np.zeros(in_feature,out_feature)
        self.db = np.zeros(1,out_feature)

        self.momentum_W = np.zeros(in_feature,out_feature)
        self.momentum_b = np.zeros(1,out_feature)



    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):

        self.x = x
        self.out = np.dot(self.x,self.W)
        for i in range(self.x.shape([0])):
            self.out[i] = self.out[i,:] + self.b
        Linear.for_back_Linear.append(self.out)
        return self.out

    def backward(self, delta):

        self.dW = np.dot(np.transpose(self.x),delta)
        self.db = np.dot(np.transpose(self.x),delta)
        Linear.for_back_Linear_dw.append(self.dW)
        Linear.for_back_Linear_db.append(self.db)
        self.dx = np.sum(delta*np.transpose(self.W),axis=0)
        return  self.dx

