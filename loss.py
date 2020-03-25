
import numpy as np
import os

class Criterion(object):

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented

class SoftmaxCrossEntropy(Criterion):
    """
    Softmax loss
    """
    for_back_loss=[]
    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()

    def forward(self, x, y):

        self.logits = x
        self.labels = y
        prob=[]
        self.prob = prob
        self.log_likehood=0
        for i in range(self.logits.shape[0]):
            ex = np.exp(self.logits[i] - np.max(self.logits[i], axis=1, keepdims=True))
            prob1 = ex / (np.sum(ex, axis=1, keepdims=True))
            self.prob.append(prob1)

        for i in range(x.shape[0]):
            self.log_likehood = self.log_likehood + -1*np.log(prob[i])*self.labels[i].argmax(axis=1)
        self.log_likehood = self.log_likehood/x.shape[0]
        SoftmaxCrossEntropy.for_back_loss.append(self.log_likehood)
        return self.log_likehood

    def derivative(self):

        m = self.labels.argmax(axis =1)
        grad = self.log_likehood
        grad[range(m), m] -= 1
        grad = grad / m
        return grad

