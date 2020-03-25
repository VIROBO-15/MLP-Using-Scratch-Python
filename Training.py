
import sys
import os
import numpy as np


sys.path.append('/mytorch/')
from activation import *
from loss import *
from batchnorm import *
from linear import *
class MLP(object):

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn,
                 bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0):


        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum

        layer=hiddens
        layer.insert(0,self.input_size)
        layer.append(self.output_size)
        self.linear_layers = [Linear(layer[i],layer[i+1],weight_init_fn,bias_init_fn) for i in range(self.nlayers)]

        if self.bn:
            self.bn_layers = [BatchNorm(layer[i+1]) for i in range(self.num_bn_layers)]
        self.stores=[]
        #self.weight_init_fn = weight_init_fn
        #self.bias_init_fn = bias_init_fn
    def forward(self, x):

        self.x=x
        self.output =x
        self.stores.append(self.output)
        for i in range(len(self.linear_layers)-2):
            self.output = self.linear_layers[i].forward(self.output)
            z_n = self.bn_layers[i].forward(self.output)
            self.output = self.activations[i].forward(z_n)
            self.stores.append(self.output)

        return self.output


    def zero_grads(self):
        Linear.dW.fill(0.0)
        Linear.db.fill(0.0)
        BatchNorm.dbeta.fill(0.0)
        BatchNorm.dgamma.fill(0.0)

    def step(self):

        for i in range(len(self.linear_layers)):
            # Update weights and biases here
            self.linear_layers[i].W= self.linear_layers[i].W    - self.lr* (self.linear_layers[i].for_back_Linear_dw[i])
            self.bias_init_fn = self.bias_init_fn - self.lr *(self.linear_layers[i].for_back_Linear_db[i])
        # Do the same for batchnorm layers
        for i in range(len(self.bn_layers)):
            self.bn_layers[i].gamma = self.bn_layers[i].gamma - self.lr * (self.bn_layers[i].for_back_BatchNorm_gamma[i])
            self.bn_layers[i].beta = self.bn_layers[i].beta - self.lr * (self.bn_layers[i].for_back_BatchNorm_beta[i])

    def backward(self, labels):
        self.criterion.forward(self.stores[-1],labels)
        d_y = self.criterion.derviative()
        length = len(self.linear_layers)-1
        for i in range(1,len(self.linear_layers)):
            self.activations[length-i].forward(self.bn_layers[length-i].for_back_BatchNorm[length-i])
            d_z_das = d_y * self.activations[length-i].derivative()
            self.bn_layers[length-i].forward(self.linear_layers[length-i].for_back_Linear[length-i])
            d_z = self.bn_layers[length-i].backward(d_z_das)
            if length-i-1 >=0:
                self.linear_layers[length-i].forward(self.activations[length-i-1].for_back_activation[length-i-1])
                d_y = self.linear_layers[length-i].backward(d_z)
            else:
                self.linear_layers[length - i].forward(self.stores[0])
                d_y = self.linear_layers[length-i].backward(d_z)

    def error(self, labels):
        return (np.argmax(self.output , axis = 1) != np.argmax(labels, axis = 1)).sum()

    def total_loss(self, labels):
        return self.criterion(self.output , labels).sum()

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False

def get_training_stats(mlp, dset, nepochs, batch_size):

    train, val, _ = dset
    trainx, trainy = train
    valx, valy = val

    idxs = np.arange(len(trainx))

    training_losses = np.zeros(nepochs)
    training_errors = np.zeros(nepochs)
    validation_losses = np.zeros(nepochs)
    validation_errors = np.zeros(nepochs)


    for e in range(nepochs):
        err1 = 0
        los1 = 0
        err2 = 0
        los2 = 0
        for b in range(0, len(trainx)-batch_size, batch_size):
            output = mlp.forward(trainx[b:batch_size])
            mlp.backward(trainy[b:batch_size])
            mlp.steps()
            err1 = err1 + mlp.error(trainy[b:batch_size])
            los1 = los1+ mlp.total_loss(trainy[b:batch_size])

        for b in range(0, len(valx)-batch_size, batch_size):
            output = mlp.forward(trainx[b:batch_size])
            mlp.backward(trainy[b:batch_size])
            mlp.steps()
            err2 = err2+ mlp.error(trainy[b:batch_size])
            los2 = los2+ mlp.total_loss(trainy[b:batch_size])
        training_errors[e] = err1
        training_losses[e] = los1
        validation_errors[e] = err1
        validation_losses[e] = los2
    return (training_losses, training_errors, validation_losses, validation_errors)

    #raise NotImplemented
