
import numpy as np
import os


class Activation(object):


    def __init__(self):
        self.state = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class Identity(Activation):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        self.state = x
        return x

    def derivative(self):
        return 1.0


class Sigmoid(Activation):

    for_back_activation= []

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        self.state = 1/(1+np.exp(-1*x))
        Sigmoid.for_back_activation.append(self.state)
        return self.state

    def derivative(self):
        self.state = self.state*(1-self.state)

        return self.state


class Tanh(Activation):

    for_back_activation = []
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        self.state = (np.exp(x) - np.exp(-1*x))/(np.exp(x) + np.exp(-1*x))
        Tanh.for_back_activation.append(self.state)
        return self.state

    def derivative(self):
        self.state = 1 - (self.state*self.state)

        return self.state


class ReLU(Activation):

    """
    ReLU non-linearity
    """
    for_back_activation = []
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        self.x =x
        if x > 0:
            self.state = x
        else:
            self.state=0
        ReLU.for_back_activation.append(self.state)
        return self.state

    def derivative(self):
        if self.x >0:
            self.state = 1
        else:
            self.state = 0
        ReLU.for_back_activation.append(self.state)
        return self.state