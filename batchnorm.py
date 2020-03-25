
import numpy as np

class BatchNorm(object):
    for_back_BatchNorm = []
    for_back_BatchNorm_gamma = []
    for_back_BatchNorm_beta = []

    def __init__(self, in_feature, alpha=0.9):

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested
        self.var = np.ones((1, in_feature))
        self.mean = np.zeros((1, in_feature))

        self.gamma = np.ones((1, in_feature))
        self.dgamma = np.zeros((1, in_feature))

        self.beta = np.zeros((1, in_feature))
        self.dbeta = np.zeros((1, in_feature))

        # inference parameters
        self.running_mean = np.zeros((1, in_feature))
        self.running_var = np.ones((1, in_feature))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):

        self.x = x

        self.mean = np.mean(self.x , axis = 0)
        self.var = np.mean((self.x - self.mean)**2 , axis= 0)
        self.norm = (self.x - self.mean) * 1.0 / np.sqrt(self.var + self.eps)
        self.out = self.norm * self.gamma + self.beta

        # Update running batch statistics
        self.running_mean =  self.running_mean * self.alpha + (1 - self.alpha)*self.mean
        self.running_var =  self.alpha * self.running_var + (1 - self.alpha)*self.var
        BatchNorm.for_back_BatchNorm.append(self.out)

        return self.out


    def backward(self, delta):

        std_inv = 1. / np.sqrt(self.var + 1e-8)
        dx_norm = delta * self.gamma
        x_mu = self.x - self.mean
        dvar = np.sum(dx_norm * x_mu, axis=0) * -.5 * std_inv ** 3
        dmu = np.sum(dx_norm * -std_inv, axis=0) + dvar * np.mean(-2. * x_mu, axis=0)
        dX = (dx_norm * std_inv) + (dvar * 2 * x_mu / self.x.shape([0])) + (dmu / self.x.shape([0]))

        dgamma = np.sum(delta * self.norm, axis=0)
        BatchNorm.for_back_BatchNorm_gamma.append(dgamma)
        dbeta = np.sum(delta, axis=0)
        BatchNorm.for_back_BatchNorm_beta.append(dbeta)
        return dX
