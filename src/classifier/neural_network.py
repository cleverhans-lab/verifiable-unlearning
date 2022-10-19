
import numpy as np
from itertools import product

def weight_iterator(weights):
    for w in weights:
        print(w)
        yield w
class NeuralNetwork:

    def __init__(self, neurons=2):
        self.no_neurons = neurons
        self.w_0 = None
        self.b_0 = None
        self.w_1 = None
        self.b_1 = None

    def sigmoid(self, x, shift):
        W0 = self.add_shift(0.50, shift)
        W1 = self.add_shift(0.1501, shift)
        W3 = self.add_shift(0.0016, shift)
        v = W0 + self.remove_shift(W1*x, shift) - self.remove_shift(W3*self.remove_shift(x*self.remove_shift(x*x, shift), shift), shift)
        return v

    @property
    def weights(self):
        weights = []
        weights += [ w_i for w in self.w_0 for w_i in w ]
        weights += [ b_i for b_i in self.b_0 ]
        weights += [ w_i for w in self.w_1 for w_i in w ]
        weights += [ b_i for b_i in self.b_1 ]
        return weights
    
    def format_weights_init(self, config, dataset, weights_init):
        weights_init_itr = weight_iterator(weights_init)
        weights_str = ''
        for j in range(self.no_neurons):
            w_0_j = [ self.add_shift(w_i, config['precision']) 
                        for i, w_i, in zip(range(dataset.no_features), weights_init_itr) ]
            weights_str += f'    u64[{dataset.no_features}] mut w_0_{j} = {w_0_j};\n'
        w_1_0 = [ self.add_shift(w_i, config['precision']) 
                  for i, w_i, in zip(range(self.no_neurons), weights_init_itr) ]
        weights_str += f'    u64[{self.no_neurons}] mut w_1_0 = {w_1_0};\n'
        weights_str += f'    u64[{self.no_neurons}] mut b_0 = {[0]*self.no_neurons};\n'
        weights_str += f'    u64 mut b_1 = 0;'
        return weights_str.strip()

    def no_weights(self, dataset, bias=True):
        if bias:
            return self.no_neurons * (dataset.no_features+1) + self.no_neurons+1
        else:
            return self.no_neurons * dataset.no_features + self.no_neurons

    def add_shift(self, input, shift):
        if shift != 1:
            return int(input * shift)
        else:
            return input 

    def remove_shift(self, input, shift):
        if shift != 1:
            return int(input / shift)
        else:
            return input

    def score(self, dataset, config):
        no_correct = 0
        for x, y in zip(dataset.X_test, dataset.Y_test):
            a_0 = np.zeros(self.no_neurons).tolist()
            a_1 = np.zeros(1).tolist()
            z_0 = np.zeros(self.no_neurons).tolist()
            z_1 = np.zeros(1).tolist()
            for j in range(len(self.w_0)):
                z_0[j] = np.sum([ self.remove_shift(x_i*w_i, config['precision']) 
                                    for x_i, w_i in zip(x, self.w_0[j]) ]) + self.b_0[j]
                a_0[j] = self.sigmoid(z_0[j], config['precision'])
            # layer 1
            z_1[0] = np.sum([ self.remove_shift(a_i*w_i, config['precision'])  
                                for a_i, w_i in zip(a_0, self.w_1[0]) ]) + self.b_1[0]
            a_1[0] = self.sigmoid(z_1[0], config['precision'])
            y_pred = a_1[0]
            thresh = self.add_shift(0.5, config['precision'])
            if (y_pred >= thresh and y == config['precision']) or \
                (y_pred <  thresh and y == 0):
                no_correct += 1
        return no_correct / len(dataset.X_test)

    def train(self, config, dataset, weights_init):
        weights_init = weight_iterator(weights_init)

        self.w_0 = np.zeros((self.no_neurons, dataset.no_features)).tolist()
        for (j, i), w_j_i in zip(product(range(self.no_neurons), range(dataset.no_features)), weights_init):
            self.w_0[j][i] = self.add_shift(w_j_i, config['precision'])

        self.w_1 = np.zeros((1, self.no_neurons)).tolist()
        for w_0_i, i in zip(weights_init, range(self.no_neurons)):
            self.w_1[0][i] = self.add_shift(w_0_i, config['precision'])

        self.b_0 = [0]*self.no_neurons
        self.b_1 = [0]

        for epoch in range(config['epochs']):

            loss = 0
            for sample_idx in range(len(dataset)):
                # init accumulators
                a_0 = np.zeros(self.no_neurons).tolist()
                a_1 = np.zeros(1).tolist()
                z_0 = np.zeros(self.no_neurons).tolist()
                z_1 = np.zeros(1).tolist()
                da_0 = np.zeros(self.no_neurons).tolist()
                # da_1 = np.zeros(1).tolist()
                dz_0 = np.zeros(self.no_neurons).tolist()
                dz_1 = np.zeros(1).tolist()
                dw_0 = [ np.zeros(dataset.no_features).tolist() 
                            for _ in range(self.no_neurons)    ]
                dw_1 = [ np.zeros(self.no_neurons).tolist() ]
                db_0 = np.zeros(self.no_neurons).tolist()
                db_1 = np.zeros(1).tolist()
                # get data sample
                x = dataset.X[sample_idx]
                y = dataset.Y[sample_idx]
                ## forward
                # layer 0
                for j in range(len(self.w_0)):
                    z_0[j] = np.sum([ self.remove_shift(x_i*w_i, config['precision']) 
                                      for x_i, w_i in zip(x, self.w_0[j]) ]) + self.b_0[j]
                    a_0[j] = self.sigmoid(z_0[j], config['precision'])
                # layer 1
                z_1[0] = np.sum([ self.remove_shift(a_i*w_i, config['precision'])  
                                  for a_i, w_i in zip(a_0, self.w_1[0]) ]) + self.b_1[0]
                a_1[0] = self.sigmoid(z_1[0], config['precision'])
                y_pred = a_1[0]
                # loss
                loss += - (  y/config['precision'] *np.log(  y_pred/config['precision']) \
                        + (1-y/config['precision'])*np.log(1-y_pred/config['precision']))
                ## backward
                # layer 1
                dz_1[0] = a_1[0] - y
                for i in range(len(dw_1[0])):
                    dw_1[0][i] += self.remove_shift(dz_1[0]*a_0[i], config['precision'])
                db_1[0] += dz_1[0]
                # layer 0
                for j in range(len(self.w_0)):
                    da_0[j] = self.remove_shift(dz_1[0]*self.w_1[0][j], config['precision'])
                for j in range(len(self.w_0)):
                    dz_0[j] = self.remove_shift(da_0[j] * self.remove_shift(a_0[j] * (1-a_0[j]), config['precision']), config['precision'])
                for j in range(len(dw_0)):
                    for i in range(len(dw_0[j])):
                        dw_0[j][i] += self.remove_shift(dz_0[j] * x[i], config['precision'])
                    db_0[j] += dz_0[j]
                # update
                for i in range(len(self.w_1[0])):
                    self.w_1[0][i] -= self.remove_shift(self.add_shift(config['lr'], config['precision'])*dw_1[0][i], config['precision'])
                self.b_1[0] -= self.remove_shift(self.add_shift(config['lr'], config['precision'])*db_1[0], config['precision'])
                
                for j in range(len(self.w_0)):
                    for i in range(len(self.w_0[j])):
                        self.w_0[j][i] -= self.remove_shift(self.add_shift(config['lr'], config['precision'])*dw_0[j][i], config['precision'])
                    self.b_0[j] -= self.remove_shift(self.add_shift(config['lr'], config['precision'])*db_0[j], config['precision'])
            print(f'[{epoch:>2}] loss: {loss/len(dataset)/config["precision"]:.5f} acc: {self.score(dataset, config):.2f}')
