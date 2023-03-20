
import numpy as np
import copy

class LinearRegression:

    def __init__(self):
        self.w = None
        self.b = None

    def init_model(self, config, no_features):
        np.random.seed(config['model_seed'])
        self.w = [ self.add_shift(w, config['precision']) 
                   for w in (np.random.rand(no_features)-0.5).tolist() ]
        self.b = 0
        return self.weights

    @property
    def weights(self):
        return copy.copy(self.w) + [self.b]

    @property
    def deltas(self):
        return copy.deepcopy(self._deltas)

    # def no_weights(self, dataset, bias=True):
    #     if bias:
    #         return dataset.no_features+1
    #     else:
    #         return dataset.no_features

    def format_weights(self, as_zokrates, from_variable):
        assert as_zokrates
        if from_variable:
            return f"u64[{len(self.w)}] w = [{', '.join([ f'm_prev[{idx}]' for idx in range(len(self.w)) ])}]\n\tu64 b = m_prev[{len(self.w)}]"
        else:
            return f"u64[{len(self.w)}] w = {[ w for w in self.w ]}\n\tu64 b = {self.b}"
     
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
            y_pred = np.sum([ self.remove_shift(x_i*w_i, config['precision']) 
                              for x_i, w_i in zip(x, self.w) ]) + self.b
            thresh = self.add_shift(0.5, config['precision'])
            if (y_pred >= thresh and y == config['precision']) or \
                (y_pred <  thresh and y == 0):
                no_correct += 1
        return no_correct / len(dataset.X_test)

    def train(self, config, dataset, weights, deltas=None):
        self.w = [ w_i for w_i in weights[:-1] ]
        self.b = weights[-1]
        if deltas is not None:
            self._deltas = deltas
        else:
            self._deltas = np.zeros((len(dataset), dataset.no_features+1))
        for epoch in range(config['epochs']):
            loss = 0
            for sample_idx in range(len(dataset)):
                # init accumulators
                dw = np.zeros(dataset.no_features).tolist()
                db = 0
                # get data sample
                x = dataset.X[sample_idx]
                y = dataset.Y[sample_idx]
                # forward
                y_pred = np.sum([ self.remove_shift(x_i*w_i, config['precision']) 
                                  for x_i, w_i in zip(x, self.w) ]) + self.b
                # loss
                loss += 1/2*(y_pred - y)**2/config['precision']
                # backward
                dy_pred = y_pred - y
                for i in range(len(dw)):
                    dw[i] += self.remove_shift(dy_pred * x[i], config['precision'])
                db += dy_pred
                # update
                for i in range(len(self.w)):
                    delta = self.remove_shift(self.add_shift(config['lr'], config['precision'])*dw[i], config['precision'])
                    self.w[i] -= delta
                    self._deltas[sample_idx][i] += delta
                delta = self.remove_shift(self.add_shift(config['lr'], config['precision'])*db, config['precision'])
                self.b -= delta
                self._deltas[sample_idx][-1] += delta
            print(f'[{epoch:>2}] loss: {loss/len(dataset)/config["precision"]:.5f} acc: {self.score(dataset, config):.2f}')

    def amnesiac(self, deltas):
        for delta in deltas:
            for i in range(len(self.w)):
                self.w[i] += delta[i] 
            self.b += delta[-1]        

    def optimization_unlearning(self, config, dataset, weights):
        self.w = [ w_i for w_i in weights[:-1] ]
        self.b = weights[-1]
        for epoch in range(config['unlearning_epochs']):
            loss = 0
            for sample_idx in range(len(dataset)):
                # init accumulators
                dw = np.zeros(dataset.no_features).tolist()
                db = 0
                # get data sample
                x = dataset.X[sample_idx]
                y = dataset.Y[sample_idx]
                # forward
                y_pred = np.sum([ self.remove_shift(x_i*w_i, config['precision']) 
                                  for x_i, w_i in zip(x, self.w) ]) + self.b
                # loss
                loss += 1/2*(y_pred - y)**2/config['precision']
                # backward
                dy_pred = y_pred - y
                for i in range(len(dw)):
                    dw[i] += self.remove_shift(dy_pred * x[i], config['precision'])
                db += dy_pred
                # update
                for i in range(len(self.w)):
                    self.w[i] += self.remove_shift(self.add_shift(config['lr'], config['precision'])*dw[i], config['precision'])
                self.b += self.remove_shift(self.add_shift(config['lr'], config['precision'])*db, config['precision'])


