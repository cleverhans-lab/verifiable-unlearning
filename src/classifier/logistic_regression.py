import numpy as np


class LogisticRegression:

    def __init__(self):
        self.w = None
        self.b = None

    def sigmoid(self, x, shift):
        assert x < self.add_shift(8, shift) and x > self.add_shift(-8, shift), x
        W0 = self.add_shift(0.50, shift)
        W1 = self.add_shift(0.1501, shift)
        W3 = self.add_shift(0.0016, shift)
        v = W0 + self.remove_shift(W1*x, shift) - self.remove_shift(W3*self.remove_shift(x*self.remove_shift(x*x, shift), shift), shift)
        return v

    @property
    def weights(self):
        return self.w + [self.b]

    def format_weights_init(self, config, dataset, weights_init):
        weights = [ self.add_shift(w, config['precision']) for w in weights_init ]
        return f"u64[{dataset.no_features}] mut w = {weights};\n u64 mut b = 0;\n"

    def no_weights(self, dataset):
        return dataset.no_features

    def no_weights(self, dataset, bias=True):
        if bias:
            return dataset.no_features+1
        else:
            return dataset.no_features

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
            y_pred = self.sigmoid(y_pred, config['precision'])
            thresh = self.add_shift(0.5, config['precision'])
            if (y_pred >= thresh and y == config['precision']) or \
                (y_pred <  thresh and y == 0):
                no_correct += 1
        return no_correct / len(dataset.X_test)

    def train(self, config, dataset, weights_init):
        self.w = [ self.add_shift(w_i, config['precision']) for w_i in weights_init ]
        self.b = 0
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
                y_pred = self.sigmoid(y_pred, config['precision'])
                # loss
                loss += - (  y/config['precision'] *np.log(  y_pred/config['precision']) \
                        + (1-y/config['precision'])*np.log(1-y_pred/config['precision']))
                # backward
                dy_pred = y_pred - y
                for i in range(len(dw)):
                    dw[i] += self.remove_shift(dy_pred * x[i], config['precision'])
                db += dy_pred
                # update
                for i in range(len(self.w)):
                    self.w[i] -= self.remove_shift(self.add_shift(config['lr'], config['precision'])*dw[i], config['precision'])
                self.b -= self.remove_shift(self.add_shift(config['lr'], config['precision'])*db, config['precision'])
            print(f'[{epoch:>2}] loss: {loss/len(dataset)/config["precision"]:.5f} acc: {self.score(dataset, config):.2f}')
