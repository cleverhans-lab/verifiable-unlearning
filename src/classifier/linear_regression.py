
import numpy as np

class LinearRegression:

    def __init__(self):
        self.w = None
        self.b = None

    @property
    def weights(self):
        return self.w + [self.b]

    def no_weights(self, dataset, bias=True):
        if bias:
            return dataset.no_features+1
        else:
            return dataset.no_features

    def format_weights_init(self, config, dataset, weights_init):
        weights = [ self.add_shift(w, config['precision']) for w in weights_init ]
        return f"u64[{dataset.no_features}] mut w = {weights};\n u64 mut b = 0;\n"
     
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
                # loss
                loss += 1/2*(y_pred - y)**2/config['precision']
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
