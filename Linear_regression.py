import numpy as np
import pandas as pd


class LinearRegression_GradientDescent:

    def __init__(self, learning_rate, early_stopping, weight=0, bias=0, verbose=True):

        self.weight = weight
        self.bias = bias
        self.learning_rate = learning_rate
        self.early_stopping = early_stopping
        self.verbose = verbose

    # fitting the algorithm the dataset

    def fit(self, X, y, epochs):

        weight = np.zeros(X.shape[1])
        bias = 0
        stop = 1
        previous_mse = -100

        for epoch in range(epochs):

            predictii = []
            nr = []
            gradient = [0 for x in X.columns]
            bias_gradient = 0

            for i in X.index:
                predictie = 0

                for j in range(len(X.columns)):
                    predictie += weight[j]*X.loc[i][j]

                predictii.append(predictie+bias)

            bias_gradient, gradient = self._update(y, predictii, gradient, X)

            bias += bias_gradient*self.learning_rate

            for i in range(len(gradient)):
                weight[i] += self.learning_rate*gradient[i]

            if self.early_stopping:
                mean_se = self._loss(y, predictii)
                stop = self._EarlyStopping(
                    stop=stop, mean_se=mean_se, previous_mse=previous_mse)

                previous_mse = mean_se

                if self.verbose:
                    print("Epoch: ", epoch+1, " completed")

                if not stop:
                    break

        self.weight = weight
        self.bias = bias

    # predicting the values after training

    def predict(self, X):

        predictii = []

        for z in X.index:
            calc = 0

            for i in range(len(X.columns)):
                calc += X.loc[z][i]*self.weight[i]

            predictii.append(calc+self.bias)

        return predictii

    # updateing the weights

    def _update(self, actual, pred, gradient, X):
        bias_gradient = 0
        lungime = len(pred)
        nr = 0

        for i in X.index:
            for j in range(len(gradient)):
                gradient[j] += (actual[i]-pred[nr])*X.loc[i][j]

            bias_gradient += (actual[i]-pred[nr])

            nr += 1

        for i in range(len(gradient)):
            gradient[i] = gradient[i]/lungime*2

        return bias_gradient/lungime*2, gradient

    # Stopping

    def _EarlyStopping(self, mean_se, previous_mse, min_delta=0.001, wait=5, stop=1):

        if min_delta > np.abs(mean_se-previous_mse):
            stop += 1

        else:
            stop = 1

        if (stop > wait):
            return 0

        return stop

    # calculationg the loss

    def _loss(self, y, predictii):
        sum = 0
        nr = 0

        for i in y:
            sum += (i - predictii[nr])**2
            nr += 1

        return sum/len(y)
