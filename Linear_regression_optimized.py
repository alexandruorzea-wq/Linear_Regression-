import numpy as np
import pandas as pd


class LinearRegression_GradientDescent_optimized:

    def __init__(self, learning_rate, early_stopping, weight=0, bias=0, verbose=True):

        self.weight = weight
        self.bias = bias
        self.learning_rate = learning_rate
        self.early_stopping = early_stopping
        self.verbose = verbose

    # fitting the algorithm the dataset

    def fit(self, X, y, epochs):
        X_array = X.to_numpy()
        y_array = y.to_numpy()

        weight = np.zeros(X_array.shape[1])
        bias = 0

        stop = 1
        previous_mse = -100

        for epoch in range(epochs):

            # Calculate all predictions at once
            predictions = X_array @ weight + bias

            # Calculate gradients
            bias_gradient, gradient = self._update(
                y_array,
                predictions,
                X_array
            )

            # Update parameters
            bias += self.learning_rate * bias_gradient
            weight += self.learning_rate * gradient

            if self.early_stopping:
                mean_se = self._loss(y_array, predictions)

                stop = self._EarlyStopping(
                    stop=stop,
                    mean_se=mean_se,
                    previous_mse=previous_mse
                )

                previous_mse = mean_se

                if self.verbose:
                    print("Epoch:", epoch + 1, "completed")

                if not stop:
                    break

        self.weight = weight
        self.bias = bias

    # predicting the values after training

    def predict(self, X):
        X_array = X.to_numpy()

        predictions = X_array @ self.weight + self.bias

        return predictions

    # updateing the weights

    def _update(self, actual, pred, X):

        errors = actual - pred

        gradient = (2 / len(X)) * (X.T @ errors)

        bias_gradient = (2 / len(X)) * np.sum(errors)

        return bias_gradient, gradient

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

    def _loss(self, y, predictions):
        return np.mean((y - predictions) ** 2)
