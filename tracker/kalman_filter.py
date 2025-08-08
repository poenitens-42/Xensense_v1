import numpy as np

class KalmanFilter:
    def initiate(self, measurement):
        mean = measurement
        cov = np.eye(4) * 10
        return mean, cov

    def predict(self, mean, cov):
        return mean, cov

    def update(self, mean, cov, measurement):
        new_mean = 0.8 * mean + 0.2 * measurement
        return new_mean, cov
