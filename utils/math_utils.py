import numpy as np

# ===========================================================
# Lightweight Kalman filter for position-velocity prediction
# ===========================================================
class Kalman2D:
    def __init__(self, process_var=1e-3, measure_var=1e-1):
        # State: [x, y, vx, vy]
        self.x = np.zeros((4, 1))
        self.P = np.eye(4) * 1.0
        self.F = np.eye(4)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        self.Q = np.eye(4) * process_var
        self.R = np.eye(2) * measure_var
        self.initialized = False

    def predict(self, dt=1.0):
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        return self.x[:2].ravel()

    def update(self, z):
        z = np.array(z).reshape(2, 1)
        if not self.initialized:
            self.x[0, 0], self.x[1, 0] = z[0, 0], z[1, 0]
            self.initialized = True
            return
        y = z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x += np.dot(K, y)
        I = np.eye(4)
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

# ===========================================================
# Helper Functions
# ===========================================================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def exponential_smooth(prev, new, alpha=0.7):
    if prev is None:
        return new
    return alpha * prev + (1 - alpha) * new

def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
